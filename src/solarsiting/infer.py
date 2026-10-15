"""Run the released model on any region and AEF year, straight from the public AEF tiles (no Earth Engine).

For every AEF tile intersecting the region: read only the 1024x1024-pixel blocks that overlap it (HTTP range reads),
de-quantise, predict P(solar) for pixels inside the region, and write a north-up uint8 Cloud-Optimised GeoTIFF in the
tile's native UTM projection at 10 m, encoded exactly like the released dataset (P = code * 0.004, 255 = no data).

Outputs in <out>/:
  tiles/<utm_zone>/<label>_<year>_<tile>.tif   per-tile suitability (only the window covering the region)
  mosaic_<utm_zone>.vrt                        one virtual mosaic per UTM zone (open directly in QGIS / rasterio)
  mosaic_<utm_zone>_pct.vrt                    the same, as within-region percentiles (0-100; see percentiles.csv)
  percentiles.csv                              code -> P(solar) -> share of the region's pixels it outscores
  quicklook.png                                lon/lat preview of the whole region
  run.json                                     inputs, model, timings and pixel counts (the run is resumable)
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import aef
from .model import NODATA, SCALE, encode_uint8

BLOCK = 1024


def _window_for(region_tile, tr, H, W):
    """North-up pixel window (r0, r1, c0, c1) covering the region's bounds in a tile, or None."""
    minx, miny, maxx, maxy = region_tile.bounds
    c0 = int(np.floor((minx - tr.c) / tr.a)); c1 = int(np.ceil((maxx - tr.c) / tr.a))
    r0 = int(np.floor((tr.f - maxy) / -tr.e)); r1 = int(np.ceil((tr.f - miny) / -tr.e))
    r0, r1, c0, c1 = max(r0, 0), min(r1, H), max(c0, 0), min(c1, W)
    return (r0, r1, c0, c1) if r1 > r0 and c1 > c0 else None


def _write_cog(path, arr, crs, transform, tags):
    import rasterio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part.tif"
    with rasterio.open(tmp, "w", driver="COG", width=arr.shape[1], height=arr.shape[0], count=1, dtype="uint8",
                       nodata=NODATA, crs=crs, transform=transform, compress="DEFLATE", predictor=2, blocksize=512,
                       overviews="AUTO", overview_resampling="average", BIGTIFF="IF_SAFER") as d:
        d.write(arr, 1)
        d.scales, d.offsets = (SCALE,), (0.0,)
        d.set_band_description(1, "suitability")
        d.update_tags(**tags)
    os.replace(tmp, path)


def _file_blocks(H, W, flip, win):
    """The window split along the file's own 1024 x 1024 block grid, in north-up pixel coordinates (a0, a1, b0, b1)."""
    r0, r1, c0, c1 = win
    out = []
    for fr in range(0, H, BLOCK):
        bh = min(BLOCK, H - fr)
        top = (H - fr - bh) if flip else fr                    # north-up row of this file block's top edge
        a0, a1 = max(top, r0), min(top + bh, r1)
        if a1 <= a0:
            continue
        for fc in range(0, W, BLOCK):
            b0, b1 = max(fc, c0), min(fc + min(BLOCK, W - fc), c1)
            if b1 > b0:
                out.append((a0, a1, b0, b1))
    return out


def predict_tile(model, row, region, label, year, out_dir, io_threads=8):
    """Score one AEF tile over the part of `region` it covers. Returns a stats dict (skips if already written)."""
    import threading
    from collections import deque

    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from rasterio.windows import Window

    tid = aef.tile_id(row.path); zone = row.utm_zone
    out = os.path.join(out_dir, "tiles", zone, f"{label}_{year}_{tid}.tif")
    if os.path.exists(out):
        return dict(tile=tid, zone=zone, file=os.path.relpath(out, out_dir), skipped=True)
    t0 = time.time()
    src = aef.open_tile(row.path)
    flip, tr = aef.north_up(src); H, W = src.height, src.width; crs = src.crs
    reg = gpd.GeoSeries([region], crs=4326).to_crs(crs).iloc[0]
    win = _window_for(reg, tr, H, W)
    if win is None:
        src.close()
        return dict(tile=tid, zone=zone, file=None, n_valid=0, note="region misses tile pixels")
    r0, r1, c0, c1 = win; h, w = r1 - r0, c1 - c0
    wtr = tr * Affine.translation(c0, r0)
    inside = rasterize([(reg, 1)], out_shape=(h, w), transform=wtr, fill=0, all_touched=True, dtype="uint8").astype(bool)

    # Which blocks hold AEF data at all: one cheap read of band 1 from a 1/16 overview (no data is -128 in every band).
    f = 16
    ov = src.read(1, out_shape=(-(-H // f), -(-W // f)))
    if flip:
        ov = ov[::-1]
    has = ov != aef.QNODATA
    src.close()
    blocks = [b for b in _file_blocks(H, W, flip, win)
              if inside[b[0] - r0:b[1] - r0, b[2] - c0:b[3] - c0].any() and has[b[0] // f:-(-b[1] // f), b[2] // f:-(-b[3] // f)].any()]

    local, handles = threading.local(), []

    def read(b):                                       # one north-up block as (64, bh, bw) int8; own connection per thread
        if not hasattr(local, "src"):
            local.src = aef.open_tile(row.path); handles.append(local.src)
        a0, a1, b0, b1 = b
        frow = (H - a1) if flip else a0                # bottom-up file: north-up row r is file row H-1-r
        a = local.src.read(window=Window(b0, frow, b1 - b0, a1 - a0))
        return a[:, ::-1, :] if flip else a

    codes = np.full((h, w), NODATA, np.uint8)
    n_valid = 0
    with ThreadPoolExecutor(max(1, io_threads)) as io:  # keep io_threads blocks downloading while the model predicts
        pending, it = deque(), iter(blocks)
        for b in it:
            pending.append((b, io.submit(read, b)))
            if len(pending) >= io_threads:
                break
        while pending:
            (a0, a1, b0, b1), fut = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, io.submit(read, nxt)))
            a = fut.result(); bh, bw = a.shape[1:]
            q = a.reshape(aef.NBANDS, -1).T                                   # (pixels, 64) int8
            ok = (q != aef.QNODATA).all(1) & inside[a0 - r0:a1 - r0, b0 - c0:b1 - c0].ravel()
            if ok.any():
                p = np.full(bh * bw, np.nan, np.float32)
                p[ok] = model.predict(aef.dequantise(q[ok]))
                codes[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = encode_uint8(p).reshape(bh, bw)
                n_valid += int(ok.sum())
    for s in handles:
        s.close()
    if n_valid == 0:
        return dict(tile=tid, zone=zone, file=None, n_valid=0, note="no AEF data inside the region", sec=round(time.time() - t0))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tags = dict(PRODUCT="Solar PV siting suitability (10 m), user run", AEF_YEAR=str(year), REGION=label,
                SOURCE_AEF_TILE=row.path, MODEL=model.meta.get("name", ""), ENCODING="P(solar) = value * 0.004 ; 255 = no data",
                ATTRIBUTION=aef.ATTRIBUTION)
    _write_cog(out, codes, crs, wtr, tags)
    hist = np.bincount(codes[codes != NODATA], minlength=251)[:251]
    return dict(tile=tid, zone=zone, file=os.path.relpath(out, out_dir), n_valid=n_valid, crs=str(crs),
                hist=hist.tolist(), sec=round(time.time() - t0, 1))


def _vrt(paths, dst, pct_lut=None):
    """Virtual mosaic of same-CRS tiles (optionally Float32 percentiles through a LUT)."""
    import rasterio
    infos = []
    for p in paths:
        with rasterio.open(p) as s:
            infos.append((p, s.transform, s.width, s.height, s.crs))
    res = infos[0][1].a
    minx = min(t.c for _, t, _, _, _ in infos); maxy = max(t.f for _, t, _, _, _ in infos)
    maxx = max(t.c + w * t.a for _, t, w, _, _ in infos); miny = min(t.f + h * t.e for _, t, _, h, _ in infos)
    W, H = int(round((maxx - minx) / res)), int(round((maxy - miny) / res))
    dtype, nod = ("Float32", -1) if pct_lut is not None else ("Byte", NODATA)
    lut = "" if pct_lut is None else "<LUT>" + ",".join(f"{k}:{pct_lut[k]:.4f}" for k in range(251)) + "</LUT>"
    src_xml = []
    for p, t, w, h, _ in infos:
        xo, yo = int(round((t.c - minx) / res)), int(round((maxy - t.f) / res))
        src_xml.append(f'    <ComplexSource><SourceFilename relativeToVRT="1">{os.path.relpath(p, os.path.dirname(dst))}'
                       f'</SourceFilename><SourceBand>1</SourceBand><NODATA>{NODATA}</NODATA>{lut}'
                       f'<SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/><DstRect xOff="{xo}" yOff="{yo}" xSize="{w}" ySize="{h}"/>'
                       f'</ComplexSource>')
    scale = "" if pct_lut is not None else f"<Offset>0</Offset><Scale>{SCALE}</Scale>"
    xml = (f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">\n  <SRS>{infos[0][4].to_wkt()}</SRS>\n'
           f'  <GeoTransform>{minx}, {res}, 0, {maxy}, 0, {-res}</GeoTransform>\n'
           f'  <VRTRasterBand dataType="{dtype}" band="1">\n    <NoDataValue>{nod}</NoDataValue>{scale}\n'
           + "\n".join(src_xml) + "\n  </VRTRasterBand>\n</VRTDataset>\n")
    open(dst, "w").write(xml)


def _quicklook(paths, region, dst, width=1600):
    """Lon/lat preview of all tiles (read from their overviews), magma colour map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    w, s, e, n = region.bounds
    height = max(1, int(width * (n - s) / max(e - w, 1e-9) / max(np.cos(np.radians((n + s) / 2)), 0.2)))
    dtr = from_bounds(w, s, e, n, width, height)
    canvas = np.full((height, width), np.nan, np.float32)
    for p in paths:
        with rasterio.open(p) as src:
            f = max(1, int(max(src.width, src.height) / 2048))
            a = src.read(1, out_shape=(max(1, src.height // f), max(1, src.width // f)), resampling=Resampling.nearest)
            atr = src.transform * src.transform.scale(src.width / a.shape[1], src.height / a.shape[0])
            dst_a = np.full((height, width), np.nan, np.float32)
            reproject(np.where(a == NODATA, np.nan, a * SCALE).astype(np.float32), dst_a, src_transform=atr, src_crs=src.crs,
                      dst_transform=dtr, dst_crs="EPSG:4326", src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
            m = np.isfinite(dst_a); canvas[m] = dst_a[m]
    fig, ax = plt.subplots(figsize=(10, 10 * height / width))
    im = ax.imshow(canvas, cmap="magma", vmin=0, vmax=1, extent=(w, e, s, n), interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.6, label="P(solar)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.savefig(dst, dpi=150, bbox_inches="tight"); plt.close(fig)


def run(model, region, label, year, out_dir, tiles=None, io_threads=8, log=print):
    """Score `region` (EPSG:4326 geometry) for AEF `year` into `out_dir`. Resumable: finished tiles are skipped."""
    os.makedirs(out_dir, exist_ok=True)
    T = aef.tiles_for(region, year) if tiles is None else tiles
    if T.empty:
        raise ValueError("no AEF tiles intersect this region (is it all ocean, or outside the AEF coverage?)")
    log(f"{label} {year}: {len(T)} AEF tile(s) to process")
    rows, t0 = [], time.time()
    for k, row in enumerate(T.itertuples(), 1):
        if callable(model) and not hasattr(model, "predict"):       # lazy loader: load only when a tile needs scoring
            done = os.path.exists(os.path.join(out_dir, "tiles", row.utm_zone, f"{label}_{year}_{aef.tile_id(row.path)}.tif"))
            if not done:
                model = model()
        st = predict_tile(model, row, region, label, year, out_dir, io_threads=io_threads)
        rows.append(st)
        msg = "already done" if st.get("skipped") else (f"{st.get('n_valid', 0)/1e6:.1f} M px in {st.get('sec', 0):.0f}s"
                                                         if st.get("file") else st.get("note", ""))
        log(f"  [{k}/{len(T)}] {st['zone']}/{st['tile']}: {msg} | elapsed {(time.time()-t0)/60:.1f} min")
    # histograms of finished tiles (also for tiles skipped on a resumed run)
    import rasterio
    files = [os.path.join(out_dir, r["file"]) for r in rows if r.get("file")]
    if not files:
        raise RuntimeError("no pixels were scored (no AEF data inside the region for this year?)")
    hist = np.zeros(251, np.int64)
    for f in files:
        with rasterio.open(f) as s:
            a = s.read(1); hist += np.bincount(a[a != NODATA], minlength=251)[:251]
    below = np.r_[0, np.cumsum(hist)[:-1]]; pct = 100.0 * below / hist.sum()
    with open(os.path.join(out_dir, "percentiles.csv"), "w") as f:
        f.write("code,p_solar,percentile,pixel_share\n")
        for c in range(251):
            f.write(f"{c},{c * SCALE:.3f},{pct[c]:.4f},{hist[c] / hist.sum():.8f}\n")
    by_zone = {}
    for f in files:
        by_zone.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    for zone, fs in by_zone.items():
        _vrt(fs, os.path.join(out_dir, f"mosaic_{zone}.vrt"))
        _vrt(fs, os.path.join(out_dir, f"mosaic_{zone}_pct.vrt"), pct_lut=pct)
    _quicklook(files, region, os.path.join(out_dir, "quicklook.png"))
    name = model.meta.get("name", "") if hasattr(model, "meta") else "(all tiles already present)"
    info = dict(region=label, aef_year=int(year), model=name, tiles=len(T),
                pixels_scored=int(hist.sum()), area_km2=round(hist.sum() * 1e-4, 1),
                minutes=round((time.time() - t0) / 60, 1), tile_stats=[{k: v for k, v in r.items() if k != "hist"} for r in rows])
    json.dump(info, open(os.path.join(out_dir, "run.json"), "w"), indent=1)
    log(f"done: {info['pixels_scored']/1e6:.1f} M pixels ({info['area_km2']:,} km²) in {info['minutes']} min -> {out_dir}")
    return info
