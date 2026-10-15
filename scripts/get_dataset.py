"""Get part of the released global dataset (solar_suitability v1.0, AEF 2025): a country or any area, without
downloading the whole ~850 GB archive.

    python scripts/get_dataset.py --country Kenya                     # raw scores, clipped to the country
    python scripts/get_dataset.py --country KEN --percentile          # + within-country percentile view
    python scripts/get_dataset.py --bbox 23.5 37.8 24.2 38.2          # any area (percentiles: global table)
    python scripts/get_dataset.py --region-file area.geojson
    python scripts/get_dataset.py --country Chile --whole-tiles       # whole tiles as archived (no clipping)
    python scripts/get_dataset.py --country Kenya --list               # just list the files it would fetch

Reads the dataset's tile index (index/tiles.parquet, ~60 MB, cached), selects the tiles intersecting the region,
reads only the needed windows over HTTPS (or copies whole tiles with --whole-tiles), and writes to
outputs/dataset_<region>/ the same layout as scripts/predict.py: tiles/<zone>/*.tif (uint8, P = value * 0.004,
255 = no data), mosaic_<zone>.vrt, mosaic_<zone>_pct.vrt (with --percentile), quicklook.png, run.json.

Percentiles come from the dataset's own lookup tables: within-country (lookup/percentiles_by_country.csv) for
--country, global (lookup/percentiles_global.csv) otherwise. They are applied on the fly (VRT lookup tables), so they
take no extra disk space. Use --root to read a local or mirrored copy of the dataset.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

# TODO(release): replace with the CEDA HTTPS root of solar_suitability_v1.0 once the DOI is minted.
DEFAULT_ROOT = "https://data.ceda.ac.uk/PLACEHOLDER/solar_suitability_v1.0"


def is_url(p):
    return p.startswith("http://") or p.startswith("https://")


def fetch_to(root, rel, dst):
    from solarsiting.aef import _download
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if is_url(root):
        _download(root.rstrip("/") + "/" + rel, dst)
    else:
        import shutil
        shutil.copyfile(os.path.join(root, rel), dst + ".part"); os.replace(dst + ".part", dst)


def gdal_path(root, rel):
    return ("/vsicurl/" + root.rstrip("/") + "/" + rel) if is_url(root) else os.path.join(root, rel)


def load_index(root, refresh=False):
    import geopandas as gpd
    from solarsiting.aef import cache_dir
    key = "".join(ch if ch.isalnum() else "_" for ch in root)[-80:]
    p = os.path.join(cache_dir(), f"dataset_index_{key}.parquet")
    if refresh or not os.path.exists(p):
        print("downloading the dataset tile index (once) ...", flush=True)
        fetch_to(root, "index/tiles.parquet", p)
    return gpd.read_parquet(p)


def lut(root, iso3=None):
    """code (0-250) -> percentile, from the dataset's lookup tables (within-country if iso3 has a table)."""
    import pandas as pd
    from solarsiting.aef import cache_dir
    name = "percentiles_by_country.csv" if iso3 else "percentiles_global.csv"
    p = os.path.join(cache_dir(), "dataset_" + name)
    if not os.path.exists(p):
        fetch_to(root, f"lookup/{name}", p)
    d = pd.read_csv(p)
    if iso3:
        d = d[d.iso3 == iso3]
        if d.empty:
            return None, None
        kind = str(d.table.iloc[0])
    else:
        kind = "global"
    t = np.zeros(251); t[d.code.values.astype(int)] = d.percentile.values
    return t, kind


def clip_tile(root, row, region, out):
    """Read only the window of a released tile covering `region`, set pixels outside it to no data, write a COG."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from rasterio.windows import Window
    from solarsiting.aef import gdal_env
    from solarsiting.infer import _window_for, _write_cog
    from solarsiting.model import NODATA
    gdal_env()
    with rasterio.open(gdal_path(root, row.file)) as s:
        reg = gpd.GeoSeries([region], crs=4326).to_crs(s.crs).iloc[0]
        win = _window_for(reg, s.transform, s.height, s.width)
        if win is None:
            return 0
        r0, r1, c0, c1 = win
        a = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
        wtr = s.transform * Affine.translation(c0, r0)
        inside = rasterize([(reg, 1)], out_shape=a.shape, transform=wtr, fill=0, all_touched=True, dtype="uint8").astype(bool)
        a = np.where(inside, a, NODATA).astype(np.uint8)
        n = int((a != NODATA).sum())
        if n:
            tags = dict(s.tags(), CLIPPED_TO="region requested with scripts/get_dataset.py")
            _write_cog(out, a, s.crs, wtr, tags)
        return n


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("region (give exactly one)")
    g.add_argument("--country", help="country name or ISO3 code")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    g.add_argument("--region-file", help="vector file; features are unioned")
    ap.add_argument("--percentile", action="store_true", help="also write percentile views (within-country for --country)")
    ap.add_argument("--whole-tiles", action="store_true", help="copy whole archived tiles instead of clipping")
    ap.add_argument("--list", action="store_true", help="only list the tiles that would be fetched")
    ap.add_argument("--out", help="output folder (default: outputs/dataset_<region>)")
    ap.add_argument("--root", default=os.environ.get("SOLARSITING_DATASET", DEFAULT_ROOT),
                    help="dataset root: the CEDA URL (default), a mirror URL, or a local folder")
    ap.add_argument("--refresh-index", action="store_true")
    a = ap.parse_args()
    if "PLACEHOLDER" in a.root:
        sys.exit("The dataset's CEDA link is not public yet; it will be set here and in the README at release.\n"
                 "If you have a copy (local folder or mirror URL), pass --root <folder or URL>.")
    import shapely
    from solarsiting import regions
    from solarsiting.infer import _quicklook, _vrt

    region, label = regions.resolve(a.country, a.bbox, a.region_file)
    iso3 = None
    if a.country:
        g_ = regions._countries(); key = a.country.strip().lower()
        for c in ("ADM0_A3", "ISO_A3", "ADMIN", "NAME", "NAME_LONG"):
            hit = g_[g_[c].astype(str).str.lower() == key]
            if len(hit):
                iso3 = str(hit.iloc[0]["ADM0_A3"]); break
    label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label.lower()).strip("_")
    idx = load_index(a.root, a.refresh_index)
    w, s, e, n = region.bounds
    cand = idx[(idx.lon_max >= w) & (idx.lon_min <= e) & (idx.lat_max >= s) & (idx.lat_min <= n)]
    sel = cand[shapely.intersects(cand.geometry.values, region)].sort_values(["zone", "tile_id"])
    gb = sel.bytes.sum() / 1e9
    print(f"region: {label}{f' ({iso3})' if iso3 else ''} | {len(sel)} archived tile(s) "
          f"| {gb:.1f} GB as whole tiles{'' if a.whole_tiles else ' (only the clipped part is read)'}")
    if a.list or sel.empty:
        for r in sel.itertuples():
            print(f"  {r.file}  ({r.bytes / 1e6:.0f} MB)")
        return
    out = a.out or os.path.join("outputs", f"dataset_{label}")
    t0, files, npx = time.time(), [], 0
    for k, r in enumerate(sel.itertuples(), 1):
        dst = os.path.join(out, "tiles", r.zone, os.path.basename(r.file))
        if not os.path.exists(dst):
            if a.whole_tiles:
                fetch_to(a.root, r.file, dst); m = int(r.n_valid)
            else:
                m = clip_tile(a.root, r, region, dst)
        else:
            m = -1
        if os.path.exists(dst):
            files.append(dst); npx += max(m, 0)
        print(f"  [{k}/{len(sel)}] {r.zone}/{r.tile_id}: " + ("already present" if m < 0 else f"{m / 1e6:.1f} M px") +
              f" | {(time.time() - t0) / 60:.1f} min")
    if not files:
        sys.exit("no released pixels inside this region (all sea, or outside the land mask)")
    table, kind = (lut(a.root, iso3) if (a.percentile and iso3) else (lut(a.root) if a.percentile else (None, None)))
    if a.percentile and table is None:
        table, kind = lut(a.root); print(f"  no within-country table for {iso3}; using the global percentiles")
    by_zone = {}
    for f in files:
        by_zone.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    for zone, fs in by_zone.items():
        _vrt(fs, os.path.join(out, f"mosaic_{zone}.vrt"))
        if table is not None:
            _vrt(fs, os.path.join(out, f"mosaic_{zone}_pct.vrt"), pct_lut=table)
    _quicklook(files, region, os.path.join(out, "quicklook.png"))
    json.dump(dict(source="solar_suitability v1.0 (AEF 2025)", root=a.root, region=label, iso3=iso3, tiles=len(files),
                   whole_tiles=a.whole_tiles, percentile_table=kind, minutes=round((time.time() - t0) / 60, 1)),
              open(os.path.join(out, "run.json"), "w"), indent=1)
    print(f"done: {len(files)} tile(s) -> {out}" + (f" | percentiles: {kind} table" if table is not None else ""))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:     # input mistakes: a clear message, not a traceback
        sys.exit(f"error: {e}")
