"""Access to the public AlphaEarth Foundations (AEF) annual embeddings, with no Earth Engine account needed.

The AEF Satellite Embedding dataset (Google / Google DeepMind) is published as Cloud-Optimised GeoTIFFs: one file per
UTM tile and year (2017 onwards), 64 int8 bands at 10 m, mirrored on Google Cloud Storage and on Source Cooperative.
A tile index (GeoParquet, ~80 MB) lists every file with its footprint; it is downloaded once and cached.

Pixels are read with HTTP range requests (GDAL /vsicurl/), so only the parts of a tile that cover your region are
transferred. Values are de-quantised exactly as Earth Engine serves them:  x = sign(q) * (q / 127.5)^2,  q = -128 = no data.
Note: the COGs are stored bottom-up (positive y pixel size); `read_window` returns north-up arrays and transforms.
"""
import os
import time

import numpy as np

INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
_SC_S3 = "s3://us-west-2.opendata.source.coop/tge-labs/aef/"
MIRRORS = ("https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/",   # Google Cloud Storage
           "https://data.source.coop/tge-labs/aef/")                                      # Source Cooperative
YEARS = range(2017, 2026)            # years available when this release was made (check the index for newer ones)
NBANDS, QNODATA = 64, -128
ATTRIBUTION = "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."


def cache_dir():
    d = os.environ.get("SOLARSITING_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "solarsiting"))
    os.makedirs(d, exist_ok=True)
    return d


def gdal_env():
    """GDAL settings for fast, robust reads of remote COGs (set once per process)."""
    # The AEF COGs store each of the 64 bands separately, so one pixel block = 64 HTTP ranges. MULTIRANGE=PARALLEL fetches
    # them concurrently (13x faster than one-by-one on a home connection: ~2.5 s instead of ~33 s per 1024 x 1024 block).
    for k, v in dict(GDAL_HTTP_MULTIRANGE="PARALLEL", GDAL_NUM_THREADS="ALL_CPUS",
                     GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif",
                     GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_VERSION="2",
                     GDAL_HTTP_MAX_RETRY="8", GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_RETRY_CODES="ALL", GDAL_HTTP_USERAGENT=USER_AGENT, VSI_CACHE="TRUE", VSI_CACHE_SIZE="33554432",
                     GDAL_CACHEMAX="128").items():
        os.environ.setdefault(k, v)


USER_AGENT = "solarsiting/1.0 (+https://github.com/Abdullahr29/Global-Solar-Siting)"   # some hosts reject Python's default


def _download(url, dst):
    import urllib.request
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    os.replace(tmp, dst)


def load_index(year=None, refresh=False):
    """The AEF tile index as a DataFrame (columns: crs, path, year, utm_zone, wgs84_* bounds, geom [WKB, EPSG:4326])."""
    import pandas as pd
    p = os.path.join(cache_dir(), "aef_index.parquet")
    if refresh or not os.path.exists(p):
        print(f"downloading the AEF tile index (~80 MB, once) -> {p}", flush=True)
        _download(INDEX_URL, p)
    cols = ["crs", "path", "year", "utm_zone", "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north", "geom"]
    d = pd.read_parquet(p, columns=cols)
    if year is not None:
        d = d[d.year == int(year)]
        if d.empty:
            have = sorted(pd.read_parquet(p, columns=["year"]).year.unique())
            raise ValueError(f"AEF year {year} not in the index (available: {have[0]}-{have[-1]}); "
                             f"run with --refresh-index if a newer year has just been published")
    return d.reset_index(drop=True)


def tiles_for(region, year):
    """Index rows of the AEF tiles that intersect `region` (shapely geometry in EPSG:4326) for `year`."""
    import shapely
    d = load_index(year)
    w, s, e, n = region.bounds
    d = d[(d.wgs84_east >= w) & (d.wgs84_west <= e) & (d.wgs84_north >= s) & (d.wgs84_south <= n)]
    if d.empty:
        return d
    geoms = shapely.from_wkb(d.geom.values)
    keep = shapely.intersects(geoms, region)
    return d[keep].reset_index(drop=True)


def tile_urls(path):
    rel = path[len(_SC_S3):] if path.startswith(_SC_S3) else path
    return [m + rel for m in MIRRORS]


def tile_id(path):
    return os.path.basename(path).rsplit(".", 1)[0]


def open_tile(path, tries=3):
    """Open a remote AEF tile, falling back to the second mirror if the first fails."""
    import rasterio
    gdal_env()
    err = None
    for k in range(tries):
        for url in tile_urls(path):
            try:
                return rasterio.open("/vsicurl/" + url)
            except Exception as e:          # network hiccup or mirror outage: try the other mirror, then retry
                err = e
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"could not open AEF tile {path}: {err}")


def dequantise(q):
    """int8 AEF codes (pixels, 64), all valid -> float32 embeddings, exactly as Earth Engine serves them."""
    v = np.asarray(q).astype(np.float32)
    return np.sign(v) * (v / 127.5) ** 2


def north_up(src):
    """(flip, transform) for a dataset: AEF COGs are bottom-up, outputs are north-up."""
    from rasterio.transform import Affine
    tr = src.transform
    if tr.e > 0:
        return True, Affine(tr.a, tr.b, tr.c, tr.d, -tr.e, tr.f + tr.e * src.height)
    return False, tr
