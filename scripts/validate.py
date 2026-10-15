"""Forward validation of a suitability map: do solar sites built AFTER the map's AEF year score high?

    python scripts/predict.py  --country Greece --year 2019
    python scripts/validate.py --run outputs/greece_2019 --sites data/validation/sites_greece.csv

As in the paper: the map uses AEF year Y; validation sites are those installed in year >= Y + 2 (so they were absent
from the scored imagery, matching the two-year lag the model was trained with); each is compared with randomly
sampled pixels of the same region (random land). Reports ROC-AUC (site vs random land), the share of sites in the
most-suitable 20 / 10 / 5 % of the region, the median site percentile, and a score-distribution figure.

Sites CSV columns: lon, lat, year (WGS84). Bundled: data/validation/sites_<country>.csv for the paper's 13 countries
(from the GPVF inventory, CC-BY-4.0; see data/README.md). Any inventory in that format works, e.g. TZ-SAM points.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def sample_points(vrts, lon, lat):
    """Suitability codes at lon/lat points from per-UTM-zone mosaics (255 where outside every mosaic)."""
    import rasterio
    from pyproj import Transformer
    out = np.full(len(lon), 255, np.uint8)
    for v in vrts:
        with rasterio.open(v) as s:
            x, y = Transformer.from_crs(4326, s.crs, always_xy=True).transform(lon, lat)
            r, c = rasterio.transform.rowcol(s.transform, x, y)
            r, c = np.asarray(r), np.asarray(c)
            ok = (r >= 0) & (r < s.height) & (c >= 0) & (c < s.width) & (out == 255)
            if ok.any():
                vals = np.array([v_[0] for v_ in s.sample(zip(np.asarray(x)[ok], np.asarray(y)[ok]))], np.uint8)
                out[np.nonzero(ok)[0]] = vals
    return out


def random_land(tiles, n, seed=0):
    """n codes drawn uniformly from all valid pixels of the run (area-weighted across tiles)."""
    import rasterio
    rng = np.random.default_rng(seed); arrays = []
    for f in tiles:
        with rasterio.open(f) as s:
            a = s.read(1); arrays.append(a[a != 255])
    allv = np.concatenate(arrays)
    return allv[rng.choice(len(allv), min(n, len(allv)), replace=False)]


def main():
    sys.stdout.reconfigure(line_buffering=True)          # progress shows up live, also in log files
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output folder of scripts/predict.py")
    ap.add_argument("--sites", required=True, help="CSV with lon, lat, year")
    ap.add_argument("--min-year", type=int, help="earliest installation year to validate on (default: AEF year + 2)")
    ap.add_argument("--n-random", type=int, default=20000, help="random land pixels (default 20,000)")
    a = ap.parse_args()
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from solarsiting.model import SCALE

    info = json.load(open(os.path.join(a.run, "run.json")))
    year = info["aef_year"]; min_year = a.min_year or year + 2
    vrts = sorted(glob.glob(os.path.join(a.run, "mosaic_*.vrt")))
    vrts = [v for v in vrts if not v.endswith("_pct.vrt")]
    tiles = sorted(glob.glob(os.path.join(a.run, "tiles", "*", "*.tif")))
    S = pd.read_csv(a.sites)
    S = S[S.year >= min_year]
    site = sample_points(vrts, S.lon.values, S.lat.values)
    site = site[site != 255]
    if len(site) < 5:
        sys.exit(f"only {len(site)} sites installed >= {min_year} fall inside the mapped region; need more for a meaningful test")
    rand = random_land(tiles, a.n_random)
    y = np.r_[np.ones(len(site)), np.zeros(len(rand))]; x = np.r_[site, rand].astype(float)
    roc = roc_auc_score(y, x)
    pct = np.array([(rand < s).mean() + 0.5 * (rand == s).mean() for s in site]) * 100     # site percentile in the region
    cap = {k: float((pct >= 100 - k).mean()) for k in (20, 10, 5)}
    print(f"{info['region']} | AEF {year} map | {len(site)} sites installed >= {min_year} vs {len(rand):,} random pixels")
    print(f"  ROC-AUC {roc:.3f} | median site percentile {np.median(pct):.0f} | "
          f"sites in top 20% / 10% / 5% of the region: {cap[20]:.0%} / {cap[10]:.0%} / {cap[5]:.0%}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2)); bins = np.linspace(0, 1, 26)
    ax.hist(rand * SCALE, bins, density=True, alpha=0.75, color="#3b0f70", label="random land")
    ax.hist(site * SCALE, bins, density=True, alpha=0.7, color="#fcac6d", label=f"sites installed {min_year}+ (n={len(site)})")
    ax.set_xlabel("P(solar)"); ax.set_yticks([]); ax.legend(frameon=False)
    ax.set_title(f"{info['region']}, AEF {year}: ROC-AUC {roc:.3f}", fontsize=10)
    fig.tight_layout(); f = os.path.join(a.run, "validation.png"); fig.savefig(f, dpi=150); plt.close(fig)
    json.dump(dict(region=info["region"], aef_year=year, min_install_year=min_year, sites=int(len(site)), random=int(len(rand)),
                   roc_auc=round(float(roc), 4), median_site_percentile=round(float(np.median(pct)), 1),
                   top20=round(cap[20], 4), top10=round(cap[10], 4), top5=round(cap[5], 4), sites_file=a.sites),
              open(os.path.join(a.run, "validation.json"), "w"), indent=1)
    print(f"  -> {f}, validation.json")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:     # input mistakes: a clear message, not a traceback
        sys.exit(f"error: {e}")
