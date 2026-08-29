"""Country-scale solar-suitability map with the released random forest, computed
entirely inside Google Earth Engine (no imagery download).

The trained scikit-learn forest is injected into Earth Engine as decision-tree
strings (via geemap), used to classify a whole country's AlphaEarth annual mosaic
server-side, and then validated against real solar sites that were installed *after*
the imagery year (so they were absent from the scored imagery, matching the two-year
pre-installation gap the model was trained with). A national suitability heatmap and
a site-vs-random-land score distribution are saved to ./outputs/.

Requires Earth Engine access (free): a Google account, a cloud project with the
Earth Engine API enabled, and a one-time `earthengine authenticate`. See the README.

    export EE_PROJECT=your-earthengine-project-id
    python run_country.py Greece
    python run_country.py Germany --year 2019

Notes / caveats:
  * Map year is the AlphaEarth embedding year (default 2019). Validation uses sites
    installed >= year+2, so they did not exist in the scored imagery.
  * "Random land" points only estimate the country-wide score distribution; they are
    not confirmed non-solar, so the reported country ROC understates the model.
  * Validation here uses all inventory sites for the country installed after the
    imagery year; the paper's per-country tables use a deduplicated, country-balanced
    validation subset, so exact counts/ROCs differ slightly from these indicative runs.
  * The forest must be trained without max_leaf_nodes (best-first trees serialise to
    strings Earth Engine rejects); size trees with min_samples_leaf / max_depth.
"""
import argparse
import io
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

DEFAULT_MODEL = "models/rf30_v3_R.joblib"


def run_country(country, year=2019, min_install=None, n_rand=6000, max_sites=3000,
                model_path=DEFAULT_MODEL, sites_csv=None, ee_project=None, seed=42):
    import ee
    import pandas as pd
    import joblib
    import requests
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from geemap import ml
    from PIL import Image
    from shapely.geometry import shape, Point
    from sklearn.metrics import roc_auc_score, average_precision_score

    ee_project = ee_project or os.environ.get("EE_PROJECT")
    if not ee_project:
        sys.exit("Set your Earth Engine cloud project id: `export EE_PROJECT=your-project-id` "
                 "(or pass --ee-project). See the README for one-time setup.")
    if min_install is None:
        min_install = year + 2
    if sites_csv is None:
        sites_csv = f"data/sites_{country.lower()}.csv"
    os.makedirs("outputs", exist_ok=True)
    out_prefix = f"outputs/{country.lower().replace(' ', '_')}_rf_{year}"

    ee.Initialize(project=ee_project)
    bands = [f"A{i:02d}" for i in range(64)]

    # ---- country boundary (public LSIB table) ----
    lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    fc = lsib.filter(ee.Filter.eq("country_na", country))
    if fc.size().getInfo() == 0:
        names = lsib.aggregate_array("country_na").distinct().getInfo()
        close = [n for n in names if country.lower() in n.lower()]
        sys.exit(f"'{country}' not in LSIB. Close matches: {close}")
    geom = fc.geometry()

    # ---- trained random forest -> Earth Engine classifier (inject as tree strings) ----
    rf = joblib.load(model_path)
    t0 = time.time()
    trees = ml.rf_to_strings(rf, bands, output_mode="PROBABILITY")
    clf = ml.strings_to_classifier(trees)
    print(f"classifier injected: {len(trees)} trees "
          f"({sum(len(s) for s in trees) / 1e6:.1f} MB, {time.time() - t0:.0f}s)", flush=True)

    # ---- classify the country's AlphaEarth mosaic (public catalog) ----
    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic().select(bands))
    score = img.classify(clf).clip(geom).rename("score")

    # ---- validation sites: installed after the imagery, unseen by the map ----
    if not os.path.exists(sites_csv):
        sys.exit(f"no site file {sites_csv}. Bundled: data/sites_greece.csv, data/sites_germany.csv. "
                 "Provide your own with --sites (columns: lon,lat,year).")
    sdf = pd.read_csv(sites_csv)
    sdf = sdf[sdf["year"] >= min_install]
    if len(sdf) == 0:
        sys.exit(f"no sites in {sites_csv} installed >= {min_install}")
    site_xy = sdf[["lon", "lat"]].to_numpy(dtype=float)
    if len(site_xy) > max_sites:
        idx = np.random.default_rng(seed).choice(len(site_xy), max_sites, replace=False)
        site_xy = site_xy[idx]
        print(f"validation sites: {len(sdf):,} -> sampled {max_sites:,}", flush=True)
    else:
        print(f"validation sites (installed >= {min_install}): {len(site_xy):,}", flush=True)

    # ---- random land points (country-wide score distribution) ----
    shp = shape(geom.getInfo())
    minx, miny, maxx, maxy = shp.bounds
    rng = np.random.default_rng(seed)
    rand_xy = []
    while len(rand_xy) < n_rand:
        cand = np.c_[rng.uniform(minx, maxx, 4000), rng.uniform(miny, maxy, 4000)]
        rand_xy += [tuple(c) for c in cand if shp.contains(Point(c))]
    rand_xy = np.array(rand_xy[:n_rand])

    # ---- score the points server-side (chunked; only touched tiles compute) ----
    def pts_scores(xy, tag):
        out = []
        for i in range(0, len(xy), 1000):
            pfc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(p[0]), float(p[1])]))
                                        for p in xy[i:i + 1000]])
            vals = (score.reduceRegions(pfc, ee.Reducer.first(), scale=10)
                    .aggregate_array("first").getInfo())
            out += [v for v in vals if v is not None]
            print(f"  {tag} {min(i + 1000, len(xy))}/{len(xy)}", flush=True)
        return np.array(out, dtype=float)

    t0 = time.time()
    site_s = pts_scores(site_xy, "sites")
    rand_s = pts_scores(rand_xy, "random")
    print(f"scored {len(site_s)} sites + {len(rand_s)} random ({time.time() - t0:.0f}s)", flush=True)

    # ---- metrics ----
    y = np.r_[np.ones(len(site_s)), np.zeros(len(rand_s))]
    x = np.r_[site_s, rand_s]
    pct = np.array([(rand_s < s).mean() for s in site_s])
    roc, pr = roc_auc_score(y, x), average_precision_score(y, x)
    print(f"\n=== {country} {year}: RF map vs sites installed >= {min_install} ===")
    print(f"ROC-AUC {roc:.3f} | PR-AUC {pr:.3f} (chance {len(site_s) / len(x):.3f}) | "
          f"median site percentile {np.median(pct) * 100:.1f}")
    print(f"sites in top 20/10/5% of ranked land: "
          f"{(pct >= 0.8).mean() * 100:.1f}% / {(pct >= 0.9).mean() * 100:.1f}% / "
          f"{(pct >= 0.95).mean() * 100:.1f}%", flush=True)

    # ---- save scores first (rendering can fail for huge countries) ----
    b = geom.bounds().coordinates().getInfo()[0]
    xs, ys = [c[0] for c in b], [c[1] for c in b]
    ext = [min(xs), max(xs), min(ys), max(ys)]
    np.savez(f"{out_prefix}_scores.npz", site_s=site_s, rand_s=rand_s,
             site_xy=site_xy, rand_xy=rand_xy, ext=np.array(ext))
    print(f"saved {out_prefix}_scores.npz", flush=True)

    # ---- national heatmap + score distributions ----
    lo, hi = np.percentile(rand_s, [2, 98])
    palette = ["000004", "2c115f", "721f81", "b73779", "f1605d", "feb078", "fcfdbf"]
    im, png = None, b""
    for dim in (1400, 900, 600):
        try:
            url = score.getThumbURL({"region": geom.bounds(), "dimensions": dim,
                                     "min": float(lo), "max": float(hi), "palette": palette})
            png = requests.get(url, timeout=600).content
            im = np.array(Image.open(io.BytesIO(png)))
            with open(f"{out_prefix}.png", "wb") as f:
                f.write(png)
            break
        except Exception as e:
            print(f"thumbnail at {dim}px failed ({e}); retrying smaller", flush=True)
    if im is None:
        print("no thumbnail rendered; scores npz already saved", flush=True)
        return

    fig, ax = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].imshow(im, extent=ext, aspect="auto")
    ax[0].scatter(site_xy[:, 0], site_xy[:, 1], s=4, c="cyan", alpha=0.6,
                  label=f"sites installed >= {min_install} (n={len(site_xy):,})")
    ax[0].legend(loc="lower right", fontsize=9)
    ax[0].set_title(f"{country} - RF suitability, AlphaEarth {year}")
    ax[1].hist(rand_s, bins=60, density=True, alpha=0.6, label=f"random {country} land")
    ax[1].hist(site_s, bins=60, density=True, alpha=0.6, label="future solar sites")
    ax[1].set_xlabel("RF P(solar)")
    ax[1].set_ylabel("density")
    ax[1].legend(fontsize=9)
    ax[1].set_title(f"ROC-AUC {roc:.3f}")
    fig.savefig(f"{out_prefix}_fig.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_prefix}.png / _fig.png / _scores.npz", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Country-scale RF solar-suitability run (Earth Engine)")
    ap.add_argument("country", help="country name as in the LSIB boundary table, e.g. Greece")
    ap.add_argument("--year", type=int, default=2019, help="AlphaEarth embedding year of the map")
    ap.add_argument("--min-install", type=int, default=None,
                    help="validate sites installed >= this year (default: year+2)")
    ap.add_argument("--n-rand", type=int, default=6000, help="random land points")
    ap.add_argument("--max-sites", type=int, default=3000, help="cap on validation sites")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="path to the trained RF joblib")
    ap.add_argument("--sites", default=None, help="site CSV (columns: lon,lat,year); "
                    "default data/sites_<country>.csv")
    ap.add_argument("--ee-project", default=None, help="Earth Engine cloud project id "
                    "(or set EE_PROJECT)")
    a = ap.parse_args()
    run_country(a.country, year=a.year, min_install=a.min_install, n_rand=a.n_rand,
                max_sites=a.max_sites, model_path=a.model, sites_csv=a.sites, ee_project=a.ee_project)
