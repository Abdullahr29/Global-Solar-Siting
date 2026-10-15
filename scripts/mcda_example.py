"""Multi-criteria decision analysis (MCDA) vs the RF+MLP ensemble, for Greece, from bundled ~1 km data (seconds).

    python scripts/mcda_example.py
    python scripts/mcda_example.py --wdpa WDPA_Greece.gpkg      # + protected-area exclusion (download WDPA yourself)

Reconstructs the paper's MCDA benchmark (Appendix H): two published expert weightings (Richards et al. 2025, an
infrastructure-led AHP; Chen Jong & Mohamud Ahmed 2024, a resource-and-terrain scheme), each criterion scaled to [0, 1]
by country-relative 2-98 % min-max (cost criteria inverted), combined by weighted linear combination (WLC) or TOPSIS,
with water / herbaceous wetland (ESA WorldCover 80, 90) and, optionally, protected areas forced to the minimum score.
Each map is scored like the ensemble: solar sites installed from 2021 (after the AEF-2019 map) vs random land cells.

Bundled data: data/mcda_example/greece_0p01deg.npz (0.01-degree grid; sources and licences in data/README.md).
WDPA is not bundled (its terms forbid redistribution): get it from https://www.protectedplanet.net and pass --wdpa.
Other countries: build the same layers from the sources in docs/data_sources.md and pass --data (see data/README.md).
Note: results use the bundled GPVF sites only and the paper's figure-resolution grid, so they differ slightly from
the paper's point-based Table 5 (which also used TZ-SAM sites and WDPA), but the gap is the same story.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIRECTION = {"ghi": "b", "pvout": "b", "gti": "b", "temp": "c", "slope": "c", "elevation": "b",
             "equatorwardness": "b", "road_dist_m": "c", "grid_dist_m": "c", "pop_density": "b"}
WEIGHTS = {   # published weights, renormalised over the criteria available (paper Table 4)
    "Richards": {"road_dist_m": 0.5401, "grid_dist_m": 0.1392, "slope": 0.0605, "elevation": 0.0225, "ghi": 0.0352,
                 "pvout": 0.0253, "gti": 0.0169, "equatorwardness": 0.0759, "temp": 0.0844},
    "Chen": {"ghi": 0.1924, "temp": 0.1701, "slope": 0.1576, "elevation": 0.1594, "grid_dist_m": 0.1661, "pop_density": 0.1544}}
EXCLUDE_WORLDCOVER = (80, 90)       # permanent water, herbaceous wetland


def normalise(a, mask, direction):
    lo, hi = np.nanpercentile(a[mask], [2, 98])
    x = np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)
    return x if direction == "b" else 1 - x


def wlc(N, w):
    return sum(w[c] * N[c] for c in w)


def topsis(N, w):
    ws = np.array([w[c] for c in w]); X = np.stack([N[c] for c in w], -1) * ws
    dpos = np.sqrt(((X - ws) ** 2).sum(-1)); dneg = np.sqrt((X ** 2).sum(-1))
    return dneg / (dpos + dneg + 1e-12)


def percentile_map(a, mask):
    from scipy.stats import rankdata
    o = np.full(a.shape, np.nan); v = mask & np.isfinite(a); o[v] = rankdata(a[v]) / v.sum(); return o


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "mcda_example", "greece_0p01deg.npz"))
    ap.add_argument("--sites", default=os.path.join(HERE, "..", "data", "validation", "sites_greece.csv"))
    ap.add_argument("--min-year", type=int, default=2021, help="validation sites installed from this year (map is AEF 2019)")
    ap.add_argument("--wdpa", help="protected-area polygons (any vector file) to exclude, as in the paper")
    ap.add_argument("--out", default=os.path.join("outputs", "mcda_example"))
    a = ap.parse_args()
    import pandas as pd
    from rasterio.transform import Affine, rowcol
    from sklearn.metrics import roc_auc_score

    z = np.load(a.data, allow_pickle=False); mask = z["mask"]; tr = Affine(*z["transform"])
    excl = np.isin(z["worldcover"], EXCLUDE_WORLDCOVER) & mask
    if a.wdpa:
        import geopandas as gpd
        from rasterio.features import rasterize
        g = gpd.read_file(a.wdpa).to_crs(4326)
        excl |= rasterize(((x, 1) for x in g.geometry if x is not None), out_shape=mask.shape, transform=tr, fill=0).astype(bool) & mask
    print(f"{int(mask.sum()):,} land cells (~1 km) | excluded: {int(excl.sum()):,} "
          f"(WorldCover water/wetland{' + WDPA' if a.wdpa else ''})")

    S = pd.read_csv(a.sites); S = S[S.year >= a.min_year]
    r, c = rowcol(tr, S.lon.values, S.lat.values); r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1]); r, c = r[ok], c[ok]
    ok = mask[r, c]; r, c = r[ok], c[ok]
    rng = np.random.default_rng(0); land = np.flatnonzero(mask.ravel())
    rand = rng.choice(land, min(20000, land.size), replace=False)

    def score(m):
        s, n = m[r, c], m.ravel()[rand]; s, n = s[np.isfinite(s)], n[np.isfinite(n)]
        return roc_auc_score(np.r_[np.ones(len(s)), np.zeros(len(n))], np.r_[s, n])

    print(f"validation: {len(r)} sites installed >= {a.min_year} vs {len(rand):,} random land cells\n")
    res = {"RF+MLP ensemble (AEF 2019)": (score(z["ensemble_2019"]), z["ensemble_2019"])}
    for scheme, w0 in WEIGHTS.items():
        have = [k for k in w0 if k in z.files]
        if len(have) < len(w0):
            print(f"  {scheme}: missing {sorted(set(w0) - set(have))} in the data -> skipped"); continue
        w = {k: w0[k] / sum(w0[x] for x in have) for k in have}
        N = {k: normalise(z[k], mask, DIRECTION[k]) for k in w}
        for agg, fn in (("WLC", wlc), ("TOPSIS", topsis)):
            m = np.where(mask, fn(N, w), np.nan)
            m[excl] = np.nanmin(m[mask])
            res[f"MCDA {scheme} {agg}"] = (score(m), m)
    for k, (roc, _) in res.items():
        print(f"  {k:<28} ROC-AUC {roc:.3f}")
    best = max((k for k in res if k.startswith("MCDA")), key=lambda k: res[k][0])
    ens = res["RF+MLP ensemble (AEF 2019)"][0]
    print(f"\nbest MCDA: {best} ({res[best][0]:.3f}) | ensemble {ens:.3f} | gap {ens - res[best][0]:+.3f}")

    os.makedirs(a.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pe, pm = percentile_map(res["RF+MLP ensemble (AEF 2019)"][1], mask), percentile_map(res[best][1], mask)
    w_, s_, e_, n_ = z["extent"]                                          # stored as west, south, east, north
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for k, (arr, title, cmap, lim) in enumerate([(pe, f"RF+MLP ensemble (ROC-AUC {ens:.2f})", "magma", (0, 1)),
                                                 (pm, f"{best} (ROC-AUC {res[best][0]:.2f})", "magma", (0, 1)),
                                                 (pe - pm, "ensemble - MCDA (percentiles)", "RdBu_r", (-0.7, 0.7))]):
        im = ax[k].imshow(arr, extent=(w_, e_, s_, n_), cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
        ax[k].set_aspect(1 / np.cos(np.radians((s_ + n_) / 2)))         # lon/lat grid shown at true shape
        ax[k].set_title(title, fontsize=10); ax[k].set_xticks([]); ax[k].set_yticks([]); fig.colorbar(im, ax=ax[k], shrink=0.7)
    x_, y_ = tr * (c + 0.5, r + 0.5)                                    # the validation sites used above
    ax[0].scatter(x_, y_, s=2, c="#2b8cff", lw=0, label=f"sites installed {a.min_year}+ (n={len(r)})")
    ax[0].legend(loc="lower left", fontsize=8, frameon=False, markerscale=4)
    fig.tight_layout(); f = os.path.join(a.out, "mcda_vs_ensemble_greece.png"); fig.savefig(f, dpi=150); plt.close(fig)
    pd.DataFrame([(k, round(v[0], 4)) for k, v in res.items()], columns=["map", "roc_auc"]).to_csv(
        os.path.join(a.out, "mcda_results.csv"), index=False)
    print(f"-> {f}, mcda_results.csv")


if __name__ == "__main__":
    main()
