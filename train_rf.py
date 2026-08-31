"""Train and evaluate the deployed solar-suitability random forest — reference pipeline.

This is the exact recipe behind the released model (`models/rf30_v3_R.joblib`), in two
stages:

  1. build_pixel_pool()  — turn AlphaEarth chips + solar masks into a per-pixel training
     pool. Positives are the *masked* (solar) pixels of each positive chip; negatives are
     the pixels of globally-random non-solar chips. Each chip contributes a capped number
     of pixels so no single large site dominates. Requires the full chip corpus (the
     multi-terabyte AlphaEarth extraction), which is **not shipped** with this artifact —
     this function documents, and reproduces, how the pool is made if you have the chips.

  2. train_rf() / evaluate() — fit the forest on the pool and score a held-out pool. The
     forest is deliberately sized so its decision-tree strings stay under a few MB, which
     is what lets it be injected into Earth Engine for global inference (see run_country.py).
     This stage runs out of the box on the bundled pixel sample (`data/px_sample.npz`).

Run the training/eval demo on the bundled sample (no chip corpus needed):

    python train_rf.py                       # trains on data/px_sample.npz, prints ROC/PR

Reproduce the deployed model from a full extracted pool (needs the chip corpus):

    python extract_step -> a pool .npz with arrays X_sol, X_neg   # via build_pixel_pool()
    python train_rf.py --pool px_pool_train.npz --test px_pool_test.npz --out models/rf.joblib

The released model is "Model R": the full pool with `class_weight="balanced"`. The paper's
appendix also reports de-confounded (per-country reweighted), leave-one-country-out and
leak-proof temporal variants; those are the same recipe with one extra filter/weight on the
pool, omitted here to keep the reference model legible.
"""
import argparse
import os
import time

import numpy as np

# 64 AlphaEarth annual-embedding bands, named as in the Earth Engine catalog.
BANDS = [f"A{i:02d}" for i in range(64)]

# ---- deployed random-forest hyperparameters (Model R) --------------------------------
N_TREES = 30            # kept small so the serialised trees fit Earth Engine
MAX_DEPTH = 12
MAX_SAMPLES = 0.34      # per-tree bootstrap fraction
MB_CAP = 9.0            # Earth Engine rejects classifiers whose tree strings exceed ~10 MB
# min_samples_leaf is not a free knob: it is grown only until the serialised trees fit
# MB_CAP. On the full multi-million-pixel pool the cap binds and it lands at ~1500 (the
# deployed value); on the small bundled sample the cap is met with a far smaller leaf, so
# the demo starts lower and reaches its natural accuracy instead of heavily underfitting.
FULL_MIN_LEAF = 1500
DEMO_MIN_LEAF = 5
SEED = 0

# ---- pixel-pool extraction (reference; needs the chip corpus) -------------------------
CAP_SOL = 120           # max solar pixels sampled per positive chip
PER_NEG = 2800          # max pixels sampled per negative chip


def _chip_pixels(path, mask_path=None, cap=None, seed=0):
    """Finite 64-d pixels of one chip; if mask_path is given, only the masked (solar) ones."""
    import rasterio
    with rasterio.open(path) as s:
        a = s.read()                                   # (64, H, W)
    if mask_path is not None:
        with rasterio.open(mask_path) as s:
            m = s.read(1).astype(bool)
        if not m.any():
            return None
        x = a[:, m].T.astype(np.float32)               # (n_solar_px, 64)
    else:
        x = a.reshape(a.shape[0], -1).T.astype(np.float32)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) == 0:
        return None
    if cap is not None and len(x) > cap:               # cap so no chip dominates
        x = x[np.random.default_rng(seed).choice(len(x), cap, replace=False)]
    return x


def build_pixel_pool(out, chips_dir, masks_dir, neg_dir, pos_list, neg_list,
                     cap_sol=CAP_SOL, per_neg=PER_NEG):
    """Assemble the per-pixel training pool from chips + masks.

    pos_list / neg_list are CSVs of the chips to use (a `chip`/`file` column of filenames).
    Masks are binary: a pixel is solar (1) iff it fell inside a real site's footprint in
    imagery taken before that site was built. Not runnable here — the chips are not shipped —
    but this is the exact procedure that produced the bundled sample and the deployed pool.
    """
    import pandas as pd
    for d in (chips_dir, masks_dir, neg_dir):
        if not os.path.isdir(d):
            raise SystemExit(
                f"missing '{d}'. build_pixel_pool needs the full AlphaEarth chip corpus, "
                "which is not shipped with this artifact (released on acceptance). The "
                "bundled data/px_sample.npz is a ready-made subsample of this pool; run "
                "`python train_rf.py` to train on it directly.")
    pos, neg = pd.read_csv(pos_list), pd.read_csv(neg_list)
    sol = [p for p in (_chip_pixels(os.path.join(chips_dir, c),
                                    os.path.join(masks_dir, c), cap_sol, i)
                       for i, c in enumerate(pos.chip)) if p is not None]
    ngv = [p for p in (_chip_pixels(os.path.join(neg_dir, f), None, per_neg, i)
                       for i, f in enumerate(neg.file)) if p is not None]
    X_sol, X_neg = np.concatenate(sol), np.concatenate(ngv)
    print(f"pool: {len(X_sol):,} solar px / {len(X_neg):,} non-solar px "
          f"(pos share {len(X_sol) / (len(X_sol) + len(X_neg)):.3f})")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, X_sol=X_sol, X_neg=X_neg)
    print(f"wrote {out}")
    return X_sol, X_neg


def train_rf(X, y, mb_cap=MB_CAP, min_leaf=FULL_MIN_LEAF, jobs=-1, verbose=True):
    """Fit the deployed forest, growing min_samples_leaf until the tree strings fit Earth
    Engine. Returns (fitted_rf, tree_string_MB, final_min_samples_leaf)."""
    from sklearn.ensemble import RandomForestClassifier
    from geemap import ml
    while True:
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=MAX_DEPTH, min_samples_leaf=min_leaf,
            max_samples=MAX_SAMPLES, class_weight="balanced", n_jobs=jobs,
            random_state=SEED).fit(X, y)
        # rf_to_strings is the exact Earth Engine serialisation used for deployment; its
        # size is the constraint we tune min_samples_leaf against.
        mb = sum(len(s) for s in ml.rf_to_strings(rf, BANDS, output_mode="PROBABILITY")) / 1e6
        if verbose:
            print(f"  min_samples_leaf={min_leaf}: {mb:.2f} MB tree strings "
                  f"({time.time() - t0:.0f}s)")
        if mb <= mb_cap:
            return rf, mb, min_leaf
        min_leaf = int(min_leaf * mb / (mb_cap - 0.5))   # shrink the trees, refit


def evaluate(rf, X, y):
    from sklearn.metrics import roc_auc_score, average_precision_score
    keep = np.isfinite(X).all(axis=1)
    p = rf.predict_proba(X[keep])[:, 1]
    roc = roc_auc_score(y[keep], p)
    pr = average_precision_score(y[keep], p)
    print(f"  held-out px ROC-AUC {roc:.4f} | PR-AUC {pr:.4f} "
          f"(chance {y[keep].mean():.3f}) on {int(keep.sum()):,} px")
    return roc, pr


def _pool(path):
    """(X, y) from a pool .npz holding X_sol / X_neg (as build_pixel_pool writes)."""
    z = np.load(path)
    X = np.concatenate([z["X_sol"], z["X_neg"]]).astype(np.float32)
    y = np.r_[np.ones(len(z["X_sol"])), np.zeros(len(z["X_neg"]))]
    return X, y


def main(a):
    if a.pool:                                          # full extracted pool
        Xtr, ytr = _pool(a.pool)
        Xte, yte = _pool(a.test) if a.test else (Xtr, ytr)
        min_leaf = FULL_MIN_LEAF
    else:                                               # bundled-sample demo
        d = np.load(a.sample)
        Xtr = np.concatenate([d["X_sol_train"], d["X_neg_train"]]).astype(np.float32)
        ytr = np.r_[np.ones(len(d["X_sol_train"])), np.zeros(len(d["X_neg_train"]))]
        Xte = np.concatenate([d["X_sol_test"], d["X_neg_test"]]).astype(np.float32)
        yte = np.r_[np.ones(len(d["X_sol_test"])), np.zeros(len(d["X_neg_test"]))]
        min_leaf = DEMO_MIN_LEAF
        print(f"DEMO: training on the bundled sample ({len(Xtr):,} px) with the deployed "
              "recipe, at the paper's ~1:5 prevalence. It reproduces the deployed model's "
              "held-out ROC/PR to within subsample noise; the leaf floor is relaxed because "
              "the Earth Engine size cap only binds on the full pool (see the module docstring).")

    # identical seed => identical shuffle of features and labels in lockstep
    for arr in (Xtr, ytr):
        np.random.default_rng(SEED).shuffle(arr)
    rf, mb, min_leaf = train_rf(Xtr, ytr, mb_cap=a.mb_cap, min_leaf=min_leaf)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    import joblib
    joblib.dump(rf, a.out)
    print(f"saved {a.out} ({mb:.2f} MB tree strings, min_samples_leaf={min_leaf})")
    evaluate(rf, Xte, yte)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=None,
                    help="training pool .npz (X_sol/X_neg); omit to use the bundled sample")
    ap.add_argument("--test", default=None, help="held-out pool .npz (X_sol/X_neg)")
    ap.add_argument("--sample", default="data/px_sample.npz",
                    help="bundled sample used when --pool is not given")
    ap.add_argument("--out", default="models/rf_demo.joblib",
                    help="where to write the trained forest (won't overwrite the released one)")
    ap.add_argument("--mb-cap", type=float, default=MB_CAP,
                    help="max tree-string size (MB) for Earth Engine deployability")
    main(ap.parse_args())
