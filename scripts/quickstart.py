"""Check that everything works, in about a minute: model, AlphaEarth access, and a tiny map.

  1. loads the released model and scores the bundled pixel sample (a functional check: the released model was
     trained on all labelled data, so this sample is in-sample for it; held-out accuracy is what scripts/benchmark.py
     and the paper's validation measure);
  2. maps an ~11 km x 6 km area of Kozani, northern Greece (AEF 2025), straight from the public AlphaEarth tiles,
     writing outputs/quickstart/ (GeoTIFF + quicklook.png).

    python scripts/quickstart.py            # full model (needs ~6-8 GB RAM)
    python scripts/quickstart.py --lite     # MLP-only variant (< 1 GB RAM)
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def main():
    sys.stdout.reconfigure(line_buffering=True)          # progress shows up live, also in log files
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("SOLARSITING_MODEL", "models/solar_suitability_model_v1.0"))
    ap.add_argument("--lite", action="store_true", help="MLP-only variant (< 1 GB RAM)")
    a = ap.parse_args()
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    from solarsiting import infer, regions
    from solarsiting.model import load_model

    if not os.path.isdir(a.model):
        sys.exit(f"model folder not found: {a.model}\n  download it first:  python scripts/download_model.py")
    t = time.time(); m = load_model(a.model, lite=a.lite)
    print(f"[1/2] model loaded in {time.time()-t:.0f}s ({'lite: MLP seeds only' if a.lite else 'RF + MLP ensemble'})")
    z = np.load(os.path.join(HERE, "..", "data", "px_sample.npz"))
    X = np.vstack([z["X_sol_test"], z["X_neg_test"]]); y = np.r_[np.ones(len(z["X_sol_test"])), np.zeros(len(z["X_neg_test"]))]
    p = m.predict(X)
    print(f"      bundled sample: ROC-AUC {roc_auc_score(y, p):.3f} | PR-AUC {average_precision_score(y, p):.3f} "
          f"(in-sample for the released model: a functional check, not an accuracy estimate)")

    # Kozani, northern Greece: 17 solar sites in the bundled inventory sample (16 built 2021-2024), farmland and hills
    region = regions.bbox(21.67, 40.35, 21.80, 40.40)
    out = os.path.join("outputs", "quickstart" + ("_lite" if a.lite else ""))
    print("[2/2] mapping ~11 km x 6 km of Kozani, Greece, for AEF 2025 from the public AlphaEarth tiles ...")
    info = infer.run(m, region, "quickstart", 2025, out, log=lambda s: print("      " + s))
    print(f"\nall good. Open {out}/quicklook.png, or {out}/mosaic_*.vrt in QGIS. Next: scripts/predict.py --help")


if __name__ == "__main__":
    main()
