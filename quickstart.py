"""Quick sanity check (no Earth Engine, no training, runs in seconds).

Loads the released random forest and scores the bundled held-out pixel sample,
so you can confirm the model works before setting anything else up.

    python quickstart.py
"""
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

z = np.load("data/px_sample.npz")
X = np.vstack([z["X_sol_test"], z["X_neg_test"]])
y = np.r_[np.ones(len(z["X_sol_test"])), np.zeros(len(z["X_neg_test"]))]

rf = joblib.load("models/rf30_v3_R.joblib")
p = rf.predict_proba(X)[:, 1]

print(f"Random forest scored on {len(y):,} held-out AlphaEarth pixels "
      f"({int(y.sum()):,} solar / {int((y == 0).sum()):,} non-solar):")
print(f"  ROC-AUC {roc_auc_score(y, p):.3f} | PR-AUC {average_precision_score(y, p):.3f}")
print("\nThe model maps each 64-dimensional AlphaEarth pixel embedding to P(solar).")
print("Next: `python benchmark_extractors.py` (method comparison) or "
      "`python run_country.py Greece` (map a whole country, needs Earth Engine).")
