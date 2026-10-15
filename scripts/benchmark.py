"""Extraction-method benchmark (paper Table 1) on the bundled AlphaEarth pixel sample. CPU only, ~1-2 minutes.

Nine ways of turning a 64-d AlphaEarth pixel embedding into P(solar), all trained on the same training sample and
scored on the same held-out sample (solar : non-solar ~ 1 : 5, the paper's test prevalence), with the paper's settings:

  mean-embedding cosine | QDA | logistic probe | MaxEnt | random Fourier features | random forest |
  histogram gradient boosting | MLP | RF+MLP ensemble (equal-weight average of the RF and MLP above)

The paper trains on a 1.4 M-pixel balanced subsample of the full pool; the bundled sample is ~20x smaller, so values
are a little lower for the high-capacity models, but the ranking reproduces Table 1 (see docs/reproduce.md).
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = {  # Table 1, pixel ROC-AUC / PR-AUC (held-out test pixels)
    "Mean-embedding cosine": (0.702, 0.318), "QDA": (0.854, 0.518), "Logistic probe": (0.866, 0.551),
    "MaxEnt": (0.874, 0.577), "Random Fourier features": (0.891, 0.628), "Random forest": (0.914, 0.696),
    "Hist. gradient boosting": (0.919, 0.713), "MLP": (0.924, 0.728), "RF+MLP ensemble": (0.932, 0.752)}


def main():
    sys.stdout.reconfigure(line_buffering=True)          # progress shows up live, also in log files
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "px_sample.npz"))
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    z = np.load(a.data)
    X = np.vstack([z["X_sol_train"], z["X_neg_train"]]).astype(np.float32)
    y = np.r_[np.ones(len(z["X_sol_train"])), np.zeros(len(z["X_neg_train"]))]
    Xt = np.vstack([z["X_sol_test"], z["X_neg_test"]]).astype(np.float32)
    yt = np.r_[np.ones(len(z["X_sol_test"])), np.zeros(len(z["X_neg_test"]))]
    print(f"train {len(y):,} px | test {len(yt):,} px (PR-AUC chance {yt.mean():.3f})\n")
    sc = StandardScaler().fit(X); S, St = sc.transform(X), sc.transform(Xt)
    rows = []

    def report(name, p, t0, note=""):
        r, pr = roc_auc_score(yt, p), average_precision_score(yt, p)
        rows.append((name, r, pr))
        pp = PAPER[name]
        print(f"  {name:<24} ROC {r:.3f} | PR {pr:.3f}   (paper {pp[0]:.3f} / {pp[1]:.3f})  {time.time()-t0:5.1f}s {note}", flush=True)
        return p

    t = time.time(); proto = z["X_sol_train"].mean(0); proto /= np.linalg.norm(proto)
    report("Mean-embedding cosine", Xt @ proto, t, "(no training: similarity to the mean solar embedding)")
    t = time.time(); report("QDA", QuadraticDiscriminantAnalysis().fit(X, y).predict_proba(Xt)[:, 1], t)
    t = time.time(); lr = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced").fit(S, y)
    report("Logistic probe", lr.predict_proba(St)[:, 1], t)
    t = time.time(); me = LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=200, class_weight="balanced")
    me.fit(np.hstack([S, S ** 2]), y)
    report("MaxEnt", me.predict_proba(np.hstack([St, St ** 2]))[:, 1], t, "(IWLR-equivalent: L1 logistic on [x, x^2])")
    t = time.time(); rbf = RBFSampler(n_components=512, gamma=1.0 / 64, random_state=0).fit(S)
    lr2 = LogisticRegression(max_iter=300, class_weight="balanced").fit(rbf.transform(S), y)
    report("Random Fourier features", lr2.predict_proba(rbf.transform(St))[:, 1], t)
    t = time.time(); rf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=50, class_weight="balanced",
                                                 n_jobs=a.threads, random_state=0).fit(X, y)
    p_rf = report("Random forest", rf.predict_proba(Xt)[:, 1], t)
    t = time.time(); hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=0).fit(X, y)
    report("Hist. gradient boosting", hgb.predict_proba(Xt)[:, 1], t)
    t = time.time(); mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=60, early_stopping=True, random_state=0).fit(S, y)
    p_mlp = report("MLP", mlp.predict_proba(St)[:, 1], t)
    report("RF+MLP ensemble", 0.5 * p_rf + 0.5 * p_mlp, time.time(), "(equal-weight average of the RF and MLP above)")

    best = max(rows, key=lambda r: r[1])
    order_ok = [r[0] for r in sorted(rows, key=lambda r: r[1])] == list(PAPER)
    print(f"\nbest: {best[0]} (ROC {best[1]:.3f}) | ranking identical to Table 1: {'yes' if order_ok else 'no (see docs/reproduce.md)'}")


if __name__ == "__main__":
    main()
