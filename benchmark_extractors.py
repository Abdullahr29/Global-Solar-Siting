"""Extraction-method benchmark on the bundled AlphaEarth pixel sample.

Reproduces the comparison in Table 1 of the paper: eight ways of turning a 64-d
AlphaEarth pixel embedding into a solar-suitability score, all trained on a common
balanced subsample and scored on the same held-out sample. The deployed random forest
is loaded from disk (models/rf30_v3_R.joblib); the other methods are trained here.

Numbers here are computed on the small bundled sample (40k/class train, 10k/class
test), so they approximate the paper, which uses the full multi-million-pixel pools
(released with the dataset). The method *ranking* is what this reproduces. The sample
is class-balanced, so PR-AUC reads higher than the paper's (~1:5 imbalanced) pool;
ROC-AUC is comparable.

    python benchmark_extractors.py
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

SEED = 0
RF_PATH = "models/rf30_v3_R.joblib"

z = np.load("data/px_sample.npz")
X = np.vstack([z["X_sol_train"], z["X_neg_train"]]).astype(np.float32)
y = np.r_[np.ones(len(z["X_sol_train"])), np.zeros(len(z["X_neg_train"]))]
perm = np.random.default_rng(SEED).permutation(len(y))
X, y = X[perm], y[perm]
Xt = np.vstack([z["X_sol_test"], z["X_neg_test"]]).astype(np.float32)
yt = np.r_[np.ones(len(z["X_sol_test"])), np.zeros(len(z["X_neg_test"]))]
print(f"train {len(y):,} px / test {len(yt):,} px (test chance {yt.mean():.3f})\n", flush=True)

sc = StandardScaler().fit(X)
Xs, Xts = sc.transform(X), sc.transform(Xt)
rocs, deployable = {}, {}


def report(name, proba, note, dep):
    roc = roc_auc_score(yt, proba)
    pr = average_precision_score(yt, proba)
    rocs[name], deployable[name] = roc, dep
    print(f"  {name:24s} ROC {roc:.3f} | PR {pr:.3f}   {note}", flush=True)


# 1. mean-embedding prototype, cosine similarity (normalise the prototype, not inputs)
proto = z["X_sol_train"].mean(0)
proto = proto / (np.linalg.norm(proto) + 1e-9)
report("Mean-embedding cosine", Xt @ proto, "prototype", True)

# 2. QDA
report("QDA", QuadraticDiscriminantAnalysis().fit(X, y).predict_proba(Xt)[:, 1], "discriminant", True)

# 3. logistic probe
lr = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced").fit(Xs, y)
report("Logistic probe", lr.predict_proba(Xts)[:, 1], "linear", True)

# 4. MaxEnt (elapid if present, else L1 logistic on [x, x^2] = IWLR-equivalent)
try:
    from elapid import MaxentModel
    pme = MaxentModel().fit(X, y).predict(Xt)
    note = "log-linear (SDM), elapid"
except Exception:
    Xq, Xtq = np.hstack([Xs, Xs ** 2]), np.hstack([Xts, Xts ** 2])
    pme = LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=200,
                             class_weight="balanced").fit(Xq, y).predict_proba(Xtq)[:, 1]
    note = "log-linear (SDM), L1 logistic"
report("MaxEnt", pme, note, True)

# 5. random Fourier features + logistic
rbf = RBFSampler(n_components=512, gamma=1.0 / X.shape[1], random_state=SEED).fit(Xs)
lr2 = LogisticRegression(max_iter=300, class_weight="balanced").fit(rbf.transform(Xs), y)
report("Random Fourier features", lr2.predict_proba(rbf.transform(Xts))[:, 1], "kernel", True)

# 6. the deployed random forest (loaded from disk)
if os.path.exists(RF_PATH):
    rf = joblib.load(RF_PATH)
    report("Random forest (deployed)", rf.predict_proba(Xt)[:, 1], "ensemble, from models/", True)

# 7. histogram gradient boosting
hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                     class_weight="balanced", random_state=0).fit(X, y)
report("Histogram grad. boosting", hgb.predict_proba(Xt)[:, 1], "ensemble", False)

# 8. MLP
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=60, early_stopping=True,
                    random_state=0).fit(Xs, y)
report("MLP", mlp.predict_proba(Xts)[:, 1], "neural", False)

print("\nRanked by held-out ROC-AUC (Deployable = has a native global-scale "
      "inference path in Earth Engine):")
print(f"  {'Method':24s} {'ROC-AUC':>8s}  Deployable")
for name in sorted(rocs, key=rocs.get, reverse=True):
    print(f"  {name:24s} {rocs[name]:8.3f}  {'yes' if deployable[name] else 'no'}")
print("\nThe random forest is the most accurate method that still admits native "
      "global inference in Earth Engine, so it is the one deployed (see run_country.py).")
