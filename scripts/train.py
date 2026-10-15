"""Train the RF+MLP ensemble with the release recipe and save it in the released model format (usable by predict.py).

    python scripts/train.py --preset sample --out models/my_model        # bundled sample, a few minutes on a laptop
    python scripts/train.py --data my_pool.npz --out models/my_model     # your own pixel pool, release recipe

Recipe (paper Appendix B; selected on held-out country validation, never on test data):
  random forest  80 trees, max_features=12, min_samples_leaf=14, bootstrap, class_weight="balanced", random_state=0
  MLP            64 -> 256 -> 256 -> 1 (ReLU), standardised inputs, class-balanced BCE, AdamW (lr 3e-3, wd 1e-5),
                 batch 16,384, one-cycle schedule over 8 epochs stopped after the first epoch; 5 seeds (0-4), averaged
  ensemble       P = 0.5 * P_RF + 0.5 * mean(P_MLP seeds)

The released model was trained on ~45 M labelled pixels from 38,591 sites and 13,440 non-solar parcels (AEF two years before
installation); that pool is not redistributable at this size. `--preset sample` adapts the MLP schedule to the small
bundled sample (batch 1,024, all 8 epochs) and reports held-out ROC/PR on the bundled test pixels.
Input format (.npz): X_sol_train, X_neg_train [, X_sol_test, X_neg_test], float32 (n, 64) de-quantised AEF embeddings.

MLP training needs PyTorch (CPU is fine):  pip install torch  (or: conda install -c conda-forge pytorch)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

RELEASE = dict(hidden=[256, 256], lr=3e-3, wd=1e-5, bs=16384, epochs=8, pct=0.05, stop_epoch=0, seeds=[0, 1, 2, 3, 4])
PRESETS = {"release": {}, "sample": dict(bs=1024, stop_epoch=7)}


def train_mlp(X, y, cfg, seed, log):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y.astype(np.float32))
    mu = Xt.mean(0, keepdim=True); sd = Xt.std(0, keepdim=True).clamp_min(1e-6); Xt = (Xt - mu) / sd
    layers, d = [], 64
    for h in cfg["hidden"]:
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    net = nn.Sequential(*layers, nn.Linear(d, 1))
    npos = int(y.sum()); nneg = len(y) - npos
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / npos))         # == class_weight="balanced"
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    spe = -(-len(y) // cfg["bs"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * spe, pct_start=cfg["pct"])
    g = torch.Generator().manual_seed(seed)
    for ep in range(cfg["stop_epoch"] + 1):
        net.train(); perm = torch.randperm(len(yt), generator=g); tl = 0.0
        for i in range(0, len(yt), cfg["bs"]):
            j = perm[i:i + cfg["bs"]]; opt.zero_grad(set_to_none=True)
            loss = lossf(net(Xt[j]).squeeze(1), yt[j]); loss.backward(); opt.step(); sch.step(); tl += loss.item() * len(j)
        log(f"    seed {seed} epoch {ep}: loss {tl / len(yt):.4f}")
    lin = [m for m in net if isinstance(m, nn.Linear)]
    return [(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy()) for m in lin], mu.numpy()[0], sd.numpy()[0]


def main():
    sys.stdout.reconfigure(line_buffering=True)          # progress shows up live, also in log files
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "px_sample.npz"))
    ap.add_argument("--preset", choices=list(PRESETS), default="release")
    ap.add_argument("--out", required=True, help="output model folder (model.json, rf.joblib, mlp.npz)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    try:
        import torch
        torch.set_num_threads(a.threads)
    except ImportError:
        sys.exit("MLP training needs PyTorch (CPU is fine):  pip install torch")
    import joblib
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from solarsiting.model import NumpyMLP, SuitabilityModel, sha256_16

    cfg = dict(RELEASE, **PRESETS[a.preset]); log = lambda s: print(s, flush=True)
    z = np.load(a.data)
    X = np.vstack([z["X_sol_train"], z["X_neg_train"]]).astype(np.float32)
    y = np.r_[np.ones(len(z["X_sol_train"])), np.zeros(len(z["X_neg_train"]))]
    log(f"training pool: {len(y):,} px ({int(y.sum()):,} solar / {int(len(y) - y.sum()):,} non-solar) | preset {a.preset}")
    t = time.time()
    rf = RandomForestClassifier(n_estimators=80, max_features=12, min_samples_leaf=14, bootstrap=True, class_weight="balanced",
                                n_jobs=a.threads, random_state=0).fit(X, y)
    log(f"  random forest: {time.time() - t:.0f}s")
    t = time.time(); seeds = []
    for s in cfg["seeds"]:
        seeds.append(train_mlp(X, y, cfg, s, log))
    log(f"  MLP x{len(cfg['seeds'])} seeds: {time.time() - t:.0f}s")

    os.makedirs(a.out, exist_ok=True)
    joblib.dump(rf, os.path.join(a.out, "rf.joblib"), compress=3)
    arrs = {"mu": seeds[0][1], "sd": seeds[0][2]}
    if not all(np.allclose(s_[1], arrs["mu"]) for s_ in seeds):
        raise RuntimeError("seeds must share the input standardiser (they are fit on the same data)")
    for s, (layers, _, _) in zip(cfg["seeds"], seeds):
        for k, (W, b) in enumerate(layers):
            arrs[f"W{k}_s{s}"] = W.astype(np.float32); arrs[f"b{k}_s{s}"] = b.astype(np.float32)
    np.savez(os.path.join(a.out, "mlp.npz"), **arrs)
    meta = dict(name=f"user-trained RF+MLP ({os.path.basename(a.out)})", version="user", w_rf=0.5,
                mlp=dict(hidden=cfg["hidden"], seeds=cfg["seeds"], activation="relu", output="sigmoid", standardised_inputs=True),
                rf=dict(n_estimators=80, max_features=12, min_samples_leaf=14), training=dict(data=os.path.abspath(a.data),
                preset=a.preset, mlp_cfg=cfg), requires=dict(scikit_learn=sklearn.__version__),
                output=dict(encoding="uint8 code = round(P * 250); P = code * 0.004; 255 = no data"))
    meta["sha256_16"] = {f: sha256_16(os.path.join(a.out, f)) for f in ("rf.joblib", "mlp.npz")}
    json.dump(meta, open(os.path.join(a.out, "model.json"), "w"), indent=1)
    log(f"saved -> {a.out}")

    if "X_sol_test" in z.files:
        m = SuitabilityModel(rf, [NumpyMLP(l, mu, sd) for l, mu, sd in seeds], 0.5)
        Xt = np.vstack([z["X_sol_test"], z["X_neg_test"]]); yt = np.r_[np.ones(len(z["X_sol_test"])), np.zeros(len(z["X_neg_test"]))]
        for name, p in (("RF", rf.predict_proba(Xt)[:, 1]), ("MLP (5 seeds)", np.mean([mm.predict_p(Xt) for mm in m.mlps], 0)),
                        ("RF+MLP ensemble", m.predict(Xt))):
            log(f"  held-out {name:<16} ROC-AUC {roc_auc_score(yt, p):.3f} | PR-AUC {average_precision_score(yt, p):.3f}")


if __name__ == "__main__":
    main()
