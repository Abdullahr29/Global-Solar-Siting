"""The released solar-suitability model: an equal-weight ensemble of a random forest and a 5-seed MLP.

    P(solar | x) = w_rf * P_RF(x) + (1 - w_rf) * mean_s P_MLP_s(x),    w_rf = 0.5

`x` is a 64-dimensional AlphaEarth Foundations (AEF) annual embedding of one 10 m pixel. Everything runs on the CPU:
the random forest is scikit-learn, and the MLP forward pass (two hidden layers of 256 ReLU units) is plain NumPy, so
no deep-learning framework or GPU is needed.

Model directory layout (as distributed alongside the dataset):
    model.json   metadata: weights, MLP architecture, input standardisation, versions, checksums
    rf.joblib    scikit-learn RandomForestClassifier (80 trees; requires scikit-learn == 1.8.0 to unpickle)
    mlp.npz      MLP weights for every seed (W0_s, b0_s, W1_s, b1_s, W2_s, b2_s) + input mean/std (mu, sd)
"""
import hashlib
import json
import os

import numpy as np

# Dataset encoding: uint8 code = round(P * 250); P = code * SCALE; 255 = no data.
SCALE = 0.004
NODATA = 255


def encode_uint8(p):
    """P(solar) in [0, 1] (NaN = no data) -> uint8 codes, exactly as in the released dataset."""
    p = np.asarray(p, np.float32)
    return np.where(np.isnan(p), NODATA, np.round(p * 250)).astype(np.uint8)


def decode_uint8(code):
    """uint8 codes -> P(solar) as float32 (NaN where no data)."""
    code = np.asarray(code)
    return np.where(code == NODATA, np.nan, code * np.float32(SCALE)).astype(np.float32)


def _sigmoid(z):
    z = np.clip(z, -80.0, 80.0)                     # avoids exp overflow; identical uint8 codes
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


class NumpyMLP:
    """Forward pass of one trained MLP seed: standardise -> [Linear -> ReLU] x n -> Linear -> sigmoid."""

    def __init__(self, layers, mu=None, sd=None):
        self.layers = [(np.asarray(W, np.float32), np.asarray(b, np.float32)) for W, b in layers]
        self.mu = None if mu is None else np.asarray(mu, np.float32)
        self.sd = None if sd is None else np.asarray(sd, np.float32)

    def predict_p(self, X, batch=1 << 18):
        X = np.asarray(X, np.float32)
        out = np.empty(len(X), np.float32)
        for i in range(0, len(X), batch):
            h = X[i:i + batch]
            if self.mu is not None:
                h = (h - self.mu) / self.sd
            for k, (W, b) in enumerate(self.layers):
                h = h @ W.T + b
                if k < len(self.layers) - 1:
                    np.maximum(h, 0, out=h)
            out[i:i + batch] = _sigmoid(h[:, 0])
        return out


class SuitabilityModel:
    """RF + MLP-seed ensemble. `predict(X)` takes (n, 64) de-quantised AEF embeddings and returns P(solar)."""

    MLP_CHUNK = 16384                               # pixels per MLP work item: cache-sized, so all cores stay busy

    def __init__(self, rf, mlps, w_rf=0.5, meta=None):
        self.rf, self.mlps, self.w_rf = rf, list(mlps), float(w_rf)
        self.meta = meta or {}
        self.threads = 1

    def set_threads(self, n):
        """Threads used by the random forest (the dominant cost). NumPy's BLAS uses its own thread settings."""
        self.threads = max(1, int(n))
        if self.rf is not None:
            self.rf.n_jobs = self.threads
        return self

    def _mlp_mean(self, X):
        """Seed-averaged MLP probability, split into chunks across threads (each with single-threaded BLAS, so this
        scales on every BLAS: OpenBLAS, MKL, Apple Accelerate). Output is bit-identical to the unchunked computation."""
        b = self.MLP_CHUNK
        one = lambda i: np.mean([m.predict_p(X[i:i + b], batch=b) for m in self.mlps], axis=0)
        if self.threads == 1 or len(X) <= b:
            return np.concatenate([one(i) for i in range(0, len(X), b)])
        from concurrent.futures import ThreadPoolExecutor
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1, user_api="blas"), ThreadPoolExecutor(self.threads) as ex:
            return np.concatenate(list(ex.map(one, range(0, len(X), b))))

    def predict(self, X):
        X = np.asarray(X, np.float32)
        if len(X) == 0:
            return np.empty(0, np.float32)
        p_mlp = self._mlp_mean(X)
        if self.rf is None:                                   # lite variant: MLP seeds only
            return p_mlp.astype(np.float32)
        p_rf = self.rf.predict_proba(X)[:, 1].astype(np.float32)
        return (self.w_rf * p_rf + (1 - self.w_rf) * p_mlp).astype(np.float32)

    def predict_proba(self, X):
        p = self.predict(X)
        return np.c_[1 - p, p]


def sha256_16(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def load_model(model_dir, threads=None, verify=False, lite=False):
    """Load the released model from `model_dir` (see module docstring). `verify=True` checks file checksums.

    lite=True loads only the 5 MLP seeds (no random forest): < 1 GB of RAM instead of ~11 GB and several times faster,
    at a small accuracy cost (see README). Outputs are then labelled as the lite variant, not the released model."""
    import joblib

    meta = json.load(open(os.path.join(model_dir, "model.json")))
    if verify:
        for name, expected in meta.get("sha256_16", {}).items():
            got = sha256_16(os.path.join(model_dir, name))
            if got != expected:
                raise ValueError(f"{name}: checksum {got} != expected {expected} (corrupt or partial download?)")
    rf = None if lite else joblib.load(os.path.join(model_dir, "rf.joblib"))
    z = np.load(os.path.join(model_dir, "mlp.npz"))
    n_layers = len(meta["mlp"]["hidden"]) + 1
    mu = z["mu"] if "mu" in z.files else None
    sd = z["sd"] if "sd" in z.files else None
    mlps = [NumpyMLP([(z[f"W{k}_s{s}"], z[f"b{k}_s{s}"]) for k in range(n_layers)], mu, sd) for s in meta["mlp"]["seeds"]]
    if lite:
        meta = dict(meta, name=meta.get("name", "model") + " [lite: MLP seeds only, not the released model]")
    m = SuitabilityModel(rf, mlps, 0.0 if lite else meta["w_rf"], meta)
    m.set_threads(threads or os.cpu_count() or 1)
    return m
