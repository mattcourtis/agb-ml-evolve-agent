"""
DI — Dissimilarity Index in two flavours, in training-codec space.

Primary: importance-weighted CAST DI (Meyer & Pebesma 2021, "Predicting into unknown
space"). Each predictor is standardised on the training stats and scaled by its model
gain-importance weight; DI of a point = nearest-neighbour weighted-Euclidean distance to
the training set, normalised by the mean pairwise distance within training. The AOA
threshold is CAST's Q75 + 1.5·IQR of the (fold-aware) training DI.

Cross-check: Mahalanobis DI over the 64 embeddings, lifted from the Ireland
run_bias_characterisation.py, with the 99th-pct training radius.

Importable (DISpace) and runnable (fits both spaces on the deployed training cloud,
prints thresholds, saves a reusable bundle to the data-space).

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/trust/di.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

RNG = np.random.default_rng(42)


@dataclass
class DISpace:
    """Fitted CAST DI space: standardisation + weights + normaliser + threshold."""

    features: list[str]
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray  # per-feature importance weight (mean 1)
    dbar: float  # mean pairwise distance among training (normaliser)
    threshold_cast: float
    p95: float
    p99: float
    train_di: np.ndarray = field(repr=False)
    _nn: NearestNeighbors = field(repr=False, default=None)
    _zt: np.ndarray = field(repr=False, default=None)  # transformed training (for BLAS path)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) * self.w

    def di(self, X: np.ndarray) -> np.ndarray:
        """DI for query points: NN weighted distance to training / dbar."""
        d, _ = self._nn.kneighbors(self.transform(X), n_neighbors=1)
        return d[:, 0] / self.dbar

    def di_fast(self, X: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """Raster-scale DI via chunked BLAS min-distance (||x-t||^2 expansion)."""
        Zq = self.transform(X)
        zt = self._zt
        tnorm = (zt**2).sum(1)  # ||t||^2
        out = np.empty(len(Zq))
        for s in range(0, len(Zq), chunk):
            q = Zq[s : s + chunk]
            d2 = (q**2).sum(1)[:, None] + tnorm[None, :] - 2.0 * q @ zt.T
            out[s : s + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0)) / self.dbar
        return out

    def inside(self, X: np.ndarray) -> np.ndarray:
        return self.di(X) <= self.threshold_cast


def _mean_pairwise(Zt: np.ndarray, sample: int = 3000) -> float:
    idx = RNG.choice(len(Zt), size=min(sample, len(Zt)), replace=False)
    S = Zt[idx]
    # mean of pairwise Euclidean distances within the sample
    sq = np.maximum(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1), 0.0)
    d = np.sqrt(sq)
    iu = np.triu_indices(len(S), k=1)
    return float(d[iu].mean())


def fit(X: np.ndarray, projects: np.ndarray, features: list[str], w: np.ndarray) -> DISpace:
    """Fit a CAST DI space with fold-aware (leave-project-out) training DI."""
    finite = np.isfinite(X).all(1)
    X, projects = X[finite], np.asarray(projects)[finite]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Zt = ((X - mu) / sd) * w
    dbar = _mean_pairwise(Zt)

    # fold-aware training DI: NN distance to OTHER-project training points
    train_di = np.empty(len(Zt))
    for p in np.unique(projects):
        mask = projects == p
        nn = NearestNeighbors(n_neighbors=1).fit(Zt[~mask])
        d, _ = nn.kneighbors(Zt[mask], n_neighbors=1)
        train_di[mask] = d[:, 0] / dbar

    q75, q25 = np.percentile(train_di, [75, 25])
    thr = float(q75 + 1.5 * (q75 - q25))
    nn_all = NearestNeighbors(n_neighbors=1).fit(Zt)
    return DISpace(
        features=features,
        mu=mu,
        sd=sd,
        w=w,
        dbar=dbar,
        threshold_cast=thr,
        p95=float(np.percentile(train_di, 95)),
        p99=float(np.percentile(train_di, 99)),
        train_di=train_di,
        _nn=nn_all,
        _zt=Zt,
    )


def mahalanobis_emb(
    X_train_emb: np.ndarray, X_query_emb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mahalanobis DI over the 64 embeddings (cross-check). Returns (train_di, query_di)."""
    mu = X_train_emb.mean(0)
    cov_inv = np.linalg.pinv(np.cov(X_train_emb, rowvar=False))

    def md(X):
        diff = X - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))

    return md(X_train_emb), md(X_query_emb)


def main() -> None:
    out = common.TRUST_OUT / "trust"
    out.mkdir(parents=True, exist_ok=True)
    tr = common.load_full_training()
    summary = {}
    for space in ("full", "embonly"):
        features = common.SPACES[space][0]
        X = tr[features].astype(float).to_numpy()
        w = common.gain_weights(space)
        dsp = fit(X, tr["project_name"].to_numpy(), features, w)
        # persist a reusable bundle
        np.savez(
            out / f"di_space_{space}.npz",
            features=np.array(features),
            mu=dsp.mu,
            sd=dsp.sd,
            w=dsp.w,
            dbar=dsp.dbar,
            threshold_cast=dsp.threshold_cast,
            p95=dsp.p95,
            p99=dsp.p99,
            train_di=dsp.train_di,
        )
        summary[space] = {
            "n_features": len(features),
            "n_train": int(np.isfinite(X).all(1).sum()),
            "dbar": dsp.dbar,
            "threshold_cast": dsp.threshold_cast,
            "p95": dsp.p95,
            "p99": dsp.p99,
            "top_weighted_features": [features[i] for i in np.argsort(dsp.w)[::-1][:5]],
        }
        print(
            f"[{space}] {len(features)} feats | dbar={dsp.dbar:.3f} | "
            f"CAST thr={dsp.threshold_cast:.3f} | p95={dsp.p95:.3f} p99={dsp.p99:.3f}"
        )
    (out / "di_thresholds.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved DI bundles + thresholds to {out}")


if __name__ == "__main__":
    main()
