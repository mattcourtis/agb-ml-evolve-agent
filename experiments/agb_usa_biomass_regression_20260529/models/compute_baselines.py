"""Simple-baseline floor for agb_usa iteration 0.

Computes leave-one-project-out (LOPO) reference baselines on the reused joint_v2 feature
table, to sit beneath the LightGBM reproduction:
  - mean-predictor: predict each held-out project with the training-fold global mean.
  - ridge-on-PC20: 20-component PCA on the 64-dim embeddings + Ridge, fit per fold.

This is a generic evaluation-side comparator (sklearn LeaveOneGroupOut). It does NOT
reimplement the sibling LightGBM trainer; it exists only to quantify the embeddings linear
floor required by configs (require_simple_baseline: true).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = Path(
    "/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet"
)
OUT = Path(__file__).resolve().parent / "baseline_metrics.json"
SEED = 42


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def main() -> None:
    df = pd.read_parquet(FEATURES)
    if "failure" in df.columns:
        # match the sibling trainer: drop rows with any failure marker (non-null).
        df = df[df["failure"].isna()].copy()
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].to_numpy()
    y = df["target"].to_numpy()
    groups = df["project_name"].to_numpy()

    logo = LeaveOneGroupOut()
    oof_mean = np.full(len(df), np.nan)
    oof_ridge = np.full(len(df), np.nan)

    for tr, va in logo.split(X, y, groups):
        oof_mean[va] = y[tr].mean()
        pipe = make_pipeline(
            StandardScaler(), PCA(n_components=20, random_state=SEED), Ridge(alpha=1.0)
        )
        pipe.fit(X[tr], y[tr])
        oof_ridge[va] = pipe.predict(X[va])

    result = {
        "n_folds": int(len(np.unique(groups))),
        "mean_predictor": _metrics(y, oof_mean),
        "ridge_pc20": _metrics(y, oof_ridge),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
