"""
SPATIAL CV — honest out-of-fold residuals for the uncertainty surface.

Refits the full-feature LightGBM head (same params as the deployed model) under two
spatial schemes on the existing 23-project training data (validation only — not a new
deployed model, per scope):
  - LOPO: leave-one-project-out (23 folds) — near/adjacent-expansion analogue.
  - leave-bloc-out: 3 folds (mw/ne/wv) — far-expansion floor.

Also records each plot's fold-aware full-space CAST DI (NN distance to OTHER-project
training points), so the OOF residuals can be paired with DI in uncertainty.py.

Output (data-space): trust/oof_residuals.parquet + per-fold metrics json.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/trust/spatial_cv.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import di as di_mod  # noqa: E402

OUT = common.TRUST_OUT / "trust"
N_ESTIMATORS = 143  # deployed full model's selected n_estimators (inference_features.json)
PARAMS = dict(
    num_leaves=31, learning_rate=0.05, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1
)


def cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions, refitting per leave-one-group-out fold."""
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, **PARAMS)
        model.fit(pd.DataFrame(X[~te], columns=common.FULL_FEATURES), y[~te])
        oof[te] = model.predict(pd.DataFrame(X[te], columns=common.FULL_FEATURES))
    return oof


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = common.load_full_training()
    feats = common.FULL_FEATURES
    X = df[feats].astype(float).to_numpy()
    y = df["target"].to_numpy()
    finite = np.isfinite(X).all(1)
    df, X, y = df[finite].reset_index(drop=True), X[finite], y[finite]
    proj = df["project_name"].to_numpy()
    region = df["region"].to_numpy()

    # fold-aware full-space DI (reuse the CAST fit; train_di is leave-project-out NN)
    dsp = di_mod.fit(X, proj, feats, common.gain_weights("full"))
    df["di_full"] = dsp.train_di

    df["oof_lopo"] = cv_predict(X, y, proj)
    df["oof_bloc"] = cv_predict(X, y, region)
    df["resid_lopo"] = df["oof_lopo"] - y
    df["resid_bloc"] = df["oof_bloc"] - y

    keep = [
        "project_name",
        "region",
        "target",
        "di_full",
        "oof_lopo",
        "oof_bloc",
        "resid_lopo",
        "resid_bloc",
    ]
    df[keep].to_parquet(OUT / "oof_residuals.parquet", index=False)

    def metrics(resid, mask=None):
        r = resid if mask is None else resid[mask]
        rmse = float(np.sqrt(np.mean(r**2)))
        return rmse

    summary = {
        "n": int(len(df)),
        "lopo_rmse": metrics(df["resid_lopo"].to_numpy()),
        "bloc_rmse": metrics(df["resid_bloc"].to_numpy()),
        "lopo_rmse_by_region": {
            reg: metrics(df["resid_lopo"].to_numpy(), (region == reg)) for reg in np.unique(region)
        },
        "bloc_rmse_by_region": {
            reg: metrics(df["resid_bloc"].to_numpy(), (region == reg)) for reg in np.unique(region)
        },
        "di_full_threshold_cast": dsp.threshold_cast,
    }
    (OUT / "spatial_cv_metrics.json").write_text(json.dumps(summary, indent=2))
    print(
        f"LOPO RMSE = {summary['lopo_rmse']:.2f} | leave-bloc-out RMSE = {summary['bloc_rmse']:.2f}"
    )
    print(
        "  LOPO RMSE by region:",
        {k: round(v, 1) for k, v in summary["lopo_rmse_by_region"].items()},
    )
    print(
        "  bloc RMSE by region:",
        {k: round(v, 1) for k, v in summary["bloc_rmse_by_region"].items()},
    )
    print(f"Saved {OUT / 'oof_residuals.parquet'}")


if __name__ == "__main__":
    main()
