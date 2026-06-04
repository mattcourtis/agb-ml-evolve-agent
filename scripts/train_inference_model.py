"""
Train the deployment LightGBM model for wall-to-wall AGB inference.

Feature set (74) — "embeddings + disturbance-aware co-features":
  emb_00..63 (64) + chm_m + topo_{elevation,slope,aspect_cos,aspect_sin,tpi} (5)
  + dstx_pre_ysd (corrected survey-relative dist) + dstx_pre_loss_5yr + dstx_loss_frac_buf
  + dstx_lt_mag (4)

Trained on ALL 4,636 modelled plots (no held-out fold — this is the deployed model). n_estimators
is chosen via early stopping on a random 15% holdout, then the model is refit on all data with
that fixed count. Saves the Booster + the exact feature order so inference uses identical columns.

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/train_inference_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
)
PARQUET = EXPDIR / "preprocessing/features_iter3.parquet"
DSTX_CSV = EXPDIR / "preprocessing/disturbance_timing_features.csv"
MODEL_DIR = Path("/home/mattc/code/agb-ml-agent-evolve/models")
MODEL_TXT = MODEL_DIR / "inference_model.txt"
FEATS_JSON = MODEL_DIR / "inference_features.json"

# NB: dstx_lt_mag (LandTrendr) dropped — it is ~97% nodata wall-to-wall over Bayfield's
# lake-spanning bbox, so it cannot be rasterised reliably for inference (planned fallback).
EMB = [f"emb_{i:02d}" for i in range(64)]
TOPO = ["topo_elevation", "topo_slope", "topo_aspect_cos", "topo_aspect_sin", "topo_tpi"]
DSTX = ["dstx_pre_ysd", "dstx_pre_loss_5yr", "dstx_loss_frac_buf"]
FULL_FEATURES = EMB + ["chm_m"] + TOPO + DSTX  # 73, fixed order

SEED = 42


def load() -> pd.DataFrame:
    base = pd.read_parquet(PARQUET).reset_index(drop=True)
    base["row_key"] = base.index.astype(str)
    dstx = pd.read_csv(DSTX_CSV, dtype={"row_key": str})
    df = base.merge(dstx[["row_key"] + DSTX], on="row_key", how="left")
    df = df[df["failure"].isna()].reset_index(drop=True)
    df["dstx_pre_ysd"] = df["dstx_pre_ysd"].fillna(100.0)
    for c in ["dstx_pre_loss_5yr", "dstx_loss_frac_buf"]:
        df[c] = df[c].fillna(0.0)
    print(f"Loaded {len(df)} plots.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embonly", action="store_true", help="train on the 64 embeddings only")
    ap.add_argument("--embdstx", action="store_true", help="embeddings + dstx (no static layers)")
    args = ap.parse_args()
    if args.embonly:
        features, tag = EMB, "embonly"
    elif args.embdstx:
        features, tag = EMB + DSTX, "embdstx"  # dynamic-only: no chm/topo
    else:
        features, tag = FULL_FEATURES, None
    suffix = f"_{tag}" if tag else ""
    model_txt = MODEL_DIR / f"inference_model{suffix}.txt"
    feats_json = MODEL_DIR / f"inference_features{suffix}.json"
    print(f"feature set: {tag or 'full'} ({len(features)})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    missing = [c for c in features if c not in df.columns]
    assert not missing, f"missing feature cols: {missing}"
    X = df[features].astype("float32").to_numpy()
    y = df["target"].to_numpy()

    # pick n_estimators via early stopping on a random holdout
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=SEED)
    probe = lgb.LGBMRegressor(
        n_estimators=3000,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    probe.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    best_iter = probe.best_iteration_ or 500
    print(f"early-stopping best_iteration = {best_iter}")

    # refit on ALL data with the fixed count
    model = lgb.LGBMRegressor(
        n_estimators=best_iter,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)

    # train-fit sanity (in-sample, not a generalisation estimate)
    pred = model.predict(X)
    r2_insample = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"in-sample R²={r2_insample:.3f} (sanity only); target range {y.min():.0f}-{y.max():.0f}")

    model.booster_.save_model(str(model_txt))
    feats_json.write_text(
        json.dumps(
            {
                "features": features,
                "n_features": len(features),
                "n_estimators": int(best_iter),
                "target": "CO2 standing stock, tCO2/acre",
                "target_range": [float(y.min()), float(y.max())],
                "trained_on": "all 4636 plots (incl. Bayfield — in-sample there)",
            },
            indent=2,
        )
    )
    print(f"Saved {model_txt}\nSaved {feats_json}")

    # top feature importances (gain)
    imp = pd.Series(model.booster_.feature_importance("gain"), index=features).sort_values(
        ascending=False
    )
    print("\nTop 12 features by gain:")
    print(imp.head(12).round(0).to_string())


if __name__ == "__main__":
    main()
