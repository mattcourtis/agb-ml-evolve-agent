"""Shared loaders and constants for the trust (DI/AOA) module.

Everything here works in TRAINING-CODEC space (what the deployed LightGBM model sees).
No GEE. See plan compiled-scribbling-umbrella.md.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
DATASPACE = Path("/home/mattc/data-space/carbonmap-embeddings")
USA_EXP = REPO / "experiments/agb_usa_biomass_regression_20260529"
TRUST_OUT = DATASPACE / "agb_trust_aoa_20260626"

CANONICAL = TRUST_OUT / "preprocessing/anew_canonical_codec.parquet"
MODELS = REPO / "models"

EMB = [f"emb_{i:02d}" for i in range(64)]
TOPO = ["topo_elevation", "topo_slope", "topo_aspect_cos", "topo_aspect_sin", "topo_tpi"]
DSTX = ["dstx_pre_ysd", "dstx_pre_loss_5yr", "dstx_loss_frac_buf"]
FULL_FEATURES = EMB + ["chm_m"] + TOPO + DSTX  # 73, fixed order (matches train_inference_model.py)

# feature space -> (feature list, deployed model used for importance weights)
SPACES = {
    "full": (FULL_FEATURES, MODELS / "inference_model.txt"),
    "embdstx": (EMB + DSTX, MODELS / "inference_model_embdstx.txt"),
    "embonly": (EMB, MODELS / "inference_model_embonly.txt"),
}


def load_full_training() -> pd.DataFrame:
    """Deployed-model training cloud (23 projects, codec) with the corrected dstx merged.

    Replicates scripts/train_inference_model.py::load() exactly.
    """
    base = pd.read_parquet(USA_EXP / "preprocessing/features_iter3.parquet").reset_index(drop=True)
    base["row_key"] = base.index.astype(str)
    dstx = pd.read_csv(
        USA_EXP / "preprocessing/disturbance_timing_features.csv", dtype={"row_key": str}
    )
    df = base.merge(dstx[["row_key"] + DSTX], on="row_key", how="left")
    df = df[df["failure"].isna()].reset_index(drop=True)
    df["dstx_pre_ysd"] = df["dstx_pre_ysd"].fillna(100.0)
    for c in ["dstx_pre_loss_5yr", "dstx_loss_frac_buf"]:
        df[c] = df[c].fillna(0.0)
    return df


def load_canonical() -> pd.DataFrame:
    """All 52 ANEW projects in codec space (embeddings only + CO2 + eco + lon/lat)."""
    return pd.read_parquet(CANONICAL)


def gain_weights(space: str) -> np.ndarray:
    """Per-feature gain importance from the matching deployed model, normalised to mean 1.

    Falls back to uniform weights if the model file is absent.
    """
    features, model_path = SPACES[space]
    if not model_path.exists():
        return np.ones(len(features))
    booster = lgb.Booster(model_file=str(model_path))
    imp = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    names = booster.feature_name()
    # align to our feature order
    idx = {n: i for i, n in enumerate(names)}
    aligned = np.array([imp[idx[f]] if f in idx else 0.0 for f in features])
    if aligned.sum() <= 0:
        return np.ones(len(features))
    return aligned / aligned.mean()
