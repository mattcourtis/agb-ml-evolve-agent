"""Compute the biomass-specific evaluation metrics for agb_usa iteration 0.

Reads the out-of-fold (OOF) predictions from the joint_v2 baseline run (bit-identical to the
iteration-0 retrain — see reports/training_run.md), joins true ecoregion (ECO_NAME) from the
ANEW gpkg on (project_name, Plot_ID), and emits the full metric set required by
references/evaluation.md.

Metric definitions (references/evaluation.md):
  per_quintile_bias            mean signed residual (pred - true) per true-target quintile Q1..Q5
  predicted_range_discrimination
                               (pred_Q5_mean - pred_Q1_mean) / (true_Q5_mean - true_Q1_mean)
  per_ecoregion_r2             R2 within each ECO_NAME group
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OOF = Path("/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/oof.parquet")
GPKG = Path("/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg")
OUT = Path(__file__).resolve().parent / "biomass_metrics.json"


def _agg(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "bias": float(np.mean(p - y)),
        "n": int(len(y)),
    }


def main() -> None:
    oof = pd.read_parquet(OOF)
    y = oof["target"].to_numpy()
    p = oof["pred"].to_numpy()

    overall = _agg(y, p)

    # --- per_quintile_bias + predicted_range_discrimination ---
    q = pd.qcut(oof["target"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    d = oof.assign(quintile=q, residual=oof["pred"] - oof["target"])
    per_quintile_bias = {
        str(k): float(v) for k, v in d.groupby("quintile", observed=True)["residual"].mean().items()
    }
    true_means = d.groupby("quintile", observed=True)["target"].mean()
    pred_means = d.groupby("quintile", observed=True)["pred"].mean()
    prd = float((pred_means["Q5"] - pred_means["Q1"]) / (true_means["Q5"] - true_means["Q1"]))

    # --- per_ecoregion_r2 (join ECO_NAME on (project_name, Plot_ID)) ---
    eco = gpd.read_file(GPKG)[["project_name", "Plot_ID", "ECO_NAME"]]
    m = oof.copy()
    m["Plot_ID"] = m["plot_id"]
    m = m.merge(eco, on=["project_name", "Plot_ID"], how="left")
    assert m["ECO_NAME"].isna().sum() == 0, "ecoregion join produced nulls"
    per_ecoregion_r2 = {}
    for name, grp in m.groupby("ECO_NAME"):
        if len(grp) >= 2:
            per_ecoregion_r2[str(name)] = {
                "r2": float(r2_score(grp["target"], grp["pred"])),
                "rmse": float(np.sqrt(mean_squared_error(grp["target"], grp["pred"]))),
                "mae": float(mean_absolute_error(grp["target"], grp["pred"])),
                "bias": float(np.mean(grp["pred"] - grp["target"])),
                "n": int(len(grp)),
            }

    # --- error_by_region (bloc) ---
    error_by_region = {}
    for name, grp in oof.groupby("region"):
        error_by_region[str(name)] = {
            "r2": float(r2_score(grp["target"], grp["pred"])),
            "rmse": float(np.sqrt(mean_squared_error(grp["target"], grp["pred"]))),
            "mae": float(mean_absolute_error(grp["target"], grp["pred"])),
            "bias": float(np.mean(grp["pred"] - grp["target"])),
            "n": int(len(grp)),
        }

    # --- predicted-range discrimination per region (for error analysis) ---
    prd_by_region = {}
    for name, grp in oof.groupby("region"):
        gq = pd.qcut(grp["target"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        gg = grp.assign(quintile=gq)
        tm = gg.groupby("quintile", observed=True)["target"].mean()
        pm = gg.groupby("quintile", observed=True)["pred"].mean()
        prd_by_region[str(name)] = float((pm["Q5"] - pm["Q1"]) / (tm["Q5"] - tm["Q1"]))

    # --- calibration: residual mean per predicted-value decile ---
    pdec = pd.qcut(oof["pred"], 10, labels=False, duplicates="drop")
    calib = (
        oof.assign(dec=pdec, residual=oof["pred"] - oof["target"]).groupby("dec")["residual"].mean()
    )
    calibration_max_abs_resid = float(calib.abs().max())

    result = {
        "overall": overall,
        "per_quintile_bias": per_quintile_bias,
        "per_quintile_true_mean": {str(k): float(v) for k, v in true_means.items()},
        "per_quintile_pred_mean": {str(k): float(v) for k, v in pred_means.items()},
        "predicted_range_discrimination": prd,
        "predicted_range_discrimination_by_region": prd_by_region,
        "per_ecoregion_r2": per_ecoregion_r2,
        "error_by_region": error_by_region,
        "calibration_max_abs_decile_residual": calibration_max_abs_resid,
        "external_holdout_r2": overall["r2"],  # LOPO aggregate = new-project expectation
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
