"""
Apply the ship/no-ship rule to the weighting comparison, then train the final model.

Decision rule (vs S0 baseline) -- ship the best frontier scheme only if ALL hold, else S0:
  G1 regional-frontier RMSE improves >= 10% relative,
  G2 interior RMSE degrades <= 3% relative,
  G3 no regional-frontier range-compression regression,
  G4 n_eff >= 0.85 N.
Self-standing-frontier not improving is NOT a veto (documented "needs local GT" boundary).

Trains the final LightGBM on ALL 51 projects with the chosen scheme's weights and the fixed
n_estimators, then saves the model + feature schema (candidate; data-space).

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/anew_emb_model/decide_and_train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_ladder as cvl  # noqa: E402
import data as data_mod  # noqa: E402

EXP = common.DATASPACE / "agb_anew_emb_weighted_20260630"
CV = EXP / "cv"
MODELS = EXP / "models"

G1_FRONTIER_GAIN = 0.10
G2_INTERIOR_BUDGET = 0.03
G4_NEFF_FLOOR = 0.85


def decide(comp: pd.DataFrame) -> tuple[str, list[dict]]:
    """Return (chosen_scheme, per-scheme gate audit). Falls back to S0."""
    s0 = comp[comp.scheme == "S0"].iloc[0]
    audit = []
    eligible = []
    for _, r in comp[comp.scheme != "S0"].iterrows():
        g1 = (s0.rmse_regional_frontier - r.rmse_regional_frontier) / s0.rmse_regional_frontier
        g2 = (r.rmse_interior - s0.rmse_interior) / s0.rmse_interior
        g3 = r.range_regional_frontier >= s0.range_regional_frontier
        g4 = r.n_eff_frac >= G4_NEFF_FLOOR
        passed = (g1 >= G1_FRONTIER_GAIN) and (g2 <= G2_INTERIOR_BUDGET) and g3 and g4
        audit.append(
            {
                "scheme": r.scheme,
                "frontier_gain": round(g1, 4),
                "interior_cost": round(g2, 4),
                "no_range_regression": bool(g3),
                "n_eff_ok": bool(g4),
                "passes": bool(passed),
            }
        )
        if passed:
            eligible.append((r.rmse_regional_frontier, r.scheme))
    chosen = min(eligible)[1] if eligible else "S0"
    return chosen, audit


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    comp = pd.read_parquet(CV / "comparison_matrix.parquet")
    n_trees = json.loads((CV / "n_trees.json").read_text())["n_estimators"]
    chosen, audit = decide(comp)

    print("gate audit (vs S0):")
    print(pd.DataFrame(audit).to_string(index=False))
    print(
        f"\nchosen scheme: {chosen}"
        + ("" if chosen != "S0" else "  (no weighting scheme cleared the gates -> unweighted)")
    )

    # train final model on all 51 with chosen scheme
    df = data_mod.load_eligible()
    X = df[cvl.EMB].astype(float).to_numpy()
    y = df["CO2"].to_numpy()
    w, info = data_mod.build_weights(df, chosen)
    model = lgb.LGBMRegressor(n_estimators=n_trees, **cvl.LGB_PARAMS)
    model.fit(pd.DataFrame(X, columns=cvl.EMB), y, sample_weight=w)
    model.booster_.save_model(str(MODELS / "anew_emb51_model.txt"))

    schema = {
        "features": cvl.EMB,
        "n_features": len(cvl.EMB),
        "n_estimators": n_trees,
        "feature_space": "embonly-64-codec",
        "n_train": int(len(df)),
        "n_projects": int(df.project_name.nunique()),
        "target": "CO2 (tCO2/acre, raw, uncapped)",
        "target_stats": {
            "min": float(y.min()),
            "median": float(np.median(y)),
            "mean": float(y.mean()),
            "max": float(y.max()),
        },
        "weighting_scheme": chosen,
        "n_eff_frac": round(info["n_eff_frac"], 3),
        "lgb_params": cvl.LGB_PARAMS,
        "note": "candidate; ships with trust layer (see trust_fit.py). NOT auto-promoted to repo models/.",
    }
    (MODELS / "anew_emb51_features.json").write_text(json.dumps(schema, indent=2))
    (CV / "decision.json").write_text(
        json.dumps(
            {"chosen_scheme": chosen, "n_estimators": n_trees, "gate_audit": audit}, indent=2
        )
    )
    print(f"\nTrained final model (scheme={chosen}, n_trees={n_trees}) -> {MODELS}")


if __name__ == "__main__":
    main()
