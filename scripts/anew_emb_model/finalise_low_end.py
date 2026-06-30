"""
Finalise the low-end work: train + save the chosen variant (log1p) candidate, make figures.

The bake-off (low_end.py) selected the log1p target transform: it cuts the <100 bias ~45% and
the <100 RMSE ~24% while slightly improving low-band discrimination, at an accepted high-end
trade. Here we (1) retrain that variant on all 51 projects and save a candidate booster + schema
(predictions need expm1 then clip-at-0), and (2) render the low-end figures from the CV artifacts.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/anew_emb_model/finalise_low_end.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_ladder as cvl  # noqa: E402
import data as data_mod  # noqa: E402

EXP = common.DATASPACE / "agb_anew_emb_weighted_20260630"
LE = EXP / "low_end"
MODELS = EXP / "models"
FIG_DIR = common.REPO / "experiments/agb_anew_emb_weighted_20260630/figures"
HIGHLIGHT = ["S0_raw", "T_log1p", "T_sqrt", "tweedie_1.5", "hurdle_t50", "Cal_isotonic"]


def train_chosen() -> str:
    chosen = json.loads((LE / "decision.json").read_text())["chosen"]
    df = data_mod.load_eligible()
    X = pd.DataFrame(df[cvl.EMB].astype(float).to_numpy(), columns=cvl.EMB)
    y = df["CO2"].to_numpy()
    n_trees = json.loads((cvl.OUT / "n_trees.json").read_text())["n_estimators"]
    assert chosen == "T_log1p", f"final-train wired for log1p; chosen={chosen}"
    model = lgb.LGBMRegressor(n_estimators=n_trees, **cvl.LGB_PARAMS).fit(X, np.log1p(y))
    model.booster_.save_model(str(MODELS / "anew_emb51_log1p_model.txt"))
    schema = {
        "features": cvl.EMB,
        "n_features": len(cvl.EMB),
        "n_estimators": n_trees,
        "feature_space": "embonly-64-codec",
        "target": "CO2 (tCO2/acre)",
        "target_transform": "log1p",
        "inverse": "expm1 then clip at 0",
        "n_train": int(len(df)),
        "n_projects": int(df.project_name.nunique()),
        "note": "low-end de-biased candidate; predict = clip(expm1(booster.predict(X)), 0). "
        "Ships with the same DI/AOA trust layer. NOT auto-promoted.",
    }
    (MODELS / "anew_emb51_log1p_features.json").write_text(json.dumps(schema, indent=2))
    return chosen


def fig_band_bias(band: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = [b for b in band[band.variant == "S0_raw"]["band"]]
    for v in HIGHLIGHT:
        s = band[band.variant == v].set_index("band").reindex(order)
        lw = 3 if v in ("S0_raw", "T_log1p") else 1.3
        ax.plot(order, s["bias"], marker="o", lw=lw, label=v)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel("bias = mean(pred − true) (tCO2/acre)")
    ax.set_xlabel("true CO2 band")
    ax.set_title("Conditional bias by biomass band — log1p halves the <100 over-prediction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "low_end_band_bias.png", dpi=120)
    plt.close(fig)


def fig_pred_vs_true(oof: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3), sharex=True, sharey=True)
    y = oof["CO2"].to_numpy()
    lim = [0, float(np.percentile(y, 99.5))]
    for ax, v, t in [
        (axes[0], "S0_raw", "S0 raw-L2 (mean-seeking)"),
        (axes[1], "T_log1p", "log1p (de-biased low end)"),
    ]:
        ax.scatter(y, oof[v], s=5, alpha=0.15, c="#1f77b4")
        ax.plot(lim, lim, ls="--", c="k")
        ax.axvline(100, ls=":", c="#d62728")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("true CO2 (tCO2/acre)")
        ax.set_title(t)
    axes[0].set_ylabel("LOPO OOF prediction")
    fig.suptitle("Predicted vs true — low-end lift toward the 1:1 line under log1p")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "low_end_pred_vs_true.png", dpi=120)
    plt.close(fig)


def fig_roc(oof: pd.DataFrame) -> None:
    y = oof["CO2"].to_numpy()
    lab = (y < 50).astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for v in ["S0_raw", "T_log1p", "hurdle_t50"]:
        fpr, tpr, _ = roc_curve(lab, -oof[v].to_numpy())
        ax.plot(fpr, tpr, label=v)
    ax.plot([0, 1], [0, 1], ls="--", c="#999")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Separating true < 50 tCO2/acre (low biomass is moderately separable, AUC ~0.88)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "low_end_roc_lt50.png", dpi=120)
    plt.close(fig)


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    chosen = train_chosen()
    fig_band_bias(pd.read_parquet(LE / "band_bias.parquet"))
    fig_pred_vs_true(pd.read_parquet(LE / "oof_by_variant.parquet"))
    fig_roc(pd.read_parquet(LE / "oof_by_variant.parquet"))
    print(f"Trained {chosen} candidate -> {MODELS}; saved 3 low-end figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
