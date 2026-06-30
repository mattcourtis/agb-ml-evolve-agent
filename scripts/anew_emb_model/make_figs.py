"""
Figures for the emb-only ANEW model experiment, from the CV artifacts.

  scheme_per_tier_rmse.png   weighting scheme x DI-tier LOPO RMSE (shows weighting doesn't help)
  per_project_lopo_vs_bloc.png  per-project LOPO vs leave-bloc-out RMSE, coloured by regdep class
  pred_vs_true.png           S0 LOPO out-of-fold predictions vs truth, coloured by DI tier

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/anew_emb_model/make_figs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402

EXP = common.DATASPACE / "agb_anew_emb_weighted_20260630"
CV = EXP / "cv"
FIG_DIR = common.REPO / "experiments/agb_anew_emb_weighted_20260630/figures"

TIERS = ["interior", "regional_frontier", "self_standing_frontier"]
TIER_C = {
    "interior": "#1f77b4",
    "regional_frontier": "#ff7f0e",
    "self_standing_frontier": "#d62728",
}


def scheme_per_tier(comp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(comp))
    cols = [
        ("rmse_interior", "interior"),
        ("rmse_regional_frontier", "regional_frontier"),
        ("rmse_self_standing_frontier", "self_standing_frontier"),
    ]
    width = 0.25
    for i, (col, tier) in enumerate(cols):
        ax.bar(x + (i - 1) * width, comp[col], width, color=TIER_C[tier], label=tier)
    ax.set_xticks(x)
    ax.set_xticklabels(comp["scheme"])
    ax.set_ylabel("LOPO RMSE (tCO2/acre, unweighted scoring)")
    ax.set_title("Weighting scheme x DI tier — no scheme improves the frontier over S0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scheme_per_tier_rmse.png", dpi=120)
    plt.close(fig)


def per_project(pp: pd.DataFrame) -> None:
    cmap = {
        "interior": "#1f77b4",
        "regional_frontier": "#ff7f0e",
        "self_standing_frontier": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(8, 8))
    for cls, c in cmap.items():
        s = pp[pp.regdep_class == cls]
        ax.scatter(s["rmse_lopo"], s["rmse_bloc"], c=c, s=40, label=cls)
    lim = [0, max(pp["rmse_bloc"].max(), pp["rmse_lopo"].max()) * 1.05]
    ax.plot(lim, lim, ls="--", c="#999", label="y = x")
    for _, r in pp.iterrows():
        if r["regdep_class"] != "interior":
            ax.annotate(r["project_name"], (r["rmse_lopo"], r["rmse_bloc"]), fontsize=6, alpha=0.7)
    ax.set_xlabel("LOPO RMSE (near transfer)")
    ax.set_ylabel("leave-bloc-out RMSE (far transfer)")
    ax.set_title("Per-project error: near vs far transfer, by regional-dependence class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_project_lopo_vs_bloc.png", dpi=120)
    plt.close(fig)


def pred_vs_true(oof: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for tier in TIERS:
        s = oof[oof.tier == tier]
        ax.scatter(s["CO2"], s["oof_S0"], s=6, alpha=0.3, c=TIER_C[tier], label=tier)
    lim = [0, float(np.percentile(oof["CO2"], 99.5))]
    ax.plot(lim, lim, ls="--", c="k")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("true CO2 (tCO2/acre)")
    ax.set_ylabel("LOPO OOF prediction (S0)")
    ax.set_title("Predicted vs true (LOPO OOF) — range compression at the high end")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pred_vs_true.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    scheme_per_tier(pd.read_parquet(CV / "comparison_matrix.parquet"))
    per_project(pd.read_parquet(CV / "per_project.parquet"))
    pred_vs_true(pd.read_parquet(CV / "oof_S0.parquet"))
    print(f"Saved 3 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
