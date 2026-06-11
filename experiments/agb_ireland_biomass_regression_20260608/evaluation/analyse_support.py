"""Analyse the change-of-support gap: stats, correlations, figures.

Reads support_sensitivity_stands.parquet (written by scripts/per_pixel_inference.py) and
produces the gap distribution, correlations with heterogeneity, and the two figures.
    uv run --project /home/mattc/code/agb-ml-agent-evolve python evaluation/analyse_support.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_ireland_biomass_regression_20260608"
)
STANDS = EXPDIR / "evaluation/support_sensitivity_stands.parquet"
FIGDIR = EXPDIR / "evaluation/figures"


def main() -> None:
    df = pd.read_parquet(STANDS).sort_values("n_subcpt").reset_index(drop=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    print(f"n stands: {len(df)}")
    print("\nConsistency (emb codec pixel-mean vs iter0 polygon-mean, max abs per band):")
    print(df["emb_consistency_max_abs"].describe()[["min", "50%", "max"]].to_dict())
    print("\nf(mean) reproduction vs reported iter0 (should be ~0):")
    df["f_mean_repro_err"] = (df["f_mean"] - df["f_mean_iter0"]).abs()
    print("  max abs err:", df["f_mean_repro_err"].max())

    print("\nGap stats (mean(f) - f(mean)):")
    print(
        "  tCO2/acre:", df["gap"].describe()[["min", "25%", "50%", "75%", "max"]].round(3).to_dict()
    )
    print(
        "  %        :",
        df["gap_pct"].describe()[["min", "25%", "50%", "75%", "max"]].round(2).to_dict(),
    )
    print(
        "  mean gap%:",
        round(df["gap_pct"].mean(), 2),
        "| mean |gap%|:",
        round(df["gap_pct"].abs().mean(), 2),
    )
    print("  n positive (mean(f)>f(mean)):", int((df["gap"] > 0).sum()), "/", len(df))
    print(
        "  n |gap%|>5:",
        int((df["gap_pct"].abs() > 5).sum()),
        "| >10:",
        int((df["gap_pct"].abs() > 10).sum()),
    )

    print("\nCorrelations of |gap%| with drivers (Spearman):")
    for col in ["n_subcpt", "area_ha", "pix_pred_std", "pix_pred_iqr", "age_at_survey"]:
        rho, p = stats.spearmanr(df["gap_pct"].abs(), df[col])
        print(f"  |gap%| vs {col:14s}: rho={rho:+.3f} p={p:.3f}")
    print("Correlations of signed gap% with drivers (Spearman):")
    for col in ["n_subcpt", "area_ha", "pix_pred_std", "pix_pred_iqr"]:
        rho, p = stats.spearmanr(df["gap_pct"], df[col])
        print(f"  gap%  vs {col:14s}: rho={rho:+.3f} p={p:.3f}")

    # ---- Figure 1: f(mean) vs mean(f) scatter with 1:1 line ----
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(
        df["f_mean"], df["mean_f"], c=df["n_subcpt"], cmap="viridis", s=60, edgecolor="k", zorder=3
    )
    lo = min(df["f_mean"].min(), df["mean_f"].min()) - 5
    hi = max(df["f_mean"].max(), df["mean_f"].max()) + 5
    ax.plot([lo, hi], [lo, hi], "r--", label="1:1 (no support gap)", zorder=1)
    for _, r in df.iterrows():
        ax.annotate(r["Location_Name"][:10], (r["f_mean"], r["mean_f"]), fontsize=6, alpha=0.6)
    ax.set_xlabel("f(mean) — reported polygon-mean estimator (tCO₂/acre)")
    ax.set_ylabel("mean(f) — per-pixel-then-aggregate (tCO₂/acre)")
    ax.set_title("Change-of-support: f(mean) vs mean(f), embdstx head, Ireland")
    fig.colorbar(sc, ax=ax, label="n_subcpt (heterogeneity)")
    ax.legend()
    ax.set_aspect("equal")
    fig.savefig(FIGDIR / "support_scatter_fmean_vs_meanf.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: gap vs heterogeneity ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(df["n_subcpt"], df["gap_pct"], s=60, c="tab:blue", edgecolor="k")
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].axhline(5, color="r", ls=":", lw=0.8)
    axes[0].axhline(-5, color="r", ls=":", lw=0.8)
    axes[0].set_xlabel("n_subcpt (heterogeneity proxy)")
    axes[0].set_ylabel("gap % = 100·(mean(f) − f(mean))/f(mean)")
    axes[0].set_title("Gap vs heterogeneity")
    axes[1].scatter(df["pix_pred_std"], df["gap_pct"], s=60, c="tab:orange", edgecolor="k")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("within-stand per-pixel prediction std (tCO₂/acre)")
    axes[1].set_ylabel("gap %")
    axes[1].set_title("Gap vs within-stand dispersion")
    fig.savefig(FIGDIR / "support_gap_vs_heterogeneity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote figures to {FIGDIR}")
    cols = [
        "Location_Name",
        "n_subcpt",
        "area_ha",
        "n_pixels",
        "f_mean",
        "mean_f",
        "gap",
        "gap_pct",
        "pix_pred_min",
        "pix_pred_median",
        "pix_pred_max",
        "pix_pred_std",
    ]
    print(df[cols].round(2).to_string())


if __name__ == "__main__":
    main()
