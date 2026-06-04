"""
Observed-vs-predicted AGB with and without the disturbance-timing features, focused on whether
low-biomass (Q1) predictions are pulled down toward the 1:1 line.

Configs (LOPO out-of-fold, LightGBM):
  A. baseline           = features_iter3 (as-is)
  B. baseline + dstx     = + survey-relative disturbance-timing features
                           (dstx_pre_loss_5yr, dstx_pre_ysd, dstx_loss_frac_buf, dstx_lt_mag)

Outputs:
  reports/figures/disturbance_pred_vs_obs.png   — 3 panels:
     (A) pred-vs-obs baseline, (B) pred-vs-obs +dstx, (C) Δprediction vs observed
  reports/disturbance_pred_vs_obs.md            — Q1 summary (how many low plots reduced)

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/plot_disturbance_pred_vs_obs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_disturbance_timing_experiment import (  # noqa: E402
    metrics,
    qbias,
    quintile_label,
    run_lopo,
)

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
)
PARQUET = EXPDIR / "preprocessing/features_iter3.parquet"
DSTX_CSV = EXPDIR / "preprocessing/disturbance_timing_features.csv"
FIGDIR = EXPDIR / "reports/figures"
OUT_MD = EXPDIR / "reports/disturbance_pred_vs_obs.md"

BASE_PREFIXES = ("emb_", "gedi_", "chm_", "topo_", "dist_", "agbd_", "clim_")
DSTX = ["dstx_pre_loss_5yr", "dstx_pre_ysd", "dstx_loss_frac_buf", "dstx_lt_mag"]


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(PARQUET).reset_index(drop=True)
    base["row_key"] = base.index.astype(str)
    dstx = pd.read_csv(DSTX_CSV, dtype={"row_key": str})
    df = base.merge(dstx[["row_key"] + DSTX], on="row_key", how="left")
    df = df[df["failure"].isna()].reset_index(drop=True)
    for c in DSTX:
        df[c] = df[c].fillna(100.0 if c.endswith("_ysd") else 0.0)

    base_cols = [c for c in df.columns if c.startswith(BASE_PREFIXES)]
    y = df["target"].to_numpy()

    print("[A] baseline ...")
    _, oof_a = run_lopo(df, base_cols)
    print("[B] baseline + dstx ...")
    _, oof_b = run_lopo(df, base_cols + DSTX)

    mA, qA = metrics(y, oof_a), qbias(y, oof_a)
    mB, qB = metrics(y, oof_b), qbias(y, oof_b)
    q = quintile_label(y)  # 0..4
    q1 = q == 0

    # --- Q1 reduction summary ---
    delta = oof_b - oof_a  # change in prediction from adding dstx
    q1_reduced = (delta[q1] < 0).mean()
    q1_mean_delta = delta[q1].mean()
    lines = ["# Disturbance-timing features — observed vs predicted (low-biomass focus)\n"]
    lines.append(
        f"{len(df)} plots, LOPO. A = baseline (`features_iter3`); B = baseline + {DSTX}.\n"
    )
    lines.append("| config | R² | RMSE | Q1 bias | Q5 bias |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        f"| A baseline | {mA['r2']:.4f} | {mA['rmse']:.2f} | {qA['q1']:+.1f} | {qA['q5']:+.1f} |"
    )
    lines.append(
        f"| B + dstx | {mB['r2']:.4f} | {mB['rmse']:.2f} | {qB['q1']:+.1f} | {qB['q5']:+.1f} |"
    )
    lines.append(
        f"\n**Q1 (lowest-biomass quintile, n={int(q1.sum())}):** "
        f"adding dstx **reduced the prediction for {100 * q1_reduced:.0f}% of Q1 plots** "
        f"(mean change {q1_mean_delta:+.1f} tCO₂/acre). Q1 over-prediction "
        f"{qA['q1']:+.1f} → {qB['q1']:+.1f}.\n"
    )
    # how many of the most over-predicted plots improved
    big_over = q1 & (oof_a - y > 50)
    if big_over.sum():
        improved = ((oof_b - y)[big_over] < (oof_a - y)[big_over]).mean()
        lines.append(
            f"Among Q1 plots the baseline over-predicts by >50 tCO₂/acre (n={int(big_over.sum())}), "
            f"{100 * improved:.0f}% improved with dstx.\n"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")

    # --- figure ---
    qcol = q + 1
    lim = [0, float(np.percentile(y, 99.5))]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    for ax, oof, m, title in [
        (axes[0], oof_a, mA, "A. Baseline (no disturbance features)"),
        (axes[1], oof_b, mB, "B. + disturbance-timing features"),
    ]:
        ax.scatter(y, oof, c=qcol, cmap="viridis", s=8, alpha=0.4)
        ax.plot(lim, lim, "k--", lw=1.2, label="1:1")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("observed AGB (tCO₂/acre)")
        ax.set_ylabel("predicted AGB (tCO₂/acre)")
        ax.set_title(f"{title}\nR²={m['r2']:.3f}  Q1 bias={qbias(y, oof)['q1']:+.1f}")
        ax.legend(loc="upper left")

    # Panel C: change in prediction vs observed — does adding dstx pull low plots DOWN?
    axC = axes[2]
    sc = axC.scatter(y, delta, c=qcol, cmap="viridis", s=8, alpha=0.4)
    axC.axhline(0, color="k", lw=1.0)
    axC.set_xlim(lim)
    axC.set_xlabel("observed AGB (tCO₂/acre)")
    axC.set_ylabel("Δ prediction  (B − A, tCO₂/acre)")
    axC.set_title("C. Prediction change from dstx\n(below 0 = pulled down)")
    cb = fig.colorbar(sc, ax=axC, fraction=0.046, pad=0.02)
    cb.set_label("observed biomass quintile")

    fig.suptitle(
        "Observed vs predicted AGB (LOPO) — effect of disturbance-timing features on low biomass",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "disturbance_pred_vs_obs.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIGDIR / 'disturbance_pred_vs_obs.png'}")


if __name__ == "__main__":
    main()
