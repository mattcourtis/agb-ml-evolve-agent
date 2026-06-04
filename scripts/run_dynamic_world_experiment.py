"""
Dynamic World experiment for the AGB USA pilot: observed-vs-predicted, with and without DW.

Two things demonstrated:
  1. The `dist_years_since` FIX — the baseline here drops the broken `dist_years_since`
     (from features_iter3.parquet) and substitutes the corrected, survey-relative, leakage-safe
     `dstx_pre_ysd` (from disturbance_timing_features.csv). A before/after table quantifies the
     change, focused on the post-survey-contaminated plots.
  2. DYNAMIC WORLD as a low-biomass / forest<->non-forest prior. Two LOPO configs:
        A. baseline (corrected dist, no DW)
        B. baseline + DW probability bands
     reported with R²/RMSE/MAE + per-quintile bias, and the headline deliverable: a side-by-side
     observed-vs-predicted scatter (OOF LOPO) without vs with DW.

Reuses the LOPO harness from run_disturbance_timing_experiment.py.

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/run_dynamic_world_experiment.py
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
DW_CSV = EXPDIR / "preprocessing/dynamic_world_features.csv"
FIGDIR = EXPDIR / "reports/figures"
OUT_MD = EXPDIR / "reports/dynamic_world_experiment.md"

# iter3 baseline prefixes WITHOUT dist_ (the broken feature) — corrected dist added explicitly.
BASE_PREFIXES = ("emb_", "gedi_", "chm_", "topo_", "agbd_", "clim_")
CORRECTED_DIST = "dstx_pre_ysd"


def load() -> pd.DataFrame:
    base = pd.read_parquet(PARQUET).reset_index(drop=True)
    base["row_key"] = base.index.astype(str)
    dstx = pd.read_csv(DSTX_CSV, dtype={"row_key": str})
    dw = pd.read_csv(DW_CSV, dtype={"row_key": str})
    dstx_cols = [c for c in dstx.columns if c.startswith("dstx_")]
    dw_cols = [c for c in dw.columns if c.startswith("dw_")]
    df = base.merge(dstx[["row_key"] + dstx_cols], on="row_key", how="left")
    df = df.merge(dw[["row_key"] + dw_cols], on="row_key", how="left")
    df = df[df["failure"].isna()].reset_index(drop=True)
    # neutral fills
    df[CORRECTED_DIST] = df[CORRECTED_DIST].fillna(100.0)
    for c in dw_cols:
        df[c] = df[c].fillna(df[c].median())
    print(f"Loaded {len(df)} modelled rows; dw cols: {dw_cols}")
    return df, dw_cols


def dist_fix_table(df: pd.DataFrame) -> list[str]:
    """Before (broken dist_years_since) vs after (corrected dstx_pre_ysd)."""
    lines = ["## 1. `dist_years_since` fix (before → after)\n"]
    broken = df["dist_years_since"]
    fixed = df[CORRECTED_DIST]
    changed = (broken != fixed).sum()
    post = df["dstx_post_survey_flag"] == 1
    post_zero_before = ((broken == 0) & post).sum()
    post_fixed_to_100 = ((fixed == 100) & post).sum()
    lines.append(f"- plots whose value changed: **{int(changed)}** / {len(df)}")
    lines.append(
        f"- post-survey-harvest plots (n={int(post.sum())}): "
        f"{int(post_zero_before)} had broken `years_since==0` (a 'just-disturbed' signal on "
        f"high-biomass plots); after the fix {int(post_fixed_to_100)} are correctly set to the "
        f"undisturbed sentinel (100)."
    )
    ex = df[post & (broken == 0)].head(1)
    if len(ex):
        r = ex.iloc[0]
        lines.append(
            f"- example post-survey plot (plot_id={r.get('plot_id')}): "
            f"broken={r['dist_years_since']:.0f} → fixed={r[CORRECTED_DIST]:.0f}, "
            f"target={r['target']:.0f} tCO₂/acre (legitimately high)."
        )
    lines.append("")
    return lines


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df, dw_cols = load()

    base_cols = [c for c in df.columns if c.startswith(BASE_PREFIXES)] + [CORRECTED_DIST]
    print(f"baseline feature count: {len(base_cols)} (incl. corrected dist); +{len(dw_cols)} DW")

    y = df["target"].to_numpy()
    print("\n[config A] baseline (corrected dist, no DW) ...")
    _, oof_base = run_lopo(df, base_cols)
    print("[config B] baseline + Dynamic World ...")
    _, oof_dw = run_lopo(df, base_cols + dw_cols)

    mA, qA = metrics(y, oof_base), qbias(y, oof_base)
    mB, qB = metrics(y, oof_dw), qbias(y, oof_dw)

    # --- report ---
    lines = ["# Dynamic World Experiment — observed vs predicted\n"]
    lines.append(
        f"Baseline = `features_iter3.parquet` with the **disturbance fix** "
        f"(broken `dist_years_since` replaced by survey-relative `{CORRECTED_DIST}`), "
        f"{len(df)} plots, LightGBM 23-project LOPO. DW = survey-year growing-season buffer-mean "
        f"probability bands {dw_cols}.\n"
    )
    lines += dist_fix_table(df)

    lines.append("## 2. LOPO: with vs without Dynamic World\n")
    lines.append("| config | R² | RMSE | MAE | bias | Q1 | Q2 | Q3 | Q4 | Q5 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, m, q in [("baseline (no DW)", mA, qA), ("baseline + DW", mB, qB)]:
        lines.append(
            f"| {name} | {m['r2']:.4f} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['bias']:+.1f} | "
            f"{q['q1']:+.1f} | {q['q2']:+.1f} | {q['q3']:+.1f} | {q['q4']:+.1f} | {q['q5']:+.1f} |"
        )
    lines.append(
        f"\nΔR² from DW = **{mB['r2'] - mA['r2']:+.4f}**; ΔQ1 bias = "
        f"{qB['q1'] - qA['q1']:+.1f} (negative = less over-prediction of low-biomass plots).\n"
    )

    # Q1 correlation of DW bands with baseline residual
    resid = oof_base - y
    q = quintile_label(y)
    q1m = q == 0
    lines.append("## 3. Within-Q1 correlation of DW bands with baseline residual\n")
    lines.append("(+resid = over-prediction; −corr ⇒ band flags the over-predicted low plots)\n")
    lines.append("| band | corr (Q1) | corr (all) |")
    lines.append("| --- | --- | --- |")
    for c in dw_cols:
        v = df[c].to_numpy()
        cq = np.corrcoef(v[q1m], resid[q1m])[0, 1] if q1m.sum() > 2 else np.nan
        lines.append(f"| {c} | {cq:+.3f} | {np.corrcoef(v, resid)[0, 1]:+.3f} |")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")

    # --- headline figure: observed vs predicted, without vs with DW ---
    qcol = quintile_label(y) + 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    lim = [0, float(np.percentile(y, 99.5))]
    for ax, oof, m, title in [
        (axes[0], oof_base, mA, "Without Dynamic World"),
        (axes[1], oof_dw, mB, "With Dynamic World"),
    ]:
        sc = ax.scatter(y, oof, c=qcol, cmap="viridis", s=8, alpha=0.4)
        ax.plot(lim, lim, "k--", lw=1.2, label="1:1")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("observed AGB (tCO₂/acre)")
        ax.set_title(
            f"{title}\nR²={m['r2']:.3f}  RMSE={m['rmse']:.1f}  Q1 bias={qbias(y, oof)['q1']:+.1f}"
        )
        ax.legend(loc="upper left")
    axes[0].set_ylabel("predicted AGB (tCO₂/acre)")
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("observed biomass quintile")
    fig.suptitle("Observed vs predicted AGB (LOPO out-of-fold) — effect of Dynamic World", y=1.02)
    fig.savefig(FIGDIR / "dw_pred_vs_obs.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIGDIR / 'dw_pred_vs_obs.png'}")


if __name__ == "__main__":
    main()
