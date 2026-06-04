"""
Disturbance-timing audit + LOPO experiment for the AGB USA pilot.

Consumes preprocessing/disturbance_timing_features.csv (from extract_disturbance_timing.py)
joined to the baseline preprocessing/features_iter3.parquet.

Part A — Contamination audit (no model):
  Bucket every plot by harvest timing relative to its field-survey year:
    undisturbed / pre-or-at-survey (delta<=0) / post-survey (delta>0).
  Report counts per region/project, target (biomass) distribution per bucket, how the
  existing `dist_years_since` mishandles post-survey plots, Hansen-vs-LandTrendr agreement.
  -> reports/disturbance_audit.md + reports/figures/dstx_audit_*.png

Part B — LOPO experiment (reuses the investigate_extended.py harness):
  4 configs, reporting R2/RMSE/MAE + per-quintile bias Q1..Q5:
    1 baseline (iter3)              2 + dstx (leakage-safe predictive)
    3 baseline on cleaned data      4 + dstx on cleaned data
  (cleaned = drop dstx_post_survey_flag==1 plots). Plus a Q1-focused analysis: within-Q1
  correlation of each dstx predictor with the baseline residual.
  -> reports/disturbance_timing_experiment.md + reports/figures/dstx_q1_*.png

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/run_disturbance_timing_experiment.py
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
)
PARQUET = EXPDIR / "preprocessing/features_iter3.parquet"
DSTX_CSV = EXPDIR / "preprocessing/disturbance_timing_features.csv"
FIGDIR = EXPDIR / "reports/figures"
AUDIT_MD = EXPDIR / "reports/disturbance_audit.md"
EXP_MD = EXPDIR / "reports/disturbance_timing_experiment.md"

# Baseline feature prefixes (matches investigate_extended.py) + new dstx predictive features.
BASE_PREFIXES = ("emb_", "gedi_", "chm_", "topo_", "dist_", "agbd_", "clim_")
DSTX_PREDICTIVE = [
    "dstx_pre_loss_5yr",
    "dstx_pre_ysd",
    "dstx_loss_frac_buf",
    "dstx_lt_mag",
]
SEED = 42


# ---------------------------------------------------------------------------
# Shared LOPO harness (copied from scripts/investigate_extended.py)
# ---------------------------------------------------------------------------


def lopo_folds(df: pd.DataFrame):
    projects = sorted(df["project_name"].unique())
    p2i = {p: i for i, p in enumerate(projects)}
    return df["project_name"].map(p2i).to_numpy(), projects


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "bias": round(float((y_pred - y_true).mean()), 2),
    }


def qbias(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    bins = np.quantile(y_true, [0.2, 0.4, 0.6, 0.8])
    lbls = np.digitize(y_true, bins)
    return {
        f"q{i + 1}": round(float((y_pred[lbls == i] - y_true[lbls == i]).mean()), 1)
        for i in range(5)
    }


def lgbm_model(**kw):
    return lgb.LGBMRegressor(
        n_estimators=3000,
        n_jobs=-1,
        random_state=SEED,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        verbose=-1,
        **kw,
    )


def lgbm_fit(model, X_tr, y_tr, X_va, y_va):
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )


def run_lopo(df: pd.DataFrame, feat_cols: list[str]):
    """Out-of-fold LOPO predictions for the given feature columns."""
    folds, projects = lopo_folds(df)
    X = df[feat_cols].astype("float32").to_numpy()
    y = df["target"].to_numpy()
    oof = np.zeros(len(y))
    for fid in range(len(projects)):
        tr, va = folds != fid, folds == fid
        m = lgbm_model()
        lgbm_fit(m, X[tr], y[tr], X[va], y[va])
        oof[va] = m.predict(X[va])
    return y, oof


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load() -> pd.DataFrame:
    base = pd.read_parquet(PARQUET).reset_index(drop=True)
    base["row_key"] = base.index.astype(str)
    dstx = pd.read_csv(DSTX_CSV, dtype={"row_key": str})
    dstx_cols = [c for c in dstx.columns if c.startswith("dstx_")]
    df = base.merge(dstx[["row_key"] + dstx_cols], on="row_key", how="left")
    df = df[df["failure"].isna()].reset_index(drop=True)
    print(f"Loaded {len(df)} modelled rows; dstx cols merged: {dstx_cols}")
    return df


def bucket(df: pd.DataFrame) -> pd.Series:
    """undisturbed / pre_at_survey / post_survey (Hansen calendar timing)."""
    b = pd.Series("undisturbed", index=df.index)
    has_loss = df["dstx_hansen_loss_year"].notna()
    delta = df["dstx_delta_survey"]
    b[has_loss & (delta <= 0)] = "pre_at_survey"
    b[has_loss & (delta > 0)] = "post_survey"
    # post-survey flag also catches LandTrendr-only post events
    b[(df["dstx_post_survey_flag"] == 1) & (b == "undisturbed")] = "post_survey"
    return b


# ---------------------------------------------------------------------------
# Part A — audit
# ---------------------------------------------------------------------------


def quintile_label(y: np.ndarray) -> np.ndarray:
    return np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))


def write_audit(df: pd.DataFrame) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    y = df["target"].to_numpy()
    df = df.copy()
    df["bucket"] = bucket(df)
    df["quintile"] = quintile_label(y) + 1

    lines = ["# Disturbance-Timing Contamination Audit\n"]
    lines.append(
        f"Dataset: `features_iter3.parquet` ⨝ `disturbance_timing_features.csv`, "
        f"{len(df)} modelled plots. Survey years: {sorted(df['year'].unique())}. "
        f"Buckets by Hansen loss year relative to plot survey year.\n"
    )

    # 1. bucket counts
    counts = df["bucket"].value_counts()
    lines.append("## 1. Bucket counts\n")
    lines.append("| bucket | n | % |")
    lines.append("| --- | --- | --- |")
    for bkt in ["undisturbed", "pre_at_survey", "post_survey"]:
        n = int(counts.get(bkt, 0))
        lines.append(f"| {bkt} | {n} | {100 * n / len(df):.1f}% |")
    n_post = int(counts.get("post_survey", 0))
    lines.append(
        f"\n**Post-survey contamination: {n_post} plots ({100 * n_post / len(df):.1f}%)** — "
        f"harvested *after* their field survey, so their field biomass is legitimately high "
        f"but 'current land cover' reads non-forest.\n"
    )

    # 2. counts per region
    lines.append("## 2. Bucket × region\n")
    ct = pd.crosstab(df["region"], df["bucket"])
    lines.append("| region | " + " | ".join(ct.columns) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(ct.columns)) + " |")
    for region, row in ct.iterrows():
        lines.append(f"| {region} | " + " | ".join(str(int(v)) for v in row) + " |")
    lines.append("")

    # 3. target distribution per bucket
    lines.append("## 3. Target (tCO₂/acre) by bucket\n")
    lines.append("| bucket | n | mean | median | Q1-share | Q5-share |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for bkt in ["undisturbed", "pre_at_survey", "post_survey"]:
        sub = df[df["bucket"] == bkt]
        if len(sub) == 0:
            continue
        q1 = (sub["quintile"] == 1).mean()
        q5 = (sub["quintile"] == 5).mean()
        lines.append(
            f"| {bkt} | {len(sub)} | {sub['target'].mean():.1f} | "
            f"{sub['target'].median():.1f} | {100 * q1:.0f}% | {100 * q5:.0f}% |"
        )
    lines.append(
        "\n*Hypothesis check:* pre-or-at-survey harvest should skew low (high Q1-share); "
        "post-survey plots should sit higher (measured before the cut).\n"
    )

    # 4. existing dist_years_since mishandling
    if "dist_years_since" in df.columns:
        lines.append("## 4. How the existing `dist_years_since` encodes each bucket\n")
        lines.append("| bucket | mean dist_years_since | % with years_since==0 |")
        lines.append("| --- | --- | --- |")
        for bkt in ["undisturbed", "pre_at_survey", "post_survey"]:
            sub = df[df["bucket"] == bkt]
            if len(sub) == 0:
                continue
            z = (sub["dist_years_since"] == 0).mean()
            lines.append(f"| {bkt} | {sub['dist_years_since'].mean():.1f} | {100 * z:.0f}% |")
        lines.append(
            "\nPost-survey plots receiving `years_since==0` are the feature–label inversion: "
            "a 'just disturbed' signal on a high-biomass plot.\n"
        )

    # 5. Hansen vs LandTrendr agreement
    if "dstx_lt_mag" in df.columns:
        LT_THRESH = 150.0  # ×1000 inverted-NBR rise ~ stand-replacing-scale disturbance
        lines.append("## 5. Hansen vs LandTrendr (pre/at-survey detection)\n")
        h = (df["dstx_pre_loss_5yr"] == 1) | (df["bucket"] == "pre_at_survey")
        lt = df["dstx_lt_mag"].fillna(0) > LT_THRESH
        lines.append(f"- Hansen pre/at-survey loss: {int(h.sum())} plots")
        lines.append(
            f"- LandTrendr pre/at-survey disturbance (mag>{LT_THRESH:.0f}): {int(lt.sum())} plots"
        )
        lines.append(
            f"- both: {int((h & lt).sum())}; LandTrendr-only: {int((lt & ~h).sum())} "
            f"(partial harvest / degradation Hansen's stand-replacing loss misses)\n"
        )

    AUDIT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {AUDIT_MD}")

    # --- figures ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["undisturbed", "pre_at_survey", "post_survey"]
    data = [df.loc[df["bucket"] == b, "target"].to_numpy() for b in order]
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_ylabel("target (tCO₂/acre)")
    ax.set_title("Biomass by disturbance-timing bucket")
    fig.tight_layout()
    fig.savefig(FIGDIR / "dstx_audit_target_by_bucket.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ct_q = pd.crosstab(df["bucket"], df["quintile"], normalize="index").reindex(order)
    ct_q.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_ylabel("share")
    ax.set_title("Biomass-quintile composition per bucket")
    ax.legend(title="quintile", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "dstx_audit_quintile_composition.png", dpi=120)
    plt.close(fig)
    print(f"Wrote audit figures to {FIGDIR}")


# ---------------------------------------------------------------------------
# Part B — LOPO experiment
# ---------------------------------------------------------------------------


def write_experiment(df: pd.DataFrame) -> None:
    base_cols = [c for c in df.columns if c.startswith(BASE_PREFIXES)]
    dstx_cols = [c for c in DSTX_PREDICTIVE if c in df.columns]
    # fill dstx nulls (a few plots may miss LandTrendr) with neutral sentinels
    for c in dstx_cols:
        if c.endswith("_ysd"):
            df[c] = df[c].fillna(100.0)
        else:
            df[c] = df[c].fillna(0.0)

    configs = []

    print("\n[config 1] baseline (iter3) ...")
    y, oof = run_lopo(df, base_cols)
    configs.append(("1_baseline", metrics(y, oof), qbias(y, oof), oof, df.index))

    print("[config 2] + dstx ...")
    y, oof = run_lopo(df, base_cols + dstx_cols)
    configs.append(("2_baseline+dstx", metrics(y, oof), qbias(y, oof), oof, df.index))

    clean = df[df["dstx_post_survey_flag"] != 1].reset_index(drop=True)
    print(f"[config 3] baseline on cleaned data (n={len(clean)}) ...")
    yc, oofc = run_lopo(clean, base_cols)
    configs.append(("3_baseline_clean", metrics(yc, oofc), qbias(yc, oofc), oofc, clean.index))

    print("[config 4] + dstx on cleaned data ...")
    yc, oofc = run_lopo(clean, base_cols + dstx_cols)
    configs.append(("4_clean+dstx", metrics(yc, oofc), qbias(yc, oofc), oofc, clean.index))

    lines = ["# Disturbance-Timing LOPO Experiment\n"]
    lines.append(
        f"Baseline `features_iter3.parquet` ({len(df)} plots), LightGBM 23-project LOPO. "
        f"dstx predictive features: {dstx_cols}. Cleaned configs drop "
        f"{int((df['dstx_post_survey_flag'] == 1).sum())} post-survey-contaminated plots.\n"
    )
    lines.append("| config | n | R² | RMSE | MAE | bias | Q1 | Q2 | Q3 | Q4 | Q5 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, m, qb, oof, idx in configs:
        n = len(oof)
        lines.append(
            f"| {name} | {n} | {m['r2']:.4f} | {m['rmse']:.2f} | {m['mae']:.2f} | "
            f"{m['bias']:+.1f} | {qb['q1']:+.1f} | {qb['q2']:+.1f} | {qb['q3']:+.1f} | "
            f"{qb['q4']:+.1f} | {qb['q5']:+.1f} |"
        )
    lines.append(
        "\n**Read:** success = Q1 bias shrinks toward 0 (config 2 vs 1) without harming Q5 or "
        "overall R²; config 3 lifts R² if contamination is material.\n"
    )

    # --- Q1-focused: within-Q1 correlation of dstx predictors with baseline residual ---
    y = df["target"].to_numpy()
    _, oof_base = run_lopo(df, base_cols)
    resid = oof_base - y  # +ve = over-prediction
    q = quintile_label(y)
    q1m = q == 0
    lines.append("## Within-Q1 correlation of dstx predictors with baseline residual\n")
    lines.append(
        "(+resid = over-prediction; a strong −corr means the feature flags the plots "
        "the baseline over-predicts.)\n"
    )
    lines.append("| feature | corr (Q1 only) | corr (all) |")
    lines.append("| --- | --- | --- |")
    for c in dstx_cols:
        v = df[c].to_numpy()
        cq = np.corrcoef(v[q1m], resid[q1m])[0, 1] if q1m.sum() > 2 else np.nan
        ca = np.corrcoef(v, resid)[0, 1]
        lines.append(f"| {c} | {cq:+.3f} | {ca:+.3f} |")
    lines.append("")

    EXP_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {EXP_MD}")

    # --- figure: Q1 bias across configs ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [c[0] for c in configs]
    q1v = [c[2]["q1"] for c in configs]
    q5v = [c[2]["q5"] for c in configs]
    x = np.arange(len(names))
    ax.bar(x - 0.2, q1v, 0.4, label="Q1 bias")
    ax.bar(x + 0.2, q5v, 0.4, label="Q5 bias")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("mean residual (tCO₂/acre)")
    ax.set_title("Per-quintile bias across configs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "dstx_q1_bias_by_config.png", dpi=120)
    plt.close(fig)
    print(f"Wrote experiment figure to {FIGDIR}")


def main() -> None:
    df = load()
    print("\n=== PART A: AUDIT ===")
    write_audit(df)
    print("\n=== PART B: LOPO EXPERIMENT ===")
    write_experiment(df)


if __name__ == "__main__":
    main()
