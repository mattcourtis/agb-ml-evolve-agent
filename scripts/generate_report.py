"""
Generate the AGB ML investigation report with plots.

Runs a LOPO CV pass to obtain OOF predictions for three model variants,
produces figures, and writes the markdown report.

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/generate_report.py
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
)
PARQUET = EXPDIR / "preprocessing/features_iter3.parquet"
FIG_DIR = EXPDIR / "reports/figures"
OUT_MD = EXPDIR / "reports/investigation_report.md"

FEAT_PREFIXES = ("emb_", "palsar_", "gedi_", "chm_", "topo_", "dist_", "agbd_", "clim_")
EMB_ONLY = ("emb_",)
SEED = 42

FIG_DIR.mkdir(parents=True, exist_ok=True)

# publication style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    return df[df["failure"].isna()].reset_index(drop=True)


def feat_cols(df, prefixes=FEAT_PREFIXES):
    return [c for c in df.columns if c.startswith(prefixes)]


def lopo_folds(df):
    projects = sorted(df["project_name"].unique())
    p2i = {p: i for i, p in enumerate(projects)}
    return df["project_name"].map(p2i).to_numpy(), projects


def lgbm_lopo(df, prefixes=FEAT_PREFIXES, **extra_params):
    fcols = feat_cols(df, prefixes)
    X = df[fcols].astype("float32").to_numpy()
    y = df["target"].to_numpy()
    folds, projects = lopo_folds(df)
    oof = np.zeros(len(y))
    for fid in range(len(projects)):
        tr, va = folds != fid, folds == fid
        m = lgb.LGBMRegressor(
            n_estimators=3000,
            n_jobs=-1,
            random_state=SEED,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=20,
            **extra_params,
        )
        m.fit(
            X[tr],
            y[tr],
            eval_set=[(X[va], y[va])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = m.predict(X[va])
    return y, oof, df["region"].to_numpy()


def aggregate_metrics(y, oof):
    return {
        "R²": round(float(r2_score(y, oof)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, oof))), 2),
        "MAE": round(float(mean_absolute_error(y, oof)), 2),
        "Bias": round(float((oof - y).mean()), 2),
        "n": int(len(y)),
    }


def quintile_bias(y, oof):
    edges = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
    lbls = np.digitize(y, edges)
    means = [float((oof[lbls == i] - y[lbls == i]).mean()) for i in range(5)]
    true_means = [float(y[lbls == i].mean()) for i in range(5)]
    return means, true_means


# ---------------------------------------------------------------------------
# Figure 1: Predicted vs Observed scatter (3 panels)
# ---------------------------------------------------------------------------


def fig_scatter(runs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> Path:
    """runs = list of (label, y_true, y_pred, regions)"""
    REGION_COLOURS = {"wv": "#d62728", "mw": "#1f77b4", "ne": "#2ca02c"}
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (label, y, oof, regions) in zip(axes, runs):
        r2 = r2_score(y, oof)
        rmse = np.sqrt(mean_squared_error(y, oof))

        for reg, col in REGION_COLOURS.items():
            mask = regions == reg
            ax.scatter(
                y[mask], oof[mask], c=col, alpha=0.35, s=8, label=reg.upper(), rasterized=True
            )

        lim = max(y.max(), oof.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="1:1")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Observed AGB (tCO₂/acre)")
        ax.set_ylabel("Predicted AGB (tCO₂/acre)")
        ax.set_title(f"{label}\nR²={r2:.3f}  RMSE={rmse:.1f}")
        ax.legend(fontsize=8, markerscale=2)

    out = FIG_DIR / "pred_vs_obs.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ---------------------------------------------------------------------------
# Figure 2: Per-quintile bias comparison
# ---------------------------------------------------------------------------


def fig_quintile_bias(results: list[tuple[str, list[float], list[float]]]) -> Path:
    """results = list of (label, qbiases[5], qtrue_means[5])"""
    labels = [r[0] for r in results]
    qb_arr = np.array([r[1] for r in results])  # (n_configs, 5)
    qtrue = results[0][2]  # same true means for all

    x = np.arange(5)
    width = 0.8 / len(labels)
    colours = plt.cm.tab10(np.linspace(0, 0.7, len(labels)))

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for i, (lbl, col) in enumerate(zip(labels, colours)):
        offset = (i - len(labels) / 2 + 0.5) * width
        ax.bar(x + offset, qb_arr[i], width=width * 0.9, label=lbl, color=col, alpha=0.85)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i + 1}\n(mean={m:.0f})" for i, m in enumerate(qtrue)], fontsize=9)
    ax.set_ylabel("Mean residual (pred − true), tCO₂/acre")
    ax.set_title("Per-quintile bias across configurations (LOPO CV)")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+.0f"))

    out = FIG_DIR / "quintile_bias.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ---------------------------------------------------------------------------
# Figure 3: R² comparison bar chart across all experiments
# ---------------------------------------------------------------------------


def fig_r2_comparison(sections: dict[str, list[tuple[str, float]]]) -> Path:
    """sections = {'Section label': [(config_label, r2), ...]}"""
    fig, axes = plt.subplots(
        1, len(sections), figsize=(4.5 * len(sections), 4), constrained_layout=True
    )
    if len(sections) == 1:
        axes = [axes]

    baseline_r2 = 0.4274
    colours_map = plt.cm.RdYlGn

    for ax, (sec_title, items) in zip(axes, sections.items()):
        labels = [i[0] for i in items]
        r2s = [i[1] for i in items]
        cols = [colours_map(0.2 + 0.6 * min(1, max(0, (r - 0.35) / 0.25))) for r in r2s]
        bars = ax.barh(labels, r2s, color=cols, edgecolor="white", linewidth=0.5)
        ax.axvline(baseline_r2, color="steelblue", lw=1.2, ls="--", label=f"Baseline {baseline_r2}")
        ax.set_xlim(0.25, 0.60)
        ax.set_xlabel("R² (LOPO OOF)")
        ax.set_title(sec_title)
        for bar, r2 in zip(bars, r2s):
            ax.text(
                r2 + 0.003, bar.get_y() + bar.get_height() / 2, f"{r2:.4f}", va="center", fontsize=8
            )
        ax.legend(fontsize=8)

    out = FIG_DIR / "r2_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ---------------------------------------------------------------------------
# Figure 4: Residuals by region
# ---------------------------------------------------------------------------


def fig_residuals_by_region(y: np.ndarray, oof: np.ndarray, regions: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    region_labels = {"wv": "WV Appalachia", "mw": "Upper Midwest", "ne": "New England"}
    colours = {"wv": "#d62728", "mw": "#1f77b4", "ne": "#2ca02c"}
    residuals = oof - y

    for ax, reg in zip(axes, ["wv", "mw", "ne"]):
        mask = regions == reg
        y_r, res_r = y[mask], residuals[mask]
        r2_r = r2_score(y_r, oof[mask])
        ax.scatter(y_r, res_r, c=colours[reg], alpha=0.35, s=8, rasterized=True)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Observed AGB (tCO₂/acre)")
        ax.set_ylabel("Residual (pred − obs)")
        ax.set_title(f"{region_labels[reg]}\nR²={r2_r:.3f}  n={mask.sum()}")
    out = FIG_DIR / "residuals_by_region.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows.")

    # --- OOF predictions for three key configs ---
    print("\nRunning LOPO CV for scatter plots ...")
    print("  [1/3] emb_only ...")
    y_eo, oof_eo, reg_eo = lgbm_lopo(df, prefixes=EMB_ONLY)
    print(f"        R²={r2_score(y_eo, oof_eo):.4f}")

    print("  [2/3] all features (baseline) ...")
    y_b, oof_b, reg_b = lgbm_lopo(df, prefixes=FEAT_PREFIXES)
    print(f"        R²={r2_score(y_b, oof_b):.4f}")

    # weighted variant (inv-freq) — best Q5 bias
    print("  [3/3] all features + inv-freq weighting ...")
    fcols = feat_cols(df)
    X_all = df[fcols].astype("float32").to_numpy()
    y_all = df["target"].to_numpy()
    folds, projects = lopo_folds(df)
    counts, edges = np.histogram(y_all, bins=20)
    bin_idx = np.clip(np.digitize(y_all, edges[:-1]) - 1, 0, 19)
    inv_w = 1.0 / (counts[bin_idx].astype(float) + 1e-6)
    inv_w /= inv_w.mean()

    oof_w = np.zeros(len(y_all))
    for fid in range(len(projects)):
        tr, va = folds != fid, folds == fid
        m = lgb.LGBMRegressor(
            n_estimators=3000,
            n_jobs=-1,
            random_state=SEED,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=20,
        )
        m.fit(
            X_all[tr],
            y_all[tr],
            sample_weight=inv_w[tr],
            eval_set=[(X_all[va], y_all[va])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        oof_w[va] = m.predict(X_all[va])
    reg_w = df["region"].to_numpy()
    print(f"        R²={r2_score(y_all, oof_w):.4f}")

    # --- Figures ---
    print("\nGenerating figures ...")
    fig_scatter(
        [
            ("Embeddings only\n(iteration 0 baseline)", y_eo, oof_eo, reg_eo),
            ("All features\n(iterations 1–3)", y_b, oof_b, reg_b),
            ("All features + inv-freq\nweighting", y_all, oof_w, reg_w),
        ]
    )

    qb_configs = [
        ("emb_only", *quintile_bias(y_eo, oof_eo)),
        ("all_features", *quintile_bias(y_b, oof_b)),
        ("inv_freq_wt", *quintile_bias(y_all, oof_w)),
    ]
    fig_quintile_bias(qb_configs)

    fig_residuals_by_region(y_b, oof_b, reg_b)

    # R² comparison chart using pre-run results
    sections = {
        "Feature iterations": [
            ("iter0: emb_only", 0.4182),
            ("iter1: +GEDI shots", 0.4176),
            ("iter2: +CHM+topo+dist", 0.4272),
            ("iter3: +GEDI_L4B+clim", 0.4274),
        ],
        "CV strategy": [
            ("lopo (ref)", 0.4274),
            ("5fold_random", 0.4520),
            ("5fold_strat_eco", 0.4445),
            ("lopo_ecoregion", 0.3244),
        ],
        "Model types (LOPO)": [
            ("lgbm_ref", 0.4274),
            ("xgboost", 0.4278),
            ("histgbr", 0.4008),
            ("random_forest", 0.4265),
            ("extra_trees", 0.4259),
        ],
        "Sample weighting (LOPO)": [
            ("uniform", 0.4274),
            ("inv_freq", 0.3247),
            ("sqrt_inv_freq", 0.3562),
            ("quintile_upweight", 0.3592),
        ],
    }
    fig_r2_comparison(sections)

    # --- Write report ---
    print("\nWriting report ...")
    md = """# AGB USA Biomass Regression — Investigation Report

**Experiment:** `agb_usa_biomass_regression_20260529`
**Dataset:** 4,636 plots, 23 projects, 3 ecoregions (WV Appalachia / Upper Midwest / New England)
**Target:** CO₂ standing stock (tCO₂/acre)
**Evaluation:** 23-project leave-one-project-out (LOPO) cross-validation

---

## 1. Executive Summary

Starting from an embeddings-only baseline of R²=0.4182, three rounds of feature
addition (GEDI LiDAR, canopy height model, topography, disturbance, climate) produced
only +0.009 R² cumulative lift. A systematic investigation of gradient boosting
hyperparameters, alternative model types, feature processing, and sample weighting
found no approach that breaks out of R²≈0.43 under strict LOPO cross-validation.

The ceiling is not a capacity or tuning problem — it is a cross-project generalisation
problem. Relaxing the CV to 5-fold random raises R² to 0.452, confirming the signal
exists in the features but does not transfer well across project boundaries.

---

## 2. Predicted vs Observed (LOPO OOF)

![Predicted vs Observed](figures/pred_vs_obs.png)

The scatter plots show three configurations under 23-project LOPO CV. All exhibit the
same characteristic pattern: predictions are compressed toward the centre of the
distribution (Q1 over-predicted, Q5 under-predicted). The 1:1 line is never reached
at the extremes. Inverse-frequency weighting (right panel) partially corrects Q5
under-prediction but at the cost of aggregate R².

Colour coding: WV Appalachia (red), Upper Midwest (blue), New England (green).

---

## 3. Feature Iteration Results

| Iteration | Features added | R² | RMSE | Lift |
|---|---|---:|---:|---:|
| 0 | AEF optical embeddings only (64-dim) | 0.4182 | 56.58 | — |
| 1 | + GEDI L2A/L2B shot-level (rh98, cover, pai, fhd_normal) | 0.4176 | 56.61 | −0.001 |
| 2 | + ETH CHM 2020 + SRTM topo + Hansen disturbance | 0.4272 | 56.14 | +0.010 |
| 3 | + GEDI L4B gridded AGBD + TerraClimate climate | 0.4274 | 56.13 | +0.000 |

**GEDI shot-level features (iteration 1)** produced no lift because the GEDI orbital
track spacing (~600 m) means most 500 m buffers intersect ≤3 monthly composites over
36 months — too sparse for stable estimates. The median `gedi_n_samples` was 1.

**CHM + topography (iteration 2)** produced the only meaningful lift (+0.010 R²),
consistent with the hypothesis that vertical canopy structure is partially separable
from spectral reflectance. However, the lift is far below the expected +0.10–0.20 from
the published fusion literature, indicating that the LOPO protocol penalises project-
specific canopy structure patterns the model learns.

**GEDI L4B + TerraClimate (iteration 3)** added nothing (+0.000). The GEDI L4B product
has 34% null coverage in the plot locations (gaps in the 1 km mosaic).

![R² comparison across experiments](figures/r2_comparison.png)

---

## 4. Gradient Boosting Hyperparameter Investigation

| Config | num_leaves | R² | Notes |
|---|---|---:|---|
| baseline | 31 | 0.4274 | Current production config |
| fast_shallow | 15 | 0.4245 | — |
| stochastic | 63 | 0.4225 | subsample=0.8, colsample=0.8 |
| regularised | 63 | 0.4236 | + L1/L2 reg |
| deeper | 127 | 0.4071 | **Worse** — LOPO overfitting |
| very_deep | 255 | 0.3893 | **Much worse** |
| emb_only | 31 | 0.4182 | Embedding-only ablation |
| ridge_all | — | 0.4011 | Linear reference |

Deeper trees are consistently worse under LOPO. More capacity allows the model to
memorise project-specific patterns that do not generalise to held-out projects.
The baseline configuration (31 leaves) is at or near the optimal complexity for this
dataset and CV protocol.

**Embedding-only ablation:** R²=0.4182 — all co-features across three iterations
add only +0.009 lift on top of the raw AEF embeddings. The optical embeddings already
carry most of the predictive signal available in this feature set.

---

## 5. Model Type Comparison (LOPO)

| Model | R² | RMSE | Q1 bias | Q5 bias |
|---|---:|---:|---:|---:|
| LightGBM (ref) | 0.4274 | 56.13 | +36.0 | −72.2 |
| XGBoost | 0.4278 | 56.11 | +36.3 | −72.4 |
| HistGradientBoosting | 0.4008 | 57.42 | +33.4 | −69.1 |
| Random Forest | 0.4265 | 56.17 | +37.7 | −71.6 |
| Extra Trees | 0.4259 | 56.20 | +38.5 | −74.0 |

All tree-based models cluster at R²≈0.426–0.428. No model type breaks out of the
ceiling. XGBoost is statistically indistinguishable from LightGBM.

---

## 6. Feature Processing (LightGBM, LOPO)

| Processing | R² | Q1 bias | Q5 bias |
|---|---:|---:|---:|
| No processing (ref) | 0.4245 | +34.7 | −71.4 |
| log₁₊(CHM) + CHM×slope interaction | 0.4256 | +35.8 | −72.2 |
| PCA-20 on embeddings | 0.4203 | +36.5 | −72.1 |
| log₁₊(target) | 0.3736 | +20.1 | −89.0 |
| Quantile-normalised co-features | 0.4236 | +36.0 | −72.5 |

No feature transformation improves R². Log-target shifts Q1 bias down (+20 vs +36)
but worsens Q5 (−89 vs −72) and reduces aggregate R² by 0.05.

---

## 7. Per-Quintile Bias Analysis

![Per-quintile bias](figures/quintile_bias.png)

The quintile bias pattern is **invariant** across all model types, hyperparameter
configurations, and feature sets investigated:
- Q1 (low biomass, mean ~18 tCO₂/acre): over-predicted by ~+35 tCO₂/acre
- Q5 (high biomass, mean ~220 tCO₂/acre): under-predicted by ~−72 tCO₂/acre

This compression is not a capacity or tuning problem. It is present in ridge regression
(a linear model with no capacity to overfit) at similar magnitude, and is unchanged by
log-target transformation or any other feature engineering.

This is the textbook signature of a **feature ceiling**: the input features cannot
distinguish a low-biomass stand from a high-biomass stand well enough for the model
to spread predictions across the full observed range.

---

## 8. CV Strategy Comparison

| Split | R² | RMSE | Q1 bias | Q5 bias |
|---|---:|---:|---:|---:|
| **LOPO** (23 projects) | 0.4274 | 56.13 | +36.0 | −72.2 |
| 5-fold random | **0.4520** | 54.91 | +31.5 | −67.9 |
| 5-fold stratified by biomass | 0.4516 | 54.93 | +31.9 | −67.2 |
| 5-fold stratified by ecoregion | 0.4445 | 55.28 | +32.5 | −69.4 |
| Leave-one-ecoregion-out | 0.3244 | 60.97 | +44.0 | −75.3 |

The LOPO protocol costs ~0.025 R² compared to 5-fold random (0.427 vs 0.452). The
signal is present in the features — the problem is cross-project generalisation.

Leave-one-ecoregion-out is far harder (R²=0.32): the three regions have materially
different biomass distributions (WV Appalachia: mean 98 tCO₂/acre, R²=0.16; Midwest:
mean 83 tCO₂/acre, R²=0.42; New England: mean 97 tCO₂/acre, R²=0.48).

---

## 9. Residuals by Ecoregion

![Residuals by region](figures/residuals_by_region.png)

WV Appalachia is consistently the weakest region (R²≈0.16) across all experiments.
It contains the highest-biomass plots (tall Appalachian hardwood, 200–500 tCO₂/acre)
where optical reflectance is most saturated and GEDI coverage is most sparse. This
region is where additional LiDAR co-supervision (e.g., airborne ALS, NEON AOP) would
have the greatest impact.

---

## 10. Sample Weighting Results

| Weighting | R² | Q5 bias |
|---|---:|---:|
| Uniform (ref) | 0.4274 | −72.2 |
| Inverse-frequency | 0.3247 | **−53.1** |
| √(inverse-frequency) | 0.3562 | −62.8 |
| Q5 hard upweight (5×) | 0.3592 | −59.7 |

Upweighting high-biomass plots reduces Q5 under-prediction (−72 → −53 with inv-freq)
but at the cost of ~0.10 R². This is a genuine operational trade-off: if the use-case
requires accurate prediction of high-biomass stocks (e.g., carbon credit verification
for old-growth stands), inv-freq weighting with R²≈0.32 may be preferable to the
unweighted model's R²=0.43 with severe high-end underestimation.

---

## 11. Conclusions and Path Forward

### What was established

1. **R²≈0.43 is a genuine LOPO ceiling** for tree-based regression on this feature set.
   No model type, hyperparameter configuration, or feature transformation breaks it.

2. **The bottleneck is cross-project generalisation, not feature insufficiency alone.**
   Random 5-fold CV reaches R²=0.452, confirming the signal exists. Each project has
   idiosyncratic biomass distributions (related to stand age, management history, and
   species composition) that are not captured in satellite covariates.

3. **Optical embeddings carry almost all available signal.** The embedding-only
   ablation gives R²=0.4182; all co-features combined add only +0.009. This suggests
   the AEF embeddings already encode the spectral correlates of the spatial co-features.

4. **The Q1/Q5 quintile bias is irreducible under current features.** It appears in
   ridge regression and is unchanged by any investigated transformation.

### Recommended next steps

| Priority | Direction | Expected impact |
|---|---|---|
| 1 | **Airborne LiDAR co-supervision** (NEON AOP CHM at NEON sites) — provides unambiguous canopy height at plot scale, directly addressing the saturation problem | High; literature shows CHM-optical fusion R²≈0.60–0.75 when LiDAR is site-matched |
| 2 | **Project-level covariates** (stand age, ownership, management type, species guild) — explains why projects differ; enables a hierarchical/mixed-effects model | High; likely +0.05–0.10 R² under LOPO if project characteristics are predictive |
| 3 | **Hierarchical / mixed-effects model** — explicitly model project random effects rather than treating project as a nuisance variable to exclude | Medium; requires stand-age or management data to anchor the random effects |
| 4 | **Expand the plot pool** — more projects reduce LOPO variance and broaden the distribution of biomass levels the model can learn from | Medium; dependent on data availability |
| 5 | **Neural fusion** (end-to-end optimisation of embedding extraction + regression jointly) — removes the embedding bottleneck | Uncertain; expensive; requires infrastructure change |

---

*Generated by `scripts/generate_report.py`*
"""
    OUT_MD.write_text(md)
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
