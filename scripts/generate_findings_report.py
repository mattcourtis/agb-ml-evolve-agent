"""
Concise findings report — AGB USA biomass regression investigation.

Embeds existing figures and generates a summary progression chart.
Output: reports/findings_report.html

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \\
        python scripts/generate_findings_report.py
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

EXPDIR = Path("experiments/agb_usa_biomass_regression_20260529")
FIG_DIR = EXPDIR / "reports/figures"
OUT = EXPDIR / "reports/findings_report.html"

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


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def fig_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img(src: str, w="100%") -> str:
    return f'<img src="data:image/png;base64,{src}" style="width:{w};max-width:100%">'


# ---------------------------------------------------------------------------
# Figure A: R² progression — feature iterations + GB sweep + PALSAR
# ---------------------------------------------------------------------------


def fig_r2_progression() -> str:
    sections = {
        "Feature\niterations": [
            ("Embeddings only\n(baseline)", 0.4182, "#aec7e8"),
            ("+ GEDI shots\n(iter-1)", 0.4176, "#aec7e8"),
            ("+ CHM+topo+dist\n(iter-2)", 0.4272, "#1f77b4"),
            ("+ GEDI L4B+climate\n(iter-3)", 0.4274, "#1f77b4"),
            ("+ Gaussian PALSAR\n(iter-3 + SAR)", 0.4347, "#d62728"),
        ],
        "GB config\nsweep": [
            ("very_deep\n(255 leaves)", 0.3893, "#ffbb78"),
            ("deeper\n(127 leaves)", 0.4071, "#ffbb78"),
            ("baseline\n(31 leaves)", 0.4274, "#1f77b4"),
            ("stochastic\n(63 leaves)", 0.4225, "#aec7e8"),
        ],
        "CV strategy\n(LightGBM)": [
            ("LOPO\n(23 projects)", 0.4274, "#1f77b4"),
            ("5-fold strat.\necoregion", 0.4445, "#2ca02c"),
            ("5-fold\nrandom", 0.4520, "#2ca02c"),
        ],
        "Sample\nweighting": [
            ("Uniform\n(baseline)", 0.4274, "#1f77b4"),
            ("√(inv-freq)", 0.3562, "#ff7f0e"),
            ("Inv-freq\n(Q5↑, R²↓)", 0.3247, "#d62728"),
        ],
    }

    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)

    x = 0
    xticks, xlabels = [], []
    section_spans = []
    BASELINE = 0.4182

    for sec_label, items in sections.items():
        x_start = x
        for label, r2, col in items:
            ax.bar(x, r2, color=col, edgecolor="white", linewidth=0.5, width=0.8)
            ax.text(
                x,
                r2 + 0.002,
                f"{r2:.4f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )
            xticks.append(x)
            xlabels.append(label)
            x += 1
        section_spans.append((x_start, x - 1, sec_label))
        x += 0.8  # gap between sections

    ax.axhline(BASELINE, color="black", lw=1, ls="--", alpha=0.6, label=f"Baseline R²={BASELINE}")
    ax.axhline(0.55, color="purple", lw=1, ls=":", alpha=0.7, label="Realistic target R²=0.55")

    # Section labels below axis
    for x_start, x_end, label in section_spans:
        mid = (x_start + x_end) / 2
        ax.annotate(
            label,
            xy=(mid, 0.25),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="#333",
        )
        if x_start > 0:
            ax.axvline(x_start - 0.4, color="#ccc", lw=1, ls="-", alpha=0.5)

    # Legend patches
    patches = [
        mpatches.Patch(color="#d62728", label="Best result (+Gaussian PALSAR)"),
        mpatches.Patch(color="#1f77b4", label="Key results"),
        mpatches.Patch(color="#2ca02c", label="Less-strict CV (ceiling reference)"),
        mpatches.Patch(color="#ffbb78", label="Worse than baseline"),
    ]
    ax.legend(
        handles=patches
        + [
            plt.Line2D([0], [0], color="black", ls="--", lw=1, label=f"Baseline R²={BASELINE}"),
            plt.Line2D([0], [0], color="purple", ls=":", lw=1, label="Target R²=0.55"),
        ],
        fontsize=8,
        loc="upper right",
        ncol=2,
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7.5, rotation=0, ha="center")
    ax.set_ylim(0.3, 0.60)
    ax.set_ylabel("R² (LOPO OOF unless noted)")
    ax.set_title(
        "AGB regression R² across all experiments — ANEW CONUS 4,636 plots, 23-project LOPO",
        fontsize=11,
    )
    result = fig_b64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Figure B: PALSAR correlation by extraction method and region
# ---------------------------------------------------------------------------


def fig_palsar_correlation() -> str:
    regions = ["WV\nAppalachia", "Upper\nMidwest", "New\nEngland", "Pooled"]
    r_point = [0.1212, 0.2360, 0.2714, 0.2252]
    r_gauss = [0.1338, 0.3160, 0.3977, 0.3055]
    r_boxcar = [0.1210, 0.3281, 0.4144, 0.3157]

    x = np.arange(4)
    w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Panel 1: correlation bars
    ax = axes[0]
    ax.bar(x - w, r_point, w * 0.9, label="Point centroid", color="#aec7e8")
    ax.bar(x, r_gauss, w * 0.9, label="Gaussian σ=25 m", color="#1f77b4")
    ax.bar(x + w, r_boxcar, w * 0.9, label="Boxcar 125 m", color="#9467bd")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Pearson r (HV γ₀ vs AGB)")
    ax.set_title("PALSAR HV correlation by region & extraction")
    ax.legend(fontsize=8)
    for xi, rp, rg in zip(x, r_point, r_gauss):
        ax.annotate(
            f"+{rg - rp:.3f}",
            (xi, rg + 0.005),
            ha="center",
            fontsize=7.5,
            color="#d62728",
            fontweight="bold",
        )

    # Panel 2: LOPO R² with/without Gaussian PALSAR
    configs = [
        "Baseline\n(no PALSAR)",
        "+HV\nGaussian",
        "+HH+HV+CR\nGaussian",
        "+HH+HV+CR\nBoxcar",
        "+All SAR\nvariants",
    ]
    r2s = [0.4274, 0.4344, 0.4347, 0.4264, 0.4319]
    cols = ["#aec7e8", "#d62728", "#d62728", "#ffbb78", "#ff7f0e"]
    ax2 = axes[1]
    bars = ax2.bar(range(5), r2s, color=cols, edgecolor="white")
    ax2.axhline(0.4274, color="black", lw=1, ls="--", alpha=0.6, label="Baseline")
    for i, (b, r) in enumerate(zip(bars, r2s)):
        delta = r - 0.4274
        ax2.text(i, r + 0.0005, f"{r:.4f}", ha="center", va="bottom", fontsize=8)
        if delta != 0:
            ax2.text(
                i,
                r - 0.003,
                f"{delta:+.4f}",
                ha="center",
                va="top",
                fontsize=7.5,
                color="#d62728" if delta > 0 else "#ff7f0e",
                fontweight="bold",
            )
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(configs, fontsize=8)
    ax2.set_ylim(0.415, 0.445)
    ax2.set_ylabel("R² (LOPO CV)")
    ax2.set_title("LOPO R² gain from Gaussian PALSAR features")
    ax2.legend(fontsize=8)

    result = fig_b64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Figure C: Saturation curve summary (text-based annotation)
# ---------------------------------------------------------------------------


def fig_saturation_curve() -> str:
    agb = np.linspace(0, 400, 500)
    hv_sat = np.where(agb < 45, -12.5 + (agb / 45) * 1.0, -11.5 + 0.5 * np.exp(-(agb - 45) / 30))
    hv_sat = np.clip(hv_sat, -14, -10)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    ax = axes[0]
    ax.plot(agb, hv_sat, "b-", lw=2.5, label="HV backscatter (schematic)")
    ax.axvline(45, color="orange", lw=1.5, ls="--", label="~L-band saturation (100 Mg/ha)")
    ax.fill_betweenx(
        [-14, -10], 0, 45, alpha=0.12, color="green", label="Discriminable range (21% of plots)"
    )
    ax.fill_betweenx(
        [-14, -10], 45, 400, alpha=0.07, color="red", label="Saturated range (79% of plots)"
    )

    # Quintile markers
    q_agb = [18, 63, 101, 139, 220]
    q_hv = [-12.52, -11.57, -11.27, -11.04, -11.00]
    ax.scatter(q_agb, q_hv, s=80, c="black", zorder=5)
    for i, (xa, ya) in enumerate(zip(q_agb, q_hv)):
        ax.annotate(
            f"Q{i + 1}\n{xa:.0f}",
            (xa, ya),
            fontsize=7.5,
            xytext=(10, 8 * (1 if i % 2 == 0 else -1)),
            textcoords="offset points",
        )

    ax.set_xlim(0, 380)
    ax.set_ylim(-14, -9.5)
    ax.set_xlabel("Observed AGB (tCO₂/acre)")
    ax.set_ylabel("PALSAR HV γ₀ (dB)")
    ax.set_title("L-band SAR saturation — schematic + observed quintile means")
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: project-level correlations
    ax2 = axes[1]
    projects = [
        "NorthMaine\nWoods",
        "BigSix",
        "Tomah\nHighlands",
        "Cassidy",
        "Eagle\nMountain",
        "100Mile\nWilderness",
        "Kanawha\nRiver",
        "Superior\nWatershed",
    ]
    r_pt_proj = [0.420, 0.187, 0.302, 0.218, 0.163, 0.073, 0.022, 0.039]
    r_gau_proj = [0.557, 0.351, 0.339, 0.315, 0.299, 0.203, 0.022, 0.039]  # last 2 unchanged

    y = np.arange(len(projects))
    ax2.barh(y + 0.2, r_gau_proj, 0.35, label="Gaussian σ=25 m", color="#1f77b4", alpha=0.85)
    ax2.barh(y - 0.2, r_pt_proj, 0.35, label="Point centroid", color="#aec7e8", alpha=0.85)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(projects, fontsize=8)
    ax2.set_xlabel("r(AGB, HV)")
    ax2.set_title("Per-project HV correlation (NE projects + WV/MW comparison)")
    ax2.legend(fontsize=8)

    result = fig_b64(fig)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# CSS + HTML
# ---------------------------------------------------------------------------

CSS = """
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       max-width:1300px;margin:0 auto;padding:24px;color:#222;line-height:1.5 }
h1 { border-bottom:3px solid #e94560;padding-bottom:8px;color:#1a1a2e }
h2 { color:#16213e;margin-top:32px;border-left:4px solid #e94560;padding-left:10px }
h3 { color:#444;margin-top:20px }
table { border-collapse:collapse;width:100%;font-size:0.88em;margin:12px 0 }
th,td { border:1px solid #ddd;padding:6px 10px }
th { background:#16213e;color:white;text-align:center }
td { text-align:left }
td.n { text-align:right;font-family:'Courier New',monospace }
tr:nth-child(even) { background:#f7f7f7 }
.box { background:#f8f9fa;border-left:4px solid #e94560;
       padding:12px 16px;margin:14px 0;border-radius:0 8px 8px 0 }
.box.green  { border-left-color:#2ca02c;background:#f0fff0 }
.box.amber  { border-left-color:#ff7f0e;background:#fff8f0 }
.box.red    { border-left-color:#d62728;background:#fff0f0 }
.box.purple { border-left-color:#7b2f8a;background:#faf0ff }
.fig { margin:18px 0;border:1px solid #e0e0e0;border-radius:8px;
       padding:12px;background:white }
.cap { font-size:0.83em;color:#666;margin-top:6px;font-style:italic }
.grid2 { display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0 }
kbd { background:#eee;border:1px solid #ccc;border-radius:3px;
      padding:1px 5px;font-family:monospace;font-size:0.9em }
"""


def build_html(fig_prog: str, fig_palsar: str, fig_sat: str) -> str:
    # Load existing figures from disk
    pred_obs = b64(FIG_DIR / "pred_vs_obs.png")
    q_bias = b64(FIG_DIR / "quintile_bias.png")
    resid_reg = b64(FIG_DIR / "residuals_by_region.png")
    palsar_sat = b64(FIG_DIR / "palsar_saturation.png")
    palsar_qst = b64(FIG_DIR / "palsar_quintile_stats.png")

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AGB Investigation — Findings & Next Steps</title>
<style>{CSS}</style>
</head>
<body>

<h1>AGB USA Biomass Regression — Findings & Next Steps</h1>
<p><strong>Dataset:</strong> 4,636 ANEW field plots · 23 projects · CONUS
(WV Appalachia / Upper Midwest / New England) · Target: CO₂ standing stock (tCO₂/acre)<br>
<strong>CV protocol:</strong> 23-project leave-one-project-out (LOPO) — the strictest
spatial holdout, ~0.025 R² harder than 5-fold random<br>
<strong>Baseline model:</strong> LightGBM (31 leaves, lr=0.05) on 64-dim AEF optical embeddings</p>

<h2>1. Summary of R² Progression</h2>

<div class="fig">
{img(fig_prog)}
<p class="cap">R² across all experiments under 23-project LOPO CV (except the CV-strategy
section which varies the split). Dashed black = optical-embedding baseline. Purple dotted =
realistic target R²=0.55. Red bars = best results achieved.</p>
</div>

<div class="box green">
<strong>Best LOPO R² achieved: 0.4347</strong> (+0.0165 over baseline) — LightGBM with
AEF embeddings + ETH CHM + SRTM topography + Hansen disturbance + Gaussian-extracted
PALSAR-2 (HH+HV+CrossRatio, σ=25 m). The realistic target of R²≥0.55 has not been reached.
</div>

<h3>Feature iteration results</h3>
<table>
<tr><th>Iteration</th><th>Features added</th><th>R²</th><th>RMSE</th><th>ΔR²</th><th>Key finding</th></tr>
<tr><td>0 (baseline)</td><td>AEF embeddings only (64-dim)</td>
  <td class="n">0.4182</td><td class="n">56.58</td><td class="n">—</td>
  <td>Reproduces joint_v2 exactly; feature ceiling confirmed</td></tr>
<tr><td>1</td><td>+ GEDI L2A/L2B shot-level via GEE (500 m buffer)</td>
  <td class="n">0.4176</td><td class="n">56.61</td><td class="n">−0.001</td>
  <td>Median n_samples=1 (orbital sparsity); no usable signal</td></tr>
<tr><td>2</td><td>+ ETH CHM 2020 + SRTM topo + Hansen disturbance</td>
  <td class="n">0.4272</td><td class="n">56.14</td><td class="n">+0.010</td>
  <td>13.5% SHAP; only meaningful lift from spatial co-features</td></tr>
<tr><td>3</td><td>+ GEDI L4B 1 km AGBD + TerraClimate climate</td>
  <td class="n">0.4274</td><td class="n">56.13</td><td class="n">+0.000</td>
  <td>34% nulls in GEDI L4B; climate adds zero under LOPO</td></tr>
<tr style="background:#fff0f0"><td>+SAR</td><td>+ Gaussian PALSAR HH+HV+CrossRatio (σ=25 m)</td>
  <td class="n"><strong>0.4347</strong></td><td class="n">55.77</td><td class="n"><strong>+0.007</strong></td>
  <td>Point centroid PALSAR gave ~0; Gaussian extraction reveals sub-saturation signal</td></tr>
</table>

<h2>2. The Core Problem — Dynamic Range Compression</h2>

<div class="fig">
{img(pred_obs)}
<p class="cap">Predicted vs observed AGB for three configurations under LOPO CV.
The 1:1 line is never reached at either extreme. Predictions are compressed toward
the centre regardless of features or model type.</p>
</div>

<div class="grid2">
<div class="fig">
{img(q_bias, "100%")}
<p class="cap">Per-quintile bias (mean residual) across three configurations. The pattern
— Q1 over-predicted +35 tCO₂/acre, Q5 under-predicted −72 tCO₂/acre — is invariant
across all model types, hyperparameters, and feature sets investigated.</p>
</div>
<div class="fig">
{img(resid_reg, "100%")}
<p class="cap">Residuals by ecoregion. WV Appalachia (R²=0.17) shows a steep negative
slope — predictions barely track observed values above ~150 tCO₂/acre. The dominant
tall-hardwood stands are optically indistinguishable above canopy closure.</p>
</div>
</div>

<div class="box red">
<strong>Quintile bias is irreducible with the current model class.</strong>
Ridge regression, Random Forest, ExtraTrees, XGBoost, and LightGBM across 7 hyperparameter
configs all produce Q1≈+35 and Q5≈−72 tCO₂/acre. The compression is not a capacity or tuning
problem — it is a fundamental property of the input features relative to the target range.
</div>

<h2>3. Model Architecture Investigation</h2>

<table>
<tr><th>Config</th><th>num_leaves</th><th>R²</th><th>Finding</th></tr>
<tr><td>baseline</td><td>31</td><td class="n">0.4274</td><td>Optimal for LOPO</td></tr>
<tr><td>stochastic (subsample=0.8)</td><td>63</td><td class="n">0.4225</td><td>Slight degradation</td></tr>
<tr><td>deeper</td><td>127</td><td class="n">0.4071</td><td>LOPO overfitting</td></tr>
<tr><td>very_deep</td><td>255</td><td class="n">0.3893</td><td>Strong LOPO overfitting</td></tr>
<tr><td><strong>emb_only</strong></td><td>31</td><td class="n"><strong>0.4182</strong></td>
  <td><strong>Embeddings alone = 0.4182; all co-features add only +0.009 total</strong></td></tr>
<tr><td>XGBoost</td><td>max_depth=6</td><td class="n">0.4278</td><td>Indistinguishable from LightGBM</td></tr>
<tr><td>Random Forest</td><td>n=500</td><td class="n">0.4265</td><td>Indistinguishable</td></tr>
<tr><td>Ridge regression</td><td>linear</td><td class="n">0.4011</td><td>Similar quintile bias pattern</td></tr>
</table>

<div class="box amber">
<strong>Deeper trees are consistently worse under LOPO.</strong> More capacity memorises
project-specific biomass distributions that don't generalise. The LOPO protocol costs
~0.025 R² vs 5-fold random (0.427 vs 0.452) — confirming the signal exists in the features
but does not transfer well across project boundaries.
</div>

<h2>4. PALSAR-2 SAR Investigation</h2>

<div class="fig">
{img(fig_palsar)}
<p class="cap">Left: PALSAR-2 HV correlation with AGB by region and extraction method.
Red annotations show the Gaussian uplift vs point centroid. Right: LOPO R² with different PALSAR
feature configurations — Gaussian HH+HV+CrossRatio gives the best lift (+0.007).</p>
</div>

<div class="fig">
{img(fig_sat)}
<p class="cap">Left: L-band SAR saturation schematic with observed quintile means overlaid.
79% of ANEW plots are above the saturation threshold (~45 tCO₂/acre ≈ 100 Mg/ha). Right:
Per-project HV correlation — NorthMaineWoods (NE) achieves r=+0.557 with Gaussian extraction;
the weakest projects (KanawhaRiver WV, SuperiorWatershed MW) are concentrated above saturation.
</p>
</div>

<div class="fig">
{img(palsar_sat)}
<p class="cap">PALSAR HV γ₀ vs AGB scatter for three extraction methods. The quintile mean line
(black) rises from Q1 to Q2 then plateaus — the classic saturation signature. Gaussian and boxcar
smoothing shift the cloud but cannot recover the absent high-biomass signal.</p>
</div>

<div class="fig">
{img(palsar_qst)}
<p class="cap">Per-quintile HV mean (left) and within-quintile noise (right). Gaussian cuts
noise by 30–40% across all quintiles; means shift by &lt;0.1 dB — cleaner extraction of a
still-flat saturated signal. The critical sub-saturation (Q1) noise reduction is what drives
the +0.007 LOPO lift.</p>
</div>

<table>
<tr><th>Metric</th><th>Point centroid</th><th>Gaussian σ=25 m</th><th>Boxcar 125 m</th></tr>
<tr><td>Pooled r(AGB, HV)</td><td class="n">+0.225</td><td class="n"><strong>+0.306</strong></td><td class="n">+0.316</td></tr>
<tr><td>WV r(AGB, HV)</td><td class="n">+0.121</td><td class="n">+0.134</td><td class="n">+0.121</td></tr>
<tr><td>MW r(AGB, HV)</td><td class="n">+0.236</td><td class="n">+0.316</td><td class="n">+0.328</td></tr>
<tr><td>NE r(AGB, HV)</td><td class="n">+0.271</td><td class="n"><strong>+0.398</strong></td><td class="n">+0.414</td></tr>
<tr><td>LOPO ΔR² vs baseline</td><td class="n">~0 (prior exp.)</td><td class="n"><strong>+0.007</strong></td><td class="n">≈0</td></tr>
<tr><td>Plots above saturation</td><td colspan="3" class="n">79% (WV 94%, MW 77%, NE 76%)</td></tr>
</table>

<div class="box green">
<strong>Key reversal of prior conclusion:</strong> Point-centroid PALSAR showed ~+0.02 R²
"within noise" in the prior tf-deep-landcover investigation. With Gaussian σ=25 m extraction,
LOPO R² lift is <strong>+0.007</strong> (reproducibly above zero). The prior result was noise-limited
by single-pixel extraction of a 25 m speckled SAR image at a ~7 m radius plot. Boxcar 125 m
achieves higher raw correlation but no LOPO lift — too aggressive, averaging across stand boundaries.
</div>

<h2>5. Deep Research Synthesis</h2>
<p>Multi-source literature search (105 subagents, 970 tool uses, 2025 peer-reviewed papers)
confirmed four high-confidence findings:</p>

<div class="grid2">
<div class="box green">
<strong>Buffer extraction &gt; point centroid</strong> (3-0 verified, Cai et al. PLOS ONE 2025,
Sci. Reports 2020). Standard operational standard: 25.82 m radius mean for FIA-scale satellite
calibration. Our 1/24-acre plots (~7.3 m radius) are substantially sub-pixel for all features
at 10–30 m resolution. <em>Implication: re-extract all co-features with Gaussian kernel.</em>
</div>
<div class="box green">
<strong>GEDI height is #1 predictor</strong> in CONUS RF models (3-0 verified, Lu et al. 2025,
<em>Forest Ecology &amp; Management</em>). LANDFIRE EVT (stand type) is #2, EVI is #3.
PALSAR SAR and ICESat-2 rank last.
<em>Implication: LANDFIRE EVT is untested here and could address project-level generalisation.</em>
</div>
<div class="box green">
<strong>Quantile normalisation hurts GBTs</strong> (3-0 verified, Nuyts &amp; Davis arXiv 2025).
QN improves linear models by 11–63% RSE but degrades gradient boosted trees by +19% RSE.
<em>Confirms our empirical result (log-target dropped R² from 0.427 to 0.374).</em>
</div>
<div class="box amber">
<strong>Zero-AGB augmentation</strong>: Theoretically sound (model has no sub-zero-biomass
anchor), empirically unverified in the literature. Could reduce Q1 over-prediction by
extending the training distribution downward.
</div>
</div>

<h2>6. Proposed Next Steps</h2>

<table>
<tr><th>Priority</th><th>Experiment</th><th>Expected ΔR²</th><th>Effort</th><th>Rationale</th></tr>
<tr style="background:#f0fff0">
  <td><strong>1</strong></td>
  <td><strong>LANDFIRE EVT (stand type)</strong> as categorical feature<br>
  GEE: <kbd>LANDFIRE/Fire/EVT/v1_4_0</kbd></td>
  <td class="n">+0.05–0.10?</td><td>Low</td>
  <td>Literature #2 predictor after GEDI; directly addresses project-level biomass
  distribution differences that LOPO penalises. Stand type explains WHY projects differ.</td>
</tr>
<tr style="background:#f0fff0">
  <td><strong>2</strong></td>
  <td><strong>Gaussian PALSAR in production feature set</strong><br>
  HH + HV + CrossRatio, σ=25 m, 2022/2023</td>
  <td class="n">+0.007 (confirmed)</td><td>Done</td>
  <td>Reproducibly adds lift from sub-saturation (Q1) signal. Boxcar does not help.</td>
</tr>
<tr style="background:#fff8f0">
  <td><strong>3</strong></td>
  <td><strong>Sentinel-1 seasonal VH differential</strong><br>
  Leaf-on (May–Oct) vs leaf-off (Nov–Apr) VH difference; temporal variance<br>
  GEE: <kbd>COPERNICUS/S1_GRD</kbd></td>
  <td class="n">+0.005–0.015?</td><td>Medium</td>
  <td>Unlike PALSAR/CHM, seasonal VH encodes deciduousness and phenology — not
  captured by annual optical embeddings. Literature supports ~+0.10 for temperate forests.</td>
</tr>
<tr style="background:#fff8f0">
  <td><strong>4</strong></td>
  <td><strong>Zero-AGB augmentation</strong><br>
  Synthetic non-forest plots from Hansen clearcuts + NLCD non-forest within ANEW AOIs</td>
  <td class="n">Unknown (Q1 bias reduction)</td><td>Low</td>
  <td>Model has no low/zero biomass anchor. Adding confirmed-zero locations could pull
  Q1 predictions down without proportional Q5 damage (unlike inverse-freq weighting).</td>
</tr>
<tr>
  <td><strong>5</strong></td>
  <td><strong>Re-extract CHM with larger Gaussian</strong> (σ=25–30 m)<br>
  Current σ=15 m showed ΔR²=−0.003. σ=25 m may avoid stand contamination.</td>
  <td class="n">+0.001–0.005?</td><td>Low</td>
  <td>The Gaussian PALSAR finding (σ=25 m at 25 m native) works better than boxcar.
  Applying same logic to CHM (10 m native) deserves a matched test.</td>
</tr>
<tr>
  <td><strong>6</strong></td>
  <td><strong>NEON AOP airborne LiDAR CHM</strong> at NEON sites within ANEW AOIs</td>
  <td class="n">+0.10–0.20?</td><td>High</td>
  <td>Site-matched 1 m airborne LiDAR directly addresses the optical saturation problem.
  Literature: CHM-optical fusion R²=0.60–0.75 in temperate forest. Highest potential
  uplift but requires data access and site overlap check.</td>
</tr>
</table>

<div class="box purple">
<strong>Fundamental constraint:</strong> The R²=0.43 LOPO ceiling is driven by
cross-project generalisation failure (~0.025 R² cost vs random CV), not feature
insufficiency alone. All co-features together add only +0.009 R² on top of optical
embeddings (emb_only R²=0.4182). The highest-leverage intervention is likely
<strong>LANDFIRE EVT</strong> (stand-type categorisation explaining project differences)
or <strong>airborne LiDAR co-supervision</strong> (removing the optical saturation
problem directly). Short of those, further satellite feature additions will continue to
yield diminishing returns at +0.005–0.010 R² increments.
</div>

<h2>7. Current Best Feature Set</h2>
<table>
<tr><th>Feature group</th><th>Source</th><th>Columns</th><th>SHAP %</th><th>R² contribution</th></tr>
<tr><td>AEF optical embeddings</td><td>Source Coop (10 m, annual)</td><td>emb_00..63 (64)</td>
  <td class="n">83–87%</td><td class="n">0.4182 (baseline)</td></tr>
<tr><td>ETH Canopy Height Model</td><td>users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1</td>
  <td>chm_m (1)</td><td class="n" rowspan="3">13.5% combined</td><td class="n" rowspan="3">+0.010</td></tr>
<tr><td>SRTM topography</td><td>USGS/SRTMGL1_003 (30 m)</td>
  <td>topo_elevation, topo_slope, topo_aspect_cos, topo_aspect_sin, topo_tpi (5)</td></tr>
<tr><td>Hansen disturbance</td><td>UMD/hansen/global_forest_change_2025_v1_13 (30 m)</td>
  <td>dist_years_since (1)</td></tr>
<tr><td>GEDI L4B AGBD</td><td>LARSE/GEDI/GEDI04_B_002 (1 km)</td><td>agbd_mu (1)</td>
  <td class="n">~0%</td><td class="n">~0</td></tr>
<tr><td>TerraClimate climate</td><td>IDAHO_EPSCOR/TERRACLIMATE (4 km)</td>
  <td>clim_pr, clim_tmmx, clim_aet (3)</td><td class="n">~0%</td><td class="n">~0</td></tr>
<tr style="background:#f0fff0"><td><strong>Gaussian PALSAR (NEW)</strong></td>
  <td>JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH (25 m) + Gaussian σ=25 m</td>
  <td>palsar_hh_gauss, palsar_hv_gauss, palsar_cr_gauss (3)</td>
  <td class="n">TBD</td><td class="n"><strong>+0.007</strong></td></tr>
<tr><th colspan="2">Total</th><th>80 base + 3 Gaussian SAR = 83 features</th>
  <th class="n" colspan="2">R²=0.4347</th></tr>
</table>

<hr>
<p style="font-size:0.8em;color:#999">
Generated by <code>scripts/generate_findings_report.py</code> ·
Experiment: agb_usa_biomass_regression_20260529 ·
4,636 ANEW plots · 23-project LOPO CV · LightGBM baseline
</p>
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Generating figures ...")
    fp = fig_r2_progression()
    print("  R² progression done")
    fpal = fig_palsar_correlation()
    print("  PALSAR correlation done")
    fsat = fig_saturation_curve()
    print("  Saturation curve done")

    print("Writing HTML ...")
    html = build_html(fp, fpal, fsat)
    OUT.write_text(html, encoding="utf-8")
    print(f"  Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
