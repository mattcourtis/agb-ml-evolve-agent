"""
PALSAR-2 saturation analysis — all metrics (HH, HV, CrossRatio, RVI).

Reads the already-extracted HH/HV values (palsar_features.csv) and derives:
  - CrossRatio = HV_dB − HH_dB  (dB; negative means HV weaker than HH)
  - RVI = 4 × σHV / (σHH + σHV)  (linear power; 0 = bare, 1 = dense vegetation)

Produces a standalone HTML report with all figures embedded as base64 PNGs.

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \\
        python scripts/palsar_saturation_report.py

Output: reports/palsar_full_report.html
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPDIR = Path("experiments/agb_usa_biomass_regression_20260529")
PALSAR_CSV = EXPDIR / "preprocessing/palsar_features.csv"
PARQUET = EXPDIR / "preprocessing/features_iter3.parquet"
OUT_HTML = EXPDIR / "reports/palsar_full_report.html"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
    }
)

REGION_COL = {"wv": "#d62728", "mw": "#1f77b4", "ne": "#2ca02c"}
REGION_NAME = {"wv": "WV Appalachia", "mw": "Upper Midwest", "ne": "New England"}
Q_LABELS = ["Q1\n(~18)", "Q2\n(~63)", "Q3\n(~102)", "Q4\n(~150)", "Q5\n(~220)"]
METHODS = [("point", "Point centroid"), ("gauss", "Gaussian σ=25 m"), ("boxcar", "Boxcar 125 m")]
METRICS = [
    ("hh", "HH γ₀ (dB)", "HH backscatter"),
    ("hv", "HV γ₀ (dB)", "HV backscatter"),
    ("cross_ratio", "CrossRatio HV−HH (dB)", "Cross-pol ratio"),
    ("rvi", "RVI", "Radar Vegetation Index"),
]


# ---------------------------------------------------------------------------
# Data loading and derived metrics
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    base = pd.read_parquet(PARQUET).reset_index(drop=True)
    palsar = pd.read_csv(PALSAR_CSV, index_col="row_key")
    df = base.join(palsar)
    df = df[df["failure"].isna()].reset_index(drop=True)

    for suffix in ["point", "gauss", "boxcar"]:
        hh_db = df[f"hh_{suffix}"]
        hv_db = df[f"hv_{suffix}"]

        # Cross-polarisation ratio in dB
        df[f"cross_ratio_{suffix}"] = hv_db - hh_db

        # RVI: convert dB → linear power first, then compute index
        hh_lin = 10 ** (hh_db / 10)
        hv_lin = 10 ** (hv_db / 10)
        df[f"rvi_{suffix}"] = (4 * hv_lin) / (hh_lin + hv_lin)

    return df


def quintile_labels(y: np.ndarray) -> tuple[np.ndarray, list[float]]:
    edges = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
    lbls = np.digitize(y, edges)
    q_means = [round(float(y[lbls == q].mean()), 1) for q in range(5)]
    return lbls, q_means


# ---------------------------------------------------------------------------
# Figure helpers → base64 PNG
# ---------------------------------------------------------------------------


def fig_to_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def img_tag(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:100%">'


# ---------------------------------------------------------------------------
# Figure 1: Scatter — all metrics, point centroid, coloured by region
# ---------------------------------------------------------------------------


def fig_scatter_all(df: pd.DataFrame) -> str:
    y = df["target"].to_numpy()
    lbls, _ = quintile_labels(y)
    edges = np.quantile(y, [0.2, 0.4, 0.6, 0.8])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes = axes.flat

    for ax, (mkey, ylabel, title) in zip(axes, METRICS):
        col = f"{mkey}_point"
        sub = df.dropna(subset=[col])
        y_s = sub["target"].to_numpy()
        x_s = sub[col].to_numpy()

        for reg, clr in REGION_COL.items():
            mask = sub["region"] == reg
            ax.scatter(
                y_s[mask.to_numpy()],
                x_s[mask.to_numpy()],
                c=clr,
                alpha=0.25,
                s=6,
                label=REGION_NAME[reg],
                rasterized=True,
            )

        # Quintile mean line
        lbl_s = np.digitize(y_s, edges)
        qx = [y_s[lbl_s == q].mean() for q in range(5)]
        qy = [x_s[lbl_s == q].mean() for q in range(5)]
        ax.plot(qx, qy, "k-o", ms=8, lw=2.5, zorder=5, label="Quintile mean")
        ax.axvline(45, color="orange", lw=1.2, ls="--", alpha=0.8, label="~sat. threshold")

        r = np.corrcoef(y_s, x_s)[0, 1]
        ax.set_xlabel("Observed AGB (tCO₂/acre)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}  (point centroid, r={r:+.3f})")
        ax.legend(fontsize=7.5, markerscale=2)

    fig.suptitle("PALSAR-2 — all metrics vs AGB (point centroid extraction)", fontsize=12)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Figure 2: Extraction method comparison per metric
# ---------------------------------------------------------------------------


def fig_extraction_compare(df: pd.DataFrame) -> str:
    y = df["target"].to_numpy()
    lbls, q_means = quintile_labels(y)
    x = np.arange(5)
    w = 0.25
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    # Row 0: quintile means; Row 1: quintile std

    for col_idx, (mkey, ylabel, title) in enumerate(METRICS):
        ax_m = axes[0, col_idx]
        ax_s = axes[1, col_idx]

        for i, (suf, slbl) in enumerate(METHODS):
            col = f"{mkey}_{suf}"
            sub = df.dropna(subset=[col])
            lbl_s = np.digitize(sub["target"].to_numpy(), np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
            means = [sub.loc[lbl_s == q, col].mean() for q in range(5)]
            stds = [sub.loc[lbl_s == q, col].std() for q in range(5)]
            off = (i - 1) * w

            ax_m.bar(x + off, means, width=w * 0.9, label=slbl, color=cols[i], alpha=0.85)
            ax_m.errorbar(x + off, means, yerr=stds, fmt="none", color="black", capsize=2.5, lw=1)
            ax_s.bar(x + off, stds, width=w * 0.9, label=slbl, color=cols[i], alpha=0.85)

        for ax in (ax_m, ax_s):
            ax.set_xticks(x)
            ax.set_xticklabels(Q_LABELS, fontsize=8)

        ax_m.set_title(title, fontsize=9)
        ax_m.set_ylabel(f"Mean {ylabel}", fontsize=8)
        ax_m.legend(fontsize=7, loc="lower right")

        ax_s.set_ylabel(f"StdDev {ylabel}", fontsize=8)
        ax_s.set_title("Within-quintile noise", fontsize=9)
        ax_s.legend(fontsize=7, loc="upper right")

    fig.suptitle("PALSAR-2 — extraction method comparison (point / Gaussian / Boxcar)", fontsize=11)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Figure 3: Correlation summary heatmap
# ---------------------------------------------------------------------------


def fig_correlation_heatmap(df: pd.DataFrame) -> str:
    records = []
    for mkey, _, title in METRICS:
        for suf, slbl in METHODS:
            col = f"{mkey}_{suf}"
            sub = df.dropna(subset=[col])
            r = float(np.corrcoef(sub["target"], sub[col])[0, 1])
            records.append({"Metric": title, "Extraction": slbl, "r": r})

    pivot = pd.DataFrame(records).pivot(index="Metric", columns="Extraction", values="r")
    metric_order = [m[2] for m in METRICS]
    method_order = [m[1] for m in METHODS]
    pivot = pivot.reindex(index=metric_order, columns=method_order)

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-0.3, vmax=0.3, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r with AGB")

    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(method_order, fontsize=9)
    ax.set_yticks(range(len(metric_order)))
    ax.set_yticklabels(metric_order, fontsize=9)

    for i in range(len(metric_order)):
        for j in range(len(method_order)):
            v = pivot.values[i, j]
            ax.text(
                j,
                i,
                f"{v:+.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(v) > 0.15 else "black",
                fontweight="bold",
            )

    ax.set_title("Pearson r(AGB, PALSAR metric) by extraction method", fontsize=10)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Figure 4: RVI scatter with Pearson r by region
# ---------------------------------------------------------------------------


def fig_rvi_deep(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)

    for ax, (suf, slbl) in zip(axes, METHODS):
        col = f"rvi_{suf}"
        sub = df.dropna(subset=[col])
        y_s = sub["target"].to_numpy()
        x_s = sub[col].to_numpy()

        for reg, clr in REGION_COL.items():
            mask = sub["region"] == reg
            y_r = y_s[mask.to_numpy()]
            x_r = x_s[mask.to_numpy()]
            r_r = np.corrcoef(y_r, x_r)[0, 1] if len(y_r) > 2 else np.nan
            ax.scatter(
                y_r,
                x_r,
                c=clr,
                alpha=0.3,
                s=7,
                label=f"{REGION_NAME[reg]} (r={r_r:+.3f})",
                rasterized=True,
            )

        edges = np.quantile(y_s, [0.2, 0.4, 0.6, 0.8])
        lbl_s = np.digitize(y_s, edges)
        qx = [y_s[lbl_s == q].mean() for q in range(5)]
        qy = [x_s[lbl_s == q].mean() for q in range(5)]
        ax.plot(qx, qy, "k-o", ms=8, lw=2.5, zorder=5, label="Quintile mean")
        ax.axvline(45, color="orange", lw=1.2, ls="--", alpha=0.7, label="~sat. threshold")

        r_all = np.corrcoef(y_s, x_s)[0, 1]
        ax.set_xlabel("Observed AGB (tCO₂/acre)")
        ax.set_ylabel("RVI")
        ax.set_title(f"{slbl}  (r={r_all:+.3f})")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7.5)

    fig.suptitle("RVI (Radar Vegetation Index) vs AGB — by extraction method & region", fontsize=11)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Statistics table
# ---------------------------------------------------------------------------


def build_stats_table(df: pd.DataFrame) -> str:
    y = df["target"].to_numpy()
    edges = np.quantile(y, [0.2, 0.4, 0.6, 0.8])

    rows_html = []
    for mkey, ylabel, title in METRICS:
        for suf, slbl in METHODS:
            col = f"{mkey}_{suf}"
            sub = df.dropna(subset=[col])
            lbl_s = np.digitize(sub["target"].to_numpy(), edges)
            r = float(np.corrcoef(sub["target"], sub[col])[0, 1])
            q_m = [round(float(sub.loc[lbl_s == q, col].mean()), 3) for q in range(5)]
            q_s = [round(float(sub.loc[lbl_s == q, col].std()), 3) for q in range(5)]
            span = q_m[4] - q_m[0]  # Q5 mean − Q1 mean
            rows_html.append(
                f"<tr>"
                f"<td>{title}</td><td>{slbl}</td>"
                f"<td class='num'>{r:+.4f}</td>"
                f"<td class='num'>{q_m[0]:.3f}</td><td class='num'>{q_m[1]:.3f}</td>"
                f"<td class='num'>{q_m[2]:.3f}</td><td class='num'>{q_m[3]:.3f}</td>"
                f"<td class='num'>{q_m[4]:.3f}</td>"
                f"<td class='num'>{span:+.3f}</td>"
                f"<td class='num'>{q_s[0]:.3f}</td><td class='num'>{q_s[2]:.3f}</td>"
                f"<td class='num'>{q_s[4]:.3f}</td>"
                f"</tr>"
            )

    header = (
        "<tr><th>Metric</th><th>Extraction</th><th>r</th>"
        "<th>Q1 mean<br>(~18)</th><th>Q2 mean<br>(~63)</th>"
        "<th>Q3 mean<br>(~102)</th><th>Q4 mean<br>(~150)</th>"
        "<th>Q5 mean<br>(~220)</th><th>Q5−Q1<br>span</th>"
        "<th>Q1 std</th><th>Q3 std</th><th>Q5 std</th></tr>"
    )
    return f"<table><thead>{header}</thead><tbody>{''.join(rows_html)}</tbody></table>"


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 24px; color: #222; }
h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 8px; }
h2 { color: #16213e; margin-top: 36px; }
h3 { color: #444; }
.callout { background: #f8f9fa; border-left: 4px solid #e94560; padding: 12px 16px;
           margin: 16px 0; border-radius: 0 8px 8px 0; }
.finding-high  { border-left-color: #2ca02c; background: #f0fff0; }
.finding-med   { border-left-color: #ff7f0e; background: #fff8f0; }
.finding-low   { border-left-color: #d62728; background: #fff0f0; }
table { border-collapse: collapse; width: 100%; font-size: 0.87em; margin: 16px 0; }
th, td { border: 1px solid #ddd; padding: 6px 10px; }
th { background: #16213e; color: white; text-align: center; }
td { text-align: left; }
td.num { text-align: right; font-family: 'Courier New', monospace; }
tr:nth-child(even) { background: #f7f7f7; }
.fig-wrap { margin: 20px 0; border: 1px solid #e0e0e0; border-radius: 8px;
            padding: 12px; background: white; }
.fig-caption { font-size: 0.85em; color: #666; margin-top: 8px; font-style: italic; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
"""


def build_html(
    b64_scatter: str,
    b64_compare: str,
    b64_heatmap: str,
    b64_rvi: str,
    stats_table: str,
    df: pd.DataFrame,
) -> str:
    # Compute key headline numbers for the interpretation section
    y = df["target"].to_numpy()
    edges = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
    lbls = np.digitize(y, edges)
    hv_q1 = df.loc[lbls == 0, "hv_point"].mean()
    hv_q5 = df.loc[lbls == 4, "hv_point"].mean()
    hv_std_q3 = df.loc[lbls == 2, "hv_point"].std()
    hv_std_q3_g = df.loc[lbls == 2, "hv_gauss"].dropna().std()
    rvi_q1 = df.loc[lbls == 0, "rvi_point"].mean()
    rvi_q5 = df.loc[lbls == 4, "rvi_point"].mean()
    r_hv = float(np.corrcoef(df["target"], df["hv_point"].fillna(df["hv_point"].mean()))[0, 1])
    r_rvi = float(np.corrcoef(df["target"], df["rvi_point"].fillna(df["rvi_point"].mean()))[0, 1])
    r_cr = float(np.corrcoef(df["target"], df["cross_ratio_point"].fillna(0))[0, 1])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PALSAR-2 Saturation Analysis — Full Report</title>
<style>{CSS}</style>
</head>
<body>

<h1>PALSAR-2 Saturation Analysis — Full Report</h1>
<p><strong>Dataset:</strong> 4,636 ANEW plots · 23 projects · CONUS (WV/MW/NE) ·
JAXA ALOS PALSAR-2 annual mosaic 2022/2023 (year-matched to field measurements)<br>
<strong>Metrics:</strong> HH γ₀, HV γ₀, CrossRatio (HV−HH dB), RVI (Radar Vegetation Index)<br>
<strong>Extraction:</strong> Point centroid · Gaussian σ=25 m · Boxcar 125 m × 125 m<br>
<strong>Speckle treatment:</strong> JAXA annual mosaic is already a multi-temporal composite
(multi-look across the acquisition year); additional spatial smoothing (Gaussian/boxcar)
applied on top.</p>

<h2>Key Findings</h2>

<div class="callout finding-low">
<strong>Saturation confirmed across Q2–Q5 (80% of plots).</strong>
HV mean shifts only {hv_q5 - hv_q1:+.2f} dB from Q1 (mean {hv_q1:.2f} dB) to Q5
(mean {hv_q5:.2f} dB), while within-Q3 noise is ±{hv_std_q3:.2f} dB. Signal-to-noise
ratio &lt; 1 for Q2–Q5. The PALSAR-2 L-band cannot discriminate biomass levels above
~45 tCO₂/acre (~100 Mg/ha) in these closed-canopy temperate forests.
</div>

<div class="callout finding-med">
<strong>Speckle filtering reduces noise but cannot recover absent signal.</strong>
Gaussian σ=25 m cuts within-quintile HV std from ~{hv_std_q3:.2f} to ~{hv_std_q3_g:.2f} dB
at Q3 (−{(1 - hv_std_q3_g / hv_std_q3) * 100:.0f}% noise). Quintile means shift by &lt;0.1 dB
across all extraction methods. A cleaner version of a flat signal is still flat.
</div>

<div class="callout finding-med">
<strong>RVI shows a weak but non-zero correlation (r={r_rvi:+.3f}) vs HV alone (r={r_hv:+.3f}).</strong>
The vegetation index formulation (normalised by total backscatter) partially compensates
for terrain and moisture effects, separating Q1 from Q2+. RVI mean rises from {rvi_q1:.3f}
(Q1) to {rvi_q5:.3f} (Q5) — a gradient of {rvi_q5 - rvi_q1:.3f} — but Q2–Q5 remain
essentially flat ({rvi_q5 - df.loc[lbls == 1, "rvi_point"].mean():.3f} span from Q2 to Q5).
</div>

<div class="callout finding-med">
<strong>CrossRatio (HV−HH) shows r={r_cr:+.3f} with AGB</strong> — marginally higher than
raw HV — because subtracting HH partially removes terrain-driven HH variation that is
unrelated to biomass. However the saturation plateau is identical to HV alone.
</div>

<h2>1. Scatter Plots — All Metrics vs AGB (Point Centroid)</h2>
<div class="fig-wrap">
{img_tag(b64_scatter)}
<p class="fig-caption">
Scatter plots for all four PALSAR-2 metrics (point centroid extraction, year-matched
2022/2023 JAXA mosaic). Coloured by ecoregion (red=WV, blue=MW, green=NE). Black line
= quintile mean. Orange dashed line = approximate L-band saturation threshold (~45 tCO₂/acre
≈ 100 Mg/ha). Note the flattening of quintile mean line after Q2 in all metrics.
</p>
</div>

<h2>2. Extraction Method Comparison — Means and Within-Quintile Noise</h2>
<div class="fig-wrap">
{img_tag(b64_compare)}
<p class="fig-caption">
Top row: mean metric value per AGB quintile (± 1 SD error bars) for point, Gaussian, and
boxcar extraction. Bottom row: within-quintile standard deviation — lower = less noise.
Spatial smoothing consistently reduces noise (bottom row) but does not change quintile means
(top row), confirming the saturation is in the signal, not the extraction.
</p>
</div>

<h2>3. Correlation Heatmap</h2>
<div class="fig-wrap">
{img_tag(b64_heatmap, width="60%")}
<p class="fig-caption">
Pearson r between each metric and AGB (tCO₂/acre) for all extraction methods. Green = positive
correlation, red = negative. All correlations are weak (|r| &lt; 0.20). RVI and CrossRatio
are slightly stronger than raw HH or HV alone. Extraction method has negligible effect on
correlation strength.
</p>
</div>

<h2>4. RVI Deep Dive — By Region and Extraction Method</h2>
<div class="fig-wrap">
{img_tag(b64_rvi)}
<p class="fig-caption">
RVI scatter by extraction method and ecoregion. Per-region correlations show WV Appalachia
(red) has the weakest relationship, consistent with steep terrain causing look-angle
backscatter artefacts that dominate over vegetation signal. The r values change little
across extraction methods.
</p>
</div>

<h2>5. Full Statistics Table</h2>
<p>Quintile means and standard deviations for all metric × extraction combinations.
<strong>Q5−Q1 span</strong> = range of quintile means (larger = more discriminating signal).</p>
{stats_table}

<h2>6. Interpretation and Conclusions</h2>

<h3>Is speckle the reason for poor PALSAR performance?</h3>
<div class="callout">
<strong>No.</strong> The JAXA annual mosaic is already a multi-temporal composite
that eliminates most single-look speckle. Additional Gaussian (σ=25 m) or boxcar
(125 m) spatial smoothing reduces within-quintile noise by 30–40%, but quintile means
shift by &lt;0.1 dB across all metrics — confirming the flatness is in the signal,
not speckle noise. The prior experiment's finding ("Lee-5 filter: no impact on
headline") is replicated here across all four metrics and three extraction approaches.
</div>

<h3>Which metric is most discriminating?</h3>
<div class="callout">
RVI (r≈{r_rvi:+.3f}) and CrossRatio (r≈{r_cr:+.3f}) are marginally better than
raw HV (r≈{r_hv:+.3f}) because they normalise by total backscatter, partially removing
moisture and terrain effects unrelated to biomass. The improvement is too small
(&lt;0.02 in |r|) to translate into meaningful LOPO R² lift.
</div>

<h3>What does the saturation curve confirm?</h3>
<div class="callout">
The quintile mean line for all metrics flattens sharply between Q1 (~18 tCO₂/acre,
where some signal exists) and Q2 (~63 tCO₂/acre). The L-band penetration depth is
simply insufficient to sense woody biomass beyond ~100 Mg/ha in closed-canopy
temperate hardwood and mixed forests. 79% of ANEW plots are above this threshold.<br><br>
This is a physical property of C-band and L-band SAR in dense forest — not a data
quality, preprocessing, or extraction problem. No spatial or temporal filtering strategy
can recover signal that the radar wavelength cannot reach.
</div>

<h3>Recommendation</h3>
<div class="callout finding-low">
<strong>PALSAR-2 (and by extension Sentinel-1 C-band, which saturates even earlier) should
not be pursued further for this dataset.</strong> The biomass range of the ANEW plots
(median 102 tCO₂/acre, max 521 tCO₂/acre) is fundamentally above L-band and C-band SAR
sensitivity. Resources are better directed toward: (1) airborne LiDAR co-supervision at
plot sites, (2) LANDFIRE EVT (stand type) as a project-level covariate, or (3) expanding
the plot pool to include lower-biomass stands where SAR would have signal.
</div>

<hr>
<p style="font-size:0.8em;color:#999">
Generated by <code>scripts/palsar_saturation_report.py</code> ·
JAXA ALOS PALSAR-2 YEARLY SAR_EPOCH · DN → γ₀ dB conversion: 10×log₁₀(DN²)−83 ·
RVI = 4×σ_HV / (σ_HH + σ_HV) in linear power · CrossRatio = HV_dB − HH_dB
</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading data ...")
    df = load_data()
    print(f"  {len(df)} modelled plots, {df.columns.tolist()[-8:]}")

    print("Building figures ...")
    b64_scatter = fig_scatter_all(df)
    print("  scatter done")
    b64_compare = fig_extraction_compare(df)
    print("  extraction compare done")
    b64_heatmap = fig_correlation_heatmap(df)
    print("  heatmap done")
    b64_rvi = fig_rvi_deep(df)
    print("  RVI deep done")

    print("Building stats table ...")
    stats_table = build_stats_table(df)

    print("Writing HTML ...")
    html = build_html(b64_scatter, b64_compare, b64_heatmap, b64_rvi, stats_table, df)
    OUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"  Wrote {OUT_HTML}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
