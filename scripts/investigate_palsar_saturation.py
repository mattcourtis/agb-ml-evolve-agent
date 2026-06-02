"""
PALSAR-2 saturation investigation.

Extracts JAXA ALOS PALSAR-2 annual mosaic (HH + HV) in dB for all 4,646 ANEW
plots, year-matched to field measurement year (2022 or 2023). Applies:
  - Point centroid extraction (no spatial filter)
  - Gaussian-weighted extraction (σ=25 m, equivalent to ~1–2 pixel smoothing at 25 m)
  - Boxcar mean (5×5 pixels = 125 m × 125 m) as a strong spatial average for comparison

Produces:
  - preprocessing/palsar_features.csv
  - reports/figures/palsar_saturation.png  (HV dB vs CO2 scatter + quintile means)
  - reports/figures/palsar_extraction_compare.png  (point vs Gaussian per quintile)
  - reports/palsar_saturation_report.md

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \\
        python scripts/investigate_palsar_saturation.py
"""

from __future__ import annotations

from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPDIR = Path("experiments/agb_usa_biomass_regression_20260529")
PARQUET = EXPDIR / "preprocessing/features_iter3.parquet"
OUT_CSV = EXPDIR / "preprocessing/palsar_features.csv"
FIG_DIR = EXPDIR / "reports/figures"
OUT_MD = EXPDIR / "reports/palsar_saturation_report.md"

FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150}
)

SIGMA_M, RADIUS_M = 25, 75  # Gaussian kernel: σ=25 m, radius=75 m (3σ)
BOXCAR_PX = 5  # 5×5 pixel boxcar at 25 m = 125 m × 125 m

BATCH_SIZE = 300


# ---------------------------------------------------------------------------
# GEE helpers
# ---------------------------------------------------------------------------


def dn_to_db(img: ee.Image) -> ee.Image:
    """Convert PALSAR DN to gamma-naught γ₀ in dB. JAXA formula: 10*log10(DN²) - 83."""
    return img.pow(2).log10().multiply(10).subtract(83).rename(img.bandNames())


def get_palsar_db(year: int) -> ee.Image:
    """Return PALSAR HH+HV in dB for a given year (already multi-look annual composite)."""
    col = (
        ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select(["HH", "HV"])
    )
    return dn_to_db(col.mosaic())


def apply_gaussian(img: ee.Image) -> ee.Image:
    kernel = ee.Kernel.gaussian(radius=RADIUS_M, sigma=SIGMA_M, units="meters", normalize=True)
    return img.reduceNeighborhood(ee.Reducer.mean(), kernel).rename(["HH", "HV"])


def apply_boxcar(img: ee.Image) -> ee.Image:
    kernel = ee.Kernel.square(radius=BOXCAR_PX // 2, units="pixels", normalize=True)
    return img.reduceNeighborhood(ee.Reducer.mean(), kernel).rename(["HH", "HV"])


def extract_palsar_batch(
    plots_df: pd.DataFrame, img_2022: ee.Image, img_2023: ee.Image, suffix: str
) -> pd.DataFrame:
    """Extract HH+HV for each plot from year-matched image, batched."""
    rows: list[dict] = []
    n = len(plots_df)
    print(f"  Extracting {suffix} ({n} plots in batches of {BATCH_SIZE}) ...")

    for start in range(0, n, BATCH_SIZE):
        batch = plots_df.iloc[start : start + BATCH_SIZE]
        fc_22 = ee.FeatureCollection(
            [
                ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"row_key": int(i)})
                for i, r in batch[batch["year"] == 2022].iterrows()
            ]
        )
        fc_23 = ee.FeatureCollection(
            [
                ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"row_key": int(i)})
                for i, r in batch[batch["year"] == 2023].iterrows()
            ]
        )

        for fc, img, yr in [(fc_22, img_2022, 2022), (fc_23, img_2023, 2023)]:
            sz = fc.size().getInfo()
            if sz == 0:
                continue
            result = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=25)
            for feat in result.getInfo()["features"]:
                p = feat["properties"]
                rows.append(
                    {
                        "row_key": int(p["row_key"]),
                        f"hh_{suffix}": p.get("HH"),
                        f"hv_{suffix}": p.get("HV"),
                    }
                )

        end = min(start + BATCH_SIZE, n)
        print(f"    batch {start}–{end - 1}: done")

    return pd.DataFrame(rows).set_index("row_key")


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

REGION_COL = {"wv": "#d62728", "mw": "#1f77b4", "ne": "#2ca02c"}
QUINTILE_LABELS = ["Q1\n(~18)", "Q2\n(~63)", "Q3\n(~102)", "Q4\n(~150)", "Q5\n(~220)"]


def plot_saturation(df_mod: pd.DataFrame) -> Path:
    """HV dB vs CO2 scatter for three extraction methods, coloured by region."""
    methods = [
        ("hv_point", "Point centroid", "o"),
        ("hv_gauss", "Gaussian σ=25 m", "s"),
        ("hv_boxcar", "Boxcar 125 m", "^"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, constrained_layout=True)

    for ax, (col, label, marker) in zip(axes, methods):
        sub = df_mod.dropna(subset=[col, "target"])
        for reg, clr in REGION_COL.items():
            mask = sub["region"] == reg
            ax.scatter(
                sub.loc[mask, "target"],
                sub.loc[mask, col],
                c=clr,
                alpha=0.25,
                s=6,
                label=reg.upper(),
                rasterized=True,
            )
        # Quintile mean overlay
        edges = np.quantile(sub["target"], [0.2, 0.4, 0.6, 0.8])
        lbls = np.digitize(sub["target"], edges)
        q_x = [sub.loc[lbls == q, "target"].mean() for q in range(5)]
        q_y = [sub.loc[lbls == q, col].mean() for q in range(5)]
        ax.plot(q_x, q_y, "k-o", ms=7, lw=2, zorder=5, label="Quintile mean")
        ax.set_xlabel("Observed AGB (tCO₂/acre)")
        ax.set_title(f"{label}\nn={len(sub):,}")
        ax.legend(fontsize=7, markerscale=2)
        # Mark saturation region
        ax.axvline(45, color="orange", lw=1.2, ls="--", label="~100 Mg/ha sat.")
    axes[0].set_ylabel("PALSAR HV γ₀ (dB)")
    fig.suptitle("PALSAR-2 HV vs AGB — saturation diagnostic (LOPO plots)", y=1.02)
    out = FIG_DIR / "palsar_saturation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


def plot_quintile_stats(df_mod: pd.DataFrame) -> Path:
    """Per-quintile mean ± std HV dB for each extraction method."""
    methods = [
        ("hv_point", "Point centroid"),
        ("hv_gauss", "Gaussian σ=25 m"),
        ("hv_boxcar", "Boxcar 125 m"),
    ]
    y = df_mod["target"].to_numpy()
    edges = np.quantile(y, [0.2, 0.4, 0.6, 0.8])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # Panel 1: mean HV per quintile
    ax = axes[0]
    x = np.arange(5)
    w = 0.25
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (col, label) in enumerate(methods):
        sub = df_mod.dropna(subset=[col])
        ql = np.digitize(sub["target"], edges)
        means = [sub.loc[ql == q, col].mean() for q in range(5)]
        stds = [sub.loc[ql == q, col].std() for q in range(5)]
        offset = (i - 1) * w
        ax.bar(x + offset, means, width=w * 0.9, label=label, color=colours[i], alpha=0.85)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none", color="black", capsize=3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(QUINTILE_LABELS)
    ax.set_ylabel("Mean HV γ₀ (dB)")
    ax.set_title("Mean HV per AGB quintile (± 1 SD)")
    ax.legend(fontsize=8)

    # Panel 2: std HV per quintile (lower std = less noise)
    ax = axes[1]
    for i, (col, label) in enumerate(methods):
        sub = df_mod.dropna(subset=[col])
        ql = np.digitize(sub["target"], edges)
        stds = [sub.loc[ql == q, col].std() for q in range(5)]
        offset = (i - 1) * w
        ax.bar(x + offset, stds, width=w * 0.9, label=label, color=colours[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(QUINTILE_LABELS)
    ax.set_ylabel("StdDev HV γ₀ (dB)")
    ax.set_title("Within-quintile HV variability (lower = less noise)")
    ax.legend(fontsize=8)

    out = FIG_DIR / "palsar_quintile_stats.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ee.Initialize()
    df_base = pd.read_parquet(PARQUET).reset_index(drop=True)
    df_mod = df_base[df_base["failure"].isna()].copy()
    print(
        f"Loaded {len(df_mod)} modelled plots "
        f"({(df_mod['year'] == 2022).sum()} in 2022, "
        f"{(df_mod['year'] == 2023).sum()} in 2023)"
    )

    # Build three image variants per year
    print("\nBuilding PALSAR images ...")
    raw_22, raw_23 = get_palsar_db(2022), get_palsar_db(2023)
    gauss_22 = apply_gaussian(raw_22)
    gauss_23 = apply_gaussian(raw_23)
    box_22 = apply_boxcar(raw_22)
    box_23 = apply_boxcar(raw_23)
    print("  Done.")

    print("\nExtracting features ...")
    df_pt = extract_palsar_batch(df_mod, raw_22, raw_23, "point")
    df_g = extract_palsar_batch(df_mod, gauss_22, gauss_23, "gauss")
    df_bx = extract_palsar_batch(df_mod, box_22, box_23, "boxcar")

    df_mod = df_mod.join(df_pt, how="left").join(df_g, how="left").join(df_bx, how="left")

    # Null report
    for col in ["hh_point", "hv_point", "hv_gauss", "hv_boxcar"]:
        n_null = df_mod[col].isna().sum()
        pct = n_null / len(df_mod) * 100
        print(f"  {col}: {n_null} nulls ({pct:.1f}%)")

    # Save CSV for reuse
    keep = ["hh_point", "hv_point", "hh_gauss", "hv_gauss", "hh_boxcar", "hv_boxcar"]
    df_base_join = df_base.copy()
    df_base_join[keep] = np.nan
    for col in keep:
        if col in df_mod.columns:
            df_base_join.loc[df_mod.index, col] = df_mod[col].values
    df_base_join[["hh_point", "hv_point", "hh_gauss", "hv_gauss", "hh_boxcar", "hv_boxcar"]].to_csv(
        OUT_CSV, index=True, index_label="row_key"
    )
    print(f"\nWrote {OUT_CSV}")

    # Saturation statistics per quintile
    y = df_mod["target"].to_numpy()
    edges = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
    lbls = np.digitize(y, edges)
    q_means = [round(y[lbls == q].mean(), 1) for q in range(5)]

    rows_md: list[str] = []
    for q in range(5):
        mask = lbls == q
        for col, lbl in [("hv_point", "point"), ("hv_gauss", "gauss"), ("hv_boxcar", "boxcar")]:
            vals = df_mod.loc[mask, col].dropna()
            rows_md.append(
                f"| Q{q + 1} (~{q_means[q]:.0f}) | {lbl} | "
                f"{vals.mean():.2f} | {vals.std():.2f} | {len(vals)} |"
            )

    print("\n=== HV dB per quintile (mean ± std) ===")
    for q in range(5):
        mask = lbls == q
        m_pt = df_mod.loc[mask, "hv_point"].mean()
        m_g = df_mod.loc[mask, "hv_gauss"].mean()
        m_bx = df_mod.loc[mask, "hv_boxcar"].mean()
        s_pt = df_mod.loc[mask, "hv_point"].std()
        print(
            f"  Q{q + 1} ({q_means[q]:.0f} tCO2/acre): "
            f"point={m_pt:.2f}±{s_pt:.2f}  gauss={m_g:.2f}  boxcar={m_bx:.2f} dB"
        )

    # Figures
    print("\nGenerating figures ...")
    plot_saturation(df_mod)
    plot_quintile_stats(df_mod)

    # Markdown report
    corr_pt = df_mod[["target", "hv_point"]].dropna().corr().iloc[0, 1]
    corr_g = df_mod[["target", "hv_gauss"]].dropna().corr().iloc[0, 1]
    corr_bx = df_mod[["target", "hv_boxcar"]].dropna().corr().iloc[0, 1]

    md = f"""# PALSAR-2 Saturation Investigation

**Hypothesis:** The prior +0.02 R² from PALSAR-2 may have been noise-limited rather
than signal-limited. This investigation verifies whether (a) HV/HH values actually
saturate across Q2–Q5, and (b) whether speckle treatment (Gaussian or boxcar spatial
smoothing) reveals additional signal.

**Data:** JAXA ALOS PALSAR-2 annual mosaic 2022/2023, year-matched to field measurements.
DN → γ₀ dB conversion: `10 × log₁₀(DN²) − 83`.
The JAXA annual mosaic is already a multi-temporal composite (multi-look across the
acquisition year), providing inherent temporal speckle reduction.

---

## HV Pearson correlation with AGB (tCO₂/acre)

| Extraction | r |
|---|---:|
| Point centroid | {corr_pt:+.4f} |
| Gaussian σ=25 m | {corr_g:+.4f} |
| Boxcar 125 m | {corr_bx:+.4f} |

---

## HV dB statistics per AGB quintile

| Quintile (true mean) | Extraction | HV mean (dB) | HV std (dB) | n |
|---|---|---:|---:|---:|
{chr(10).join(rows_md)}

---

## Figures

![Saturation scatter](figures/palsar_saturation.png)

![Quintile stats](figures/palsar_quintile_stats.png)

---

## Interpretation

- **If HV mean is flat across Q2–Q5**: L-band saturation confirmed; Gaussian extraction
  cannot help because the signal itself is absent.
- **If HV std drops with Gaussian/boxcar**: speckle IS contributing noise, but signal
  may still be absent (flat mean with reduced std = cleaner zero signal).
- **If Gaussian/boxcar reveals a gradient**: the prior extraction was noise-limited
  and re-extraction with spatial smoothing could add value.
"""
    OUT_MD.write_text(md)
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
