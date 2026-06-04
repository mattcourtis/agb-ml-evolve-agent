"""
Extract iteration-2 co-features for AGB USA pilot plots.

Sources:
  - users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1  → chm_m (canopy height, metres)
  - COPERNICUS/DEM/GLO30                             → topo_elevation, topo_slope,
                                                        topo_aspect_cos, topo_aspect_sin,
                                                        topo_tpi
  - UMD/hansen/global_forest_change_2025_v1_13      → dist_years_since

All extractions use point-sample (ee.Reducer.mean()) at the plot centroid.
Output: preprocessing/iter2_features.csv (567 rows × 8 columns).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path

import ee

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARQUET = (
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
    "/preprocessing/features_iter1.parquet"
)
OUT_CSV = (
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
    "/preprocessing/iter2_features.csv"
)

BATCH_SIZE = 200

CHM_COLS = ["chm_m"]
TOPO_COLS = ["topo_elevation", "topo_slope", "topo_aspect_cos", "topo_aspect_sin", "topo_tpi"]
DIST_COLS = ["dist_years_since"]
ALL_FEATURE_COLS = CHM_COLS + TOPO_COLS + DIST_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_plots(parquet_path: str) -> pd.DataFrame:
    # Each row is a unique physical plot — (project_name, plot_id) is the true unique key.
    # plot_id alone is reused across projects, so drop_duplicates('plot_id') would silently
    # drop plots from NE and parts of MW. Use the full 4,646 rows, keyed on row_index.
    df = pd.read_parquet(parquet_path, columns=["plot_id", "project_name", "lon", "lat", "region"])
    df = df.reset_index(drop=True)
    df["row_key"] = df.index.astype(str)  # stable unique key for GEE join
    print(f"Loaded {len(df)} unique plots from parquet.")
    return df


def make_point_fc(df: pd.DataFrame) -> ee.FeatureCollection:
    """Build a GEE FeatureCollection of point features (no buffer — direct centroid sample)."""
    features = [
        ee.Feature(
            ee.Geometry.Point([float(row["lon"]), float(row["lat"])]),
            {"row_key": str(row["row_key"])},
        )
        for _, row in df.iterrows()
    ]
    return ee.FeatureCollection(features)


def reduce_regions_batched(
    image: ee.Image,
    plots_df: pd.DataFrame,
    label: str,
    scale: int,
    prop_names: list[str],
) -> pd.DataFrame:
    """Run reduceRegions (mean) — full FC first, fall back to BATCH_SIZE chunks.

    GEE naming note:
    - Multi-band image: output properties are named after the band names.
    - Single-band image: output property is named "mean" (not the band name).
    We detect the single-band case (len(prop_names) == 1) and read "mean" directly.
    """
    fc = make_point_fc(plots_df)
    print(f"  Running reduceRegions for {label} ({len(plots_df)} plots, scale={scale}m) ...")
    reducer = ee.Reducer.mean()
    try:
        result = image.reduceRegions(collection=fc, reducer=reducer, scale=scale)
        features = result.getInfo()["features"]
        print(f"  Success: {len(features)} features returned.")
    except Exception as exc:
        print(f"  Full-collection call failed ({exc}). Falling back to batches of {BATCH_SIZE}.")
        features = []
        n = len(plots_df)
        for start in range(0, n, BATCH_SIZE):
            batch_df = plots_df.iloc[start : start + BATCH_SIZE]
            batch_fc = make_point_fc(batch_df)
            result = image.reduceRegions(collection=batch_fc, reducer=reducer, scale=scale)
            batch_feats = result.getInfo()["features"]
            features.extend(batch_feats)
            print(f"    Batch {start}–{start + len(batch_df) - 1}: {len(batch_feats)} features")

    single_band = len(prop_names) == 1
    rows = []
    for feat in features:
        props = feat.get("properties", {})
        row = {"row_key": str(props.get("row_key"))}
        if single_band:
            # GEE names single-band reduceRegions output "mean"
            row[prop_names[0]] = props.get("mean", np.nan)
        else:
            for col in prop_names:
                row[col] = props.get(col, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Image builders
# ---------------------------------------------------------------------------


def build_chm_image() -> ee.Image:
    """ETH Global Canopy Height 2020 — single band b1, uint8, metres."""
    return ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").select("b1").rename("chm_m")


def build_topo_image() -> ee.Image:
    """SRTM v3 (USGS/SRTMGL1_003) — single Image, 30m, correct projection for terrain.

    COPERNICUS/DEM/GLO30 was tried first but ee.Terrain.slope() on the mosaicked
    ImageCollection returns near-zero slopes because .mosaic() destroys the native
    projection and GEE falls back to geographic-degree spacing for the gradient. Using
    SRTM (a single Image with intact projection) gives correct metric slopes.
    """
    dem = ee.Image("USGS/SRTMGL1_003").rename("DEM")
    dem_f = dem.toFloat()
    slope = ee.Terrain.slope(dem_f).rename("topo_slope")
    aspect = ee.Terrain.aspect(dem_f)
    aspect_cos = aspect.multiply(math.pi / 180).cos().rename("topo_aspect_cos")
    aspect_sin = aspect.multiply(math.pi / 180).sin().rename("topo_aspect_sin")
    elevation = dem_f.rename("topo_elevation")

    # TPI: local elevation minus mean within 500 m neighbourhood
    tpi = dem_f.subtract(dem_f.focal_mean(radius=500, units="meters")).rename("topo_tpi")

    return elevation.addBands(slope).addBands(aspect_cos).addBands(aspect_sin).addBands(tpi)


def build_dist_image(survey_year: int = 2023) -> ee.Image:
    """Hansen GFC — years since last *pre/at-survey* loss (100 if undisturbed by survey).

    lossyear values: 0 = no loss, 1..N = two-digit calendar year (1=2001, 25=2025).

    BUG FIX (2026-06): the previous implementation computed `max(0, 23 - lossyear)` with a
    hard-coded 2023 reference. This (a) ignored each plot's actual survey year and (b) mapped
    POST-survey loss (lossyear > 23) to 0 — i.e. it stamped a "just disturbed" signal onto plots
    whose field biomass was measured *before* the harvest and is legitimately high. The audit
    found 46% of post-survey plots received `years_since=0`, a feature-label inversion.

    Corrected, leakage-safe definition (parameterised by the plot's survey year):
      - loss in (0, survey_year_code]  -> years_since = survey_year_code - lossyear  (>= 0)
      - no loss, OR loss AFTER survey   -> 100 (treated as undisturbed as of survey date)
    Post-survey information is never encoded. Callers should pass the plot's survey year; the
    survey-relative production feature is `dstx_pre_ysd` in extract_disturbance_timing.py.
    """
    lossyear = ee.Image("UMD/hansen/global_forest_change_2025_v1_13").select("lossyear")
    code = survey_year - 2000  # e.g. 22 or 23
    pre_survey_loss = lossyear.gt(0).And(lossyear.lte(code))
    years_since = (
        ee.Image(100)  # undisturbed (or only post-survey loss) -> sentinel
        .where(pre_survey_loss, ee.Image(code).subtract(lossyear))
        .rename("dist_years_since")
    )
    return years_since


# ---------------------------------------------------------------------------
# Null-count report
# ---------------------------------------------------------------------------


def null_report(df: pd.DataFrame, region_map: pd.DataFrame) -> None:
    """Print null counts per column per region; flag any column with >1% nulls."""
    merged = df if "region" in df.columns else df.merge(region_map, on="row_key", how="left")
    print("\n=== NULL REPORT (iter2_features.csv) ===")
    threshold = 0.01
    flagged = []
    for region, grp in merged.groupby("region"):
        n_total = len(grp)
        for col in ALL_FEATURE_COLS:
            n_null = grp[col].isna().sum()
            pct = n_null / n_total
            flag = "  *** >1% NULLS ***" if pct > threshold else ""
            if pct > 0:
                print(f"  {region} | {col}: {n_null}/{n_total} ({100 * pct:.2f}%){flag}")
                if pct > threshold:
                    flagged.append((region, col, pct))
    if not flagged:
        print("  All columns within threshold (<= 1% nulls) across all regions.")
    else:
        print("\nFLAGGED (>1% nulls):")
        for region, col, pct in flagged:
            print(f"  {region} | {col}: {100 * pct:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ee.Initialize()
    print("GEE initialised.")

    plots_df = load_plots(PARQUET)
    region_map = plots_df[["row_key", "region"]].copy()

    # --- CHM (10 m scale) ---
    print("\n[1/3] CHM extraction ...")
    chm_img = build_chm_image()
    df_chm = reduce_regions_batched(chm_img, plots_df, "chm_m", scale=10, prop_names=CHM_COLS)

    # --- Topography (30 m scale) ---
    print("\n[2/3] Topography extraction ...")
    topo_img = build_topo_image()
    df_topo = reduce_regions_batched(topo_img, plots_df, "topo_*", scale=30, prop_names=TOPO_COLS)

    # --- Disturbance (30 m scale) ---
    print("\n[3/3] Disturbance extraction ...")
    dist_img = build_dist_image()
    df_dist = reduce_regions_batched(
        dist_img, plots_df, "dist_years_since", scale=30, prop_names=DIST_COLS
    )

    # --- Merge all three on row_key, then attach plot_id/region ---
    df_out = df_chm.merge(df_topo, on="row_key", how="outer").merge(
        df_dist, on="row_key", how="outer"
    )
    # Re-attach plot_id and region from the original plots_df
    df_out["row_key"] = df_out["row_key"].astype(str)
    plots_df["row_key"] = plots_df["row_key"].astype(str)
    df_out = df_out.merge(plots_df[["row_key", "plot_id", "region"]], on="row_key", how="left")

    # Enforce column order
    df_out = df_out[["row_key", "plot_id"] + ALL_FEATURE_COLS]

    # Write CSV
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df_out)} rows to {OUT_CSV}")
    print(f"Columns: {list(df_out.columns)}")

    # Null report
    null_report(df_out, region_map)

    print("\nSample rows (first 5):")
    print(df_out.head().to_string())


if __name__ == "__main__":
    main()
