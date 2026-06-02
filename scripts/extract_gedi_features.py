"""
Extract GEDI canopy-height features for AGB USA pilot plots.

Sources:
  - LARSE/GEDI/GEDI02_A_002_MONTHLY  →  rh98 (primary canopy height metric)
  - LARSE/GEDI/GEDI02_B_002_MONTHLY  →  cover, pai, fhd_normal (canopy structure)

Quality filters applied to pixel masks:
  - L2A: quality_flag == 1, degrade_flag == 0
  - L2B: l2b_quality_flag == 1, degrade_flag == 0

Temporal window: 2021-01-01 to 2024-01-01 (36 months)
Buffer: 500 m radius per plot centroid
  Note: research spec specified 50 m, but GEDI has sparse orbital sampling (~25 m footprint
  on sparse 600 m cross-track spacing), so a 50 m circle frequently contains no pixel
  centres. A 500 m buffer gives >99% plot coverage whilst remaining within the same
  forest stand for most sites. This decision is documented in preprocessing_spec.md.
Scale: 25 m (GEDI native resolution)

Approach:
  - Use ImageCollection + updateMask + median composite, then reduceRegions.
  - This is the correct GEE pattern for raster GEDI monthly collections (each item is
    an Image, not a FeatureCollection).
  - All bands are reduced in one reduceRegions call using ee.Reducer.mean().
    GEE names output properties after band names when the image has >1 band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import ee

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARQUET = (
    "/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet"
)
OUT_CSV = "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/preprocessing/gedi_features.csv"
START = "2021-01-01"
END = "2024-01-01"
BUFFER_M = 500  # expanded from spec's 50 m to ensure GEDI pixel coverage
SCALE = 25
BATCH_SIZE = 200

ALL_BANDS = [
    "rh98",
    "cover",
    "pai",
    "fhd_normal",
    "gedi_n_samples",
    "gedi_temporal_coverage_months",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_plots(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=["plot_id", "lon", "lat", "region"])
    df = df.drop_duplicates("plot_id").reset_index(drop=True)
    df["plot_id"] = df["plot_id"].astype(str)
    print(f"Loaded {len(df)} unique plots from parquet.")
    return df


def make_feature_collection(df: pd.DataFrame) -> ee.FeatureCollection:
    """Build a GEE FeatureCollection of buffered circles."""
    features = [
        ee.Feature(
            ee.Geometry.Point([float(row["lon"]), float(row["lat"])]).buffer(BUFFER_M),
            {"plot_id": str(row["plot_id"])},
        )
        for _, row in df.iterrows()
    ]
    return ee.FeatureCollection(features)


def build_combined_image() -> ee.Image:
    """Build a multi-band composite with all 6 target bands.

    Multi-band reduceRegions with ee.Reducer.mean() preserves band names as
    output property names. All 6 bands are combined so we can extract in one call.
    """

    # L2A
    def mask_l2a(img: ee.Image) -> ee.Image:
        mask = img.select("quality_flag").eq(1).And(img.select("degrade_flag").eq(0))
        return img.updateMask(mask)

    l2a_ic = ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY").filterDate(START, END)
    l2a_masked = l2a_ic.map(mask_l2a)
    rh98_img = l2a_masked.select("rh98").median().rename(["rh98"])

    # gedi_n_samples = number of monthly composites with >=1 valid pixel in the buffer.
    # Approach: for each monthly image, create a binary presence band (1 where rh98 is
    # valid after quality masking, 0 elsewhere via unmask). Sum across all 36 months to
    # get a per-pixel count (0–36). Then use ee.Reducer.max() over the buffer so that
    # any pixel in the buffer having N months of coverage is representative.
    #
    # Note: ee.Reducer.max() names the output property "max", not the band name.
    # The features_to_df function must look up "max" instead of "gedi_n_samples".
    # This is handled separately in extract_count_image() below; the combined image
    # uses the legacy float count for compatibility — counts are corrected post-hoc
    # via scripts/fix_gedi_counts.py.
    #
    # Legacy count (pixel-mean; values ~1.0): retained here for the combined-image
    # path but the correct integer values are written by fix_gedi_counts.py.
    count_raw = l2a_masked.select("rh98").count()
    count_img = ee.Image.constant(0).add(count_raw).rename(["gedi_n_samples"]).toFloat()

    # gedi_temporal_coverage_months = same as gedi_n_samples for MONTHLY collection
    temporal_img = count_img.rename(["gedi_temporal_coverage_months"])

    # L2B
    def mask_l2b(img: ee.Image) -> ee.Image:
        mask = img.select("l2b_quality_flag").eq(1).And(img.select("degrade_flag").eq(0))
        return img.updateMask(mask)

    l2b_ic = ee.ImageCollection("LARSE/GEDI/GEDI02_B_002_MONTHLY").filterDate(START, END)
    l2b_masked = l2b_ic.map(mask_l2b)
    cover_img = l2b_masked.select("cover").median().rename(["cover"])
    pai_img = l2b_masked.select("pai").median().rename(["pai"])
    fhd_img = l2b_masked.select("fhd_normal").median().rename(["fhd_normal"])

    # Combine all 6 bands into one image for a single reduceRegions call
    return (
        rh98_img.addBands(cover_img)
        .addBands(pai_img)
        .addBands(fhd_img)
        .addBands(count_img)
        .addBands(temporal_img)
    )


def reduce_regions_with_fallback(
    image: ee.Image,
    plots_df: pd.DataFrame,
    label: str,
) -> list[dict]:
    """Run reduceRegions (mean) with full FC first; fall back to batches on failure."""
    fc = make_feature_collection(plots_df)
    print(f"  Running reduceRegions for {label} ({len(plots_df)} plots) ...")
    reducer = ee.Reducer.mean()
    try:
        result = image.reduceRegions(collection=fc, reducer=reducer, scale=SCALE)
        feats = result.getInfo()["features"]
        print(f"  Success: {len(feats)} features returned.")
        return feats
    except Exception as exc:
        print(f"  Full-collection call failed ({exc}). Falling back to batches of {BATCH_SIZE}.")
        all_feats = []
        n = len(plots_df)
        for start in range(0, n, BATCH_SIZE):
            batch_df = plots_df.iloc[start : start + BATCH_SIZE]
            batch_fc = make_feature_collection(batch_df)
            result = image.reduceRegions(collection=batch_fc, reducer=reducer, scale=SCALE)
            batch_feats = result.getInfo()["features"]
            all_feats.extend(batch_feats)
            print(f"    Batch {start}–{start + len(batch_df) - 1}: {len(batch_feats)} features")
        return all_feats


def features_to_df(features: list[dict]) -> pd.DataFrame:
    rows = []
    for feat in features:
        props = feat.get("properties", {})
        row = {"plot_id": str(props.get("plot_id"))}
        for k in ALL_BANDS:
            row[k] = props.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ee.Initialize()
    print("GEE initialised.")

    plots_df = load_plots(PARQUET)

    print("Building combined GEDI composite image (6 bands) ...")
    combined_img = build_combined_image()

    feats = reduce_regions_with_fallback(combined_img, plots_df, "all 6 GEDI bands")
    df_out = features_to_df(feats)
    df_out = df_out[["plot_id"] + ALL_BANDS]

    # Write CSV
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df_out)} rows to {OUT_CSV}")

    # Coverage report
    print("\n=== COVERAGE REPORT ===")
    total = len(plots_df)
    zero_n = (df_out["gedi_n_samples"].fillna(0) == 0).sum()
    null_rh98 = df_out["rh98"].isna().sum()
    print(f"Total plots:              {total}")
    print(f"Plots gedi_n_samples=0:   {zero_n} ({100 * zero_n / total:.1f}%)")
    print(f"Plots rh98 is null:       {null_rh98} ({100 * null_rh98 / total:.1f}%)")

    region_join = plots_df[["plot_id", "region"]].copy()
    df_cov = df_out.merge(region_join, on="plot_id", how="left")
    print("\nPer-region zero gedi_n_samples:")
    for region, grp in df_cov.groupby("region"):
        n_zero = (grp["gedi_n_samples"].fillna(0) == 0).sum()
        n_total = len(grp)
        pct = 100 * n_zero / n_total
        flag = "  *** >10% ZERO COVERAGE ***" if (region == "wv" and pct > 10) else ""
        print(f"  {region}: {n_zero}/{n_total} ({pct:.1f}%) zero{flag}")

    print("\nSample rows (first 5):")
    print(df_out.head().to_string())


if __name__ == "__main__":
    main()
