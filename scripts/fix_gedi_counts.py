"""
Fix gedi_n_samples and gedi_temporal_coverage_months in gedi_features.csv.

The original extraction computed these as pixel-mean values (~1.0) because
it used ee.Reducer.mean() over a per-pixel count image. The correct values
are integers 0-36 representing the number of monthly composites that have
>=1 valid pixel within the 500 m buffer.

Approach (per Critic's instructions):
  - Build a binary presence image per month (1 where rh98 is valid, 0 elsewhere)
  - Sum across all 36 months -> temporal_sum_img (0-36 per pixel)
  - reduceRegions with ee.Reducer.max() -> max pixel value in 500 m buffer
  - This gives the number of months with valid GEDI data for each plot

Reuses existing rh98/cover/pai/fhd_normal values from gedi_features.csv.
Only replaces gedi_n_samples and gedi_temporal_coverage_months.
"""

from __future__ import annotations

import hashlib
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
GEDI_CSV = "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/preprocessing/gedi_features.csv"
FEATURES_ITER1 = "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/preprocessing/features_iter1.parquet"
DATA_VERSION = "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/preprocessing/data_version.txt"

START = "2021-01-01"
END = "2024-01-01"
BUFFER_M = 500
SCALE = 25
BATCH_SIZE = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_plots(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=["plot_id", "lon", "lat"])
    df = df.drop_duplicates("plot_id").reset_index(drop=True)
    df["plot_id"] = df["plot_id"].astype(str)
    print(f"Loaded {len(df)} unique plots from parquet.")
    return df


def make_feature_collection(df: pd.DataFrame) -> ee.FeatureCollection:
    features = [
        ee.Feature(
            ee.Geometry.Point([float(row["lon"]), float(row["lat"])]).buffer(BUFFER_M),
            {"plot_id": str(row["plot_id"])},
        )
        for _, row in df.iterrows()
    ]
    return ee.FeatureCollection(features)


def build_temporal_sum_image() -> ee.Image:
    """Build image where each pixel = number of months with >=1 valid GEDI shot.

    Per Critic's instructions:
      1. For each monthly image, apply quality mask then create binary presence:
         1 where rh98 is valid after masking, 0 elsewhere
      2. Sum across all months -> 0 to 36 per pixel

    Note: quality filters are applied as pixel masks inside map() rather than
    as collection-level filters, because the monthly composite images always
    exist as raster tiles; the quality_flag band contains per-pixel values.
    """
    gedi_l2a = ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY").filterDate(START, END)

    def make_presence(img: ee.Image) -> ee.Image:
        # Apply quality masks at pixel level
        quality_mask = img.select("quality_flag").eq(1).And(img.select("degrade_flag").eq(0))
        masked_rh98 = img.select("rh98").updateMask(quality_mask)
        # Binary: 1 where valid pixel exists, 0 elsewhere (unmask fills nodata with 0)
        return masked_rh98.mask().unmask(0).rename(["present"])

    presence_stack = gedi_l2a.map(make_presence)
    temporal_sum_img = presence_stack.sum().rename(["temporal_months"])
    return temporal_sum_img


def extract_counts(plots_df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal coverage counts for all plots using ee.Reducer.max()."""
    temporal_img = build_temporal_sum_image()

    all_feats = []
    n = len(plots_df)
    print(f"Extracting counts for {n} plots in batches of {BATCH_SIZE} ...")

    for start in range(0, n, BATCH_SIZE):
        batch_df = plots_df.iloc[start : start + BATCH_SIZE]
        batch_fc = make_feature_collection(batch_df)
        result = temporal_img.reduceRegions(
            collection=batch_fc,
            reducer=ee.Reducer.max(),
            scale=SCALE,
        )
        batch_feats = result.getInfo()["features"]
        all_feats.extend(batch_feats)
        print(f"  Batch {start}–{start + len(batch_df) - 1}: {len(batch_feats)} features")

    rows = []
    for feat in all_feats:
        props = feat.get("properties", {})
        # ee.Reducer.max() names the output "max", not the band name
        val = props.get("max", np.nan)
        rows.append(
            {
                "plot_id": str(props.get("plot_id")),
                "temporal_months": val,
            }
        )
    return pd.DataFrame(rows)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ee.Initialize(project="coral-theme-475715-f7")
    print("GEE initialised.")

    # Load existing gedi_features.csv (preserve rh98/cover/pai/fhd_normal)
    gedi = pd.read_csv(GEDI_CSV)
    gedi["plot_id"] = gedi["plot_id"].astype(str)
    print(f"Loaded gedi_features.csv: {len(gedi)} rows")
    print(
        f"  Current gedi_n_samples range: {gedi['gedi_n_samples'].min():.3f}–{gedi['gedi_n_samples'].max():.3f}"
    )

    # Load plot centroids
    plots_df = load_plots(PARQUET)

    # Extract counts from GEE
    counts_df = extract_counts(plots_df)
    print(f"\nExtracted counts for {len(counts_df)} plots.")

    # Convert to integer (0–36); NaN where no data (no pixels at all)
    counts_df["count_int"] = counts_df["temporal_months"].apply(
        lambda v: int(round(v)) if pd.notna(v) else 0
    )
    print(f"  Count range: {counts_df['count_int'].min()}–{counts_df['count_int'].max()}")
    print(f"  Zero-coverage plots: {(counts_df['count_int'] == 0).sum()}")

    # Merge counts into gedi dataframe
    counts_lookup = counts_df.set_index("plot_id")["count_int"].to_dict()
    gedi["gedi_n_samples"] = gedi["plot_id"].map(counts_lookup).fillna(0).astype(int)
    gedi["gedi_temporal_coverage_months"] = gedi["gedi_n_samples"]

    print(
        f"\nFixed gedi_n_samples range: {gedi['gedi_n_samples'].min()}–{gedi['gedi_n_samples'].max()}"
    )
    print(f"Zero-coverage: {(gedi['gedi_n_samples'] == 0).sum()}")

    # Overwrite gedi_features.csv
    gedi.to_csv(GEDI_CSV, index=False)
    print(f"\nOverwrote {GEDI_CSV}")

    csv_sha = sha256_file(GEDI_CSV)
    print(f"  SHA256: {csv_sha}")

    # Rebuild features_iter1.parquet
    print("\nRebuilding features_iter1.parquet ...")
    features = pd.read_parquet(PARQUET)
    features["plot_id"] = features["plot_id"].astype(str)
    gedi_merge = gedi.copy()
    gedi_merge["plot_id"] = gedi_merge["plot_id"].astype(str)

    merged = features.merge(gedi_merge, on="plot_id", how="left")
    assert len(merged) == len(features), f"Row count mismatch: {len(merged)} vs {len(features)}"
    merged.to_parquet(FEATURES_ITER1, index=False)
    print(f"Wrote {len(merged)} rows to {FEATURES_ITER1}")

    parquet_sha = sha256_file(FEATURES_ITER1)
    print(f"  SHA256: {parquet_sha}")

    # Update data_version.txt
    data_ver = Path(DATA_VERSION).read_text()
    # Replace old SHA lines
    import re

    data_ver = re.sub(
        r"  gedi_features_csv_sha256:.*",
        f"  gedi_features_csv_sha256: {csv_sha}",
        data_ver,
    )
    data_ver = re.sub(
        r"  features_iter1_parquet_sha256:.*",
        f"  features_iter1_parquet_sha256: {parquet_sha}",
        data_ver,
    )
    # Update timestamp
    data_ver = re.sub(
        r"  snapshot_timestamp_utc: 2026-05-29T10:00:00Z",
        "  snapshot_timestamp_utc: 2026-05-29T11:00:00Z",
        data_ver,
    )
    Path(DATA_VERSION).write_text(data_ver)
    print(f"\nUpdated {DATA_VERSION}")

    print("\nDone.")


if __name__ == "__main__":
    main()
