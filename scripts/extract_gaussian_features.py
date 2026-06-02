"""
Re-extract CHM, SRTM topographic, and Hansen disturbance features using
Gaussian-weighted neighbourhood smoothing rather than single-pixel centroid extraction.

AEF embeddings (emb_*) and coarse features (GEDI L4B at 1 km, TerraClimate at 4 km)
are intentionally excluded — they are not re-extracted here.

Kernel parameters:
  10 m features (CHM):   σ=15 m, radius=45 m  → 3×3 neighbourhood weighting
  30 m features (SRTM, Hansen): σ=25 m, radius=75 m  → 1–2 pixel GPS-robust weighting

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \\
        python scripts/extract_gaussian_features.py

Output: preprocessing/gaussian_features.csv (4,646 rows × 8 columns)
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import ee
import numpy as np
import pandas as pd

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
)
PARQUET = EXPDIR / "preprocessing/features_iter3.parquet"
OUT_CSV = EXPDIR / "preprocessing/gaussian_features.csv"

BATCH_SIZE = 500
SEED = 42

# Gaussian kernel parameters
SIGMA_10M, RADIUS_10M = 15, 45  # for CHM (10 m native)
SIGMA_30M, RADIUS_30M = 25, 75  # for SRTM / Hansen (30 m native)

CHM_COLS = ["chm_m"]
TOPO_COLS = ["topo_elevation", "topo_slope", "topo_aspect_cos", "topo_aspect_sin", "topo_tpi"]
DIST_COLS = ["dist_years_since"]
ALL_COLS = CHM_COLS + TOPO_COLS + DIST_COLS


# ---------------------------------------------------------------------------
# Helpers (same row_key = row_index pattern as iter2/iter3 scripts)
# ---------------------------------------------------------------------------


def load_plots() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET, columns=["plot_id", "project_name", "lon", "lat", "region"])
    df = df.reset_index(drop=True)
    df["row_key"] = df.index.astype(str)
    print(f"Loaded {len(df)} plots.")
    return df


def make_point_fc(df: pd.DataFrame) -> ee.FeatureCollection:
    return ee.FeatureCollection(
        [
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"row_key": str(r.row_key)})
            for _, r in df.iterrows()
        ]
    )


def reduce_regions_batched(
    image: ee.Image,
    plots_df: pd.DataFrame,
    label: str,
    scale: int,
    prop_names: list[str],
) -> pd.DataFrame:
    """Point extraction of a (pre-smoothed) image at all plot centroids."""
    fc = make_point_fc(plots_df)
    print(f"  {label} ({len(plots_df)} plots, scale={scale}m) ...")
    single_band = len(prop_names) == 1

    try:
        result = image.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale)
        features = result.getInfo()["features"]
        print(f"  OK: {len(features)} returned.")
    except Exception as exc:
        print(f"  Full call failed ({exc}). Batching {BATCH_SIZE} at a time.")
        features = []
        for start in range(0, len(plots_df), BATCH_SIZE):
            batch = plots_df.iloc[start : start + BATCH_SIZE]
            res = image.reduceRegions(
                collection=make_point_fc(batch), reducer=ee.Reducer.mean(), scale=scale
            )
            features.extend(res.getInfo()["features"])
            print(f"    batch {start}–{start + len(batch) - 1}: done")

    rows = []
    for feat in features:
        props = feat.get("properties", {})
        row = {"row_key": str(props.get("row_key"))}
        if single_band:
            row[prop_names[0]] = props.get("mean", np.nan)
        else:
            for col in prop_names:
                row[col] = props.get(col, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Image builders with Gaussian pre-smoothing
# ---------------------------------------------------------------------------


def build_chm_gaussian() -> ee.Image:
    """ETH CHM 2020, Gaussian-smoothed with σ=15 m (10 m native resolution)."""
    kernel = ee.Kernel.gaussian(radius=RADIUS_10M, sigma=SIGMA_10M, units="meters", normalize=True)
    raw = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").select("b1").rename("chm_m")
    return raw.reduceNeighborhood(ee.Reducer.mean(), kernel).rename("chm_m")


def build_topo_gaussian() -> ee.Image:
    """SRTM v3 topographic derivatives, Gaussian-smoothed with σ=25 m (30 m native)."""
    kernel = ee.Kernel.gaussian(radius=RADIUS_30M, sigma=SIGMA_30M, units="meters", normalize=True)
    dem = ee.Image("USGS/SRTMGL1_003").toFloat()
    slope = ee.Terrain.slope(dem).rename("topo_slope")
    aspect = ee.Terrain.aspect(dem)
    aspect_cos = aspect.multiply(math.pi / 180).cos().rename("topo_aspect_cos")
    aspect_sin = aspect.multiply(math.pi / 180).sin().rename("topo_aspect_sin")
    elevation = dem.rename("topo_elevation")
    tpi = dem.subtract(dem.focal_mean(radius=500, units="meters")).rename("topo_tpi")
    multi_band = elevation.addBands(slope).addBands(aspect_cos).addBands(aspect_sin).addBands(tpi)
    # reduceNeighborhood appends "_mean" to each band name; rename to drop that suffix
    return multi_band.reduceNeighborhood(ee.Reducer.mean(), kernel).rename(TOPO_COLS)


def build_dist_gaussian() -> ee.Image:
    """Hansen GFC 2025 years-since-disturbance, Gaussian-smoothed with σ=25 m (30 m native)."""
    kernel = ee.Kernel.gaussian(radius=RADIUS_30M, sigma=SIGMA_30M, units="meters", normalize=True)
    lossyear = ee.Image("UMD/hansen/global_forest_change_2025_v1_13").select("lossyear")
    years_since = (
        ee.Image(100)
        .where(lossyear.gt(0), ee.Image(23).subtract(lossyear).max(ee.Image(0)))
        .rename("dist_years_since")
    )
    return years_since.reduceNeighborhood(ee.Reducer.mean(), kernel).rename("dist_years_since")


# ---------------------------------------------------------------------------
# Null report
# ---------------------------------------------------------------------------


def null_report(df: pd.DataFrame, region: pd.Series) -> None:
    df2 = df.copy()
    df2["region"] = region.values
    print("\n=== NULL REPORT (gaussian_features.csv) ===")
    any_flag = False
    for reg, grp in df2.groupby("region"):
        for col in ALL_COLS:
            n = grp[col].isna().sum()
            pct = n / len(grp)
            if pct > 0:
                flag = " *** >1% ***" if pct > 0.01 else ""
                print(f"  {reg} | {col}: {n}/{len(grp)} ({100 * pct:.2f}%){flag}")
                if pct > 0.01:
                    any_flag = True
    if not any_flag:
        print("  All columns ≤1% nulls across all regions.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ee.Initialize()
    print(f"GEE initialised.  σ_10m={SIGMA_10M}m  σ_30m={SIGMA_30M}m")

    plots_df = load_plots()

    print("\n[1/3] CHM (Gaussian, σ=15 m, 10 m scale) ...")
    df_chm = reduce_regions_batched(build_chm_gaussian(), plots_df, "chm_m", 10, CHM_COLS)

    print("\n[2/3] SRTM topography (Gaussian, σ=25 m, 30 m scale) ...")
    df_topo = reduce_regions_batched(build_topo_gaussian(), plots_df, "topo_*", 30, TOPO_COLS)

    print("\n[3/3] Hansen disturbance (Gaussian, σ=25 m, 30 m scale) ...")
    df_dist = reduce_regions_batched(
        build_dist_gaussian(), plots_df, "dist_years_since", 30, DIST_COLS
    )

    # Merge on row_key
    df_out = df_chm.merge(df_topo, on="row_key", how="outer").merge(
        df_dist, on="row_key", how="outer"
    )
    df_out["row_key"] = df_out["row_key"].astype(int)
    df_out = df_out.sort_values("row_key").reset_index(drop=True)

    null_report(df_out, plots_df["region"])

    df_out.to_csv(OUT_CSV, index=False)
    sha = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()
    print(f"\nWrote {len(df_out)} rows → {OUT_CSV}")
    print(f"SHA256: {sha}")
    print("\nSample (first 3 rows):")
    print(df_out.head(3)[ALL_COLS].to_string())


if __name__ == "__main__":
    main()
