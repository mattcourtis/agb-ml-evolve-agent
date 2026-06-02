"""
Extract iteration-3 co-features for AGB USA pilot plots.

Sources:
  - LARSE/GEDI/GEDI04_B_002           → agbd_mu (mean AGBD, Mg/ha, 1 km)
  - IDAHO_EPSCOR/TERRACLIMATE          → clim_pr, clim_tmmx, clim_aet
                                         (2020-2023 annual means)

All extractions use point-sample (ee.Reducer.mean()) at the plot centroid,
keyed on row_key = row_index (one row per unique physical plot, 4,646 total).

Output: preprocessing/iter3_features.csv (4,646 rows × 6 columns).
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
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
    "/preprocessing/features_iter2.parquet"
)
OUT_CSV = (
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
    "/preprocessing/iter3_features.csv"
)

BATCH_SIZE = 500

GEDI_COLS = ["agbd_mu"]
CLIM_COLS = ["clim_pr", "clim_tmmx", "clim_aet"]
ALL_COLS = GEDI_COLS + CLIM_COLS


# ---------------------------------------------------------------------------
# Helpers (same pattern as extract_iter2_features.py)
# ---------------------------------------------------------------------------


def load_plots(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=["plot_id", "project_name", "lon", "lat", "region"])
    df = df.reset_index(drop=True)
    df["row_key"] = df.index.astype(str)
    print(f"Loaded {len(df)} plots from parquet.")
    return df


def make_point_fc(df: pd.DataFrame) -> ee.FeatureCollection:
    return ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point([float(row["lon"]), float(row["lat"])]),
                {"row_key": str(row["row_key"])},
            )
            for _, row in df.iterrows()
        ]
    )


def reduce_regions_batched(
    image: ee.Image,
    plots_df: pd.DataFrame,
    label: str,
    scale: int,
    prop_names: list[str],
) -> pd.DataFrame:
    """reduceRegions with full-collection call, batched fallback."""
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
# Image builders
# ---------------------------------------------------------------------------


def build_gedi_l4b_image() -> ee.Image:
    """GEDI L4B v002 — band MU = mean footprint AGBD (Mg/ha), 1 km gridded mosaic."""
    return ee.Image("LARSE/GEDI/GEDI04_B_002").select("MU").rename("agbd_mu")


def build_terraclimate_image() -> ee.Image:
    """TerraClimate annual means 2020–2023.

    Bands:
      pr   — precipitation (mm/month; annual mean of monthly values)
      tmmx — max temperature (0.1 °C units; divide by 10 for °C)
      aet  — actual evapotranspiration (mm/month)

    tmmx is stored in 0.1°C; convert to °C by dividing by 10 so the model
    sees physically interpretable values.
    """
    tc = (
        ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
        .filterDate("2020-01-01", "2024-01-01")
        .select(["pr", "tmmx", "aet"])
        .mean()
    )
    # tmmx: 0.1°C → °C
    tmmx_c = tc.select("tmmx").divide(10).rename("clim_tmmx")
    pr = tc.select("pr").rename("clim_pr")
    aet = tc.select("aet").rename("clim_aet")
    return pr.addBands(tmmx_c).addBands(aet)


# ---------------------------------------------------------------------------
# Null report
# ---------------------------------------------------------------------------


def null_report(df: pd.DataFrame, region_col: pd.Series) -> None:
    df2 = df.copy()
    df2["region"] = region_col.values
    print("\n=== NULL REPORT ===")
    any_flag = False
    for region, grp in df2.groupby("region"):
        for col in ALL_COLS:
            n_null = grp[col].isna().sum()
            pct = n_null / len(grp)
            if pct > 0:
                flag = " *** >1% ***" if pct > 0.01 else ""
                print(f"  {region} | {col}: {n_null}/{len(grp)} ({100 * pct:.2f}%){flag}")
                if pct > 0.01:
                    any_flag = True
    if not any_flag:
        print("  All columns ≤1% nulls across all regions.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ee.Initialize()
    print("GEE initialised.")

    plots_df = load_plots(PARQUET)

    # 1. GEDI L4B (1 km)
    print("\n[1/2] GEDI L4B AGBD ...")
    df_gedi = reduce_regions_batched(
        build_gedi_l4b_image(), plots_df, "agbd_mu", scale=1000, prop_names=GEDI_COLS
    )

    # 2. TerraClimate (~4 km)
    print("\n[2/2] TerraClimate climate normals ...")
    df_clim = reduce_regions_batched(
        build_terraclimate_image(), plots_df, "clim_*", scale=4000, prop_names=CLIM_COLS
    )

    # Merge on row_key
    df_out = df_gedi.merge(df_clim, on="row_key", how="outer")
    df_out = df_out.merge(plots_df[["row_key", "region"]], on="row_key", how="left")
    df_out["row_key"] = df_out["row_key"].astype(int)
    df_out = df_out.sort_values("row_key").reset_index(drop=True)

    null_report(df_out, df_out["region"])

    # Write CSV (drop region — it's in the parquet already)
    df_out = df_out.drop(columns=["region"])
    df_out = df_out[["row_key"] + ALL_COLS]
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df_out)} rows → {OUT_CSV}")
    print("Columns:", list(df_out.columns))
    print("\nSample (first 3 rows):")
    print(df_out.head(3).to_string())

    sha = hashlib.sha256(Path(OUT_CSV).read_bytes()).hexdigest()
    print(f"\nSHA256 {Path(OUT_CSV).name}: {sha}")


if __name__ == "__main__":
    main()
