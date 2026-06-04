"""
Extract Dynamic World composition features for AGB USA pilot plots.

Dynamic World (GOOGLE/DYNAMICWORLD/V1, 10 m, near-real-time Sentinel-2) is used here NOT as a
stand-type feature (it has a single undifferentiated `trees` class) but as a continuous
forest <-> non-forest *gradient* — a low-biomass / "is this even forest" prior. Within a plot
buffer the relative probabilities of trees vs shrub/grass/crop/bare encode canopy openness and
the non-forest gradient, which a GBT handles better than the argmax label.

Leakage-safety: each plot is composited over ITS OWN survey-year growing season (Jun 15–Sep 15
of `year`) — no post-survey data. Survey years are {2022, 2023}; we build one composite per
survey year and select per plot (same pattern as extract_disturbance_timing.py).

Buffer = 30 m radius (matches the disturbance extraction). Output:
preprocessing/dynamic_world_features.csv keyed on row_key (= features_iter3.parquet row index).

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/extract_dynamic_world.py            # full run (4,646 plots)
    uv run ... python scripts/extract_dynamic_world.py --limit 10   # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import pandas as pd

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
)
PARQUET = EXPDIR / "preprocessing/features_iter3.parquet"
OUT_CSV = EXPDIR / "preprocessing/dynamic_world_features.csv"

BATCH_SIZE = 200
BUFFER_M = 30
DW_SCALE = 10
SURVEY_YEARS = [2022, 2023]
DOY = ("-06-15", "-09-15")  # growing-season window

# Vegetation/structure probability bands (the forest<->non-forest gradient). Output cols dw_*.
DW_BANDS = ["trees", "shrub_and_scrub", "grass", "crops", "bare"]


def load_plots(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(
        PARQUET, columns=["plot_id", "project_name", "lon", "lat", "region", "year"]
    ).reset_index(drop=True)
    df["row_key"] = df.index.astype(str)
    if limit is not None:
        df = df.head(limit).copy()
    print(f"Loaded {len(df)} plots; survey years: {sorted(df['year'].unique())}")
    return df


def make_buffer_fc(df: pd.DataFrame) -> ee.FeatureCollection:
    feats = [
        ee.Feature(
            ee.Geometry.Point([float(r["lon"]), float(r["lat"])]).buffer(BUFFER_M),
            {"row_key": str(r["row_key"])},
        )
        for _, r in df.iterrows()
    ]
    return ee.FeatureCollection(feats)


def dw_composite(survey_year: int) -> ee.Image:
    """Median of DW probability bands over the survey-year growing season (leakage-safe)."""
    start = ee.Date(f"{survey_year}{DOY[0]}")
    end = ee.Date(f"{survey_year}{DOY[1]}")
    col = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate(start, end).select(DW_BANDS)
    # rename to dw_<band> so the trainer picks them up by prefix
    return col.median().rename([f"dw_{b}" for b in DW_BANDS])


def reduce_regions_batched(image: ee.Image, plots_df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"  reduceRegions [{label}] over {len(plots_df)} plots (scale={DW_SCALE} m) ...")

    def _run(sub_df: pd.DataFrame) -> list[dict]:
        fc = make_buffer_fc(sub_df)
        res = image.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=DW_SCALE)
        return res.getInfo()["features"]

    try:
        features = _run(plots_df)
        print(f"    full-collection OK: {len(features)} features")
    except Exception as exc:  # noqa: BLE001
        print(f"    full-collection failed ({str(exc)[:80]}); batching by {BATCH_SIZE}")
        features = []
        for start in range(0, len(plots_df), BATCH_SIZE):
            batch = plots_df.iloc[start : start + BATCH_SIZE]
            features.extend(_run(batch))
            print(f"    batch {start}-{start + len(batch) - 1}: total {len(features)}")

    rows = []
    for feat in features:
        props = feat.get("properties", {})
        row = {"row_key": str(props.pop("row_key", None))}
        row.update(props)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ee.Initialize()
    print("GEE initialised.")
    plots_df = load_plots(limit=args.limit)

    # one composite per survey year; reduce that year's plots against it.
    parts = []
    for sy in SURVEY_YEARS:
        grp = plots_df[plots_df["year"] == sy]
        if grp.empty:
            continue
        img = dw_composite(sy)
        parts.append(reduce_regions_batched(img, grp, f"dw_{sy}"))
    dw = pd.concat(parts, ignore_index=True)

    out_cols = [f"dw_{b}" for b in DW_BANDS]
    df = plots_df[["row_key", "plot_id", "region", "year"]].merge(dw, on="row_key", how="left")
    df = df[["row_key", "plot_id", "region", "year"] + out_cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")

    print("\n=== NULL REPORT ===")
    for region, g in df.groupby("region"):
        for c in out_cols:
            n = g[c].isna().sum()
            if n:
                print(f"  {region} | {c}: {n}/{len(g)} ({100 * n / len(g):.2f}%)")
    # sanity: dw_trees should be high for forest plots
    print(
        f"\n  dw_trees mean={df['dw_trees'].mean():.3f} "
        f"(expect high ~0.8-1.0 for mostly-forest plots)"
    )
    print(df[out_cols].describe().round(3).to_string())
    print("\nSample:")
    print(df.head(8).to_string())


if __name__ == "__main__":
    main()
