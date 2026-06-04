"""
Extract survey-relative disturbance-timing features for AGB USA pilot plots.

Hypothesis (investigation, 2026-06): harvest timing *relative to each plot's field-survey
year* carries critical, currently-mishandled information about low-biomass locations:
  - harvest BEFORE survey  -> field measured low biomass; the low target is real (Q1 driver)
  - harvest AFTER survey    -> field biomass legitimately high; "current land cover" features
                               read it as non-forest => feature-label inversion (contamination)

The existing `dist_years_since` (extract_iter2_features.py:159-163) collapses both cases:
`max(0, 23 - lossyear)` maps a 2025 clearcut and a 2023 clearcut both to 0, so a plot surveyed
2022 but cut 2024 gets a "just disturbed" signal attached to high biomass.

Engines (verified reachable on GEE; DIST-ALERT/GLAD have ZERO CONUS coverage so are excluded):
  - UMD/hansen/global_forest_change_2025_v1_13  -> calendar loss year, buffer fraction, recency
  - ee.Algorithms.TemporalSegmentation.LandTrendr -> greatest-disturbance magnitude/year/duration

Design:
  - Survey year is per-plot but only takes values {2022, 2023}. We extract RAW disturbance
    descriptors from GEE (timing-agnostic) plus two threshold variants for Hansen (<=2022,
    <=2023), then compute ALL survey-relative / leakage-safe features in pandas. This keeps the
    leakage logic transparent and auditable.
  - Predictive features use only information up to the plot's own survey year. Post-survey
    events are recorded separately (audit / cleaning), never fed to the model.
  - Buffer = 30 m radius (~3x3 Landsat/Hansen pixels; aligns with the ~25 m operational
    buffer-extraction standard found in deep research). Plots are ~7.3 m radius (sub-pixel).

Output: preprocessing/disturbance_timing_features.csv (keyed on row_key = parquet row index).

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/extract_disturbance_timing.py            # full run (4,646 plots)
    uv run ... python scripts/extract_disturbance_timing.py --limit 10   # smoke test
    uv run ... python scripts/extract_disturbance_timing.py --hansen-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
)
PARQUET = EXPDIR / "preprocessing/features_iter3.parquet"
OUT_CSV = EXPDIR / "preprocessing/disturbance_timing_features.csv"

BATCH_SIZE = 200
BUFFER_M = 30  # buffer radius in metres
HANSEN_SCALE = 30
LT_SCALE = 30

# Hansen lossyear threshold variants — one per distinct survey year in the data.
SURVEY_YEARS = [2022, 2023]  # lossyear codes 22, 23


# ---------------------------------------------------------------------------
# Plot loading / FeatureCollection
# ---------------------------------------------------------------------------


def load_plots(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(
        PARQUET, columns=["plot_id", "project_name", "lon", "lat", "region", "year"]
    )
    df = df.reset_index(drop=True)
    df["row_key"] = df.index.astype(str)
    if limit is not None:
        df = df.head(limit).copy()
    print(f"Loaded {len(df)} plots from {PARQUET.name}.")
    print(f"  survey years present: {sorted(df['year'].unique())}")
    return df


def make_buffer_fc(df: pd.DataFrame) -> ee.FeatureCollection:
    """Point features buffered to BUFFER_M radius, carrying row_key."""
    feats = [
        ee.Feature(
            ee.Geometry.Point([float(r["lon"]), float(r["lat"])]).buffer(BUFFER_M),
            {"row_key": str(r["row_key"])},
        )
        for _, r in df.iterrows()
    ]
    return ee.FeatureCollection(feats)


def reduce_regions_batched(
    image: ee.Image,
    plots_df: pd.DataFrame,
    label: str,
    scale: int,
    reducer: ee.Reducer,
    tile_scale: float = 1,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """reduceRegions over buffered plots; full-collection first, batch fallback.

    `tile_scale` (1-16) splits the computation into smaller tiles to avoid GEE's
    'Computed image is too large' on heavy (e.g. LandTrendr) graphs.
    Returns a DataFrame with row_key plus one column per output property the reducer emits.
    """
    print(f"  reduceRegions [{label}] over {len(plots_df)} plots (scale={scale} m) ...")

    def _run(sub_df: pd.DataFrame) -> list[dict]:
        fc = make_buffer_fc(sub_df)
        res = image.reduceRegions(collection=fc, reducer=reducer, scale=scale, tileScale=tile_scale)
        return res.getInfo()["features"]

    try:
        features = _run(plots_df)
        print(f"    full-collection OK: {len(features)} features")
    except Exception as exc:  # noqa: BLE001
        print(f"    full-collection failed ({str(exc)[:80]}); batching by {batch_size}")
        features = []
        for start in range(0, len(plots_df), batch_size):
            batch = plots_df.iloc[start : start + batch_size]
            features.extend(_run(batch))
            print(f"    batch {start}-{start + len(batch) - 1}: total {len(features)}")

    rows = []
    for feat in features:
        props = feat.get("properties", {})
        row = {"row_key": str(props.pop("row_key", None))}
        row.update(props)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hansen image — raw, timing-agnostic descriptors + per-survey-year variants
# ---------------------------------------------------------------------------


def build_hansen_image() -> ee.Image:
    """Multi-band Hansen image; reduced with mean+max over the buffer.

    Bands (all derived from `lossyear`, values 0=no loss, 1..25 = year 2001..2025):
      ly_present   : 1 where any loss              -> mean = fraction of buffer ever disturbed
      ly_year      : lossyear (0..25)              -> max  = most-recent loss code in buffer
      preYY_frac   : loss with year<=YY            -> mean = pre/at-survey disturbed fraction
      preYY_year   : lossyear where year<=YY       -> max  = most-recent pre/at-survey loss code
      postYY_pres  : 1 where loss year > YY        -> max  = any post-survey loss in buffer
    for YY in {22, 23}.
    """
    ly = ee.Image("UMD/hansen/global_forest_change_2025_v1_13").select("lossyear")
    present = ly.gt(0).rename("hn_ly_present")
    year = ly.rename("hn_ly_year")
    bands = [present, year]
    for sy in SURVEY_YEARS:
        yy = sy - 2000  # 22 or 23
        pre = ly.gt(0).And(ly.lte(yy))
        bands.append(pre.rename(f"hn_pre{yy}_frac"))
        bands.append(ly.updateMask(pre).unmask(0).rename(f"hn_pre{yy}_year"))
        bands.append(ly.gt(yy).rename(f"hn_post{yy}_pres"))
    img = bands[0]
    for b in bands[1:]:
        img = img.addBands(b)
    return img


def extract_hansen(plots_df: pd.DataFrame) -> pd.DataFrame:
    img = build_hansen_image()
    reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)
    raw = reduce_regions_batched(img, plots_df, "hansen", HANSEN_SCALE, reducer)
    # reducer emits <band>_mean and <band>_max for every band
    return raw


# ---------------------------------------------------------------------------
# LandTrendr — greatest-disturbance magnitude / year / duration (severity)
# ---------------------------------------------------------------------------

# Growing-season window (leaf-on) for annual NBR composites.
LT_START_YEAR = 1986
LT_END_YEAR = 2024
LT_DOY = ("-06-15", "-09-15")

LT_PARAMS = {
    "maxSegments": 6,
    "spikeThreshold": 0.9,
    "vertexCountOvershoot": 3,
    "preventOneYearRecovery": True,
    "recoveryThreshold": 0.25,
    "pvalThreshold": 0.05,
    "bestModelProportion": 0.75,
    "minObservationsNeeded": 6,
}


def _mask_l457(img: ee.Image) -> ee.Image:
    """Cloud/shadow mask for Landsat 4/5/7 C2 L2; scale SR; compute NBR."""
    qa = img.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))  # cloud, shadow
    sr = img.select(["SR_B4", "SR_B7"]).multiply(0.0000275).add(-0.2)
    nbr = sr.normalizedDifference(["SR_B4", "SR_B7"]).rename("NBR")
    # normalizedDifference builds a NEW image -> must re-attach time or filterDate finds nothing.
    return nbr.updateMask(mask).copyProperties(img, ["system:time_start"])


def _mask_l89(img: ee.Image) -> ee.Image:
    """Cloud/shadow mask for Landsat 8/9 C2 L2; scale SR; compute NBR (NIR=B5, SWIR2=B7)."""
    qa = img.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    sr = img.select(["SR_B5", "SR_B7"]).multiply(0.0000275).add(-0.2)
    nbr = sr.normalizedDifference(["SR_B5", "SR_B7"]).rename("NBR")
    # normalizedDifference builds a NEW image -> must re-attach time or filterDate finds nothing.
    return nbr.updateMask(mask).copyProperties(img, ["system:time_start"])


def _annual_nbr(region: ee.Geometry) -> ee.ImageCollection:
    """One growing-season median NBR image per year, harmonised across L5/7/8/9."""
    l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(_mask_l457)
    l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").map(_mask_l457)
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").map(_mask_l89)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").map(_mask_l89)
    merged = l5.merge(l7).merge(l8).merge(l9).filterBounds(region)

    # masked fallback so empty years still carry an NBR band (LandTrendr tolerates gaps).
    empty = ee.Image.constant(0).rename("NBR").updateMask(ee.Image.constant(0)).toFloat()

    def year_img(y):
        y = ee.Number(y)
        start = ee.Date.fromYMD(y, 6, 15)
        end = ee.Date.fromYMD(y, 9, 15)
        col = merged.filterDate(start, end)
        nbr = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median().select("NBR"), empty))
        # LandTrendr wants disturbance as a POSITIVE delta -> invert NBR (loss => increase).
        return (
            ee.Image(nbr)
            .multiply(-1)
            .rename("NBR_inv")
            .set("system:time_start", start.millis())
            .toFloat()
        )

    years = ee.List.sequence(LT_START_YEAR, LT_END_YEAR)
    return ee.ImageCollection(years.map(year_img))


def build_landtrendr_image(region: ee.Geometry) -> ee.Image:
    """Per-pixel greatest single-year disturbance severity, split by survey-year window.

    Uses the LandTrendr *fitted* series (row 2) and the greatest year-over-year rise in
    inverted NBR (= sharpest NBR drop = harvest/disturbance). To stay leakage-safe without
    fragile year-extraction array math, the magnitude is computed separately over the
    pre/at-survey window and the post-survey window via `arrayMask` on the (same-shape)
    year array — no length-1 broadcasting. ×1000 scaled. Bands, for YY in {22, 23}:
      lt_mag_preYY  : greatest inverted-NBR rise among deltas with year <= 20YY (predictive)
      lt_mag_postYY : greatest rise among deltas with year > 20YY (post-survey contamination)
    """
    ts = _annual_nbr(region)
    lt = ee.Algorithms.TemporalSegmentation.LandTrendr(timeSeries=ts, **LT_PARAMS)
    arr = lt.select("LandTrendr")  # rows: [year, source, fitted, isVertex]

    years_row = arr.arraySlice(0, 0, 1)  # [1, nYears]
    fitted_row = arr.arraySlice(0, 2, 3)  # [1, nYears]

    # year-over-year change of inverted NBR; positive = disturbance. Aligned with right_year.
    delta = fitted_row.arraySlice(1, 1, None).subtract(fitted_row.arraySlice(1, 0, -1))  # [1, n-1]
    right_year = years_row.arraySlice(1, 1, None)  # [1, n-1] (same shape as delta)

    out = None
    for sy in SURVEY_YEARS:
        cal = sy  # right_year holds calendar years
        pre = delta.arrayMask(right_year.lte(cal)).arrayReduce(ee.Reducer.max(), [1])
        post = delta.arrayMask(right_year.gt(cal)).arrayReduce(ee.Reducer.max(), [1])
        pre_b = pre.arrayProject([0]).arrayFlatten([[f"lt_mag_pre{sy - 2000}"]]).multiply(1000)
        post_b = post.arrayProject([0]).arrayFlatten([[f"lt_mag_post{sy - 2000}"]]).multiply(1000)
        out = pre_b.addBands(post_b) if out is None else out.addBands(pre_b).addBands(post_b)
    return out


LT_BATCH = 40  # small batches; LandTrendr array ops are heavy per request
LT_TILE_SCALE = 8  # split computation into smaller tiles to avoid 'image too large'


def extract_landtrendr(plots_df: pd.DataFrame) -> pd.DataFrame:
    """Run LandTrendr in small batches, each with a tight local AOI.

    The LandTrendr array graph (39-year fitted trajectory + array reductions) is too heavy to
    evaluate over a large region or many points per request. We therefore build the LandTrendr
    image freshly per batch over just that batch's spatial extent, with a small batch size and
    high tileScale.
    """
    out = []
    reducer = ee.Reducer.mean()
    plots_df = plots_df.reset_index(drop=True)
    n = len(plots_df)
    for start in range(0, n, LT_BATCH):
        batch = plots_df.iloc[start : start + LT_BATCH]
        aoi = ee.Geometry.Rectangle(
            [
                batch["lon"].min() - 0.05,
                batch["lat"].min() - 0.05,
                batch["lon"].max() + 0.05,
                batch["lat"].max() + 0.05,
            ]
        )
        img = build_landtrendr_image(aoi)
        label = f"landtrendr[{start}-{start + len(batch) - 1}]"
        df_b = reduce_regions_batched(
            img, batch, label, LT_SCALE, reducer, tile_scale=LT_TILE_SCALE, batch_size=LT_BATCH
        )
        out.append(df_b)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Derive survey-relative, leakage-safe features (pandas)
# ---------------------------------------------------------------------------

NO_DIST_YSD = 100.0  # sentinel: no pre/at-survey disturbance (matches existing dist_years_since)


def derive_features(
    plots_df: pd.DataFrame, hansen: pd.DataFrame, lt: pd.DataFrame | None
) -> pd.DataFrame:
    df = plots_df[["row_key", "plot_id", "project_name", "region", "year"]].merge(
        hansen, on="row_key", how="left"
    )
    if lt is not None:
        df = df.merge(lt, on="row_key", how="left")

    # --- Hansen calendar loss (audit fields; not predictors) ---
    ly_max = df.get("hn_ly_year_max")  # most-recent loss code in buffer (0 if none)
    df["dstx_hansen_loss_year"] = np.where(ly_max.fillna(0) > 0, 2000 + ly_max.fillna(0), np.nan)
    df["dstx_delta_survey"] = df["dstx_hansen_loss_year"] - df["year"]

    # --- predictive: pre/at-survey recency & fraction (leakage-safe) ---
    # Select the per-survey-year variant column (hn_pre22_* / hn_pre23_*) matching each plot.
    pre_year_max = pd.Series(np.nan, index=df.index, dtype=float)
    pre_frac = pd.Series(np.nan, index=df.index, dtype=float)
    post_pres = pd.Series(0.0, index=df.index, dtype=float)
    for sy in SURVEY_YEARS:
        code = sy - 2000
        m = df["year"] == sy
        yc = f"hn_pre{code}_year_max"
        fc = f"hn_pre{code}_frac_mean"
        pc = f"hn_post{code}_pres_max"
        if yc in df.columns:
            pre_year_max[m] = df.loc[m, yc].astype(float)
        if fc in df.columns:
            pre_frac[m] = df.loc[m, fc].astype(float)
        if pc in df.columns:
            post_pres[m] = df.loc[m, pc].astype(float)

    has_pre = pre_year_max.fillna(0) > 0
    pre_cal_year = np.where(has_pre, 2000 + pre_year_max.fillna(0), np.nan)
    df["dstx_pre_ysd"] = np.where(has_pre, df["year"] - pre_cal_year, NO_DIST_YSD)
    df["dstx_pre_loss_5yr"] = (has_pre & (df["year"] - pre_cal_year <= 5)).astype(int)
    df["dstx_loss_frac_buf"] = pre_frac.fillna(0.0)

    # --- LandTrendr severity (predictive, restricted to pre/at-survey window) ---
    LT_POST_THRESH = 150.0  # ×1000 inverted-NBR rise marking a stand-replacing-scale post event
    if lt is not None:
        lt_pre = pd.Series(np.nan, index=df.index, dtype=float)
        lt_post_mag = pd.Series(np.nan, index=df.index, dtype=float)
        for sy in SURVEY_YEARS:
            m = df["year"] == sy
            prec = f"lt_mag_pre{sy - 2000}"
            postc = f"lt_mag_post{sy - 2000}"
            if prec in df.columns:
                lt_pre[m] = df.loc[m, prec].astype(float)
            if postc in df.columns:
                lt_post_mag[m] = df.loc[m, postc].astype(float)
        # clamp negatives (recovery-only pixels) to 0 = no disturbance.
        df["dstx_lt_mag"] = lt_pre.clip(lower=0).fillna(0.0)
        lt_post = lt_post_mag.fillna(0.0) > LT_POST_THRESH
    else:
        lt_post = pd.Series(False, index=df.index)

    # --- contamination marker (audit / cleaning only; NOT a predictor) ---
    df["dstx_post_survey_flag"] = ((post_pres > 0) | lt_post).astype(int)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PREDICTIVE = [
    "dstx_pre_loss_5yr",
    "dstx_pre_ysd",
    "dstx_loss_frac_buf",
    "dstx_lt_mag",
]
AUDIT = [
    "dstx_hansen_loss_year",
    "dstx_delta_survey",
    "dstx_post_survey_flag",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on first N plots")
    ap.add_argument("--hansen-only", action="store_true", help="skip LandTrendr")
    args = ap.parse_args()

    ee.Initialize()
    print("GEE initialised.")

    plots_df = load_plots(limit=args.limit)

    print("\n[1/2] Hansen extraction ...")
    hansen = extract_hansen(plots_df)
    print(f"  hansen rows: {len(hansen)}; cols: {list(hansen.columns)}")

    lt = None
    if not args.hansen_only:
        print("\n[2/2] LandTrendr extraction ...")
        lt = extract_landtrendr(plots_df)
        print(f"  landtrendr rows: {len(lt)}; cols: {list(lt.columns)}")

    df = derive_features(plots_df, hansen, lt)

    keep = ["row_key", "plot_id", "project_name", "region", "year"] + PREDICTIVE + AUDIT
    keep = [c for c in keep if c in df.columns]
    df_out = df[keep].copy()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df_out)} rows to {OUT_CSV}")

    # --- quick leakage assertion: predictive features must not encode post-survey events ---
    if "dstx_pre_ysd" in df_out:
        bad = df_out[(df_out["dstx_pre_ysd"] < 0)]
        assert bad.empty, f"LEAKAGE: {len(bad)} rows with negative pre_ysd (post-survey)"
        print("  leakage check OK: no negative pre_ysd.")

    # --- null report (predictive cols) ---
    print("\n=== NULL REPORT (predictive cols) ===")
    for region, grp in df_out.groupby("region"):
        for col in PREDICTIVE:
            if col in grp:
                n = grp[col].isna().sum()
                if n:
                    print(f"  {region} | {col}: {n}/{len(grp)} ({100 * n / len(grp):.2f}%)")
    print("\nSample:")
    print(df_out.head(8).to_string())


if __name__ == "__main__":
    main()
