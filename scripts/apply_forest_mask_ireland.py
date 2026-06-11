"""
Apply a Dynamic World forest/clearfell mask to the EXISTING Irish per-pixel AGB predictions.

For a given per-pixel checkpoint set (preprocessing/_pixel_pred/ or _pixel_pred_yYYYY/) and a
Dynamic World year Y, sample the DW V1 `trees` GROWING-SEASON MEDIAN (Apr-Sep of year Y) at each
pixel centre (lon/lat) via GEE, then apply the mask:

    forest  if trees_prob >= 0.5  -> keep pred_tco2_acre
    else (non-forest/clearfell)   -> set pred_tco2_acre = 0

Re-aggregated stand AGB = mean over ALL pixels (masked = 0) = forest_fraction * mean(forest preds).
forest_fraction = fraction of pixels with trees >= 0.5.

DW V1 coverage starts 2015-06; the requested year is clamped to >= 2016 so a full Apr-Sep growing
season is available.

Per-stand DW samples are checkpointed under preprocessing/_dw_mask_yYYYY/<Location>.parquet
(columns: lon, lat, trees) so the run is fully RESUMABLE. A per-stand summary JSON is written under
_dw_mask_yYYYY/_summary/<Location>.json. Reuses bare ee.Initialize() and the per-stand checkpoint
pattern from scripts/per_pixel_inference.py; DW image construction + >=0.5 threshold from
scripts/apply_forest_mask.py.

    # Year-matched set: DW year == prediction year
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/apply_forest_mask_ireland.py \
        --ckpt _pixel_pred_y2022 --dw-year 2022
    # Survey-year set: DW year == each stand's survey_year (clamped >=2016)
    uv run ... python scripts/apply_forest_mask_ireland.py --ckpt _pixel_pred --dw-year survey
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PREP = EXPDIR / "preprocessing"
DISSOLVED = PREP / "ireland_locations_dissolved.gpkg"

FOREST_THRESH = 0.5
DW_SCALE = 10
TILE_SCALE = 8
DW_MIN_YEAR = 2016  # DW V1 starts 2015-06; clamp so a full Apr-Sep season exists
GROW = ("-04-01", "-10-01")  # Apr-Sep growing season (end exclusive)
POINT_BATCH = 2500  # points per reduceRegions request


def dw_trees_median(year: int) -> ee.Image:
    """DW V1 `trees` probability, growing-season (Apr-Sep) median of `year`."""
    start = ee.Date(f"{year}{GROW[0]}")
    end = ee.Date(f"{year}{GROW[1]}")
    return (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .select("trees")
        .median()
        .rename("trees")
    )


def sample_points(img: ee.Image, lon, lat) -> list[float]:
    """Sample DW `trees` at each (lon, lat) 10 m pixel centre via sampleRegions, batched.

    sampleRegions(scale=10) returns the value of the DW pixel containing each point (DW is 10 m
    native, so this is the per-pixel trees prob). reduceRegions on a zero-area POINT returns no
    value, so sampleRegions is used instead. Point order is recovered via an explicit integer
    `pidx`; points falling on a DW data gap are dropped by sampleRegions and stay None here.
    """
    out: list[float | None] = [None] * len(lon)
    for s in range(0, len(lon), POINT_BATCH):
        idx = list(range(s, min(s + POINT_BATCH, len(lon))))
        feats = [
            ee.Feature(
                ee.Geometry.Point([float(lon[i]), float(lat[i])], proj="EPSG:4326"),
                {"pidx": i},
            )
            for i in idx
        ]
        fc = ee.FeatureCollection(feats)
        res = img.sampleRegions(
            collection=fc, scale=DW_SCALE, tileScale=TILE_SCALE, geometries=False
        ).getInfo()["features"]
        for feat in res:
            p = feat["properties"]
            if "trees" in p and p["trees"] is not None:
                out[int(p["pidx"])] = p["trees"]
    return out


def dw_year_for(name: str, dw_year_arg: str, survey_year: int) -> int:
    if dw_year_arg == "survey":
        y = int(survey_year)
    else:
        y = int(dw_year_arg)
    return max(y, DW_MIN_YEAR)


def process_stand(name: str, ckpt_dir: Path, out_dir: Path, sum_dir: Path, dw_year: int) -> dict:
    """Sample DW at every pixel of one stand, write the per-stand DW checkpoint + summary."""
    pix = pd.read_parquet(ckpt_dir / f"{name}.parquet")
    lon = pix["lon"].to_numpy()
    lat = pix["lat"].to_numpy()

    img = dw_trees_median(dw_year)
    trees = sample_points(img, lon, lat)
    pix = pix.copy()
    pix["trees"] = trees
    # DW gap pixels (no growing-season obs) -> treat as non-forest (conservative, mask to 0)
    pix["trees"] = pix["trees"].astype(float)
    n_dw_null = int(pix["trees"].isna().sum())
    forest = pix["trees"].fillna(0.0) >= FOREST_THRESH

    pix["forest"] = forest
    pix["pred_masked_tco2_acre"] = pix["pred_tco2_acre"].where(forest, 0.0)
    pix[["lon", "lat", "trees", "forest", "pred_masked_tco2_acre"]].to_parquet(
        out_dir / f"{name}.parquet", index=False
    )

    forest_fraction = float(forest.mean())
    masked_density = float(pix["pred_masked_tco2_acre"].mean())
    unmasked_density = float(pix["pred_tco2_acre"].mean())
    summ = {
        "Location_Name": name,
        "dw_year": dw_year,
        "n_pixels": int(len(pix)),
        "n_dw_null": n_dw_null,
        "n_forest": int(forest.sum()),
        "forest_fraction": forest_fraction,
        "masked_pred_pixel_tCO2_acre": masked_density,
        "unmasked_pred_pixel_tCO2_acre": unmasked_density,
        "masked_pred_pixel_min": float(pix["pred_masked_tco2_acre"].min()),
        "masked_pred_pixel_max": float(pix["pred_masked_tco2_acre"].max()),
    }
    (sum_dir / f"{name}.json").write_text(json.dumps(summ))
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        required=True,
        help="checkpoint subdir under preprocessing/ (e.g. _pixel_pred_y2022, _pixel_pred)",
    )
    ap.add_argument(
        "--dw-year",
        required=True,
        help="DW year (int) or 'survey' to use each stand's survey_year (clamped >=2016)",
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ckpt_dir = PREP / args.ckpt
    assert ckpt_dir.exists(), f"missing checkpoint dir {ckpt_dir}"
    tag = args.dw_year if args.dw_year != "survey" else "survey"
    out_dir = PREP / f"_dw_mask_y{tag}"
    sum_dir = out_dir / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    ee.Initialize()
    print(f"GEE initialised. ckpt={args.ckpt} dw-year={args.dw_year} -> out {out_dir}", flush=True)

    gdf = gpd.read_file(DISSOLVED)
    sy = gdf.set_index("Location_Name")["survey_year"].astype(int).to_dict()

    names = sorted(p.stem for p in ckpt_dir.glob("*.parquet"))
    if args.limit:
        names = names[: args.limit]
    print(f"{len(names)} stands to process.", flush=True)

    failures = []
    for i, name in enumerate(names):
        summf = sum_dir / f"{name}.json"
        if summf.exists() and (out_dir / f"{name}.parquet").exists():
            continue  # resumable: already done
        dwy = dw_year_for(name, args.dw_year, sy.get(name, DW_MIN_YEAR))
        for attempt in range(3):
            try:
                t0 = time.time()
                s = process_stand(name, ckpt_dir, out_dir, sum_dir, dwy)
                print(
                    f"[{i + 1}/{len(names)}] {name} dwY={dwy} n_pix={s['n_pixels']} "
                    f"ff={s['forest_fraction']:.3f} masked={s['masked_pred_pixel_tCO2_acre']:.1f} "
                    f"unmasked={s['unmasked_pred_pixel_tCO2_acre']:.1f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name}] attempt {attempt + 1} failed: {str(exc)[:160]}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            failures.append(name)
            print(f"  [{name}] GAVE UP after 3 attempts", flush=True)

    done = len(list(sum_dir.glob("*.json")))
    print(f"\nDONE: {done}/{len(names)} stands have DW summaries. failures={failures}", flush=True)
    if failures:
        (out_dir / "_failures.json").write_text(json.dumps(failures))


if __name__ == "__main__":
    main()
