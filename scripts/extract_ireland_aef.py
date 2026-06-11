"""
Ireland AGB transfer — AEF extraction + disturbance co-features + feature assembly.

PRECONDITION: scripts/fit_aef_affine.py must have PASSED the encoding gate
(preprocessing/encoding_gate.json -> PASS). This script refuses to run otherwise.

For each of the 141 dissolved Locations:
  1. AEF: reduceRegions(mean) of GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (A00..A63) over the polygon
     for that Location's survey_year, then apply the fitted per-band affine -> training codec space
     (emb_00..63).
  2. dstx co-features: survey-relative Hansen timing (dstx_pre_ysd, dstx_pre_loss_5yr,
     dstx_loss_frac_buf) via reduceRegions over the polygon, using build_dist_image-style logic
     parameterised by each Location's survey_year.

Assembles preprocessing/ireland_features.parquet (141 x 67, exact model feature order) +
feature_schema.json + data_version.txt.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/extract_ireland_aef.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import ee
import numpy as np
import pandas as pd

import geopandas as gpd

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PREP = EXPDIR / "preprocessing"

DISSOLVED = PREP / "ireland_locations_dissolved.gpkg"
AFFINE = PREP / "aef_affine.parquet"
GATE = PREP / "encoding_gate.json"
FEATS_JSON = REPO / "models/inference_features_embdstx.json"

AEF_ASSET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"

BANDS = [f"A{i:02d}" for i in range(64)]
EMB = [f"emb_{i:02d}" for i in range(64)]
DSTX = ["dstx_pre_ysd", "dstx_pre_loss_5yr", "dstx_loss_frac_buf"]
NO_DIST_YSD = 100.0  # sentinel (matches training extract_disturbance_timing.py)

AEF_SCALE = 10
HANSEN_SCALE = 30
TILE_SCALE = 4
BATCH = 25  # small polygon batches for GEE stability (memory note)

# Inputs to hash for data_version.txt
INPUT_PATHS = {
    "db_csv": "/home/mattc/data-space/carbonmap-embeddings/dasos-ireland/deepbiomass-model-outputs/"
    "Deep Biomass - Aggregated Data & Portfolio Summary.csv",
    "gpkg": "/home/mattc/data-space/carbonmap-embeddings/boundary-files/dasos_fgl_2025ye.gpkg",
    "train_parquet": "/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/"
    "features.parquet",
    "model": str(REPO / "models/inference_model_embdstx.txt"),
    "features_json": str(FEATS_JSON),
}


def assert_gate_passed() -> dict:
    assert GATE.exists(), f"encoding gate result missing: {GATE} (run fit_aef_affine.py first)"
    g = json.loads(GATE.read_text())
    assert g.get("PASS"), "ENCODING GATE did not PASS — refusing to extract. See encoding_gate.json"
    print(f"Encoding gate PASS (mean corr {g['mean_corr_transformed']:.3f}).")
    return g


def ee_polygon(geom) -> ee.Geometry:
    """Shapely (MultiPolygon Z, EPSG:4326) -> ee.Geometry. Drop Z; build coord lists explicitly."""
    from shapely.geometry import mapping

    gj = mapping(geom)

    def drop_z(coords):
        return [[[xy[0], xy[1]] for xy in ring] for ring in coords]

    if gj["type"] == "MultiPolygon":
        polys = [drop_z(poly) for poly in gj["coordinates"]]
        return ee.Geometry.MultiPolygon(polys, proj="EPSG:4326", geodesic=False)
    polys = drop_z(gj["coordinates"])
    return ee.Geometry.Polygon(polys, proj="EPSG:4326", geodesic=False)


def aef_image(year: int) -> ee.Image:
    return (
        ee.ImageCollection(AEF_ASSET)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
        .select(BANDS)
    )


def build_dstx_image(survey_year: int) -> ee.Image:
    """Survey-relative Hansen timing bands, mean-reduced over the polygon.

    lossyear: 0=no loss, 1..25 = 2001..2025. Pre/at-survey loss = (0 < ly <= code).
      pre_year : ly masked to pre-survey (else 0)  -> max over polygon = most-recent pre-survey code
      pre_frac : pre-survey loss indicator         -> mean = disturbed area fraction in polygon
    dstx features derived in pandas from these (matches extract_disturbance_timing.derive_features).
    """
    ly = ee.Image(HANSEN_ASSET).select("lossyear")
    code = survey_year - 2000
    pre = ly.gt(0).And(ly.lte(code))
    pre_year = ly.updateMask(pre).unmask(0).rename("pre_year")  # max -> most-recent pre code
    pre_frac = pre.rename("pre_frac")  # mean -> fraction disturbed
    return pre_year.addBands(pre_frac)


def reduce_locations(
    gdf: gpd.GeoDataFrame, image_fn, props: list[str], reducer: ee.Reducer, scale: int, label: str
) -> pd.DataFrame:
    """reduceRegions(mean/combined) per survey_year group, in small polygon batches."""
    out_rows: list[dict] = []
    for year, grp in gdf.groupby("survey_year"):
        img = image_fn(int(year))
        idx = grp.index.to_list()
        for start in range(0, len(idx), BATCH):
            chunk = grp.loc[idx[start : start + BATCH]]
            feats = [
                ee.Feature(ee_polygon(r.geometry), {"loc": r.Location_Name})
                for _, r in chunk.iterrows()
            ]
            res = img.reduceRegions(
                collection=ee.FeatureCollection(feats),
                reducer=reducer,
                scale=scale,
                tileScale=TILE_SCALE,
            )
            for f in res.getInfo()["features"]:
                p = f["properties"]
                out_rows.append(
                    {"Location_Name": p.get("loc"), **{k: p.get(k, np.nan) for k in props}}
                )
            print(f"  [{label}] year {int(year)} batch {start}-{start + len(chunk) - 1}")
    return pd.DataFrame(out_rows)


def extract_aef(gdf: gpd.GeoDataFrame, affine: pd.DataFrame) -> pd.DataFrame:
    raw = reduce_locations(gdf, aef_image, BANDS, ee.Reducer.mean(), AEF_SCALE, "AEF")
    a = affine.set_index("band")["a"]
    c = affine.set_index("band")["c"]
    out = {"Location_Name": raw["Location_Name"]}
    for b, e in zip(BANDS, EMB):
        out[e] = raw[b].astype(float) * a[b] + c[b]  # affine -> training codec
    return pd.DataFrame(out)


def extract_dstx(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)
    raw = reduce_locations(
        gdf,
        build_dstx_image,
        ["pre_year_mean", "pre_year_max", "pre_frac_mean", "pre_frac_max"],
        reducer,
        HANSEN_SCALE,
        "dstx",
    )
    # reducer emits pre_year_mean/pre_year_max/pre_frac_mean/pre_frac_max
    sy = gdf.set_index("Location_Name")["survey_year"]
    raw = raw.set_index("Location_Name")
    pre_year_max = raw["pre_year_max"].astype(float)
    pre_frac_mean = raw["pre_frac_mean"].astype(float)
    has_pre = pre_year_max.fillna(0) > 0
    pre_cal_year = np.where(has_pre, 2000 + pre_year_max.fillna(0), np.nan)
    year = sy.reindex(raw.index).astype(float)
    df = pd.DataFrame(index=raw.index)
    df["dstx_pre_ysd"] = np.where(has_pre, year - pre_cal_year, NO_DIST_YSD)
    df["dstx_pre_loss_5yr"] = (has_pre & ((year - pre_cal_year) <= 5)).astype(int)
    df["dstx_loss_frac_buf"] = pre_frac_mean.fillna(0.0)
    return df.reset_index()


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def main() -> None:
    g = assert_gate_passed()
    ee.Initialize()
    print("GEE initialised.")

    gdf = gpd.read_file(DISSOLVED, layer="locations")
    affine = pd.read_parquet(AFFINE)
    feats = json.loads(FEATS_JSON.read_text())["features"]
    assert feats == EMB + DSTX, "model feature order mismatch"
    print(f"Locations: {len(gdf)}; survey_year groups: {sorted(gdf['survey_year'].unique())}")

    print("\n[1/2] AEF extraction (polygon mean -> affine) ...")
    aef_cache = PREP / "ireland_aef_raw.parquet"
    if aef_cache.exists():
        emb_df = pd.read_parquet(aef_cache)
        print(f"  loaded cached AEF: {len(emb_df)} rows")
    else:
        emb_df = extract_aef(gdf, affine)
        emb_df.to_parquet(aef_cache, index=False)
    print(f"  AEF rows: {len(emb_df)}; emb NaN rows: {emb_df[EMB].isna().any(axis=1).sum()}")

    print("\n[2/2] Disturbance co-feature extraction ...")
    dstx_df = extract_dstx(gdf)
    print(f"  dstx rows: {len(dstx_df)}")

    merged = emb_df.merge(dstx_df, on="Location_Name", how="inner")
    assert len(merged) == 141, f"expected 141 assembled rows, got {len(merged)}"
    # exact model feature order, keyed by Location
    out = merged[["Location_Name"] + feats].copy()
    n_complete = int(out[feats].notna().all(axis=1).sum())
    print(f"\nAssembled {len(out)} Locations; {n_complete}/141 with complete 67-feature vector.")

    out_path = PREP / "ireland_features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")

    # ---- feature_schema.json ----
    schema = {
        "key": "Location_Name",
        "n_locations": int(len(out)),
        "n_features": len(feats),
        "feature_order": feats,
        "columns": {
            "Location_Name": {"dtype": "string", "provenance": "DB Location Name (crosswalk key)"},
            **{
                e: {
                    "dtype": "float64",
                    "provenance": f"GEE {AEF_ASSET} band {b}, reduceRegions(mean) over "
                    "dissolved Location polygon at survey_year, per-band affine -> training codec",
                    "affine_applied": True,
                }
                for b, e in zip(BANDS, EMB)
            },
            **{
                d: {
                    "dtype": "float64",
                    "provenance": f"Hansen {HANSEN_ASSET} survey-relative timing, "
                    "reduceRegions over polygon",
                    "affine_applied": False,
                }
                for d in DSTX
            },
        },
        "target": "CO2 standing stock, tCO2/acre (model output; not present here)",
    }
    (PREP / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    print(f"Wrote {PREP / 'feature_schema.json'}")

    # ---- data_version.txt ----
    lines = [
        f"extraction_date: {date.today().isoformat()}",
        f"git_commit: {__import__('subprocess').check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD']).decode().strip()}",
        f"aef_asset: {AEF_ASSET}",
        f"hansen_asset: {HANSEN_ASSET}",
        f"encoding_gate_PASS: {g['PASS']} (mean_corr {g['mean_corr_transformed']:.4f}, "
        f"slope_median {g['slope_median']:.4f})",
        "",
        "inputs (sha256[:16] | bytes | path):",
    ]
    for name, p in INPUT_PATHS.items():
        try:
            lines.append(f"  {name}: {file_hash(p)} | {Path(p).stat().st_size} | {p}")
        except OSError as exc:
            lines.append(f"  {name}: ERROR {exc} | {p}")
    (PREP / "data_version.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote {PREP / 'data_version.txt'}")

    print("\nFeature summary (emb mean range, dstx):")
    print(
        f"  emb mean per-Location: {out[EMB].mean(axis=1).describe()[['min', 'mean', 'max']].to_dict()}"
    )
    print(f"  dstx_pre_ysd: {out['dstx_pre_ysd'].describe()[['min', 'mean', 'max']].to_dict()}")
    print(f"  dstx_pre_loss_5yr sum: {out['dstx_pre_loss_5yr'].sum()}")
    print(f"  dstx_loss_frac_buf mean: {out['dstx_loss_frac_buf'].mean():.4f}")


if __name__ == "__main__":
    main()
