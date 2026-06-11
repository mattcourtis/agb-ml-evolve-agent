"""
Iteration 1 — unified NATIVE AEF embedding matrix for the FULL ANEW pool + the 141 Irish Locations.

ONE consistent encoding (guardrail 1 of iter1_analog_selection_design.md):
  native GEE float A00..A63 from GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, NO per-band affine, NO int8
  cast — for BOTH ANEW and Ireland, so cross-project / Ireland-vs-US distances are comparable.

Plot support matches iter0:
  - ANEW: reduceRegions(mean) over a 7.3 m-radius plot footprint buffer (PLOT_RADIUS_M, same as
    fit_aef_affine.py / data_profile), survey-year aligned (parsed from `Date` = 'Mon-YYYY';
    clamped to [2017, 2025], fallback flagged — ANEW is all 2022/2023 so no fallback expected).
  - Ireland: reduceRegions(mean) over the dissolved Location MultiPolygon, survey-year aligned
    (survey_year already clamped to [2017, 2025] in preprocessing).

ROBUST + RESUMABLE: batches of BATCH plots/polys with tileScale; each batch checkpointed to
preprocessing/_pool_batches/{anew,ireland}_batch_*.parquet. Re-running skips finished batches.
Final assembly -> preprocessing/iter1_pool_embeddings.parquet.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/extract_pool_embeddings.py
"""

from __future__ import annotations

from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PREP = EXPDIR / "preprocessing"
BATCHDIR = PREP / "_pool_batches"

ANEW_GPKG = "/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg"
IRELAND_GPKG = PREP / "ireland_locations_dissolved.gpkg"

AEF_ASSET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{i:02d}" for i in range(64)]
EMB = [f"emb_{i:02d}" for i in range(64)]

PLOT_RADIUS_M = 7.3  # ANEW plot footprint (matches iter0 fit_aef_affine.py)
AEF_SCALE = 10
TILE_SCALE = 4
AEF_MIN_YEAR, AEF_MAX_YEAR = 2017, 2025
BATCH = 300  # plots/polys per checkpointed batch


# ---------------------------------------------------------------------------
# GEE helpers
# ---------------------------------------------------------------------------


def aef_image(year: int) -> ee.Image:
    return (
        ee.ImageCollection(AEF_ASSET)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
        .select(BANDS)
    )


def ee_polygon(geom) -> ee.Geometry:
    """Shapely (MultiPolygon, possibly Z, EPSG:4326) -> ee.Geometry; drop Z, explicit coords."""
    from shapely.geometry import mapping

    gj = mapping(geom)

    def drop_z(coords):
        return [[[xy[0], xy[1]] for xy in ring] for ring in coords]

    if gj["type"] == "MultiPolygon":
        polys = [drop_z(poly) for poly in gj["coordinates"]]
        return ee.Geometry.MultiPolygon(polys, proj="EPSG:4326", geodesic=False)
    polys = drop_z(gj["coordinates"])
    return ee.Geometry.Polygon(polys, proj="EPSG:4326", geodesic=False)


def reduce_batch(image: ee.Image, features: list[ee.Feature]) -> list[dict]:
    """reduceRegions(mean) of a 64-band image over a small FeatureCollection."""
    res = image.reduceRegions(
        collection=ee.FeatureCollection(features),
        reducer=ee.Reducer.mean(),
        scale=AEF_SCALE,
        tileScale=TILE_SCALE,
    )
    return res.getInfo()["features"]


# ---------------------------------------------------------------------------
# ANEW
# ---------------------------------------------------------------------------


def parse_anew(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf.reset_index(drop=True).copy()
    g["pool_id"] = g.index.astype(int)  # stable unique key (project,Plot_ID has 3 dups)
    raw_year = pd.to_datetime(g["Date"], format="%b-%Y").dt.year
    g["survey_year_raw"] = raw_year
    g["survey_year"] = raw_year.clip(AEF_MIN_YEAR, AEF_MAX_YEAR)
    g["year_fallback"] = g["survey_year"] != g["survey_year_raw"]
    return g


def extract_anew(g: gpd.GeoDataFrame) -> None:
    """Per survey_year group, batch over 7.3 m footprint buffers; checkpoint each batch."""
    BATCHDIR.mkdir(parents=True, exist_ok=True)
    for year, grp in g.groupby("survey_year"):
        idx = grp.index.to_list()
        for start in range(0, len(idx), BATCH):
            tag = f"anew_y{int(year)}_b{start:05d}"
            ckpt = BATCHDIR / f"{tag}.parquet"
            if ckpt.exists():
                print(f"  [skip] {tag} ({len(pd.read_parquet(ckpt))} rows cached)")
                continue
            chunk = grp.loc[idx[start : start + BATCH]]
            feats = [
                ee.Feature(
                    ee.Geometry.Point([float(r.geometry.x), float(r.geometry.y)]).buffer(
                        PLOT_RADIUS_M
                    ),
                    {"pool_id": int(r.pool_id)},
                )
                for _, r in chunk.iterrows()
            ]
            out_feats = reduce_batch(aef_image(int(year)), feats)
            rows = []
            for f in out_feats:
                p = f["properties"]
                rows.append({"pool_id": int(p["pool_id"]), **{b: p.get(b, np.nan) for b in BANDS}})
            pd.DataFrame(rows).to_parquet(ckpt, index=False)
            print(f"  [done] {tag}: {len(rows)}/{len(chunk)} returned")


# ---------------------------------------------------------------------------
# Ireland
# ---------------------------------------------------------------------------


def extract_ireland(ir: gpd.GeoDataFrame) -> None:
    BATCHDIR.mkdir(parents=True, exist_ok=True)
    for year, grp in ir.groupby("survey_year"):
        idx = grp.index.to_list()
        for start in range(0, len(idx), 25):  # small polygon batches (memory note)
            tag = f"ireland_y{int(year)}_b{start:05d}"
            ckpt = BATCHDIR / f"{tag}.parquet"
            if ckpt.exists():
                print(f"  [skip] {tag}")
                continue
            chunk = grp.loc[idx[start : start + 25]]
            feats = [
                ee.Feature(ee_polygon(r.geometry), {"loc": r.Location_Name})
                for _, r in chunk.iterrows()
            ]
            out_feats = reduce_batch(aef_image(int(year)), feats)
            rows = []
            for f in out_feats:
                p = f["properties"]
                rows.append({"loc": p["loc"], **{b: p.get(b, np.nan) for b in BANDS}})
            pd.DataFrame(rows).to_parquet(ckpt, index=False)
            print(f"  [done] {tag}: {len(rows)}/{len(chunk)} returned")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble(anew: gpd.GeoDataFrame, ir: gpd.GeoDataFrame) -> pd.DataFrame:
    band2emb = dict(zip(BANDS, EMB))

    # ANEW batches
    a_files = sorted(BATCHDIR.glob("anew_y*_b*.parquet"))
    a_raw = pd.concat([pd.read_parquet(f) for f in a_files], ignore_index=True)
    a_raw = a_raw.rename(columns=band2emb).drop_duplicates("pool_id")
    meta = anew[
        [
            "pool_id",
            "project_name",
            "Plot_ID",
            "CO2",
            "survey_year",
            "survey_year_raw",
            "year_fallback",
            "ECO_NAME",
            "BIOME_NAME",
            "ECO_ID",
        ]
    ]
    a = meta.merge(a_raw, on="pool_id", how="left")
    a.insert(0, "source", "anew")
    a = a.rename(columns={"Plot_ID": "plot_id"})
    a["plot_id"] = a["plot_id"].astype(str)  # mixed with Ireland string ids on concat
    a["location_id"] = a["pool_id"].astype(str)

    # Ireland batches
    i_files = sorted(BATCHDIR.glob("ireland_y*_b*.parquet"))
    i_raw = pd.concat([pd.read_parquet(f) for f in i_files], ignore_index=True)
    i_raw = i_raw.rename(columns=band2emb).drop_duplicates("loc")
    imeta = ir[["Location_Name", "survey_year", "survey_year_raw_mode", "pre2017_fallback"]].copy()
    i = imeta.merge(i_raw, left_on="Location_Name", right_on="loc", how="left").drop(columns="loc")
    i.insert(0, "source", "ireland")
    i["project_name"] = "Ireland"
    i["location_id"] = i["Location_Name"]
    i["plot_id"] = i["Location_Name"]
    i["CO2"] = np.nan
    i["ECO_NAME"] = np.nan
    i["BIOME_NAME"] = np.nan
    i["ECO_ID"] = np.nan
    i["year_fallback"] = i["pre2017_fallback"].astype(bool)
    i["survey_year_raw"] = i["survey_year_raw_mode"]

    cols = [
        "source",
        "project_name",
        "location_id",
        "plot_id",
        "CO2",
        "survey_year",
        "survey_year_raw",
        "year_fallback",
        "ECO_NAME",
        "BIOME_NAME",
        "ECO_ID",
    ] + EMB
    out = pd.concat([a[cols], i[cols]], ignore_index=True)
    return out


def main() -> None:
    ee.Initialize()
    print("GEE initialised.")

    anew = parse_anew(gpd.read_file(ANEW_GPKG))
    ir = gpd.read_file(IRELAND_GPKG, layer="locations")
    print(
        f"ANEW plots: {len(anew)} (survey_year {sorted(anew['survey_year'].unique())}, "
        f"fallbacks {int(anew['year_fallback'].sum())})"
    )
    print(f"Ireland Locations: {len(ir)}")

    print("\n[1/2] ANEW native AEF extraction (7.3 m footprint, reduceRegions mean) ...")
    extract_anew(anew)

    print("\n[2/2] Ireland native AEF extraction (polygon mean) ...")
    extract_ireland(ir)

    print("\nAssembling ...")
    out = assemble(anew, ir)
    out_path = PREP / "iter1_pool_embeddings.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}: {len(out)} rows, {len(EMB)} emb cols")

    # quick validation print
    miss = out[EMB].isna().any(axis=1)
    print(f"  rows with any missing emb: {int(miss.sum())}")
    print(f"  source counts: {out['source'].value_counts().to_dict()}")
    g_all = out.loc[~miss, EMB].values
    print(f"  emb value range: [{g_all.min():.4f}, {g_all.max():.4f}]")
    norms = np.linalg.norm(g_all, axis=1)
    print(
        f"  per-vector L2 norm: mean {norms.mean():.4f} min {norms.min():.4f} max {norms.max():.4f}"
    )


if __name__ == "__main__":
    main()
