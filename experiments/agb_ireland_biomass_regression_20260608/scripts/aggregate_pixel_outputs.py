"""
Aggregate per-pixel checkpoints (preprocessing/_pixel_pred/<Location>.parquet) to stand level
and emit the deliverables: final/ireland_agb_pixel.{csv,parquet,gpkg}, per-stand GeoTIFFs in
final/ireland_pixel_tiffs/<Location>.tif (EPSG:2157, ~10 m, band1 tCO2/acre, band2 Mg/ha,
nodata outside polygon), and a VRT index over band1.

mean(f) = stand AGB density (tCO2/acre). Mg/ha = tCO2/acre / 0.6977. total_t = Mg/ha * area_ha.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PREP = EXPDIR / "preprocessing"
FINAL = EXPDIR / "final"
EVAL = EXPDIR / "evaluation"

CKPT = PREP / "_pixel_pred"
CKPT_SUM = CKPT / "_summary"
TIFDIR = FINAL / "ireland_pixel_tiffs"

DISSOLVED = PREP / "ireland_locations_dissolved.gpkg"
DB_REF = PREP / "db_reference.parquet"
IRELAND_PRED = EVAL / "ireland_predictions.parquet"
SUPPORT = EVAL / "support_sensitivity_stands.parquet"

TCO2ACRE_TO_MGHA = 1.0 / 0.6977
RES = 10.0  # metres, EPSG:2157
NODATA = -9999.0


def build_geotiff(name: str, pix: pd.DataFrame) -> int:
    """Rasterise per-pixel predictions to a 10 m EPSG:2157 GeoTIFF (2 bands).

    Pixel centres (lon/lat, EPSG:4326) are reprojected to EPSG:2157 then snapped to a fixed
    10 m grid (origin = floor to grid). Pixels outside the polygon are nodata. Returns n cells.
    """
    pts = gpd.GeoSeries([Point(xy) for xy in zip(pix.lon, pix.lat)], crs="EPSG:4326").to_crs(
        "EPSG:2157"
    )
    x = np.array([p.x for p in pts])
    y = np.array([p.y for p in pts])
    pred = pix.pred_tco2_acre.to_numpy()

    # snap centres to a 10 m grid
    x0 = np.floor(x.min() / RES) * RES - RES / 2  # left edge so centres land mid-cell
    y1 = np.ceil(y.max() / RES) * RES + RES / 2  # top edge
    col = np.round((x - (x0 + RES / 2)) / RES).astype(int)
    row = np.round(((y1 - RES / 2) - y) / RES).astype(int)
    ncol = int(col.max()) + 1
    nrow = int(row.max()) + 1

    b1 = np.full((nrow, ncol), NODATA, dtype=np.float32)
    b2 = np.full((nrow, ncol), NODATA, dtype=np.float32)
    # last-write-wins on rare grid collisions (negligible: 10 m snap of 10 m native grid)
    b1[row, col] = pred.astype(np.float32)
    b2[row, col] = (pred * TCO2ACRE_TO_MGHA).astype(np.float32)

    transform = from_origin(x0, y1, RES, RES)
    out = TIFDIR / f"{name}.tif"
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        height=nrow,
        width=ncol,
        count=2,
        dtype="float32",
        crs="EPSG:2157",
        transform=transform,
        nodata=NODATA,
        compress="deflate",
    ) as ds:
        ds.write(b1, 1)
        ds.write(b2, 2)
        ds.set_band_description(1, "pred_AGB_tCO2_per_acre")
        ds.set_band_description(2, "pred_AGB_Mg_per_ha")
    return int((b1 != NODATA).sum())


def main() -> None:
    TIFDIR.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(DISSOLVED)
    db = pd.read_parquet(DB_REF).set_index("Location_Name")
    poly_pred = pd.read_parquet(IRELAND_PRED).set_index("Location_Name")["pred_tco2"]

    rows = []
    valid_cells = 0
    for sumf in sorted(CKPT_SUM.glob("*.json")):
        name = sumf.stem
        summ = json.loads(sumf.read_text())
        pix = pd.read_parquet(CKPT / f"{name}.parquet")
        ncells = build_geotiff(name, pix)
        valid_cells += ncells
        rows.append({**summ, "_gtiff_cells": ncells})
        print(f"{name}: n_pix={summ['n_pixels']} gtiff_cells={ncells}", flush=True)

    agg = pd.DataFrame(rows).set_index("Location_Name")

    g = gdf.set_index("Location_Name")
    out = pd.DataFrame(index=agg.index)
    out["Location_Name"] = out.index
    out["area_ha"] = g["area_ha"]
    out["survey_year"] = agg["survey_year"]
    out["n_pixels"] = agg["n_pixels"]
    out["pred_pixel_tCO2_acre"] = agg["pred_pixel_tCO2_acre"]
    out["pred_pixel_Mg_ha"] = agg["pred_pixel_Mg_ha"]
    out["pred_pixel_total_t"] = out["pred_pixel_Mg_ha"] * out["area_ha"]
    out["pred_polygonmean_tCO2_acre"] = poly_pred
    out["gap_tCO2_acre"] = out["pred_pixel_tCO2_acre"] - out["pred_polygonmean_tCO2_acre"]
    out["gap_pct"] = 100.0 * out["gap_tCO2_acre"] / out["pred_polygonmean_tCO2_acre"]
    out["pixel_pred_std"] = agg["pixel_pred_std"]
    out["pixel_pred_min"] = agg["pixel_pred_min"]
    out["pixel_pred_max"] = agg["pixel_pred_max"]
    out["db_2020_24_tCO2_acre"] = db["db_tco2acre_2020_2024_mean"]
    out["db_2020_24_Mg_ha"] = db["db_mgha_2020_2024_mean"]
    out["delta_pixel_vs_db_tCO2_acre"] = out["pred_pixel_tCO2_acre"] - out["db_2020_24_tCO2_acre"]
    out["MainSp"] = g["MainSp"]
    out["age_at_survey"] = g["age_at_survey"]
    out["Hdom"] = g["Hdom"]
    out["YC"] = g["YC"]
    out["pre2017_fallback"] = g["pre2017_fallback"]

    out = out.reset_index(drop=True).sort_values("Location_Name").reset_index(drop=True)
    out.to_csv(FINAL / "ireland_agb_pixel.csv", index=False)
    out.to_parquet(FINAL / "ireland_agb_pixel.parquet", index=False)

    # gpkg: join to polygons, EPSG:2157
    gpoly = gdf[["Location_Name", "geometry"]].to_crs("EPSG:2157")
    gout = gpoly.merge(out, on="Location_Name", how="inner")
    gout = gpd.GeoDataFrame(gout, geometry="geometry", crs="EPSG:2157")
    gout.to_file(FINAL / "ireland_agb_pixel.gpkg", driver="GPKG")

    # VRT over band1 of all tiffs
    tifs = sorted(str(p) for p in TIFDIR.glob("*.tif"))
    listf = TIFDIR / "_tiflist.txt"
    listf.write_text("\n".join(tifs) + "\n")
    import subprocess

    subprocess.run(
        ["gdalbuildvrt", "-b", "1", "-input_file_list", str(listf), str(TIFDIR / "_index.vrt")],
        check=True,
    )

    print(f"\nWrote {len(out)} stands. Total valid raster cells: {valid_cells}")
    print(f"Total mapped area_ha: {out['area_ha'].sum():.1f}")
    print(f"Total pixels (native): {int(out['n_pixels'].sum())}")
    # quick portfolio stats
    port_pixel = out["pred_pixel_tCO2_acre"].mean()
    port_poly = out["pred_polygonmean_tCO2_acre"].mean()
    port_db = out["db_2020_24_tCO2_acre"].mean()
    print(f"Portfolio mean mean(f) = {port_pixel:.2f} tCO2/acre")
    print(f"Portfolio mean f(mean) = {port_poly:.2f} tCO2/acre")
    print(
        f"Portfolio mean DB      = {port_db:.2f} tCO2/acre  ratio mean(f)/DB = {port_pixel / port_db:.2f}x"
    )
    print(
        f"H1 frac pred_pixel>=DB = {(out['pred_pixel_tCO2_acre'] >= out['db_2020_24_tCO2_acre']).mean():.4f}"
    )
    print(
        f"gap_pct median {out['gap_pct'].median():.2f}  range [{out['gap_pct'].min():.1f},{out['gap_pct'].max():.1f}]"
    )


if __name__ == "__main__":
    main()
