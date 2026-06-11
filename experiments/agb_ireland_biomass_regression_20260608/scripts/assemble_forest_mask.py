"""
Forest-Mask ASSEMBLY (continuation): assemble DW-masked per-stand checkpoints into:
  - NEW masked columns merged into final/ireland_agb_pixel.{parquet,csv,gpkg}  (survey-year set)
    and final/ireland_agb_yearmatched.{parquet,csv,gpkg}  (2022/2023/2024 + 3yr mean)
    -- existing unmasked columns are left byte-for-byte unchanged.
  - masked per-stand GeoTIFFs (band1 tCO2/acre, band2 Mg/ha; non-forest pixels = 0, outside = nodata):
    final/ireland_pixel_tiffs_masked/  (survey-year)
    final/ireland_pixel_tiffs_{2022,2023,2024}_masked/  + per-dir band1 VRT.

NO GEE. Works only from the completed checkpoints. Mg/ha = tCO2/acre / 0.6977.
"""

from __future__ import annotations

import json
import subprocess
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
DISSOLVED = PREP / "ireland_locations_dissolved.gpkg"

TCO2ACRE_TO_MGHA = 1.0 / 0.6977
RES = 10.0
NODATA = -9999.0
YEARS = [2022, 2023, 2024]

MASK_DIRS = {
    "survey": PREP / "_dw_mask_ysurvey",
    2022: PREP / "_dw_mask_y2022",
    2023: PREP / "_dw_mask_y2023",
    2024: PREP / "_dw_mask_y2024",
}


def load_summaries(maskdir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((maskdir / "_summary").glob("*.json")):
        s = json.loads(f.read_text())
        rows.append(s)
    df = pd.DataFrame(rows)
    assert len(df) == 141, f"{maskdir}: expected 141, got {len(df)}"
    return df


def build_masked_geotiff(name: str, pix: pd.DataFrame, tifdir: Path) -> tuple[int, int]:
    """Masked raster: pred_masked_tco2_acre -> band1, /0.6977 -> band2.
    Non-forest pixels are already 0 in pred_masked_tco2_acre (keep as 0, NOT nodata).
    Cells outside the polygon = nodata. Returns (n_valid_cells, n_zero_cells)."""
    pts = gpd.GeoSeries([Point(xy) for xy in zip(pix.lon, pix.lat)], crs="EPSG:4326").to_crs(
        "EPSG:2157"
    )
    x = np.array([p.x for p in pts])
    y = np.array([p.y for p in pts])
    pred = pix.pred_masked_tco2_acre.to_numpy()

    x0 = np.floor(x.min() / RES) * RES - RES / 2
    y1 = np.ceil(y.max() / RES) * RES + RES / 2
    col = np.round((x - (x0 + RES / 2)) / RES).astype(int)
    row = np.round(((y1 - RES / 2) - y) / RES).astype(int)
    ncol = int(col.max()) + 1
    nrow = int(row.max()) + 1

    b1 = np.full((nrow, ncol), NODATA, dtype=np.float32)
    b2 = np.full((nrow, ncol), NODATA, dtype=np.float32)
    b1[row, col] = pred.astype(np.float32)
    b2[row, col] = (pred * TCO2ACRE_TO_MGHA).astype(np.float32)

    transform = from_origin(x0, y1, RES, RES)
    out = tifdir / f"{name}.tif"
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
    n_valid = int((b1 != NODATA).sum())
    n_zero = int((b1 == 0.0).sum())
    return n_valid, n_zero


def build_vrt(tifdir: Path) -> None:
    tifs = sorted(str(p) for p in tifdir.glob("*.tif"))
    if not tifs:
        return
    listf = tifdir / "_tiflist.txt"
    listf.write_text("\n".join(tifs) + "\n")
    subprocess.run(
        ["gdalbuildvrt", "-b", "1", "-input_file_list", str(listf), str(tifdir / "_index.vrt")],
        check=True,
        capture_output=True,
    )


def make_tiffs(maskdir: Path, tifdir: Path) -> None:
    tifdir.mkdir(parents=True, exist_ok=True)
    for f in sorted(maskdir.glob("*.parquet")):
        pix = pd.read_parquet(f)
        # validation: every non-forest pixel must be masked to exactly 0.0
        nf = ~pix.forest.to_numpy()
        assert (pix.pred_masked_tco2_acre.to_numpy()[nf] == 0.0).all(), (
            f"{f.name}: some non-forest pixels are not 0"
        )
        build_masked_geotiff(f.stem, pix, tifdir)
    build_vrt(tifdir)


def main() -> None:
    gdf = gpd.read_file(DISSOLVED)
    area = gdf.set_index("Location_Name")["area_ha"]

    # ------------------------------------------------------------------ #
    # 1. SURVEY-YEAR set: masked stand values from _dw_mask_ysurvey
    # ------------------------------------------------------------------ #
    sv = load_summaries(MASK_DIRS["survey"]).set_index("Location_Name")
    sv_cols = pd.DataFrame(index=sv.index)
    sv_cols["forest_fraction"] = sv["forest_fraction"]
    sv_cols["pred_pixel_masked_tCO2_acre"] = sv["masked_pred_pixel_tCO2_acre"]
    sv_cols["pred_pixel_masked_Mg_ha"] = sv["masked_pred_pixel_tCO2_acre"] * TCO2ACRE_TO_MGHA
    sv_cols["pred_pixel_masked_total_t"] = sv_cols["pred_pixel_masked_Mg_ha"] * area.reindex(
        sv.index
    )

    # re-aggregation identity: masked = forest_fraction * mean(forest-pixel preds)
    for name in sv.index:
        pix = pd.read_parquet(MASK_DIRS["survey"] / f"{name}.parquet")
        ff = pix.forest.mean()
        fmean = pix.loc[pix.forest, "pred_masked_tco2_acre"].mean() if pix.forest.any() else 0.0
        ident = ff * fmean
        got = float(sv.loc[name, "masked_pred_pixel_tCO2_acre"])
        assert abs(ident - got) < 1e-6, f"{name}: identity {ident} != {got}"

    # merge into ireland_agb_pixel.* (ADD columns only, never alter unmasked)
    pix_pq = FINAL / "ireland_agb_pixel.parquet"
    before = pd.read_parquet(pix_pq)
    before_cols = list(before.columns)
    merged = before.merge(
        sv_cols.reset_index(), on="Location_Name", how="left", validate="one_to_one"
    )
    # verify unmasked columns byte-for-byte unchanged
    for c in before_cols:
        assert merged[c].equals(before[c]), f"pixel: column {c} changed!"
    merged.to_parquet(pix_pq, index=False)
    merged.to_csv(FINAL / "ireland_agb_pixel.csv", index=False)
    gpoly = gdf[["Location_Name", "geometry"]].to_crs("EPSG:2157")
    gpd.GeoDataFrame(
        gpoly.merge(merged, on="Location_Name", how="inner"), geometry="geometry", crs="EPSG:2157"
    ).to_file(FINAL / "ireland_agb_pixel.gpkg", driver="GPKG")

    # ------------------------------------------------------------------ #
    # 2. YEAR-MATCHED set: masked per-year + 3yr mean
    # ------------------------------------------------------------------ #
    ym_cols = pd.DataFrame(index=sv.index)  # placeholder index, rebuilt below
    per_year = {}
    for y in YEARS:
        s = load_summaries(MASK_DIRS[y]).set_index("Location_Name")
        per_year[y] = s
    idx = per_year[2022].index
    ym_cols = pd.DataFrame(index=idx)
    for y in YEARS:
        s = per_year[y].reindex(idx)
        ym_cols[f"our_{y}_masked_tCO2_acre"] = s["masked_pred_pixel_tCO2_acre"]
        ym_cols[f"our_{y}_masked_Mg_ha"] = s["masked_pred_pixel_tCO2_acre"] * TCO2ACRE_TO_MGHA
        ym_cols[f"forest_frac_{y}"] = s["forest_fraction"]
    ym_cols["our_mean_2022_24_masked_tCO2_acre"] = ym_cols[
        [f"our_{y}_masked_tCO2_acre" for y in YEARS]
    ].mean(axis=1)
    ym_cols["our_mean_2022_24_masked_Mg_ha"] = ym_cols[
        [f"our_{y}_masked_Mg_ha" for y in YEARS]
    ].mean(axis=1)

    ym_pq = FINAL / "ireland_agb_yearmatched.parquet"
    before_ym = pd.read_parquet(ym_pq)
    before_ym_cols = list(before_ym.columns)
    merged_ym = before_ym.merge(
        ym_cols.reset_index().rename(columns={"index": "Location_Name"}),
        on="Location_Name",
        how="left",
        validate="one_to_one",
    )
    for c in before_ym_cols:
        assert merged_ym[c].equals(before_ym[c]), f"yearmatched: column {c} changed!"
    merged_ym.to_parquet(ym_pq, index=False)
    merged_ym.to_csv(FINAL / "ireland_agb_yearmatched.csv", index=False)
    gpd.GeoDataFrame(
        gpoly.merge(merged_ym, on="Location_Name", how="inner"),
        geometry="geometry",
        crs="EPSG:2157",
    ).to_file(FINAL / "ireland_agb_yearmatched.gpkg", driver="GPKG")

    # ------------------------------------------------------------------ #
    # 3. Masked GeoTIFFs
    # ------------------------------------------------------------------ #
    make_tiffs(MASK_DIRS["survey"], FINAL / "ireland_pixel_tiffs_masked")
    for y in YEARS:
        make_tiffs(MASK_DIRS[y], FINAL / f"ireland_pixel_tiffs_{y}_masked")

    print("Assembly complete.")
    print("pixel cols added:", [c for c in merged.columns if c not in before_cols])
    print("yearmatched cols added:", [c for c in merged_ym.columns if c not in before_ym_cols])


if __name__ == "__main__":
    main()
