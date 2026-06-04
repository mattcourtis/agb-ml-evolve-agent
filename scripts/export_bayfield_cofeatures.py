"""
Export Bayfield co-feature rasters via GEE, aligned to the 30 m inference grid (grid.json),
EPSG:32615, survey_year=2023. Bands (names match the model's feature columns):

  chm_m, topo_elevation, topo_slope, topo_aspect_cos, topo_aspect_sin, topo_tpi,
  dstx_pre_ysd, dstx_pre_loss_5yr, dstx_loss_frac_buf, dstx_lt_mag

dstx bands are computed wall-to-wall with focal reducers over a ~30 m radius, matching the 30 m
buffer used when the training features were extracted. Each band is downloaded separately
(getDownloadURL size limit) on the exact target crsTransform+dimensions, then stacked into
predictions/bayfield_cofeatures_30m.tif.

Run after `infer_bayfield.py` has written predictions/grid.json:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/export_bayfield_cofeatures.py
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import ee
import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_iter2_features import build_topo_image  # noqa: E402

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
)
PRED_DIR = EXPDIR / "predictions"
GRID_JSON = PRED_DIR / "grid.json"
OUT_TIF = PRED_DIR / "bayfield_cofeatures_30m.tif"

SURVEY_YEAR = 2023
CODE = SURVEY_YEAR - 2000  # Hansen lossyear code, 23
FOCAL_M = 30  # focal radius ~ training buffer

# output band order (must be valid model feature names).
# NB: dstx_lt_mag dropped — LandTrendr was ~97% nodata wall-to-wall over the lake-spanning bbox
# (planned fallback); the deployment model is retrained on the remaining features.
BANDS = [
    "chm_m",
    "topo_elevation",
    "topo_slope",
    "topo_aspect_cos",
    "topo_aspect_sin",
    "topo_tpi",
    "dstx_pre_ysd",
    "dstx_pre_loss_5yr",
    "dstx_loss_frac_buf",
]
# nearest-neighbour bands (stepwise/categorical); the rest use bilinear
NEAREST = {"dstx_pre_ysd", "dstx_pre_loss_5yr"}


def build_cofeature_image() -> ee.Image:
    # --- canopy height (ETH 2020, 10 m) → bilinear so 30 m export is a smooth mean-ish ---
    chm = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").select("b1").rename("chm_m")
    chm = chm.resample("bilinear")

    topo = build_topo_image()  # topo_elevation/slope/aspect_cos/aspect_sin/tpi (SRTM 30 m)

    # --- Hansen survey-relative disturbance, wall-to-wall with focal reducers ---
    ly = ee.Image("UMD/hansen/global_forest_change_2025_v1_13").select("lossyear")
    pre = ly.gt(0).And(ly.lte(CODE))  # pre/at-survey loss
    kern = ee.Kernel.circle(radius=FOCAL_M, units="meters")
    # most-recent pre-survey loss code in the neighbourhood → years-since (100 if none)
    pre_year_max = ly.updateMask(pre).unmask(0).focal_max(kernel=kern)
    has_pre = pre_year_max.gt(0)
    pre_ysd = (
        ee.Image(100).where(has_pre, ee.Image(CODE).subtract(pre_year_max)).rename("dstx_pre_ysd")
    )
    pre_loss_5yr = has_pre.And(ee.Image(CODE).subtract(pre_year_max).lte(5)).rename(
        "dstx_pre_loss_5yr"
    )
    # unmask(0) so no-loss pixels count as 0 in the fraction (else focal_mean averages only the
    # masked-in loss pixels → spuriously ~1 with nodata elsewhere).
    loss_frac = pre.unmask(0).focal_mean(kernel=kern).rename("dstx_loss_frac_buf")

    return (
        chm.addBands(topo)
        .addBands(pre_ysd)
        .addBands(pre_loss_5yr)
        .addBands(loss_frac)
        .select(BANDS)
        .toFloat()
    )


def _grid_bounds_wgs84() -> list[float]:
    """County bbox in WGS84 (pad a touch) for LandTrendr filterBounds."""
    import geopandas as gpd

    g = json.loads(GRID_JSON.read_text())
    poly = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries.from_wkt(
            [
                f"POLYGON(({g['minx']} {g['miny']},{g['maxx']} {g['miny']},"
                f"{g['maxx']} {g['maxy']},{g['minx']} {g['maxy']},{g['minx']} {g['miny']}))"
            ]
        ),
        crs=g["crs"],
    ).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = poly.total_bounds
    return [minx - 0.05, miny - 0.05, maxx + 0.05, maxy + 0.05]


def download_band(img: ee.Image, band: str, g: dict) -> np.ndarray:
    """Download one band georeferenced by GEE, then reproject onto the canonical 30 m grid.

    Critical fix: we keep GEE's OWN georeferencing and reproject to the target transform — the
    previous version stamped the target transform onto GEE's array directly, which misaligned
    spatially-varying bands (chm/topo/loss_frac).
    """
    region = ee.Geometry.Rectangle(_grid_bounds_wgs84())
    url = img.select(band).getDownloadURL(
        {
            "crs": g["crs"],
            "scale": g["res"],
            "region": region,
            "format": "GEO_TIFF",
        }
    )
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    content = r.content
    if content[:2] == b"PK":  # zip-wrapped
        z = zipfile.ZipFile(io.BytesIO(content))
        content = z.read([n for n in z.namelist() if n.endswith(".tif")][0])

    dst = np.full((g["height"], g["width"]), np.nan, dtype=np.float32)
    target_tf = from_origin(g["minx"], g["maxy"], g["res"], g["res"])
    resamp = Resampling.nearest if band in NEAREST else Resampling.bilinear
    with MemoryFile(content) as mf, mf.open() as src:
        src_arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            src_arr[src_arr == src.nodata] = np.nan
        reproject(
            src_arr,
            dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_tf,
            dst_crs=g["crs"],
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resamp,
        )
    return dst


def main() -> None:
    assert GRID_JSON.exists(), "run infer_bayfield.py first (writes grid.json)"
    g = json.loads(GRID_JSON.read_text())
    ee.Initialize()
    print(f"GEE init. Target grid {g['width']}×{g['height']} @ {g['res']} m {g['crs']}")

    img = build_cofeature_image()
    H, W = g["height"], g["width"]
    stack = np.full((len(BANDS), H, W), np.nan, dtype=np.float32)
    for i, b in enumerate(BANDS):
        print(f"  [{i + 1}/{len(BANDS)}] downloading {b} ...")
        arr = download_band(img, b, g)
        if arr.shape != (H, W):
            print(f"    WARN shape {arr.shape} != {(H, W)}; cropping/padding")
            a = np.full((H, W), np.nan, np.float32)
            a[: arr.shape[0], : arr.shape[1]] = arr[:H, :W]
            arr = a
        stack[i] = arr

    transform = rasterio.transform.from_origin(g["minx"], g["maxy"], g["res"], g["res"])
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": len(BANDS),
        "height": H,
        "width": W,
        "crs": g["crs"],
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        dst.write(stack)
        for i, b in enumerate(BANDS):
            dst.set_band_description(i + 1, b)
    print(f"\nWrote {OUT_TIF}  ({len(BANDS)} bands)")
    for i, b in enumerate(BANDS):
        v = stack[i][np.isfinite(stack[i])]
        if v.size:
            print(f"  {b}: min={v.min():.2f} mean={v.mean():.2f} max={v.max():.2f}")


if __name__ == "__main__":
    main()
