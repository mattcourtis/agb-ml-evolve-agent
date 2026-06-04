"""
Apply a non-forest mask to a Bayfield AGB map: pixels that aren't forest are set to 0 tCO₂/acre.

Forest is defined from Dynamic World 2023 (growing-season median `trees` probability ≥ 0.5),
exported aligned to the 30 m inference grid. This directly addresses the low-end floor where it is
most wrong — fields, clearings, scrub, recent clearcuts (DW reads them non-tree → zeroed).

Default base map = the full 73-feature map; pass --base to mask a different one.
Writes <base>_forestmasked.tif (+ quicklook + the forest-probability raster for inspection).

    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/apply_forest_mask.py
    uv run ... python scripts/apply_forest_mask.py --base bayfield_agb_embonly_30m.tif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ee
import io
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_bayfield_cofeatures import _grid_bounds_wgs84  # noqa: E402
from infer_bayfield import PRED_DIR, RES, UTM  # noqa: E402

GRID_JSON = PRED_DIR / "grid.json"
DW_PROB_TIF = PRED_DIR / "bayfield_dw_trees_prob_30m.tif"
FOREST_THRESH = 0.5
NODATA = -9999.0


def _download_region(img: ee.Image, g: dict, region_ll: list[float], dst: np.ndarray) -> None:
    """Download img over a WGS84 sub-region and reproject into the full target grid `dst`."""
    region = ee.Geometry.Rectangle(region_ll)
    url = img.getDownloadURL(
        {"crs": g["crs"], "scale": g["res"], "region": region, "format": "GEO_TIFF"}
    )
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    content = r.content
    if content[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(content))
        content = z.read([n for n in z.namelist() if n.endswith(".tif")][0])
    tmp = np.full_like(dst, np.nan)
    target_tf = from_origin(g["minx"], g["maxy"], g["res"], g["res"])
    with MemoryFile(content) as mf, mf.open() as src:
        a = src.read(1).astype(np.float32)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        reproject(
            a,
            tmp,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_tf,
            dst_crs=g["crs"],
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    fill = np.isfinite(tmp)
    dst[fill] = tmp[fill]


def export_trees_prob(g: dict) -> np.ndarray:
    """DW 2023 growing-season median trees probability on the target grid (tiled to dodge the
    50 MB getDownloadURL cap — DW's 10 m native makes the full county too big in one request)."""
    img = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate("2023-06-15", "2023-09-15")
        .select("trees")
        .median()
        .rename("trees")
    )
    minx, miny, maxx, maxy = _grid_bounds_wgs84()
    midx, midy = (minx + maxx) / 2, (miny + maxy) / 2
    out = np.full((g["height"], g["width"]), np.nan, dtype=np.float32)
    quads = [
        [minx, miny, midx, midy],
        [midx, miny, maxx, midy],
        [minx, midy, midx, maxy],
        [midx, midy, maxx, maxy],
    ]
    for i, q in enumerate(quads):
        print(f"  DW trees quadrant {i + 1}/4 ...")
        _download_region(img, g, q, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base", default="bayfield_agb_30m.tif", help="base AGB map filename in predictions/"
    )
    ap.add_argument("--refresh-dw", action="store_true", help="re-export DW even if cached")
    args = ap.parse_args()
    base_path = PRED_DIR / args.base
    assert base_path.exists(), f"base map not found: {base_path}"
    out_path = PRED_DIR / base_path.name.replace(".tif", "_forestmasked.tif")
    out_png = out_path.with_suffix(".png")

    g = json.loads(GRID_JSON.read_text())
    print(f"base map: {base_path.name}")

    if DW_PROB_TIF.exists() and not args.refresh_dw:
        # reuse the cached DW mask so every masked map uses the identical forest definition
        with rasterio.open(DW_PROB_TIF) as ds:
            trees = ds.read(1).astype(np.float32)
        trees = np.where(trees == NODATA, np.nan, trees)
        print(f"reusing cached DW mask {DW_PROB_TIF.name}")
    else:
        ee.Initialize()
        print("GEE init; exporting DW trees probability ...")
        trees = export_trees_prob(g)
        with rasterio.open(
            DW_PROB_TIF,
            "w",
            driver="GTiff",
            dtype="float32",
            count=1,
            height=g["height"],
            width=g["width"],
            crs=UTM,
            transform=from_origin(g["minx"], g["maxy"], RES, RES),
            nodata=NODATA,
            compress="deflate",
            tiled=True,
        ) as dst:
            dst.write(np.where(np.isfinite(trees), trees, NODATA).astype(np.float32), 1)
            dst.set_band_description(1, "dw_trees_prob_2023")
        print(f"Wrote {DW_PROB_TIF}")

    with rasterio.open(base_path) as ds:
        agb = ds.read(1)
        profile = ds.profile
    base_valid = agb != NODATA

    forest = np.isfinite(trees) & (trees >= FOREST_THRESH)
    masked = agb.copy().astype(np.float32)
    # within the (county-clipped) base map, non-forest → 0; keep base nodata as nodata
    nonforest = base_valid & ~forest
    masked[nonforest] = 0.0

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(masked, 1)
        dst.set_band_description(1, "predicted_AGB_tCO2_per_acre_forestmasked")
    print(f"Wrote {out_path}")

    # stats
    n_base = int(base_valid.sum())
    n_zeroed = int(nonforest.sum())
    print(f"\nforest threshold: DW trees prob ≥ {FOREST_THRESH}")
    print(
        f"  county pixels: {n_base}; zeroed as non-forest: {n_zeroed} ({100 * n_zeroed / n_base:.1f}%)"
    )
    fv = agb[base_valid & forest]
    print(
        f"  forest AGB (kept): min {fv.min():.1f} mean {fv.mean():.1f} median {np.median(fv):.1f}"
    )
    allv = masked[base_valid]
    print(
        f"  masked map overall: min {allv.min():.1f} mean {allv.mean():.1f} "
        f"median {np.median(allv):.1f}; %<30 = {100 * (allv < 30).mean():.1f}"
    )

    # quicklook
    show = np.where(base_valid, masked, np.nan)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(show, cmap="viridis", vmin=0, vmax=float(np.nanpercentile(show, 99)))
    ax.set_title(
        f"Bayfield AGB (tCO₂/acre), 30 m — {base_path.stem}\nnon-forest masked (DW trees<{FOREST_THRESH} → 0)"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, label="tCO₂/acre")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
