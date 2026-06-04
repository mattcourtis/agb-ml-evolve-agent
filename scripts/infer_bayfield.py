"""
Wall-to-wall AGB inference over Bayfield County (QGIS), 30 m, EPSG:32615.

Two-stage orchestrator (co-features come from a separate GEE export):
  Stage 1 (default): define the 30 m target grid, build the embedding stack from the local int8
    tiles (NO dequantization — matches training), cache it, and run a correctness gate against a
    known Bayfield training plot. Writes predictions/grid.json + predictions/bayfield_emb_30m.npy.
    Then, if the co-feature GeoTIFF is missing, it stops and tells you to run
    export_bayfield_cofeatures.py.
  Stage 2 (auto, once co-features exist): stack emb + co-features → (H*W, 74), predict with the
    deployment Booster, mask to the county polygon, write the float32 AGB GeoTIFF + quicklook.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/infer_bayfield.py
    # then: uv run ... python scripts/export_bayfield_cofeatures.py
    # then: uv run ... python scripts/infer_bayfield.py    # finishes + writes the map
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rfeatures
from rasterio.transform import from_origin
from rasterio.windows import from_bounds as window_from_bounds

import geopandas as gpd

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_usa_biomass_regression_20260529"
BOUNDARY = Path("/home/mattc/data-space/carbonmap-embeddings/boundary-files/BayfieldCounty.geojson")
EMB_DIR = Path(
    "/home/mattc/data-space/carbonmap-embeddings/agb_usa_pilot_midwest/embeddings_annual/2023"
)
TRAIN_PARQUET = (
    "/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet"
)

PRED_DIR = EXPDIR / "predictions"
GRID_JSON = PRED_DIR / "grid.json"
EMB_NPY = PRED_DIR / "bayfield_emb_30m.npy"
EMB_VRT = PRED_DIR / "bayfield_emb.vrt"
COFEAT_TIF = PRED_DIR / "bayfield_cofeatures_30m.tif"
OUT_TIF = PRED_DIR / "bayfield_agb_30m.tif"
OUT_PNG = PRED_DIR / "bayfield_agb_30m_quicklook.png"

MODEL_TXT = REPO / "models/inference_model.txt"
FEATS_JSON = REPO / "models/inference_features.json"

UTM = "EPSG:32615"
RES = 30
EMB_NODATA_INT8 = -128


# ---------------------------------------------------------------------------
# Target grid (snapped to 30 m, aligned to the 10 m tile grid)
# ---------------------------------------------------------------------------


def target_grid() -> dict:
    gdf = gpd.read_file(BOUNDARY).to_crs(UTM)
    minx, miny, maxx, maxy = gdf.total_bounds
    # snap to RES multiples (tile origin 663840 is a multiple of 30 → 30 m and 10 m grids align)
    minx = np.floor(minx / RES) * RES
    miny = np.floor(miny / RES) * RES
    maxx = np.ceil(maxx / RES) * RES
    maxy = np.ceil(maxy / RES) * RES
    width = int(round((maxx - minx) / RES))
    height = int(round((maxy - miny) / RES))
    transform = from_origin(minx, maxy, RES, RES)
    return {
        "crs": UTM,
        "res": RES,
        "minx": minx,
        "miny": miny,
        "maxx": maxx,
        "maxy": maxy,
        "width": width,
        "height": height,
        "transform": list(transform)[:6],
        "gdf": gdf,
    }


def save_grid(grid: dict) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in grid.items() if k != "gdf"}
    GRID_JSON.write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# Embeddings — local int8 tiles -> 30 m float32 (NO dequant)
# ---------------------------------------------------------------------------


def build_vrt() -> Path:
    if not EMB_VRT.exists():
        tiles = sorted(str(p) for p in EMB_DIR.glob("*tile_*.tif"))
        assert tiles, f"no embedding tiles in {EMB_DIR}"
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        print(f"building VRT over {len(tiles)} tiles ...")
        subprocess.run(["gdalbuildvrt", "-q", str(EMB_VRT), *tiles], check=True)
    return EMB_VRT


STRIP_ROWS = 300  # 30 m rows per strip (×3 = 900 native rows) — bounds memory


def read_embeddings(grid: dict) -> np.ndarray:
    """Return (64, H, W) float32 on the 30 m grid; raw int8 cast to float (NO dequant), NaN nodata.

    Read in row-strips (the county bbox is ~4,900 km², so a single native-10 m read would be
    >12 GB). int8 is stored as uint8 in the VRT → reinterpret >127 → −256 BEFORE averaging,
    then 3×3 block-mean to 30 m.
    """
    vrt = build_vrt()
    H, W = grid["height"], grid["width"]
    emb30 = np.full((64, H, W), np.nan, dtype=np.float32)
    with rasterio.open(vrt) as ds:
        print(f"  VRT dtype={ds.dtypes[0]} bands={ds.count} crs={ds.crs}", flush=True)
        full = (
            window_from_bounds(grid["minx"], grid["miny"], grid["maxx"], grid["maxy"], ds.transform)
            .round_offsets()
            .round_lengths()
        )
        col_off, row_off = int(full.col_off), int(full.row_off)
        for r0 in range(0, H, STRIP_ROWS):
            h = min(STRIP_ROWS, H - r0)
            win = rasterio.windows.Window(col_off, row_off + r0 * 3, W * 3, h * 3)
            a = ds.read(window=win, boundless=True, fill_value=128)  # uint8 128 = int8 nodata −128
            a = a.astype(np.int16)
            a[a > 127] -= 256
            a = a.astype(np.float32)
            a[a == EMB_NODATA_INT8] = np.nan
            a = a.reshape(64, h, 3, W, 3)
            emb30[:, r0 : r0 + h, :] = np.nanmean(a, axis=(2, 4))
            print(f"    strip rows {r0}-{r0 + h - 1}/{H}", flush=True)
    _mask_nodata(emb30)
    print(f"  embeddings → {emb30.shape}, data frac={np.isfinite(emb30[0]).mean():.2f}", flush=True)
    return emb30


def _mask_nodata(emb30: np.ndarray) -> None:
    """All-band-zero pixels are VRT gaps (no embedding tile here) → set to NaN in place."""
    gap = (emb30 == 0).all(axis=0)
    emb30[:, gap] = np.nan


def correctness_gate(grid: dict, emb30: np.ndarray) -> None:
    """Compare sampled embeddings vs the training parquet at Bayfield plots (validates int8 cast)."""
    df = pd.read_parquet(TRAIN_PARQUET)
    bay = df[df["project_name"] == "BayfieldCounty"].copy()
    if bay.empty:
        print("  [gate] no BayfieldCounty plots in parquet — skipping")
        return
    emb_cols = [f"emb_{i:02d}" for i in range(64)]
    pts = gpd.GeoDataFrame(
        bay, geometry=gpd.points_from_xy(bay["lon"], bay["lat"]), crs="EPSG:4326"
    ).to_crs(UTM)
    corrs = []
    for _, r in pts.head(15).iterrows():
        col = int((r.geometry.x - grid["minx"]) / RES)
        row = int((grid["maxy"] - r.geometry.y) / RES)
        if not (0 <= row < grid["height"] and 0 <= col < grid["width"]):
            continue
        sampled = emb30[:, row, col]
        ref = r[emb_cols].to_numpy(dtype=float)
        if np.isfinite(sampled).all() and sampled.std() > 0:  # skip no-data (sliver) pixels
            corrs.append(np.corrcoef(sampled, ref)[0, 1])
    if corrs:
        print(
            f"  [gate] {len(corrs)} valid plots; emb corr(sampled,parquet) "
            f"mean={np.nanmean(corrs):.3f} min={np.nanmin(corrs):.3f}"
        )
        assert np.nanmean(corrs) > 0.8, "embedding transform mismatch — ABORT (check int8/dequant)"
        print("  [gate] PASS — embedding encoding matches training.")
    else:
        print("  [gate] no in-bounds data samples to check")


# ---------------------------------------------------------------------------
# Predict stage
# ---------------------------------------------------------------------------


def predict_and_write(grid: dict, emb30: np.ndarray) -> None:
    feats = json.loads(FEATS_JSON.read_text())["features"]
    booster = lgb.Booster(model_file=str(MODEL_TXT))
    H, W = grid["height"], grid["width"]

    with rasterio.open(COFEAT_TIF) as ds:
        co = ds.read().astype(np.float32)  # (n_co, H, W)
        co_names = list(ds.descriptions)
        assert ds.width == W and ds.height == H, (
            f"cofeature grid {ds.width}x{ds.height} != target {W}x{H}"
        )
    print(f"  cofeatures: {co_names}")

    # assemble (H*W, 74) in the exact training feature order
    layers = {f"emb_{i:02d}": emb30[i] for i in range(64)}
    for i, name in enumerate(co_names):
        layers[name] = co[i]
    missing = [f for f in feats if f not in layers]
    assert not missing, f"missing layers for features: {missing}"

    stack = np.stack([layers[f] for f in feats], axis=0)  # (74, H, W)
    valid = np.isfinite(stack).all(axis=0)  # pixels with all features present
    X = stack.reshape(len(feats), -1).T
    finite_rows = np.isfinite(X).all(axis=1)
    pred = np.full(X.shape[0], np.nan, dtype=np.float32)
    pred[finite_rows] = booster.predict(X[finite_rows]).astype(np.float32)
    pred_img = pred.reshape(H, W)

    # mask to county polygon
    tr = rasterio.transform.from_origin(grid["minx"], grid["maxy"], RES, RES)
    poly_mask = rfeatures.geometry_mask(
        grid["gdf"].geometry, out_shape=(H, W), transform=tr, invert=True
    )
    pred_img[~poly_mask] = np.nan
    pred_img[~valid] = np.nan

    nodata = -9999.0
    out = np.where(np.isfinite(pred_img), pred_img, nodata).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": H,
        "width": W,
        "crs": UTM,
        "transform": tr,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "predicted_AGB_tCO2_per_acre")
    vals = pred_img[np.isfinite(pred_img)]
    print(f"Wrote {OUT_TIF}")
    print(
        f"  AGB tCO2/acre: n={vals.size} min={vals.min():.1f} "
        f"mean={vals.mean():.1f} max={vals.max():.1f}"
    )

    # quicklook
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pred_img, cmap="viridis", vmin=0, vmax=float(np.nanpercentile(pred_img, 99)))
    ax.set_title("Bayfield County — predicted AGB (tCO₂/acre), 30 m")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, label="tCO₂/acre")
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")

    _write_readme(grid, feats, vals)


def _write_readme(grid: dict, feats: list[str], vals: np.ndarray) -> None:
    (PRED_DIR / "README.md").write_text(
        f"""# Bayfield County — wall-to-wall AGB inference

`bayfield_agb_30m.tif` — predicted CO₂ standing stock (tCO₂/acre), 30 m, {UTM}, nodata −9999.

- **Model:** LightGBM (`models/inference_model.txt`), {len(feats)} features =
  64 AEF embeddings + chm + topo×5 + corrected survey-relative dist + dstx disturbance features
  (`dstx_pre_ysd`, `dstx_pre_loss_5yr`, `dstx_loss_frac_buf`, `dstx_lt_mag`). Trained on all 4,636
  ANEW plots.
- **Embeddings:** local int8 AEF tiles (2023), cast to float **without dequantization** to match
  training; aggregated 10 m→30 m by mean.
- **Co-features:** GEE, survey_year=2023, focal reducers (~30 m) matching the training buffer.
- **Grid:** {grid["width"]}×{grid["height"]} px.
- **Stats:** AGB min {vals.min():.1f}, mean {vals.mean():.1f}, max {vals.max():.1f} tCO₂/acre.

**Caveats:** Bayfield is a *training* project, so predictions here are partly in-sample.
The model under-predicts high biomass and over-predicts low biomass (known dynamic-range
compression). Open in QGIS; style by the single float band.
"""
    )
    print(f"Wrote {PRED_DIR / 'README.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    assert MODEL_TXT.exists(), "train_inference_model.py first"
    grid = target_grid()
    print(f"target grid: {grid['width']}×{grid['height']} @ {RES} m  {UTM}")
    save_grid(grid)

    if EMB_NPY.exists():
        emb30 = np.load(EMB_NPY)
        _mask_nodata(emb30)  # cached array may predate the nodata fix
        print(f"loaded cached embeddings {emb30.shape}")
    else:
        emb30 = read_embeddings(grid)
        np.save(EMB_NPY, emb30)
        print(f"cached embeddings → {EMB_NPY}")
    correctness_gate(grid, emb30)

    if not COFEAT_TIF.exists():
        print(f"\nCo-feature raster not found: {COFEAT_TIF}")
        print("→ run:  uv run python scripts/export_bayfield_cofeatures.py")
        print("  then re-run this script to produce the map.")
        return

    predict_and_write(grid, emb30)


if __name__ == "__main__":
    main()
