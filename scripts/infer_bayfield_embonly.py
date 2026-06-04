"""
Embeddings-only Bayfield AGB map — isolates the stale-CHM confound.

Predicts with the 64-embedding model (models/inference_model_embonly.txt) on the cached,
correctness-gated embedding stack (predictions/bayfield_emb_30m.npy). No co-features, no GEE.
Writes predictions/bayfield_agb_embonly_30m.tif (+ quicklook) and prints a low-end comparison
against the 73-feature map to see whether the ~30 tCO₂/acre floor changes.

    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/infer_bayfield_embonly.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features as rfeatures
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_bayfield import (  # noqa: E402
    EMB_NPY,
    OUT_TIF,  # the existing 73-feature map (for comparison)
    PRED_DIR,
    REPO,
    RES,
    UTM,
    _mask_nodata,
    target_grid,
)

MODEL_TXT = REPO / "models/inference_model_embonly.txt"
FEATS_JSON = REPO / "models/inference_features_embonly.json"
OUT_EMB_TIF = PRED_DIR / "bayfield_agb_embonly_30m.tif"
OUT_EMB_PNG = PRED_DIR / "bayfield_agb_embonly_30m_quicklook.png"
NODATA = -9999.0


def low_end_stats(a: np.ndarray) -> dict:
    v = a[np.isfinite(a)]
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "p1": float(np.percentile(v, 1)),
        "p5": float(np.percentile(v, 5)),
        "median": float(np.percentile(v, 50)),
        "pct_lt30": float(100 * (v < 30).mean()),
    }


def main() -> None:
    grid = target_grid()
    H, W = grid["height"], grid["width"]
    feats = json.loads(FEATS_JSON.read_text())["features"]
    assert all(f.startswith("emb_") for f in feats), "expected embeddings-only feature set"
    booster = lgb.Booster(model_file=str(MODEL_TXT))

    emb30 = np.load(EMB_NPY)
    _mask_nodata(emb30)  # all-band-zero gaps → NaN
    # feature order = emb_00..63 (already the stack order)
    stack = emb30  # (64, H, W)
    X = stack.reshape(stack.shape[0], -1).T
    finite = np.isfinite(X).all(axis=1)
    pred = np.full(X.shape[0], np.nan, dtype=np.float32)
    pred[finite] = booster.predict(X[finite]).astype(np.float32)
    pred_img = pred.reshape(H, W)

    # mask to county polygon
    tr = from_origin(grid["minx"], grid["maxy"], RES, RES)
    poly = rfeatures.geometry_mask(grid["gdf"].geometry, (H, W), tr, invert=True)
    pred_img[~poly] = np.nan

    out = np.where(np.isfinite(pred_img), pred_img, NODATA).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": H,
        "width": W,
        "crs": UTM,
        "transform": tr,
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(OUT_EMB_TIF, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "predicted_AGB_tCO2_per_acre_embonly")
    print(f"Wrote {OUT_EMB_TIF}")

    # quicklook
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pred_img, cmap="viridis", vmin=0, vmax=float(np.nanpercentile(pred_img, 99)))
    ax.set_title("Bayfield County — predicted AGB (tCO₂/acre), 30 m — embeddings only")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, label="tCO₂/acre")
    fig.savefig(OUT_EMB_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_EMB_PNG}")

    # --- low-end comparison vs the 73-feature map ---
    emb_stats = low_end_stats(pred_img)
    print("\n=== low-end comparison (tCO₂/acre) ===")
    print(f"{'map':18s} {'n':>8s} {'min':>6s} {'p1':>6s} {'p5':>6s} {'median':>7s} {'%<30':>6s}")
    print(
        f"{'embeddings-only':18s} {emb_stats['n']:8d} {emb_stats['min']:6.1f} "
        f"{emb_stats['p1']:6.1f} {emb_stats['p5']:6.1f} {emb_stats['median']:7.1f} "
        f"{emb_stats['pct_lt30']:6.1f}"
    )
    if OUT_TIF.exists():
        with rasterio.open(OUT_TIF) as ds:
            full = ds.read(1)
        full = np.where(full == NODATA, np.nan, full)
        fs = low_end_stats(full)
        print(
            f"{'full (73-feat)':18s} {fs['n']:8d} {fs['min']:6.1f} {fs['p1']:6.1f} "
            f"{fs['p5']:6.1f} {fs['median']:7.1f} {fs['pct_lt30']:6.1f}"
        )
        print(
            f"\nfloor (min): embonly {emb_stats['min']:.1f} vs full {fs['min']:.1f} "
            f"→ {'LOWER with emb-only' if emb_stats['min'] < fs['min'] - 1 else 'essentially unchanged'}"
        )


if __name__ == "__main__":
    main()
