"""
Embeddings + disturbance ("dynamic-only") Bayfield AGB map — no static layers (no CHM/topo).

Model = models/inference_model_embdstx.txt (64 emb + dstx_pre_ysd + dstx_pre_loss_5yr +
dstx_loss_frac_buf). Reuses the cached embedding stack and the dstx bands already in
predictions/bayfield_cofeatures_30m.tif — no new GEE. Writes
predictions/bayfield_agb_embdstx_30m.tif (+ quicklook) and prints a low-end comparison against
the embeddings-only and full maps.

    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/infer_bayfield_embdstx.py
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
    COFEAT_TIF,
    EMB_NPY,
    OUT_TIF,
    PRED_DIR,
    REPO,
    RES,
    UTM,
    _mask_nodata,
    target_grid,
)

MODEL_TXT = REPO / "models/inference_model_embdstx.txt"
FEATS_JSON = REPO / "models/inference_features_embdstx.json"
OUT_EMB_TIF = PRED_DIR / "bayfield_agb_embonly_30m.tif"  # for comparison
OUT_DSTX_TIF = PRED_DIR / "bayfield_agb_embdstx_30m.tif"
OUT_DSTX_PNG = PRED_DIR / "bayfield_agb_embdstx_30m_quicklook.png"
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
    booster = lgb.Booster(model_file=str(MODEL_TXT))

    emb30 = np.load(EMB_NPY)
    _mask_nodata(emb30)
    layers = {f"emb_{i:02d}": emb30[i] for i in range(64)}

    # dstx bands from the existing co-feature raster
    with rasterio.open(COFEAT_TIF) as ds:
        names = list(ds.descriptions)
        co = ds.read().astype(np.float32)
        co[~np.isfinite(co)] = np.nan
    for n in ("dstx_pre_ysd", "dstx_pre_loss_5yr", "dstx_loss_frac_buf"):
        layers[n] = co[names.index(n)]

    missing = [f for f in feats if f not in layers]
    assert not missing, f"missing layers: {missing}"
    stack = np.stack([layers[f] for f in feats], axis=0)  # (67, H, W) in model order
    X = stack.reshape(len(feats), -1).T
    finite = np.isfinite(X).all(axis=1)
    pred = np.full(X.shape[0], np.nan, dtype=np.float32)
    pred[finite] = booster.predict(X[finite]).astype(np.float32)
    pred_img = pred.reshape(H, W)

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
    with rasterio.open(OUT_DSTX_TIF, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "predicted_AGB_tCO2_per_acre_embdstx")
    print(f"Wrote {OUT_DSTX_TIF}")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pred_img, cmap="viridis", vmin=0, vmax=float(np.nanpercentile(pred_img, 99)))
    ax.set_title("Bayfield County — predicted AGB (tCO₂/acre), 30 m — embeddings + disturbance")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, label="tCO₂/acre")
    fig.savefig(OUT_DSTX_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DSTX_PNG}")

    # comparison vs emb-only and full
    print("\n=== low-end comparison (tCO₂/acre) ===")
    print(f"{'map':22s} {'n':>8s} {'min':>6s} {'p1':>6s} {'p5':>6s} {'median':>7s} {'%<30':>6s}")
    rows = [("emb+dstx (this)", pred_img)]
    for label, path in [("embeddings-only", OUT_EMB_TIF), ("full (73-feat)", OUT_TIF)]:
        if path.exists():
            with rasterio.open(path) as ds:
                a = ds.read(1)
            rows.append((label, np.where(a == NODATA, np.nan, a)))
    for label, a in rows:
        s = low_end_stats(a)
        print(
            f"{label:22s} {s['n']:8d} {s['min']:6.1f} {s['p1']:6.1f} {s['p5']:6.1f} "
            f"{s['median']:7.1f} {s['pct_lt30']:6.1f}"
        )


if __name__ == "__main__":
    main()
