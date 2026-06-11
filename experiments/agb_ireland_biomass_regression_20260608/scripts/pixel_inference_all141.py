"""
Production-aligned per-pixel inference mean(f) for ALL 141 Irish stands.

Extends scripts/per_pixel_inference.py: per stand, at its survey_year, pull the 64-band
native-float AEF (A00..A63) at every 10 m pixel whose centre is inside the dissolved
polygon + the 3 survey-relative dstx co-features per pixel; apply the production per-band
affine (emb_b = a_b*A_b + c_b) per pixel; predict per pixel with inference_model_embdstx
(tCO2/acre). Capture per-pixel lon/lat so the predictions can be rasterised to GeoTIFF.

Resumable: each stand checkpoints to preprocessing/_pixel_pred/<Location>.parquet
(per-pixel lon, lat, pred_tco2_acre) plus a summary row in
preprocessing/_pixel_pred/_summary/<Location>.json. Done stands are skipped on restart.

Run (tmux):
    uv run python experiments/.../scripts/pixel_inference_all141.py

seed 42; AEF asset GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (V1); Hansen ...2025_v1_13.
Mg/ha = tCO2/acre / 0.6977.
"""

from __future__ import annotations

import io
import json
import time
import traceback
from pathlib import Path

import ee
import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PREP = EXPDIR / "preprocessing"

DISSOLVED = PREP / "ireland_locations_dissolved.gpkg"
AFFINE = PREP / "aef_affine.parquet"
FEATS_JSON = REPO / "models/inference_features_embdstx.json"
MODEL_TXT = REPO / "models/inference_model_embdstx.txt"

CKPT = PREP / "_pixel_pred"
CKPT_SUM = CKPT / "_summary"

AEF_ASSET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"

BANDS = [f"A{i:02d}" for i in range(64)]
EMB = [f"emb_{i:02d}" for i in range(64)]
DSTX = ["dstx_pre_ysd", "dstx_pre_loss_5yr", "dstx_loss_frac_buf"]
NO_DIST_YSD = 100.0
AEF_SCALE = 10
TCO2ACRE_TO_MGHA = 1.0 / 0.6977
SEED = 42


def ee_polygon(geom) -> ee.Geometry:
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


def dstx_pixel_image(survey_year: int) -> ee.Image:
    ly = ee.Image(HANSEN_ASSET).select("lossyear")
    code = survey_year - 2000
    pre = ly.gt(0).And(ly.lte(code))
    ysd = ee.Image(NO_DIST_YSD).where(pre, ee.Image(code).subtract(ly)).rename("dstx_pre_ysd")
    loss5 = pre.And(ee.Image(code).subtract(ly).lte(5)).rename("dstx_pre_loss_5yr")
    frac = pre.rename("dstx_loss_frac_buf")
    return ysd.addBands(loss5).addBands(frac)


def _download_npy(image: ee.Image, region: ee.Geometry, scale: int, bands: list[str]) -> dict:
    url = image.getDownloadURL({"bands": bands, "region": region, "scale": scale, "format": "NPY"})
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    arr = np.load(io.BytesIO(r.content), allow_pickle=True)
    return {b: arr[b].astype(np.float64) for b in bands}


def fetch_array(
    image: ee.Image, region: ee.Geometry, scale: int, bands: list[str], n_tiles: int = 1
) -> dict:
    """Download a region as numpy arrays (NPY), tiling if oversized.

    Adds an `inmask` band (geometry rasterised at `scale`, 1 iff pixel centre inside polygon)
    and lon/lat pixel-centre coordinate bands so predictions can be rasterised. Returns
    {band: 1D flat array} (row-major-concatenated across tiles).
    """
    mask = ee.Image.constant(1).rename("inmask").clip(region).unmask(0)
    lonlat = ee.Image.pixelLonLat()  # bands 'longitude','latitude'
    img = image.addBands(mask).addBands(lonlat)
    allb = bands + ["inmask", "longitude", "latitude"]
    if n_tiles <= 1:
        try:
            arr = _download_npy(img, region, scale, allb)
            return {b: arr[b].ravel() for b in allb}
        except ee.ee_exception.EEException as exc:
            if "request size" not in str(exc).lower():
                raise
            n_tiles = 4
    coords = region.bounds().coordinates().getInfo()[0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    while True:
        try:
            parts: dict[str, list] = {b: [] for b in allb}
            edges = np.linspace(y0, y1, n_tiles + 1)
            for i in range(n_tiles):
                strip = ee.Geometry.Rectangle(
                    [x0, edges[i], x1, edges[i + 1]], proj="EPSG:4326", geodesic=False
                )
                a = _download_npy(img, strip, scale, allb)
                for b in allb:
                    parts[b].append(a[b].ravel())
            return {b: np.concatenate(parts[b]) for b in allb}
        except ee.ee_exception.EEException as exc:
            if "request size" in str(exc).lower() and n_tiles < 64:
                n_tiles *= 2
                continue
            raise


def process_location(name: str, row, a_vec, c_vec, booster, feats) -> tuple[pd.DataFrame, dict]:
    year = int(row["survey_year"])
    region = ee_polygon(row.geometry)
    fused = aef_image(year).addBands(dstx_pixel_image(year))
    arr = fetch_array(fused, region, AEF_SCALE, BANDS + DSTX)

    emb_native = np.stack([arr[b] for b in BANDS], axis=1)
    dstx_pix = np.stack([arr[d] for d in DSTX], axis=1)
    lon = arr["longitude"]
    lat = arr["latitude"]
    inmask = arr["inmask"] >= 0.5
    valid = (
        inmask
        & np.isfinite(emb_native).all(axis=1)
        & np.isfinite(dstx_pix).all(axis=1)
        & np.isfinite(lon)
        & np.isfinite(lat)
    )
    emb_native = emb_native[valid]
    dstx_pix = dstx_pix[valid]
    lon = lon[valid]
    lat = lat[valid]
    n_pix = int(emb_native.shape[0])
    if n_pix < 1:
        raise RuntimeError("0 valid pixels")

    emb_codec = emb_native * a_vec + c_vec
    X_pix = np.concatenate([emb_codec, dstx_pix], axis=1)
    assert X_pix.shape[1] == len(feats)
    pix_pred = booster.predict(X_pix)  # tCO2/acre per pixel

    mean_f = float(pix_pred.mean())  # mean(f) = stand density tCO2/acre

    pix_df = pd.DataFrame(
        {"lon": lon.astype(np.float64), "lat": lat.astype(np.float64), "pred_tco2_acre": pix_pred}
    )
    summary = {
        "Location_Name": name,
        "survey_year": year,
        "n_pixels": n_pix,
        "pred_pixel_tCO2_acre": mean_f,
        "pred_pixel_Mg_ha": mean_f * TCO2ACRE_TO_MGHA,
        "pixel_pred_std": float(pix_pred.std()),
        "pixel_pred_min": float(pix_pred.min()),
        "pixel_pred_median": float(np.percentile(pix_pred, 50)),
        "pixel_pred_max": float(pix_pred.max()),
    }
    return pix_df, summary


def main() -> None:
    np.random.seed(SEED)
    ee.Initialize()
    print("GEE initialised.", flush=True)

    CKPT.mkdir(parents=True, exist_ok=True)
    CKPT_SUM.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(DISSOLVED)
    affine = pd.read_parquet(AFFINE)
    a = affine.set_index("band")["a"]
    c = affine.set_index("band")["c"]
    a_vec = np.array([a[b] for b in BANDS])
    c_vec = np.array([c[b] for b in BANDS])
    feats = json.loads(FEATS_JSON.read_text())["features"]
    assert feats == EMB + DSTX, "feature order mismatch"
    booster = lgb.Booster(model_file=str(MODEL_TXT))

    names = list(gdf["Location_Name"])
    print(f"Total stands: {len(names)}", flush=True)

    done, failed = [], []
    for i, name in enumerate(names, 1):
        ckpt = CKPT / f"{name}.parquet"
        sumf = CKPT_SUM / f"{name}.json"
        if ckpt.exists() and sumf.exists():
            print(f"[{i}/{len(names)}] {name}: cached, skip", flush=True)
            done.append(name)
            continue
        row = gdf.loc[gdf["Location_Name"] == name].iloc[0]
        ok = False
        for attempt in range(1, 4):
            try:
                t0 = time.time()
                pix_df, summary = process_location(name, row, a_vec, c_vec, booster, feats)
                pix_df.to_parquet(ckpt, index=False)
                sumf.write_text(json.dumps(summary))
                dt = time.time() - t0
                print(
                    f"[{i}/{len(names)}] {name}: n_pix={summary['n_pixels']} "
                    f"mean(f)={summary['pred_pixel_tCO2_acre']:.2f} tCO2/acre "
                    f"({dt:.0f}s, attempt {attempt})",
                    flush=True,
                )
                done.append(name)
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[{i}/{len(names)}] {name}: attempt {attempt} FAILED: {exc}", flush=True)
                if attempt == 3:
                    traceback.print_exc()
                time.sleep(5 * attempt)
        if not ok:
            failed.append(name)

    print(f"\nDONE: {len(done)}/{len(names)} ok, {len(failed)} failed.", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
    (CKPT / "_run_done.flag").write_text(
        json.dumps({"done": len(done), "failed": failed, "total": len(names)})
    )


if __name__ == "__main__":
    main()
