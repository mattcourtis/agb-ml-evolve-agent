"""
Per-pixel AGB inference for Ireland Dasos Locations — change-of-support analysis.

Quantifies the aggregation gap between the REPORTED stand estimator f(mean(emb))
[polygon-mean -> predict once] and the PRODUCTION estimator mean(f(emb)) [predict
per pixel -> aggregate], for the `embdstx` head.

For each requested Location:
  1. Pull the 64-band AEF (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL) NATIVE-FLOAT values at
     every 10 m pixel whose centre lies inside the dissolved polygon, as a numpy array
     (ee getDownloadURL, scale=10, clipped to the polygon).
  2. Pull the 3 survey-relative dstx co-features per 10 m pixel (build_dstx_image, Hansen
     lossyear reprojected to the same 10 m grid). dstx is constant-ish within a stand but
     evaluated per pixel for fidelity.
  3. Apply the per-band production affine (emb_b = a_b*GEE_b + c_b) per pixel.
  4. f(mean): predict once on the (pixel-mean emb + pixel-mean dstx).
  5. mean(f): predict per pixel, then average pixel predictions (stand density estimate).
  6. Consistency check: pixel-mean(native emb) -> affine should match the iter0 polygon-mean
     emb feature (reduceRegions(mean)) within tolerance.

Reuses the affine + dstx logic from scripts/extract_ireland_aef.py and build_dist_image
from scripts/extract_iter2_features.py. Parameterised by a Location list -> reusable as the
basis for a future wall-to-wall map.

    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/per_pixel_inference.py \
        --locations "Bargarriff,Carrigeeny,..." --out <parquet>

If --locations is omitted, runs the built-in stratified sample (see STRATIFIED_SAMPLE).
"""

from __future__ import annotations

import argparse
import io
import json
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
IRELAND_FEATS = PREP / "ireland_features.parquet"

AEF_ASSET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"

BANDS = [f"A{i:02d}" for i in range(64)]
EMB = [f"emb_{i:02d}" for i in range(64)]
DSTX = ["dstx_pre_ysd", "dstx_pre_loss_5yr", "dstx_loss_frac_buf"]
NO_DIST_YSD = 100.0
AEF_SCALE = 10
TILE_SCALE = 4

# Stratified sample of 18 Locations spanning the heterogeneity x size x age range.
# Stratified across n_subcpt (1=homogeneous .. 37=heterogeneous) x area_ha (0.67..151.84)
# x age (0..26.6 yr), incl. youngest/oldest, largest (Meensheefin), most-heterogeneous
# (Cloonsheever). Selection: scripts/per_pixel_inference.py docstring + support_sensitivity.md.
STRATIFIED_SAMPLE: list[str] = [
    "Carrowreagh",
    "Rathcahill West",
    "Crooderry",
    "Carrigeeny",
    "Tooreennagreana",
    "Loughros",
    "Dromreask",
    "Carrowkeel",
    "Cummeen Upper",
    "Lacka Beg",
    "Sligo Bay North_Greaghnafarna II",
    "Highmount",
    "Glanowen",
    "Knockbreenagher",
    "Sligo Bay South_Kilfree",
    "Benmore",
    "Meensheefin",
    "Cloonsheever",
]


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
    """Per-pixel survey-relative Hansen timing -> the 3 dstx features directly as bands.

    Matches extract_ireland_aef.extract_dstx but evaluated per pixel (no reduceRegions):
      dstx_pre_ysd      = survey_year_code - lossyear  if 0<ly<=code else 100
      dstx_pre_loss_5yr = 1 if pre-survey loss within 5 yr else 0
      dstx_loss_frac_buf= pre-survey loss indicator (0/1) per pixel; its polygon-mean equals
                          the iter0 pre_frac_mean fraction.
    """
    ly = ee.Image(HANSEN_ASSET).select("lossyear")
    code = survey_year - 2000
    pre = ly.gt(0).And(ly.lte(code))
    ysd = ee.Image(NO_DIST_YSD).where(pre, ee.Image(code).subtract(ly)).rename("dstx_pre_ysd")
    loss5 = pre.And(ee.Image(code).subtract(ly).lte(5)).rename("dstx_pre_loss_5yr")
    frac = pre.rename("dstx_loss_frac_buf")  # 0/1 per pixel; mean over polygon = disturbed frac
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
    """Download a region as numpy arrays via getDownloadURL (NPY), tiling if oversized.

    getDownloadURL returns the bounding-box RECTANGLE of `region` (clip does NOT drop pixels
    whose centre lies outside the polygon). We add an `inmask` band — the geometry rasterised
    at `scale` (1 iff the pixel centre is inside the polygon, GEE's default centre rule) — and
    the caller keeps only `inmask==1` pixels. This reproduces the exact pixel set that
    reduceRegions(mean) integrates over.

    For large stands the request can exceed the 50 MB getDownloadURL cap; we then split the
    bounding box into `n_tiles` latitude strips and concatenate the per-pixel (flattened)
    arrays. Tiling does not change which pixels are kept (the inmask is recomputed per tile).
    Returns {band: 1D flat array} (already row-major-concatenated across tiles).
    """
    mask = ee.Image.constant(1).rename("inmask").clip(region).unmask(0)
    img = image.addBands(mask)
    allb = bands + ["inmask"]
    if n_tiles <= 1:
        try:
            arr = _download_npy(img, region, scale, allb)
            return {b: arr[b].ravel() for b in allb}
        except ee.ee_exception.EEException as exc:
            if "request size" not in str(exc).lower():
                raise
            n_tiles = 4  # retry tiled
    # Tiled: split bbox into latitude strips.
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


def process_location(name: str, gdf, affine, booster, feats) -> dict | None:
    row = gdf.loc[gdf["Location_Name"] == name].iloc[0]
    year = int(row["survey_year"])
    region = ee_polygon(row.geometry)

    a = affine.set_index("band")["a"]
    c = affine.set_index("band")["c"]

    # Single fused image (AEF 64 + dstx 3) so all bands share one pixel grid and tiling order.
    fused = aef_image(year).addBands(dstx_pixel_image(year))
    try:
        arr = fetch_array(fused, region, AEF_SCALE, BANDS + DSTX)
    except Exception as exc:  # noqa: BLE001
        print(f"  [{name}] FAILED extraction: {exc}")
        return None

    # Stack to (n_pixels, n_bands); valid = inside polygon AND AEF finite across all 64 bands.
    # fetch_array returns flat 1D arrays (tile-concatenated, co-registered across all bands).
    emb_native = np.stack([arr[b] for b in BANDS], axis=1)  # (P, 64) raw GEE float
    dstx_pix = np.stack([arr[d] for d in DSTX], axis=1)  # (P, 3)
    inmask = arr["inmask"] >= 0.5  # pixel centre inside polygon
    valid = inmask & np.isfinite(emb_native).all(axis=1) & np.isfinite(dstx_pix).all(axis=1)
    emb_native = emb_native[valid]
    dstx_pix = dstx_pix[valid]
    n_pix = int(emb_native.shape[0])
    if n_pix < 1:
        print(f"  [{name}] 0 valid pixels")
        return None

    # Apply production affine per pixel -> training codec space.
    a_vec = np.array([a[b] for b in BANDS])
    c_vec = np.array([c[b] for b in BANDS])
    emb_codec = emb_native * a_vec + c_vec  # (P, 64)

    # --- mean(f): predict per pixel, then average ---
    X_pix = np.concatenate([emb_codec, dstx_pix], axis=1)  # (P, 67) in model order
    assert X_pix.shape[1] == len(feats)
    pix_pred = booster.predict(X_pix)
    mean_f = float(pix_pred.mean())

    # --- f(mean): pixel-mean emb (native) -> affine -> + pixel-mean dstx -> predict once ---
    emb_native_mean = emb_native.mean(axis=0)
    emb_codec_mean = emb_native_mean * a_vec + c_vec
    dstx_mean = dstx_pix.mean(axis=0)
    X_mean = np.concatenate([emb_codec_mean, dstx_mean])[None, :]
    f_mean = float(booster.predict(X_mean)[0])

    # Pixel-prediction dispersion.
    return {
        "Location_Name": name,
        "survey_year": year,
        "n_pixels": n_pix,
        "f_mean": f_mean,
        "mean_f": mean_f,
        "gap": mean_f - f_mean,
        "gap_pct": 100.0 * (mean_f - f_mean) / f_mean,
        "pix_pred_min": float(pix_pred.min()),
        "pix_pred_p25": float(np.percentile(pix_pred, 25)),
        "pix_pred_median": float(np.percentile(pix_pred, 50)),
        "pix_pred_p75": float(np.percentile(pix_pred, 75)),
        "pix_pred_max": float(pix_pred.max()),
        "pix_pred_std": float(pix_pred.std()),
        "pix_pred_iqr": float(np.percentile(pix_pred, 75) - np.percentile(pix_pred, 25)),
        # consistency: native pixel-mean emb (codec) vs iter0 polygon-mean emb feature
        "_emb_codec_mean": emb_codec_mean,  # internal, stripped before save
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations", default="", help="comma-separated Location_Name list")
    ap.add_argument("--out", default=str(EXPDIR / "evaluation/support_sensitivity_stands.parquet"))
    args = ap.parse_args()

    ee.Initialize()
    print("GEE initialised.")

    gdf = gpd.read_file(DISSOLVED, layer="locations")
    affine = pd.read_parquet(AFFINE)
    feats = json.loads(FEATS_JSON.read_text())["features"]
    assert feats == EMB + DSTX
    booster = lgb.Booster(model_file=str(MODEL_TXT))
    iter0 = pd.read_parquet(IRELAND_FEATS).set_index("Location_Name")

    if args.locations.strip():
        names = [s.strip() for s in args.locations.split(",") if s.strip()]
    else:
        names = STRATIFIED_SAMPLE
    print(f"Processing {len(names)} Locations.")

    rows, consist = [], []
    for name in names:
        print(f"[{name}] year {int(gdf.loc[gdf.Location_Name == name, 'survey_year'].iloc[0])} ...")
        res = process_location(name, gdf, affine, booster, feats)
        if res is None:
            continue
        # consistency check vs iter0 polygon-mean
        emb_codec_mean = res.pop("_emb_codec_mean")
        iter0_emb = iter0.loc[name, EMB].values.astype(float)
        max_abs = float(np.max(np.abs(emb_codec_mean - iter0_emb)))
        res["emb_consistency_max_abs"] = max_abs
        res["f_mean_iter0"] = float(
            booster.predict(
                np.concatenate([iter0_emb, iter0.loc[name, DSTX].values.astype(float)])[None, :]
            )[0]
        )
        rows.append(res)
        consist.append(max_abs)
        print(
            f"  n_pix={res['n_pixels']} f(mean)={res['f_mean']:.2f} mean(f)={res['mean_f']:.2f} "
            f"gap={res['gap']:+.2f} ({res['gap_pct']:+.1f}%) emb_consistency_max_abs={max_abs:.3f}"
        )

    df = pd.DataFrame(rows)
    # attach stratification covariates
    cov = gdf.set_index("Location_Name")[["n_subcpt", "area_ha", "age_at_survey", "Hdom", "MainSp"]]
    df = df.merge(cov, left_on="Location_Name", right_index=True, how="left")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out} ({len(df)} stands)")
    print(df[["Location_Name", "n_pixels", "f_mean", "mean_f", "gap", "gap_pct"]].to_string())


if __name__ == "__main__":
    main()
