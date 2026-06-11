"""
ENCODING GATE — per-band affine GEE AlphaEarth -> training int8-codec space.

The training parquet holds raw int8-averaged AEF (range ~[-86, 86]); GEE
GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL serves dequantised doubles (~[-0.39, 0.38]). The LightGBM
head splits on absolute parquet-scale thresholds, so GEE values must be mapped per band by an
affine `emb_j ~= a_j * A{j}_GEE + c_j` fitted at the Bayfield TRAINING overlap, where parquet
embeddings are the ground truth.

Procedure (matches training support: reduceRegions(mean) over the plot footprint, not a point):
  1. Sample GEE A00..A63 for year 2023 over each Bayfield plot footprint (7.3 m radius buffer,
     ~training plot support) via reduceRegions(mean).
  2. Split plots train/held-out (SEED=42). Fit per-band OLS a_j, c_j on the TRAIN split.
  3. Validate on HELD-OUT: apply the affine, require post-affine per-band slope ~= 1, bounded
     intercept, and the correctness gate mean corr > 0.8 (correctness_gate contract).

Outputs:
  - preprocessing/aef_affine.parquet : per-band a_j, c_j + fit diagnostics
  - preprocessing/encoding_gate.json : gate verdict (PASS/FAIL) + numbers

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/fit_aef_affine.py
"""

from __future__ import annotations

import json
from pathlib import Path

import ee
import numpy as np
import pandas as pd

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PREP = EXPDIR / "preprocessing"

TRAIN_PARQUET = (
    "/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet"
)
AEF_ASSET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

SEED = 42
PLOT_RADIUS_M = 7.3  # ANEW plot footprint (data_profile / extract_disturbance_timing buffer note)
SCALE = 10  # AEF native resolution
BATCH = 100
HELDOUT_FRAC = 0.30

# Gate thresholds. The data_profile-accepted contract is: mean corr > 0.8 AND post-affine
# per-band slope ~= 1 with a bounded intercept. Per-band OLS validated independently on the
# held-out split carries sampling + point-vs-area noise, so "slope ~= 1" is assessed on the
# CENTRAL TENDENCY (median/mean across the 64 bands) plus a per-band tolerance band, and the
# intercept is bounded RELATIVE TO each band's own parquet-scale std (an absolute parquet-unit
# cap is meaningless because per-band std spans 7.8-41.1).
GATE_CORR = 0.8
SLOPE_MED_LO, SLOPE_MED_HI = 0.95, 1.05  # median across bands must sit on 1
SLOPE_BAND_LO, SLOPE_BAND_HI = 0.8, 1.2  # >=90% of bands within this tolerance
SLOPE_BAND_FRAC = 0.9
INTERCEPT_REL_MAX = 0.5  # median |intercept| / band-std must be small (centred near 0)

BANDS = [f"A{i:02d}" for i in range(64)]
EMB = [f"emb_{i:02d}" for i in range(64)]


def load_bayfield() -> pd.DataFrame:
    df = pd.read_parquet(TRAIN_PARQUET)
    bay = df[df["project_name"] == "BayfieldCounty"].copy()
    bay = bay.dropna(subset=EMB + ["lon", "lat"]).reset_index(drop=True)
    bay["pid"] = bay.index.astype(str)
    print(f"Bayfield plots (valid emb): {len(bay)} (all year 2023)")
    return bay


def sample_gee(bay: pd.DataFrame) -> pd.DataFrame:
    """reduceRegions(mean) of GEE AEF 2023 over each plot footprint -> A00..A63 per plot."""
    img = (
        ee.ImageCollection(AEF_ASSET).filterDate("2023-01-01", "2024-01-01").mosaic().select(BANDS)
    )
    reducer = ee.Reducer.mean()
    rows: list[dict] = []
    n = len(bay)
    for start in range(0, n, BATCH):
        sub = bay.iloc[start : start + BATCH]
        feats = [
            ee.Feature(
                ee.Geometry.Point([float(r["lon"]), float(r["lat"])]).buffer(PLOT_RADIUS_M),
                {"pid": str(r["pid"])},
            )
            for _, r in sub.iterrows()
        ]
        res = img.reduceRegions(
            collection=ee.FeatureCollection(feats), reducer=reducer, scale=SCALE, tileScale=4
        )
        for f in res.getInfo()["features"]:
            p = f["properties"]
            rows.append({"pid": str(p.get("pid")), **{b: p.get(b, np.nan) for b in BANDS}})
        print(f"  sampled plots {start}-{start + len(sub) - 1}/{n}")
    return pd.DataFrame(rows)


def fit_affine(merged: pd.DataFrame, train_pids: set[str]) -> pd.DataFrame:
    tr = merged[merged["pid"].isin(train_pids)]
    recs = []
    for b, e in zip(BANDS, EMB):
        x = tr[b].to_numpy(dtype=float)
        y = tr[e].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        a, c = np.polyfit(x[m], y[m], 1)
        recs.append({"band": b, "emb": e, "a": float(a), "c": float(c)})
    return pd.DataFrame(recs)


def validate(merged: pd.DataFrame, affine: pd.DataFrame, ho_pids: set[str]) -> dict:
    ho = merged[merged["pid"].isin(ho_pids)].copy()
    a = affine.set_index("band")["a"]
    c = affine.set_index("band")["c"]

    # apply affine -> transformed GEE in parquet codec space
    trans = pd.DataFrame(
        {e: ho[b].to_numpy() * a[b] + c[b] for b, e in zip(BANDS, EMB)}, index=ho.index
    )

    # (i) post-affine per-band slope (transformed ~ parquet); should be ~1
    slopes, intercepts, band_std = [], [], []
    for e in EMB:
        x = trans[e].to_numpy(dtype=float)  # transformed GEE
        y = ho[e].to_numpy(dtype=float)  # parquet truth
        msk = np.isfinite(x) & np.isfinite(y)
        s, ic = np.polyfit(x[msk], y[msk], 1)
        slopes.append(s)
        intercepts.append(ic)
        band_std.append(float(np.std(y[msk])))
    slopes = np.array(slopes)
    intercepts = np.array(intercepts)
    band_std = np.array(band_std)
    rel_intercept = np.abs(intercepts) / np.where(band_std > 0, band_std, np.nan)

    # (ii) correctness gate: per-plot corr(transformed 64-vec, parquet 64-vec) > 0.8
    corrs = []
    for _, r in ho.iterrows():
        tv = np.array([r[b] * a[b] + c[b] for b in BANDS], dtype=float)
        pv = r[EMB].to_numpy(dtype=float)
        if np.isfinite(tv).all() and tv.std() > 0:
            corrs.append(np.corrcoef(tv, pv)[0, 1])
    corrs = np.array(corrs)

    slope_med = float(np.median(slopes))
    frac_in_band = float(((slopes >= SLOPE_BAND_LO) & (slopes <= SLOPE_BAND_HI)).mean())
    med_rel_intercept = float(np.nanmedian(rel_intercept))

    slope_ok = bool(SLOPE_MED_LO <= slope_med <= SLOPE_MED_HI and frac_in_band >= SLOPE_BAND_FRAC)
    intercept_ok = bool(med_rel_intercept <= INTERCEPT_REL_MAX)
    corr_ok = bool(np.nanmean(corrs) > GATE_CORR)

    return {
        "n_heldout": int(len(ho)),
        "n_corr_plots": int(len(corrs)),
        "mean_corr_transformed": float(np.nanmean(corrs)),
        "min_corr_transformed": float(np.nanmin(corrs)),
        "slope_min": float(slopes.min()),
        "slope_max": float(slopes.max()),
        "slope_mean": float(slopes.mean()),
        "slope_median": slope_med,
        "slope_frac_in_tolerance_band": frac_in_band,
        "intercept_abs_max": float(np.abs(intercepts).max()),
        "intercept_median_rel_to_bandstd": med_rel_intercept,
        "slope_ok": slope_ok,
        "intercept_ok": intercept_ok,
        "corr_ok": corr_ok,
        "PASS": bool(slope_ok and intercept_ok and corr_ok),
        "thresholds": {
            "corr": GATE_CORR,
            "slope_median": [SLOPE_MED_LO, SLOPE_MED_HI],
            "slope_band": [SLOPE_BAND_LO, SLOPE_BAND_HI],
            "slope_band_frac": SLOPE_BAND_FRAC,
            "intercept_median_rel_max": INTERCEPT_REL_MAX,
        },
    }


def main() -> None:
    PREP.mkdir(parents=True, exist_ok=True)
    ee.Initialize()
    print("GEE initialised.")

    cache = PREP / "bayfield_gee_vs_parquet.parquet"
    if cache.exists():
        merged = pd.read_parquet(cache)
        print(f"Loaded cached GEE-vs-parquet sample: {len(merged)} plots")
    else:
        bay = load_bayfield()
        gee = sample_gee(bay)
        merged = bay[["pid", *EMB]].merge(gee, on="pid", how="inner")
        merged = merged.dropna(subset=BANDS).reset_index(drop=True)
        print(f"Merged plots with full GEE sample: {len(merged)}")
        merged.to_parquet(cache, index=False)

    rng = np.random.default_rng(SEED)
    pids = merged["pid"].to_numpy()
    rng.shuffle(pids)
    n_ho = int(len(pids) * HELDOUT_FRAC)
    ho_pids = set(pids[:n_ho])
    train_pids = set(pids[n_ho:])
    print(f"Split: {len(train_pids)} train / {len(ho_pids)} held-out (SEED={SEED}).")

    affine = fit_affine(merged, train_pids)
    affine.to_parquet(PREP / "aef_affine.parquet", index=False)
    print(
        f"Fitted per-band affine. slope a range [{affine['a'].min():.1f}, {affine['a'].max():.1f}], "
        f"intercept c range [{affine['c'].min():.2f}, {affine['c'].max():.2f}]"
    )

    gate = validate(merged, affine, ho_pids)
    gate["seed"] = SEED
    gate["n_train"] = len(train_pids)
    (PREP / "encoding_gate.json").write_text(json.dumps(gate, indent=2))

    print("\n=== ENCODING GATE ===")
    print(
        f"  held-out plots: {gate['n_heldout']}; mean corr (transformed) = "
        f"{gate['mean_corr_transformed']:.3f} (min {gate['min_corr_transformed']:.3f})"
    )
    print(
        f"  post-affine per-band slope: median {gate['slope_median']:.3f} "
        f"(mean {gate['slope_mean']:.3f}); "
        f"{gate['slope_frac_in_tolerance_band'] * 100:.0f}% of bands in "
        f"[{SLOPE_BAND_LO},{SLOPE_BAND_HI}]"
    )
    print(
        f"  median |intercept|/band-std = {gate['intercept_median_rel_to_bandstd']:.3f} "
        f"(<= {INTERCEPT_REL_MAX})"
    )
    print(
        f"  corr_ok={gate['corr_ok']} slope_ok={gate['slope_ok']} intercept_ok={gate['intercept_ok']}"
    )
    print(f"  GATE: {'PASS' if gate['PASS'] else 'FAIL'}")

    if not gate["PASS"]:
        raise SystemExit("ENCODING GATE FAILED — STOP. See preprocessing/encoding_gate.json")


if __name__ == "__main__":
    main()
