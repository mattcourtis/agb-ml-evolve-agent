"""
PRODUCTION affine refit (revision 2) — fit the per-band AEF->codec affine on the FULL valid
Bayfield set (all 409 plots), after the held-out 287/122 gate has already PASSED.

Rationale (carry-over requirement (c)): the held-out fit-on-287 / validate-on-122 result is the
GATE VALIDATION EVIDENCE (proves generalisation: corr 0.986, slope median 1.006). Standard
production practice after a held-out gate passes is to refit on ALL available data so the applied
transform uses every plot. This script:

  1. Refits per-band OLS a_j, c_j on the FULL 409 Bayfield plots (cached
     bayfield_gee_vs_parquet.parquet has both GEE A00..A63 and parquet emb_00..63). Saves to
     preprocessing/aef_affine.parquet (this becomes the APPLIED/production affine).
  2. Recovers the RAW Ireland GEE A-values by inverting the previous (train-only) affine that
     produced the cached ireland_aef_raw.parquet  (A = (emb_old - c_old)/a_old), then re-applies the
     full-409 affine (emb_new = a_new*A + c_new). Inversion is exact, so no GEE re-extraction needed.
  3. Regenerates ireland_features.parquet (141 x 67) and re-runs the embdstx smoke prediction.
  4. Reports per-band misfit honesty stats on the held-out 122 (OLS SE z-scores, reconstruction
     RMSE relative to band sigma).

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/refit_aef_affine_production.py
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PREP = EXPDIR / "preprocessing"

BANDS = [f"A{i:02d}" for i in range(64)]
EMB = [f"emb_{i:02d}" for i in range(64)]
DSTX = ["dstx_pre_ysd", "dstx_pre_loss_5yr", "dstx_loss_frac_buf"]
SEED = 42
HELDOUT_FRAC = 0.30

FEATS_JSON = REPO / "models/inference_features_embdstx.json"
MODEL = REPO / "models/inference_model_embdstx.txt"


def fit_affine(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for b, e in zip(BANDS, EMB):
        x = df[b].to_numpy(dtype=float)
        y = df[e].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        a, c = np.polyfit(x[m], y[m], 1)
        recs.append({"band": b, "emb": e, "a": float(a), "c": float(c)})
    return pd.DataFrame(recs)


def heldout_misfit_stats(merged: pd.DataFrame, affine_full: pd.DataFrame) -> dict:
    """Honest per-band misfit on the held-out 122, using the PRODUCTION (full-409) affine.

    For each held-out plot apply the affine -> transformed GEE; OLS-regress parquet ~ transformed,
    report slope z-scores (|slope-1|/SE) and per-band reconstruction RMSE relative to band sigma.
    """
    rng = np.random.default_rng(SEED)
    pids = merged["pid"].to_numpy()
    rng.shuffle(pids)
    n_ho = int(len(pids) * HELDOUT_FRAC)
    ho = merged[merged["pid"].isin(set(pids[:n_ho]))].copy()

    a = affine_full.set_index("band")["a"]
    c = affine_full.set_index("band")["c"]

    z_list, rmse_rel, in2se = [], [], 0
    for b, e in zip(BANDS, EMB):
        xg = ho[b].to_numpy(dtype=float)
        trans = xg * a[b] + c[b]  # transformed GEE in codec space
        y = ho[e].to_numpy(dtype=float)  # parquet truth
        m = np.isfinite(trans) & np.isfinite(y)
        tt, yy = trans[m], y[m]
        n = len(tt)
        # OLS yy ~ slope*tt + b0
        slope, b0 = np.polyfit(tt, yy, 1)
        resid = yy - (slope * tt + b0)
        sigma = float(np.std(yy))
        # SE of slope
        sxx = np.sum((tt - tt.mean()) ** 2)
        s_err = np.sqrt(np.sum(resid**2) / (n - 2))
        se_slope = s_err / np.sqrt(sxx)
        z = abs(slope - 1.0) / se_slope if se_slope > 0 else np.nan
        z_list.append(z)
        if z <= 2.0:
            in2se += 1
        # reconstruction RMSE (transformed vs parquet) relative to band sigma
        rmse = float(np.sqrt(np.mean((tt - yy) ** 2)))
        rmse_rel.append(rmse / sigma if sigma > 0 else np.nan)

    z_arr = np.array(z_list, dtype=float)
    rmse_rel = np.array(rmse_rel, dtype=float)
    return {
        "bands_within_2se": int(in2se),
        "mean_abs_z": float(np.nanmean(z_arr)),
        "rmse_rel_mean": float(np.nanmean(rmse_rel)),
        "rmse_rel_max": float(np.nanmax(rmse_rel)),
    }


def main() -> None:
    merged = pd.read_parquet(PREP / "bayfield_gee_vs_parquet.parquet")
    merged = merged.dropna(subset=BANDS + EMB).reset_index(drop=True)
    print(f"Full valid Bayfield plots: {len(merged)}")

    # --- 1. production refit on ALL 409 ---
    affine_full = fit_affine(merged)
    print(
        f"Full-409 affine: a range [{affine_full['a'].min():.1f}, {affine_full['a'].max():.1f}], "
        f"c range [{affine_full['c'].min():.2f}, {affine_full['c'].max():.2f}]"
    )

    # --- recover raw Ireland A-values by inverting the OLD (train-only) affine ---
    affine_old = pd.read_parquet(PREP / "aef_affine.parquet")
    a_old = affine_old.set_index("emb")["a"]
    c_old = affine_old.set_index("emb")["c"]
    ire_old = pd.read_parquet(PREP / "ireland_aef_raw.parquet")  # post old-affine emb_*
    raw_A = pd.DataFrame({"Location_Name": ire_old["Location_Name"]})
    for b, e in zip(BANDS, EMB):
        raw_A[b] = (ire_old[e].astype(float) - c_old[e]) / a_old[e]

    # --- 2/3. re-apply full-409 affine to Ireland ---
    a_new = affine_full.set_index("band")["a"]
    c_new = affine_full.set_index("band")["c"]
    ire_new = pd.DataFrame({"Location_Name": raw_A["Location_Name"]})
    for b, e in zip(BANDS, EMB):
        ire_new[e] = raw_A[b].astype(float) * a_new[b] + c_new[b]

    # persist the production affine (overwrite) + refreshed raw cache
    affine_full.to_parquet(PREP / "aef_affine.parquet", index=False)
    ire_new.to_parquet(PREP / "ireland_aef_raw.parquet", index=False)

    # reassemble features with existing dstx
    feats = json.loads(FEATS_JSON.read_text())["features"]
    cur = pd.read_parquet(PREP / "ireland_features.parquet")
    dstx_df = cur[["Location_Name"] + DSTX]
    out = ire_new.merge(dstx_df, on="Location_Name", how="inner")
    out = out[["Location_Name"] + feats].copy()
    assert len(out) == 141, f"expected 141 rows, got {len(out)}"
    assert out[feats].notna().all().all(), "NaNs present in feature table"
    out.to_parquet(PREP / "ireland_features.parquet", index=False)
    print(f"Regenerated ireland_features.parquet: {out.shape}, 0 NaNs")

    # --- smoke prediction ---
    m = lgb.Booster(model_file=str(MODEL))
    p = m.predict(out[feats])
    pmin, pmean, pmax = float(p.min()), float(p.mean()), float(p.max())
    print(f"SMOKE prediction (full-409 affine): min {pmin:.1f} / mean {pmean:.1f} / max {pmax:.1f}")

    # --- honest misfit stats (held-out 122, production affine) ---
    mf = heldout_misfit_stats(merged, affine_full)
    print(f"MISFIT (held-out 122, full-409 affine): {mf}")

    # emb range for spec
    emb_vals = out[EMB].to_numpy()
    print(f"Ireland emb global range: [{emb_vals.min():.1f}, {emb_vals.max():.1f}]")

    # dump a small json for the revision log
    (PREP / "production_refit_summary.json").write_text(
        json.dumps(
            {
                "n_plots_production_fit": int(len(merged)),
                "affine_a_range": [float(affine_full["a"].min()), float(affine_full["a"].max())],
                "affine_c_range": [float(affine_full["c"].min()), float(affine_full["c"].max())],
                "smoke_pred": {
                    "min": round(pmin, 1),
                    "mean": round(pmean, 1),
                    "max": round(pmax, 1),
                },
                "emb_range": [float(emb_vals.min()), float(emb_vals.max())],
                "heldout_misfit": mf,
            },
            indent=2,
        )
    )
    print("Wrote production_refit_summary.json")


if __name__ == "__main__":
    main()
