"""
Low/zero-biomass de-bias + discrimination bake-off (emb-only ANEW model).

The shipped S0 model is mean-seeking: it over-predicts true CO2 < 100 tCO2/acre by +35..+49
(regression to the mean) while still rank-ordering the low band (Spearman ~0.57). This compares
variants that *re-aim the predictor* (target transforms, Tweedie objective, two-stage hurdle)
against S0, plus post-hoc calibration as a documented-weak reference, all on the SAME LOPO CV.

Scoring is low-end focused: conditional bias by true-band (primary), low-band RMSE/MAE, within-
<100 discrimination (Spearman + AUC for true<50), zero recall, and the QUANTIFIED high-end trade.

Outputs (data-space): low_end/{variant_metrics,band_bias,oof_by_variant}.parquet, decision.json.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/anew_emb_model/low_end.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_ladder as cvl  # noqa: E402
import data as data_mod  # noqa: E402

OUT = common.DATASPACE / "agb_anew_emb_weighted_20260630/low_end"
EMB = common.EMB
BANDS = [(0, 0), (0, 25), (25, 50), (50, 100), (100, 150), (150, 250), (250, np.inf)]
LOW = 100.0  # "low band" ceiling
LOW_CLASS = 50.0  # discrimination target: separate true < 50
TWEEDIE_POWERS = [1.1, 1.3, 1.5]
TAUS = [25.0, 50.0]


def cv_oof(X, y, groups, fold_fn):
    """Generic LOPO out-of-fold predictions; fold_fn(Xtr, ytr, Xte) -> preds (clipped at 0)."""
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        oof[te] = np.clip(fold_fn(X[~te], y[~te], X[te]), 0, None)
    return oof


# ----- variant fold functions ------------------------------------------------
def fold_raw(n_trees):
    def f(Xtr, ytr, Xte):
        return lgb.LGBMRegressor(n_estimators=n_trees, **cvl.LGB_PARAMS).fit(Xtr, ytr).predict(Xte)

    return f


def fold_transform(n_trees, fwd, inv):
    def f(Xtr, ytr, Xte):
        m = lgb.LGBMRegressor(n_estimators=n_trees, **cvl.LGB_PARAMS).fit(Xtr, fwd(ytr))
        return inv(m.predict(Xte))

    return f


def fold_tweedie(n_trees, power):
    params = {**cvl.LGB_PARAMS, "objective": "tweedie", "tweedie_variance_power": power}

    def f(Xtr, ytr, Xte):
        return lgb.LGBMRegressor(n_estimators=n_trees, **params).fit(Xtr, ytr).predict(Xte)

    return f


def fold_hurdle(n_trees, tau):
    """P(y>=tau) classifier * high-regressor + (1-P)*low_mean; all fit on train only."""

    def f(Xtr, ytr, Xte):
        hi = ytr >= tau
        low_mean = float(ytr[~hi].mean()) if (~hi).any() else 0.0
        if hi.sum() < 50 or (~hi).sum() < 50:  # degenerate fold -> plain regressor
            return (
                lgb.LGBMRegressor(n_estimators=n_trees, **cvl.LGB_PARAMS).fit(Xtr, ytr).predict(Xte)
            )
        clf = lgb.LGBMClassifier(n_estimators=n_trees, **cvl.LGB_PARAMS).fit(Xtr, hi.astype(int))
        reg = lgb.LGBMRegressor(n_estimators=n_trees, **cvl.LGB_PARAMS).fit(Xtr[hi], ytr[hi])
        p_hi = clf.predict_proba(Xte)[:, 1]
        return p_hi * reg.predict(Xte) + (1 - p_hi) * low_mean

    return f


# ----- metrics ---------------------------------------------------------------
def band_metrics(y, p):
    rows = []
    for lo, hi in BANDS:
        m = (y == 0) if (lo == 0 and hi == 0) else (y > lo) & (y <= hi)
        if m.sum() == 0:
            continue
        lab = (
            "==0" if (lo == 0 and hi == 0) else f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        )
        rows.append(
            {
                "band": lab,
                "n": int(m.sum()),
                "mean_true": float(y[m].mean()),
                "mean_pred": float(p[m].mean()),
                "bias": float((p[m] - y[m]).mean()),
                "rmse": cvl.rmse(y[m], p[m]),
            }
        )
    return pd.DataFrame(rows)


def summary_metrics(y, p):
    low = y < LOW
    hi = y > 150
    return {
        "bias_lt100": float((p[low] - y[low]).mean()),
        "rmse_lt100": cvl.rmse(y[low], p[low]),
        "mae_lt100": cvl.mae(y[low], p[low]),
        "spearman_lt100": float(spearmanr(p[low], y[low]).correlation),
        "auc_lt50": float(roc_auc_score(y < LOW_CLASS, -p)),  # lower pred -> more likely low
        "zero_recall_at25": float((p[y == 0] < 25).mean()),
        "bias_gt150": float((p[hi] - y[hi]).mean()),
        "rmse_gt150": cvl.rmse(y[hi], p[hi]),
        "rmse_all": cvl.rmse(y, p),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = data_mod.load_eligible()
    X = df[EMB].astype(float).to_numpy()
    y = df["CO2"].to_numpy()
    proj = df["project_name"].to_numpy()
    n_trees = json.loads((cvl.OUT / "n_trees.json").read_text())["n_estimators"]

    variants = {
        "S0_raw": fold_raw(n_trees),
        "T_log1p": fold_transform(n_trees, np.log1p, np.expm1),
        "T_sqrt": fold_transform(n_trees, np.sqrt, np.square),
        **{f"tweedie_{pw}": fold_tweedie(n_trees, pw) for pw in TWEEDIE_POWERS},
        **{f"hurdle_t{int(t)}": fold_hurdle(n_trees, t) for t in TAUS},
    }

    oof_by_variant, band_rows, summ_rows = {}, [], []
    for name, fn in variants.items():
        oof = cv_oof(X, y, proj, fn)
        oof_by_variant[name] = oof
        bm = band_metrics(y, oof)
        bm.insert(0, "variant", name)
        band_rows.append(bm)
        summ_rows.append({"variant": name, **summary_metrics(y, oof)})
        print(
            f"[{name:12s}] bias<100 {summ_rows[-1]['bias_lt100']:+6.1f}  "
            f"rmse<100 {summ_rows[-1]['rmse_lt100']:5.1f}  AUC<50 {summ_rows[-1]['auc_lt50']:.3f}  "
            f"spear<100 {summ_rows[-1]['spearman_lt100']:.3f}  bias>150 {summ_rows[-1]['bias_gt150']:+6.1f}"
        )

    # post-hoc isotonic calibration on S0 (documented-weak reference, OOF -> OOF map)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(oof_by_variant["S0_raw"], y)
    oof_cal = np.clip(iso.predict(oof_by_variant["S0_raw"]), 0, None)
    oof_by_variant["Cal_isotonic"] = oof_cal
    bm = band_metrics(y, oof_cal)
    bm.insert(0, "variant", "Cal_isotonic")
    band_rows.append(bm)
    summ_rows.append({"variant": "Cal_isotonic", **summary_metrics(y, oof_cal)})
    print(
        f"[Cal_isotonic ] bias<100 {summ_rows[-1]['bias_lt100']:+6.1f}  "
        f"rmse<100 {summ_rows[-1]['rmse_lt100']:5.1f}  (reference: L2 already self-calibrated)"
    )

    summ = pd.DataFrame(summ_rows)
    band = pd.concat(band_rows, ignore_index=True)
    summ.to_parquet(OUT / "variant_metrics.parquet", index=False)
    band.to_parquet(OUT / "band_bias.parquet", index=False)
    pd.DataFrame({"CO2": y, "project_name": proj, **oof_by_variant}).to_parquet(
        OUT / "oof_by_variant.parquet", index=False
    )

    # decision: smallest |bias<100| among variants that keep discrimination (spearman & AUC >= S0)
    s0 = summ[summ.variant == "S0_raw"].iloc[0]
    cand = summ[
        (summ.spearman_lt100 >= s0.spearman_lt100 - 1e-9) & (summ.auc_lt50 >= s0.auc_lt50 - 1e-9)
    ].copy()
    cand["abs_bias"] = cand["bias_lt100"].abs()
    chosen = cand.sort_values("abs_bias").iloc[0]["variant"] if len(cand) else "S0_raw"
    decision = {
        "chosen": chosen,
        "rule": "min |bias<100| among variants with spearman<100 and AUC<50 >= S0",
        "s0_bias_lt100": float(s0.bias_lt100),
        "chosen_bias_lt100": float(summ[summ.variant == chosen].iloc[0].bias_lt100),
        "chosen_auc_lt50": float(summ[summ.variant == chosen].iloc[0].auc_lt50),
        "chosen_bias_gt150": float(summ[summ.variant == chosen].iloc[0].bias_gt150),
        "candidates": cand[["variant", "bias_lt100", "spearman_lt100", "auc_lt50"]].to_dict(
            "records"
        ),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, default=float))
    print(
        f"\nchosen: {chosen}  (bias<100 {s0.bias_lt100:+.1f} -> "
        f"{decision['chosen_bias_lt100']:+.1f}; high-end bias>150 {decision['chosen_bias_gt150']:+.1f})"
    )
    print(f"Saved low-end artifacts to {OUT}")


if __name__ == "__main__":
    main()
