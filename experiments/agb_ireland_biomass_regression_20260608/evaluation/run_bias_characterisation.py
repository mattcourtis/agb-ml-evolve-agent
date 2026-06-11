"""Ireland AGB zero-shot transfer — bias-characterisation evaluation (model vs Deep Biomass).

NO ground truth exists. Deep Biomass (DB) is a directional lower bound, not truth.
All metrics are agreement/divergence vs DB and structural-consistency checks; NO accuracy claim.

Reuses the quintile-signed-bias + PRD logic from
  experiments/agb_usa_biomass_regression_20260529/evaluation/compute_biomass_metrics.py (lines 48-55)
but quintiles are formed on DB MAGNITUDE (not truth) and "residual" is the signed delta
  Δ = our_pred − DB_tco2.

Outputs (all absolute paths under the experiment evaluation/ dir):
  evaluation/ireland_predictions.parquet
  evaluation/evaluation_matrix.yaml
  evaluation/figures/*.png
  (the narrative bias_characterisation.md and reports/training_run.md are written separately)

SEED = 42 for any stochastic step (domain-classifier CV split).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # never plt.show
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXP = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PRE = EXP / "preprocessing"
EVAL = EXP / "evaluation"
FIG = EVAL / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL = REPO / "models/inference_model_embdstx.txt"
FEAT_JSON = REPO / "models/inference_features_embdstx.json"
TRAIN_PARQUET = Path(
    "/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet"
)

TRAINING_MAX = 520.951013322  # head target_range max
OPTICAL_CEILING_TCO2 = 80.0  # research §3 empirical optical ceiling (~115 Mg/ha)
CONUS_PRD_BASELINE = 0.468  # documented CONUS baseline PRD
FACTOR = 0.6977  # Mg/ha -> tCO2/acre

EMB_COLS = [f"emb_{i:02d}" for i in range(64)]


def load_feature_order() -> list[str]:
    return json.loads(FEAT_JSON.read_text())["features"]


def main() -> None:
    feat_order = load_feature_order()
    assert len(feat_order) == 67, feat_order

    # ---------------------------------------------------------------- INFERENCE
    feats = pd.read_parquet(PRE / "ireland_features.parquet")
    assert len(feats) == 141, len(feats)
    booster = lgb.Booster(model_file=str(MODEL))
    X = feats[feat_order].to_numpy()
    pred = booster.predict(X)
    feats = feats[["Location_Name"]].assign(pred_tco2=pred)

    db = pd.read_parquet(PRE / "db_reference.parquet")
    import geopandas as gpd

    cov = gpd.read_file(PRE / "ireland_locations_dissolved.gpkg", layer="locations").drop(
        columns="geometry"
    )

    m = feats.merge(
        db[
            [
                "Location_Name",
                "Area_Ha",
                "db_mgha_2020_2024_mean",
                "db_mgha_2024",
                "db_tco2acre_2020_2024_mean",
                "db_tco2acre_2024",
            ]
        ],
        on="Location_Name",
        how="left",
    ).merge(
        cov[
            [
                "Location_Name",
                "PlantingYe",
                "Hmean",
                "Hdom",
                "YC",
                "BA_Conifer",
                "MainSp",
                "MainSp_area_share",
                "survey_year",
                "survey_year_raw_mode",
                "pre2017_fallback",
                "age_at_survey",
            ]
        ],
        on="Location_Name",
        how="left",
    )
    assert m["db_tco2acre_2020_2024_mean"].notna().all()
    assert len(m) == 141

    m = m.rename(
        columns={
            "db_tco2acre_2020_2024_mean": "db_2020_24_tco2",
            "db_tco2acre_2024": "db_2024_tco2",
        }
    )
    m["delta_2020_24"] = m["pred_tco2"] - m["db_2020_24_tco2"]
    m["delta_2024"] = m["pred_tco2"] - m["db_2024_tco2"]

    pred_cols = [
        "Location_Name",
        "pred_tco2",
        "db_2020_24_tco2",
        "db_2024_tco2",
        "delta_2020_24",
        "delta_2024",
        "Area_Ha",
        "PlantingYe",
        "Hmean",
        "Hdom",
        "YC",
        "BA_Conifer",
        "MainSp",
        "MainSp_area_share",
        "survey_year",
        "survey_year_raw_mode",
        "pre2017_fallback",
        "age_at_survey",
    ]
    m[pred_cols].to_parquet(EVAL / "ireland_predictions.parquet", index=False)

    R = {}  # results dict for the matrix

    R["inference"] = {
        "head_id": "inference_model_embdstx.txt",
        "n_features": 67,
        "n_estimators": 73,
        "target": "tCO2/acre",
        "training_target_max": TRAINING_MAX,
        "n_locations": 141,
        "training_performed": False,
        "encoding_gate": "PASS (held-out corr 0.986, post-affine slope median 1.006); ref preprocessing/encoding_gate.json",
        "pred_min": float(m["pred_tco2"].min()),
        "pred_mean": float(m["pred_tco2"].mean()),
        "pred_median": float(m["pred_tco2"].median()),
        "pred_max": float(m["pred_tco2"].max()),
        "seed": SEED,
    }

    # ----------------------------------------------------- H1: DELTA DISTRIBUTION
    def delta_stats(d: pd.Series, pred: pd.Series, dbcol: pd.Series) -> dict:
        return {
            "mean": float(d.mean()),
            "median": float(d.median()),
            "iqr_25": float(d.quantile(0.25)),
            "iqr_75": float(d.quantile(0.75)),
            "min": float(d.min()),
            "max": float(d.max()),
            "frac_pred_ge_db": float((pred >= dbcol).mean()),
            "frac_delta_positive": float((d > 0).mean()),
        }

    R["h1_delta_distribution"] = {
        "primary_2020_24_mean": delta_stats(
            m["delta_2020_24"], m["pred_tco2"], m["db_2020_24_tco2"]
        ),
        "sensitivity_2024_only": delta_stats(m["delta_2024"], m["pred_tco2"], m["db_2024_tco2"]),
        "portfolio_pred_mean_tco2": float(m["pred_tco2"].mean()),
        "portfolio_db_2020_24_mean_tco2": float(m["db_2020_24_tco2"].mean()),
        "ratio_pred_over_db": float(m["pred_tco2"].mean() / m["db_2020_24_tco2"].mean()),
    }

    # scatter pred vs DB with 1:1 line
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = max(m["pred_tco2"].max(), m["db_2020_24_tco2"].max()) * 1.05
    ax.scatter(m["db_2020_24_tco2"], m["pred_tco2"], s=18, alpha=0.7, color="#2a6f97")
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Deep Biomass 2020-24 mean (tCO2/acre)")
    ax.set_ylabel("embdstx prediction (tCO2/acre)")
    ax.set_title("Ireland: our prediction vs Deep Biomass (n=141)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "scatter_pred_vs_db.png", dpi=130)
    plt.close(fig)

    # delta histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(m["delta_2020_24"], bins=30, color="#52b788", edgecolor="k", alpha=0.85)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(m["delta_2020_24"].median(), color="red", ls="-", lw=1.2, label="median")
    ax.set_xlabel("signed delta = pred - DB (tCO2/acre)")
    ax.set_ylabel("count of Locations")
    ax.set_title("Signed delta distribution (DB 2020-24 mean reference)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "delta_histogram.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------ H2: PER-QUINTILE SIGNED BIAS + PRD
    # quintiles by DB MAGNITUDE (reuse compute_biomass_metrics block, DB substitutes truth)
    def quintile_block(dbcol: str) -> dict:
        q = pd.qcut(m[dbcol], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        d = m.assign(quintile=q, residual=m["pred_tco2"] - m[dbcol])
        signed_bias = {
            str(k): float(v)
            for k, v in d.groupby("quintile", observed=True)["residual"].mean().items()
        }
        db_means = d.groupby("quintile", observed=True)[dbcol].mean()
        pred_means = d.groupby("quintile", observed=True)["pred_tco2"].mean()
        prd = float((pred_means["Q5"] - pred_means["Q1"]) / (db_means["Q5"] - db_means["Q1"]))
        vals = [signed_bias[f"Q{i}"] for i in range(1, 6)]
        monotone = all(vals[i] <= vals[i + 1] for i in range(4))
        return {
            "per_quintile_signed_bias": signed_bias,
            "per_quintile_db_mean": {str(k): float(v) for k, v in db_means.items()},
            "per_quintile_pred_mean": {str(k): float(v) for k, v in pred_means.items()},
            "monotone_increasing_Q1_to_Q5": bool(monotone),
            "predicted_range_discrimination": prd,
            "prd_note": "(pred_Q5_mean - pred_Q1_mean)/(DB_Q5_mean - DB_Q1_mean); DB-magnitude quintiles, no truth",
        }

    R["h2_quintile"] = {
        "primary_2020_24_mean": quintile_block("db_2020_24_tco2"),
        "sensitivity_2024_only": quintile_block("db_2024_tco2"),
        "conus_prd_baseline_vs_truth": CONUS_PRD_BASELINE,
        "prd_comparability_note": (
            "Irish PRD denominator is DB-magnitude spread (no truth); the CONUS 0.468 baseline "
            "denominator is true-target spread. Not directly comparable in level; reported as context."
        ),
    }

    # binned signed-bias bar
    qb = R["h2_quintile"]["primary_2020_24_mean"]["per_quintile_signed_bias"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(list(qb.keys()), list(qb.values()), color="#e07a5f", edgecolor="k")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("DB-magnitude quintile (Q1 low -> Q5 high)")
    ax.set_ylabel("mean signed delta pred-DB (tCO2/acre)")
    ax.set_title("Per-quintile signed bias (H2: gap widening with biomass)")
    fig.tight_layout()
    fig.savefig(FIG / "quintile_signed_bias.png", dpi=130)
    plt.close(fig)

    # ----------------------------------------------- H3: COVARIATE CUTS + SPEARMAN
    def binned_cut(col: str, bins, labels) -> dict:
        c = pd.cut(m[col], bins=bins, labels=labels, include_lowest=True)
        c = c.cat.add_categories(["missing"]).fillna("missing")
        g = m.assign(_b=c).groupby("_b", observed=True)
        out = {}
        for k, grp in g:
            out[str(k)] = {
                "n": int(len(grp)),
                "pred_mean": float(grp["pred_tco2"].mean()),
                "db_mean": float(grp["db_2020_24_tco2"].mean()),
                "delta_mean": float(grp["delta_2020_24"].mean()),
            }
        return out

    age_bins = [-0.1, 10, 20, 30, 40, 200]
    age_labels = ["0-10", "10-20", "20-30", "30-40", "40+"]
    hdom_bins = [-0.1, 5, 10, 15, 20, 40]
    hdom_labels = ["0-5", "5-10", "10-15", "15-20", "20+"]
    yc_bins = [-0.1, 10, 16, 20, 24, 50]
    yc_labels = ["0-10", "10-16", "16-20", "20-24", "24+"]

    def spearman_pair(xcol: str) -> dict:
        sub = m.dropna(subset=[xcol])
        rho_pred, p_pred = spearmanr(sub[xcol], sub["pred_tco2"])
        rho_db, p_db = spearmanr(sub[xcol], sub["db_2020_24_tco2"])
        return {
            "n": int(len(sub)),
            "rho_pred": float(rho_pred),
            "p_pred": float(p_pred),
            "rho_db": float(rho_db),
            "p_db": float(p_db),
            "pred_tracks_better": bool(rho_pred >= rho_db),
        }

    # MainSp: Sitka (SS) vs broadleaf/other vs missing
    def mainsp_group(s):
        if pd.isna(s):
            return "missing"
        return "Sitka_SS" if s == "SS" else "broadleaf_other"

    msp = m.assign(_g=m["MainSp"].map(mainsp_group)).groupby("_g", observed=True)
    mainsp_cut = {
        str(k): {
            "n": int(len(grp)),
            "pred_mean": float(grp["pred_tco2"].mean()),
            "db_mean": float(grp["db_2020_24_tco2"].mean()),
            "delta_mean": float(grp["delta_2020_24"].mean()),
        }
        for k, grp in msp
    }

    R["h3_covariate_cuts"] = {
        "age_bins": binned_cut("age_at_survey", age_bins, age_labels),
        "hdom_bins": binned_cut("Hdom", hdom_bins, hdom_labels),
        "yc_bins": binned_cut("YC", yc_bins, yc_labels),
        "mainsp": mainsp_cut,
        "spearman_age": spearman_pair("age_at_survey"),
        "spearman_hdom": spearman_pair("Hdom"),
        "spearman_yc": spearman_pair("YC"),
        "missingness": {
            "MainSp_missing_frac": float(m["MainSp"].isna().mean()),
            "PlantingYe_missing_frac": float(m["PlantingYe"].isna().mean()),
            "age_missing_frac": float(m["age_at_survey"].isna().mean()),
        },
    }

    # pred & DB vs age figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = m.dropna(subset=["age_at_survey"]).sort_values("age_at_survey")
    ax.scatter(
        sub["age_at_survey"], sub["pred_tco2"], s=18, alpha=0.7, label="our pred", color="#2a6f97"
    )
    ax.scatter(
        sub["age_at_survey"],
        sub["db_2020_24_tco2"],
        s=18,
        alpha=0.7,
        label="Deep Biomass",
        color="#e07a5f",
    )
    ax.set_xlabel("stand age at survey (yr)")
    ax.set_ylabel("tCO2/acre")
    ax.set_title("pred & DB vs stand age (H3 rank-tracking)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "pred_db_vs_age.png", dpi=130)
    plt.close(fig)

    # pred vs Hdom
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = m.dropna(subset=["Hdom"]).sort_values("Hdom")
    ax.scatter(sub["Hdom"], sub["pred_tco2"], s=18, alpha=0.7, color="#2a6f97", label="our pred")
    ax.scatter(
        sub["Hdom"], sub["db_2020_24_tco2"], s=18, alpha=0.6, color="#e07a5f", label="Deep Biomass"
    )
    ax.set_xlabel("Hdom (dominant height, m)")
    ax.set_ylabel("tCO2/acre")
    ax.set_title("pred & DB vs Hdom (H3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "pred_vs_hdom.png", dpi=130)
    plt.close(fig)

    # ----------------------------------------------------- OOD / COVARIATE SHIFT
    train = pd.read_parquet(TRAIN_PARQUET)
    train_emb = train[EMB_COLS].dropna()
    Xtr = train_emb.to_numpy()
    Xir = pd.read_parquet(PRE / "ireland_features.parquet")[EMB_COLS].to_numpy()

    mu = Xtr.mean(axis=0)
    cov_m = np.cov(Xtr, rowvar=False)
    cov_inv = np.linalg.pinv(cov_m)

    def mahal(X):
        diff = X - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))

    d_tr = mahal(Xtr)
    d_ir = mahal(Xir)
    # 99th-pct training radius
    radius99 = float(np.percentile(d_tr, 99))
    # chi2 reference (64 dof) for context
    chi2_99 = float(np.sqrt(chi2.ppf(0.99, df=64)))
    frac_beyond = float((d_ir > radius99).mean())

    R["ood_mahalanobis"] = {
        "training_radius_99pct": radius99,
        "chi2_99_sqrt_64dof_reference": chi2_99,
        "ireland_mahal_min": float(d_ir.min()),
        "ireland_mahal_median": float(np.median(d_ir)),
        "ireland_mahal_max": float(d_ir.max()),
        "frac_ireland_beyond_99pct_training_radius": frac_beyond,
    }

    # domain classifier USA vs Ireland, seed 42, cross-validated AUC
    Xdc = np.vstack([Xtr, Xir])
    ydc = np.concatenate([np.zeros(len(Xtr)), np.ones(len(Xir))])
    clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=200)
    proba = cross_val_predict(clf, Xdc, ydc, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    auc = float(roc_auc_score(ydc, proba))
    R["ood_domain_classifier"] = {
        "model": "HistGradientBoostingClassifier(max_iter=200, random_state=42)",
        "cv": "5-fold cross_val_predict",
        "n_usa": int(len(Xtr)),
        "n_ireland": int(len(Xir)),
        "auc": auc,
        "interpretation": "~0.5 indistinguishable/transferable; ~1.0 severe domain shift",
    }

    # ------------------------------------------------------------- SATURATION
    R["saturation"] = {
        "optical_ceiling_tco2": OPTICAL_CEILING_TCO2,
        "training_max_tco2": TRAINING_MAX,
        "research_saturation_onset_mgha": "150-200 Mg/ha (~105-140 tCO2/acre)",
        "pred_frac_gt_80_tco2": float((m["pred_tco2"] > OPTICAL_CEILING_TCO2).mean()),
        "pred_frac_gt_training_max": float((m["pred_tco2"] > TRAINING_MAX).mean()),
        "pred_max_tco2": float(m["pred_tco2"].max()),
        "db_frac_gt_80_tco2": float((m["db_2020_24_tco2"] > OPTICAL_CEILING_TCO2).mean()),
        "db_frac_gt_training_max": float((m["db_2020_24_tco2"] > TRAINING_MAX).mean()),
        "db_max_tco2": float(m["db_2020_24_tco2"].max()),
        "pred_frac_above_saturation_onset_105": float((m["pred_tco2"] > 105).mean()),
    }

    EVAL.joinpath("_results.json").write_text(json.dumps(R, indent=2))
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
