"""
Extended investigation: split design, model types, feature processing, sample weighting.

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/investigate_extended.py

Output: reports/extended_investigation.md
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBRegressor

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
)
PARQUET = EXPDIR / "preprocessing/features_iter3.parquet"
OUT_MD = EXPDIR / "reports/extended_investigation.md"

FEAT_PREFIXES = ("emb_", "palsar_", "gedi_", "chm_", "topo_", "dist_", "agbd_", "clim_")
EMB_COLS_PAT = "emb_"
SEED = 42

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df = df[df["failure"].isna()].reset_index(drop=True)
    print(f"Loaded {len(df)} rows.")
    return df


def feat_cols(df: pd.DataFrame, prefixes=FEAT_PREFIXES) -> list[str]:
    return [c for c in df.columns if c.startswith(prefixes)]


def lopo_folds(df: pd.DataFrame):
    projects = sorted(df["project_name"].unique())
    p2i = {p: i for i, p in enumerate(projects)}
    return df["project_name"].map(p2i).to_numpy(), projects


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "bias": round(float((y_pred - y_true).mean()), 2),
    }


def qbias(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    bins = np.quantile(y_true, [0.2, 0.4, 0.6, 0.8])
    lbls = np.digitize(y_true, bins)
    return {
        f"q{i + 1}": round(float((y_pred[lbls == i] - y_true[lbls == i]).mean()), 1)
        for i in range(5)
    }


def lgbm_model(**kw):
    return lgb.LGBMRegressor(
        n_estimators=3000,
        n_jobs=-1,
        random_state=SEED,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        **kw,
    )


def lgbm_fit(model, X_tr, y_tr, X_va, y_va):
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )


def print_row(label: str, m: dict, qb: dict):
    print(
        f"  {label:30s}  R²={m['r2']:.4f}  RMSE={m['rmse']:.2f}"
        f"  Q1={qb['q1']:+.1f}  Q5={qb['q5']:+.1f}"
    )


def row_dict(label: str, m: dict, qb: dict, **extra) -> dict:
    return {"label": label, **m, **qb, **extra}


# ---------------------------------------------------------------------------
# Section 1 — CV strategy
# ---------------------------------------------------------------------------


def run_section1(df: pd.DataFrame, folds: np.ndarray, projects: list[str]) -> list[dict]:
    print("\n=== SECTION 1: CV strategy ===")
    X_raw = np.array(df[feat_cols(df)].astype("float32"))
    y = df["target"].to_numpy()
    rows = []

    # 1a. LOPO reference
    oof = np.zeros(len(y))
    for fid in range(len(projects)):
        tr, va = folds != fid, folds == fid
        m = lgbm_model()
        lgbm_fit(m, X_raw[tr], y[tr], X_raw[va], y[va])
        oof[va] = m.predict(X_raw[va])
    m_, qb_ = metrics(y, oof), qbias(y, oof)
    print_row("lopo (ref)", m_, qb_)
    rows.append(row_dict("lopo", m_, qb_))

    # 1b. 5-fold random
    oof = np.zeros(len(y))
    for tr_idx, va_idx in KFold(n_splits=5, shuffle=True, random_state=SEED).split(X_raw):
        m = lgbm_model()
        lgbm_fit(m, X_raw[tr_idx], y[tr_idx], X_raw[va_idx], y[va_idx])
        oof[va_idx] = m.predict(X_raw[va_idx])
    m_, qb_ = metrics(y, oof), qbias(y, oof)
    print_row("5fold_random", m_, qb_)
    rows.append(row_dict("5fold_random", m_, qb_))

    # 1c. 5-fold stratified by biomass quintile
    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    oof = np.zeros(len(y))
    for tr_idx, va_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        X_raw, strat
    ):
        m = lgbm_model()
        lgbm_fit(m, X_raw[tr_idx], y[tr_idx], X_raw[va_idx], y[va_idx])
        oof[va_idx] = m.predict(X_raw[va_idx])
    m_, qb_ = metrics(y, oof), qbias(y, oof)
    print_row("5fold_strat_bio", m_, qb_)
    rows.append(row_dict("5fold_strat_bio", m_, qb_))

    # 1d. 5-fold stratified by ecoregion
    region_codes = df["region"].map({"wv": 0, "mw": 1, "ne": 2}).to_numpy()
    oof = np.zeros(len(y))
    for tr_idx, va_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        X_raw, region_codes
    ):
        m = lgbm_model()
        lgbm_fit(m, X_raw[tr_idx], y[tr_idx], X_raw[va_idx], y[va_idx])
        oof[va_idx] = m.predict(X_raw[va_idx])
    m_, qb_ = metrics(y, oof), qbias(y, oof)
    print_row("5fold_strat_eco", m_, qb_)
    rows.append(row_dict("5fold_strat_eco", m_, qb_))

    # 1e. Leave-one-ecoregion-out (3 folds: wv / mw / ne)
    eco_folds = region_codes  # 0/1/2
    oof = np.zeros(len(y))
    for eco_id in range(3):
        tr, va = eco_folds != eco_id, eco_folds == eco_id
        m = lgbm_model()
        lgbm_fit(m, X_raw[tr], y[tr], X_raw[va], y[va])
        oof[va] = m.predict(X_raw[va])
    m_, qb_ = metrics(y, oof), qbias(y, oof)
    print_row("lopo_ecoregion", m_, qb_)
    rows.append(row_dict("lopo_ecoregion", m_, qb_))

    return rows


# ---------------------------------------------------------------------------
# Section 2 — Model types (LOPO, all features)
# ---------------------------------------------------------------------------


def run_section2(df: pd.DataFrame, folds: np.ndarray, projects: list[str]) -> list[dict]:
    print("\n=== SECTION 2: Model types (LOPO) ===")
    fcols = feat_cols(df)
    X = df[fcols].astype("float32").to_numpy()
    y = df["target"].to_numpy()
    rows = []

    configs = [
        ("lgbm_ref", "lgbm"),
        ("xgboost", "xgb"),
        ("histgbr", "hgbr"),
        ("random_forest", "rf"),
        ("extra_trees", "et"),
    ]

    for label, mtype in configs:
        oof = np.zeros(len(y))
        print(f"  [{label}] ...")
        for fid in range(len(projects)):
            tr, va = folds != fid, folds == fid
            X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]

            if mtype == "lgbm":
                m = lgbm_model()
                lgbm_fit(m, X_tr, y_tr, X_va, y_va)
            elif mtype == "xgb":
                m = XGBRegressor(
                    n_estimators=3000,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    missing=np.nan,
                    random_state=SEED,
                    early_stopping_rounds=50,
                    eval_metric="rmse",
                    verbosity=0,
                    n_jobs=-1,
                )
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            elif mtype == "hgbr":
                # Fill nulls — HistGBR handles them but only with Python >=3.9 sklearn 1.0+
                m = HistGradientBoostingRegressor(
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=SEED,
                )
                m.fit(X_tr, y_tr)
            elif mtype == "rf":
                m = RandomForestRegressor(
                    n_estimators=500,
                    max_features=0.33,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=SEED,
                )
                m.fit(X_tr, y_tr)
            elif mtype == "et":
                m = ExtraTreesRegressor(
                    n_estimators=500,
                    max_features=0.33,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=SEED,
                )
                m.fit(X_tr, y_tr)

            oof[va] = m.predict(X_va)

        m_, qb_ = metrics(y, oof), qbias(y, oof)
        print_row(label, m_, qb_)
        rows.append(row_dict(label, m_, qb_))

    return rows


# ---------------------------------------------------------------------------
# Section 3 — Feature processing (LightGBM, LOPO)
# ---------------------------------------------------------------------------


def run_section3(df: pd.DataFrame, folds: np.ndarray, projects: list[str]) -> list[dict]:
    print("\n=== SECTION 3: Feature processing (LightGBM, LOPO) ===")
    y = df["target"].to_numpy()
    rows = []

    # --- helper: run lopo with given X and optional y transform ---
    def run(label: str, X_in: np.ndarray, log_target: bool = False):
        oof = np.zeros(len(y))
        y_use = np.log1p(y) if log_target else y
        for fid in range(len(projects)):
            tr, va = folds != fid, folds == fid
            m = lgbm_model()
            lgbm_fit(m, X_in[tr], y_use[tr], X_in[va], y_use[va])
            pred = m.predict(X_in[va])
            oof[va] = np.expm1(pred) if log_target else pred
        m_, qb_ = metrics(y, oof), qbias(y, oof)
        print_row(label, m_, qb_)
        rows.append(row_dict(label, m_, qb_))

    fcols = feat_cols(df)
    emb_cols = [c for c in fcols if c.startswith(EMB_COLS_PAT)]
    other_cols = [c for c in fcols if not c.startswith(EMB_COLS_PAT)]

    # 3a. Raw (reference)
    X_raw = df[fcols].astype("float32").fillna(0).to_numpy()
    run("no_proc", X_raw)

    # 3b. log1p(chm_m) and add chm_slope interaction
    df2 = df[fcols].copy()
    if "chm_m" in df2.columns:
        df2["chm_m"] = np.log1p(df2["chm_m"].fillna(0))
    if "chm_m" in df2.columns and "topo_slope" in df2.columns:
        df2["chm_slope_ix"] = df2["chm_m"] * df2["topo_slope"].fillna(0)
    X_chm = df2.astype("float32").fillna(0).to_numpy()
    run("log_chm_plus_ix", X_chm)

    # 3c. PCA-20 on embeddings (fit on train, transform val)
    X_emb = df[emb_cols].astype("float32").to_numpy()
    X_oth = df[other_cols].astype("float32").fillna(0).to_numpy()
    oof = np.zeros(len(y))
    for fid in range(len(projects)):
        tr, va = folds != fid, folds == fid
        pca = PCA(n_components=20, random_state=SEED)
        E_tr = pca.fit_transform(X_emb[tr])
        E_va = pca.transform(X_emb[va])
        X_tr = np.hstack([E_tr, X_oth[tr]])
        X_va_f = np.hstack([E_va, X_oth[va]])
        m = lgbm_model()
        lgbm_fit(m, X_tr, y[tr], X_va_f, y[va])
        oof[va] = m.predict(X_va_f)
    m_, qb_ = metrics(y, oof), qbias(y, oof)
    print_row("pca_emb_20", m_, qb_)
    rows.append(row_dict("pca_emb_20", m_, qb_))

    # 3d. Log target (log1p(y))
    run("log_target", X_raw, log_target=True)

    # 3e. QuantileTransformer on non-embedding features
    X_emb_f = df[emb_cols].astype("float32").fillna(0).to_numpy()
    X_oth_f = df[other_cols].astype("float32").fillna(0).to_numpy()
    oof = np.zeros(len(y))
    for fid in range(len(projects)):
        tr, va = folds != fid, folds == fid
        qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
        O_tr = qt.fit_transform(X_oth_f[tr])
        O_va = qt.transform(X_oth_f[va])
        X_tr = np.hstack([X_emb_f[tr], O_tr])
        X_va_f2 = np.hstack([X_emb_f[va], O_va])
        m = lgbm_model()
        lgbm_fit(m, X_tr, y[tr], X_va_f2, y[va])
        oof[va] = m.predict(X_va_f2)
    m_, qb_ = metrics(y, oof), qbias(y, oof)
    print_row("quantile_norm_cofeatures", m_, qb_)
    rows.append(row_dict("quantile_norm_cofeatures", m_, qb_))

    return rows


# ---------------------------------------------------------------------------
# Section 4 — Sample weighting (LightGBM, LOPO)
# ---------------------------------------------------------------------------


def run_section4(df: pd.DataFrame, folds: np.ndarray, projects: list[str]) -> list[dict]:
    print("\n=== SECTION 4: Sample weighting (LightGBM, LOPO) ===")
    X = df[feat_cols(df)].astype("float32").to_numpy()
    y = df["target"].to_numpy()
    rows = []

    def run_weighted(label: str, weights: np.ndarray):
        oof = np.zeros(len(y))
        for fid in range(len(projects)):
            tr, va = folds != fid, folds == fid
            m = lgbm_model()
            m.fit(
                X[tr],
                y[tr],
                sample_weight=weights[tr],
                eval_set=[(X[va], y[va])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            oof[va] = m.predict(X[va])
        m_, qb_ = metrics(y, oof), qbias(y, oof)
        print_row(label, m_, qb_)
        rows.append(row_dict(label, m_, qb_))

    # 4a. Uniform (reference)
    run_weighted("uniform (ref)", np.ones(len(y)))

    # 4b. Inverse-frequency (histogram density)
    counts, edges = np.histogram(y, bins=20)
    bin_idx = np.clip(np.digitize(y, edges[:-1]) - 1, 0, 19)
    inv_freq = 1.0 / (counts[bin_idx].astype(float) + 1e-6)
    inv_freq /= inv_freq.mean()
    run_weighted("inv_freq", inv_freq)

    # 4c. Square-root inverse frequency
    sqrt_inv = np.sqrt(inv_freq)
    sqrt_inv /= sqrt_inv.mean()
    run_weighted("sqrt_inv_freq", sqrt_inv)

    # 4d. Hard quintile upweighting (Q5=5×, Q4=2×, others=1×)
    qbins = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
    qlabels = np.digitize(y, qbins)
    hard_w = np.where(qlabels == 4, 5.0, np.where(qlabels == 3, 2.0, 1.0))
    hard_w /= hard_w.mean()
    run_weighted("quintile_upweight", hard_w)

    return rows


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(s1: list, s2: list, s3: list, s4: list) -> None:
    def md_table(rows: list[dict], cols: list[str]) -> str:
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [header, sep]
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        return "\n".join(lines)

    main_cols = ["label", "r2", "rmse", "bias", "q1", "q2", "q3", "q4", "q5"]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("# Extended Investigation\n\n")
        f.write("Dataset: `features_iter3.parquet`, 4,636 rows, LightGBM unless noted.\n\n")
        f.write("## Section 1 — CV Strategy\n\n")
        f.write(md_table(s1, main_cols))
        f.write("\n\n## Section 2 — Model Types (LOPO)\n\n")
        f.write(md_table(s2, main_cols))
        f.write("\n\n## Section 3 — Feature Processing (LightGBM, LOPO)\n\n")
        f.write(md_table(s3, main_cols))
        f.write("\n\n## Section 4 — Sample Weighting (LightGBM, LOPO)\n\n")
        f.write(md_table(s4, main_cols))
        f.write("\n")
    print(f"\nWrote {OUT_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    df = load_data()
    folds, projects = lopo_folds(df)
    print(f"LOPO: {len(projects)} projects.")

    s1 = run_section1(df, folds, projects)
    s2 = run_section2(df, folds, projects)
    s3 = run_section3(df, folds, projects)
    s4 = run_section4(df, folds, projects)

    write_report(s1, s2, s3, s4)


if __name__ == "__main__":
    main()
