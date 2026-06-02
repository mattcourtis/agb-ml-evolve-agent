"""
Gradient boosting hyperparameter investigation — LOPO CV grid search.

Compares 7 LightGBM configurations (plus a ridge baseline) on the iter3
feature set, using the same project-LOPO CV protocol as the main trainer.

Usage:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/investigate_gb.py

Output:
    reports/gb_investigation.md
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529"
)
PARQUET = EXPDIR / "preprocessing/features_iter3.parquet"
OUT_MD = EXPDIR / "reports/gb_investigation.md"

FEATURE_PREFIXES = ("emb_", "palsar_", "gedi_", "chm_", "topo_", "dist_", "agbd_", "clim_")
EMB_ONLY_PREFIXES = ("emb_",)

SEED = 42

# ---------------------------------------------------------------------------
# Configuration grid
# ---------------------------------------------------------------------------

CONFIGS = [
    # label, feature_prefixes, lgbm_kwargs (None → use RidgeCV)
    (
        "baseline",
        FEATURE_PREFIXES,
        dict(
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=20,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
        ),
    ),
    (
        "deeper",
        FEATURE_PREFIXES,
        dict(
            num_leaves=127,
            learning_rate=0.05,
            min_child_samples=10,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
        ),
    ),
    (
        "very_deep",
        FEATURE_PREFIXES,
        dict(
            num_leaves=255,
            learning_rate=0.03,
            min_child_samples=5,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
        ),
    ),
    (
        "stochastic",
        FEATURE_PREFIXES,
        dict(
            num_leaves=63,
            learning_rate=0.05,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
        ),
    ),
    (
        "regularised",
        FEATURE_PREFIXES,
        dict(
            num_leaves=63,
            learning_rate=0.03,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
        ),
    ),
    (
        "emb_only",
        EMB_ONLY_PREFIXES,
        dict(
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=20,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
        ),
    ),
    (
        "fast_shallow",
        FEATURE_PREFIXES,
        dict(
            num_leaves=15,
            learning_rate=0.1,
            min_child_samples=30,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
        ),
    ),
    ("ridge_all", FEATURE_PREFIXES, None),  # RidgeCV reference — non-tree linear model
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df = df[df["failure"].isna()].reset_index(drop=True)
    print(f"Loaded {len(df)} rows after dropping failure rows.")
    return df


def feature_cols(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in df.columns if c.startswith(prefixes)]


def lopo_folds(df: pd.DataFrame) -> np.ndarray:
    projects = sorted(df["project_name"].unique())
    p2i = {p: i for i, p in enumerate(projects)}
    return df["project_name"].map(p2i).to_numpy(), projects


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float((y_pred - y_true).mean()),
        "n": int(len(y_true)),
    }


def quintile_bias(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    """Mean signed residual per quintile of y_true."""
    bins = np.quantile(y_true, [0.2, 0.4, 0.6, 0.8])
    labels = np.digitize(y_true, bins)  # 0–4
    return [float((y_pred[labels == q] - y_true[labels == q]).mean()) for q in range(5)]


def run_lgbm_lopo(
    df: pd.DataFrame,
    folds: np.ndarray,
    projects: list[str],
    feat_cols: list[str],
    lgbm_kwargs: dict,
) -> tuple[dict, list[float]]:
    """Run LOPO CV with the given LGBM kwargs; return aggregate metrics + quintile bias."""
    X = df[feat_cols].astype("float32").to_numpy()
    y = df["target"].to_numpy()
    oof = np.zeros(len(y))

    for fold_id in range(len(projects)):
        train = folds != fold_id
        val = folds == fold_id
        model = lgb.LGBMRegressor(
            n_estimators=3000,
            n_jobs=-1,
            random_state=SEED,
            **lgbm_kwargs,
        )
        model.fit(
            X[train],
            y[train],
            eval_set=[(X[val], y[val])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val] = model.predict(X[val])

    return regression_metrics(y, oof), quintile_bias(y, oof)


def run_ridge_lopo(
    df: pd.DataFrame,
    folds: np.ndarray,
    projects: list[str],
    feat_cols: list[str],
) -> tuple[dict, list[float]]:
    """Run LOPO CV with RidgeCV (linear baseline)."""
    X = df[feat_cols].fillna(0).astype("float32").to_numpy()
    y = df["target"].to_numpy()
    oof = np.zeros(len(y))

    for fold_id in range(len(projects)):
        train = folds != fold_id
        val = folds == fold_id
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train])
        X_va = scaler.transform(X[val])
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
        model.fit(X_tr, y[train])
        oof[val] = model.predict(X_va)

    return regression_metrics(y, oof), quintile_bias(y, oof)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    df = load_data()
    folds, projects = lopo_folds(df)
    print(f"LOPO: {len(projects)} projects, {len(df)} rows.")

    rows = []
    for label, prefixes, lgbm_kw in CONFIGS:
        fcols = feature_cols(df, prefixes)
        print(f"\n[{label}] {len(fcols)} features ...")
        if lgbm_kw is None:
            metrics, qbias = run_ridge_lopo(df, folds, projects, fcols)
        else:
            metrics, qbias = run_lgbm_lopo(df, folds, projects, fcols, lgbm_kw)
        row = {"config": label, "n_features": len(fcols), **metrics}
        for i, b in enumerate(qbias):
            row[f"q{i + 1}_bias"] = round(b, 1)
        rows.append(row)
        print(
            f"  R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.2f}"
            f"  bias={metrics['bias']:.2f}"
            f"  Q-bias: {[round(b, 1) for b in qbias]}"
        )

    result_df = pd.DataFrame(rows)

    # Write markdown report
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("# Gradient Boosting Investigation\n\n")
        f.write(f"Dataset: `features_iter3.parquet`, {len(df)} rows, 23-project LOPO CV\n\n")
        f.write("## Aggregate metrics\n\n")
        agg_cols = ["config", "n_features", "r2", "rmse", "mae", "bias"]
        f.write(result_df[agg_cols].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Per-quintile bias (mean pred − true, tCO₂/acre)\n\n")
        qb_cols = ["config"] + [f"q{i}_bias" for i in range(1, 6)]
        f.write(result_df[qb_cols].to_markdown(index=False, floatfmt=".1f"))
        f.write("\n\n## Config details\n\n")
        for label, prefixes, lgbm_kw in CONFIGS:
            desc = str(lgbm_kw) if lgbm_kw else "RidgeCV(alphas=[0.1,1,10,100], cv=5)"
            f.write(f"- **{label}**: {desc}\n")
        f.write("\n")

    print(f"\nWrote {OUT_MD}")
    print("\n" + result_df[["config", "r2", "rmse", "q1_bias", "q5_bias"]].to_string(index=False))


if __name__ == "__main__":
    main()
