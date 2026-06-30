"""
CV ladder + learner baseline + weighting comparison for the emb-only ANEW model.

Three things, all in emb-only codec space over the 51 eligible projects:
  1. n_estimators chosen ONCE via a grouped-project holdout (early stopping), reused everywhere.
  2. Learner baseline check (S0, unweighted, LOPO): LightGBM vs untuned XGBoost vs ridge floor
     -- confirms the ceiling is feature-driven, not learner-driven.
  3. Weighting comparison: LOPO OOF per-tier RMSE for S0-S4 (LightGBM) + the leave-bloc-2-out
     frontier-transfer fold, plus a context transfer ladder (full leave-bloc-out / leave-biome-out
     for S0). All scheme scoring uses UNWEIGHTED OOF RMSE (weighted RMSE is not comparable).

Outputs (data-space): cv/comparison_matrix.parquet, cv/learner_check.parquet,
cv/per_project.parquet, cv/transfer_ladder.parquet, cv/oof_S0.parquet, cv/n_trees.json.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/anew_emb_model/cv_ladder.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as data_mod  # noqa: E402

OUT = common.DATASPACE / "agb_anew_emb_weighted_20260630/cv"
EMB = common.EMB
SCHEMES = ["S0", "S1", "S2", "S3", "S4"]
LGB_PARAMS = dict(
    num_leaves=31, learning_rate=0.05, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1
)


# ----- metrics ---------------------------------------------------------------
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def range_ratio(y, p):
    """Dynamic-range recovery: predicted p5-p95 span / true p5-p95 span."""
    ty = np.percentile(y, 95) - np.percentile(y, 5)
    tp = np.percentile(p, 95) - np.percentile(p, 5)
    return float(tp / ty) if ty > 0 else np.nan


# ----- model factories -------------------------------------------------------
def make_lgbm(n_trees):
    return lambda: lgb.LGBMRegressor(n_estimators=n_trees, **LGB_PARAMS)


def make_xgb(n_trees):
    return lambda: xgb.XGBRegressor(
        n_estimators=n_trees,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def make_ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


# ----- CV --------------------------------------------------------------------
def cv_predict(X, y, groups, make_model, sample_weight=None):
    """Out-of-fold predictions, refitting per leave-one-group-out fold (clipped at 0)."""
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        model = make_model()
        kw = {} if sample_weight is None else {"sample_weight": sample_weight[~te]}
        model.fit(X[~te], y[~te], **kw)
        oof[te] = np.clip(model.predict(X[te]), 0, None)
    return oof


def select_n_trees(df, X, y):
    """n_estimators via a grouped-project holdout (spans blocs, includes a frontier project)."""
    rng = np.random.default_rng(42)
    projects = df["project_name"].to_numpy()
    # holdout: one frontier project + a spread of others until ~15% of plots
    holdout = {"RainierGateway"}
    pool = [p for p in np.unique(projects) if p not in holdout]
    rng.shuffle(pool)
    for p in pool:
        if (np.isin(projects, list(holdout)).mean()) >= 0.15:
            break
        holdout.add(p)
    te = np.isin(projects, list(holdout))
    model = lgb.LGBMRegressor(n_estimators=2000, **LGB_PARAMS)
    model.fit(
        X[~te], y[~te], eval_set=[(X[te], y[te])], callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    return int(model.best_iteration_), sorted(holdout), float(te.mean())


def per_tier(df, y, oof, mask=None):
    """RMSE/MAE/range per DI tier on (optionally masked) OOF predictions."""
    rows = {}
    sub = df if mask is None else df[mask]
    yy = y if mask is None else y[mask]
    pp = oof if mask is None else oof[mask]
    for tier, idx in sub.groupby("tier").groups.items():
        loc = sub.index.get_indexer(idx)
        rows[tier] = {
            "n": len(idx),
            "rmse": rmse(yy[loc], pp[loc]),
            "mae": mae(yy[loc], pp[loc]),
            "range_ratio": range_ratio(yy[loc], pp[loc]),
        }
    rows["all"] = {
        "n": len(yy),
        "rmse": rmse(yy, pp),
        "mae": mae(yy, pp),
        "range_ratio": range_ratio(yy, pp),
    }
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = data_mod.load_eligible()
    X = df[EMB].astype(float).to_numpy()
    y = df["CO2"].to_numpy()
    proj = df["project_name"].to_numpy()
    bloc = df["bloc_id"].to_numpy()
    biome = df["biome_grp"].to_numpy()
    thr = data_mod.aoa_threshold()
    assert df["project_name"].nunique() == 51 and "Quinte" not in proj

    # 1. n_estimators (once, unweighted)
    n_trees, holdout, frac = select_n_trees(df, X, y)
    (OUT / "n_trees.json").write_text(
        json.dumps(
            {"n_estimators": n_trees, "holdout_projects": holdout, "holdout_frac": frac}, indent=2
        )
    )
    print(f"n_estimators = {n_trees} (grouped holdout {frac:.0%}, {len(holdout)} projects)")

    # 2. learner baseline check (S0, unweighted, LOPO)
    learners = {"lightgbm": make_lgbm(n_trees), "xgboost": make_xgb(n_trees), "ridge": make_ridge}
    learner_rows = []
    oof_lgb_s0 = None
    for name, mk in learners.items():
        oof = cv_predict(X, y, proj, mk)
        if name == "lightgbm":
            oof_lgb_s0 = oof
        for tier, m in per_tier(df, y, oof).items():
            learner_rows.append({"learner": name, "tier": tier, **m})
    pd.DataFrame(learner_rows).to_parquet(OUT / "learner_check.parquet", index=False)
    lc = pd.DataFrame(learner_rows)
    print("\nlearner check (interior / all LOPO RMSE):")
    for name in learners:
        s = lc[lc.learner == name].set_index("tier")["rmse"]
        print(f"  {name:9s} interior {s.get('interior', np.nan):6.1f}  all {s['all']:6.1f}")

    # 3. weighting comparison: LOPO per-tier + leave-bloc-2-out, S0-S4
    bloc2 = 2  # the all-frontier bloc (AK/PNW)
    comp_rows = []
    oof_by_scheme = {"S0": oof_lgb_s0}
    for s in SCHEMES:
        w, info = data_mod.build_weights(df, s)
        oof = (
            oof_lgb_s0 if s == "S0" else cv_predict(X, y, proj, make_lgbm(n_trees), sample_weight=w)
        )
        oof_by_scheme[s] = oof
        tiers = per_tier(df, y, oof)
        # frontier-transfer: leave-bloc-2-out, predict bloc 2
        oof_b2 = cv_predict(X, y, (bloc == bloc2).astype(int), make_lgbm(n_trees), sample_weight=w)
        b2 = bloc == bloc2
        comp_rows.append(
            {
                "scheme": s,
                "n_eff_frac": info["n_eff_frac"],
                "max_ratio": info["max_ratio"],
                "rmse_interior": tiers.get("interior", {}).get("rmse", np.nan),
                "rmse_regional_frontier": tiers.get("regional_frontier", {}).get("rmse", np.nan),
                "rmse_self_standing_frontier": tiers.get("self_standing_frontier", {}).get(
                    "rmse", np.nan
                ),
                "range_regional_frontier": tiers.get("regional_frontier", {}).get(
                    "range_ratio", np.nan
                ),
                "rmse_bloc2_transfer": rmse(y[b2], oof_b2[b2]),
                "rmse_all": tiers["all"]["rmse"],
            }
        )
    comp = pd.DataFrame(comp_rows)
    comp.to_parquet(OUT / "comparison_matrix.parquet", index=False)
    pd.DataFrame(
        {
            "CO2": y,
            "oof_S0": oof_by_scheme["S0"],
            "project_name": proj,
            "tier": df["tier"],
            "di_lopo": df["di_lopo"],
        }
    ).to_parquet(OUT / "oof_S0.parquet", index=False)
    print("\nweighting comparison (LOPO per-tier RMSE, unweighted scoring):")
    print(comp.round(2).to_string(index=False))

    # per-project table (S0 LOPO + leave-bloc-out)
    oof_bloc = cv_predict(X, y, bloc, make_lgbm(n_trees))
    rank = pd.read_parquet(data_mod.APPLIC / "analysis/project_di_ranking.parquet")
    regdep = dict(zip(rank["project_name"], rank["regional_dependence"]))
    pp_rows = []
    for p in np.unique(proj):
        m = proj == p
        rd = regdep.get(p, np.nan)
        cls = (
            "interior"
            if df.loc[m, "inside_aoa"].mean() > 0.5
            else "regional_frontier"
            if rd >= 0.3
            else "self_standing_frontier"
        )
        pp_rows.append(
            {
                "project_name": p,
                "biome_grp": df.loc[m, "biome_grp"].iloc[0],
                "bloc_id": int(bloc[m][0]),
                "n": int(m.sum()),
                "regional_dependence": rd,
                "regdep_class": cls,
                "rmse_lopo": rmse(y[m], oof_lgb_s0[m]),
                "rmse_bloc": rmse(y[m], oof_bloc[m]),
                "median_di_lopo": float(df.loc[m, "di_lopo"].median()),
                "out_of_aoa_bloc_fold": bool(df.loc[m, "di_bloc"].median() > thr),
            }
        )
    pd.DataFrame(pp_rows).sort_values("median_di_lopo").to_parquet(
        OUT / "per_project.parquet", index=False
    )

    # transfer ladder (S0): full leave-bloc-out + leave-biome-out
    ladder_rows = [{"rung": "lopo", "group": "all", "rmse": rmse(y, oof_lgb_s0), "n": len(y)}]
    for grp_name, groups in [("bloc", bloc), ("biome", biome)]:
        oof_g = cv_predict(X, y, groups, make_lgbm(n_trees))
        for g in np.unique(groups):
            mg = groups == g
            out_aoa = bool(df.loc[mg, "di_bloc"].median() > thr)
            ladder_rows.append(
                {
                    "rung": f"leave_{grp_name}_out",
                    "group": str(g),
                    "rmse": rmse(y[mg], oof_g[mg]),
                    "n": int(mg.sum()),
                    "out_of_aoa_fold": out_aoa,
                }
            )
    pd.DataFrame(ladder_rows).to_parquet(OUT / "transfer_ladder.parquet", index=False)
    print(
        f"\nbloc-2 frontier-transfer RMSE (S0) = "
        f"{comp.loc[comp.scheme == 'S0', 'rmse_bloc2_transfer'].iloc[0]:.1f}"
    )
    print(f"Saved CV artifacts to {OUT}")


if __name__ == "__main__":
    main()
