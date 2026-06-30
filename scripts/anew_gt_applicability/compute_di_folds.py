"""
ANEW GT applicability — self-referential DI & AOA over the whole training space.

Turns the trust toolkit's DI/AOA lens on the ANEW ground-truth itself instead of on the
deployed-23 reference. We score every plot of all 51 eligible projects (52 ANEW minus the
Quinte label outlier) against the *rest* of the GT cloud, in emb-only codec space, to
answer: which projects are interior (redundant) vs frontier (unique), and how much each
project's applicability depends on its regional neighbours.

Two fold-aware DI passes, both via di.fit() with a different grouping array (no new DI math):
  - LOPO       (groups = project_name): each plot's DI = NN weighted distance to OTHER
               projects. dsp.threshold_cast (Q75 + 1.5*IQR) is the GT AOA boundary.
  - leave-bloc (groups = spatial KMeans bloc): removes a whole region at once. The lift
               di_bloc - di_lopo is each project's regional dependence.

Robustness cross-check (mirrors the Ireland ranking validation): Spearman rho of the
weighted-LOPO project ranking vs (a) unweighted DI and (b) Mahalanobis DI.

Outputs (data-space, gitignored):
  analysis/project_di_ranking.parquet   one row per project, sorted by median_di_lopo
  analysis/plot_level_di.parquet        per-plot di_lopo/di_bloc/inside_aoa (+ lon/lat/CO2)
  analysis/thresholds.json              threshold, p95/p99, K, robustness rho
  preprocessing/bloc_assignments.parquet  project -> bloc (centroid in EPSG:5070)

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/anew_gt_applicability/compute_di_folds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402
import di as di_mod  # noqa: E402

OUT = common.DATASPACE / "agb_anew_gt_applicability_20260626"
DROP_PROJECTS = ["Quinte"]  # label outlier (CO2 median ~2x cohort, robust-z 5.0)
K_BLOCS = 6  # spatial blocs; validated so every bloc has >= MIN_BLOC projects
MIN_BLOC = 3


def build_blocs(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """KMeans on per-project centroids reprojected to EPSG:5070 (Albers, metres)."""
    cent = df.groupby("project_name")[["lon", "lat"]].mean().reset_index()
    tf = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    cent["x"], cent["y"] = tf.transform(cent["lon"].to_numpy(), cent["lat"].to_numpy())
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cent["bloc_id"] = km.fit_predict(cent[["x", "y"]].to_numpy())
    return cent[["project_name", "lon", "lat", "x", "y", "bloc_id"]]


def main() -> None:
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "preprocessing").mkdir(parents=True, exist_ok=True)

    canon = common.load_canonical()
    df = canon[~canon["project_name"].isin(DROP_PROJECTS)].reset_index(drop=True)
    feats = common.EMB
    X = df[feats].astype(float).to_numpy()
    ok = np.isfinite(X).all(1)
    df, X = df[ok].reset_index(drop=True), X[ok]
    proj = df["project_name"].to_numpy()
    n_proj = df["project_name"].nunique()
    print(f"{len(df)} plots, {n_proj} projects (dropped {DROP_PROJECTS})")

    # spatial blocs (validate the >= MIN_BLOC floor; nudge K down if violated)
    k = K_BLOCS
    blocs = build_blocs(df, k)
    while (blocs["bloc_id"].value_counts() < MIN_BLOC).any() and k > 2:
        k -= 1
        print(f"  bloc < {MIN_BLOC} projects at K={k + 1}; retrying K={k}")
        blocs = build_blocs(df, k)
    blocs.to_parquet(OUT / "preprocessing/bloc_assignments.parquet", index=False)
    bloc_of = dict(zip(blocs["project_name"], blocs["bloc_id"]))
    bloc_id = np.array([bloc_of[p] for p in proj])
    df["bloc_id"] = bloc_id
    print(f"  K={k} blocs, sizes: {blocs['bloc_id'].value_counts().sort_index().to_dict()}")

    # weighted CAST DI under the two folds (di.fit's train_di IS the fold-aware DI)
    w = common.gain_weights("embonly")
    dsp_lopo = di_mod.fit(X, proj, feats, w)
    dsp_bloc = di_mod.fit(X, bloc_id, feats, w)
    df["di_lopo"] = dsp_lopo.train_di
    df["di_bloc"] = dsp_bloc.train_di
    df["inside_aoa"] = df["di_lopo"] <= dsp_lopo.threshold_cast

    # robustness: project ranking under unweighted CAST and Mahalanobis (LOPO grouping)
    dsp_unw = di_mod.fit(X, proj, feats, np.ones(len(feats)))
    di_unw = dsp_unw.train_di
    # fold-aware Mahalanobis: per project, distance fit on the OTHER projects
    di_maha = np.empty(len(X))
    for p in np.unique(proj):
        m = proj == p
        _, di_maha[m] = di_mod.mahalanobis_emb(X[~m], X[m])

    plot_cols = [
        "project_name",
        "BIOME_NAME",
        "lon",
        "lat",
        "bloc_id",
        "di_lopo",
        "di_bloc",
        "inside_aoa",
        "CO2",
    ]
    df[plot_cols].to_parquet(OUT / "analysis/plot_level_di.parquet", index=False)

    # per-project ranking
    g = df.groupby("project_name")
    rank = pd.DataFrame(
        {
            "biome": g["BIOME_NAME"].first(),
            "bloc_id": g["bloc_id"].first(),
            "n": g.size(),
            "median_di_lopo": g["di_lopo"].median(),
            "iqr_di_lopo": g["di_lopo"].quantile(0.75) - g["di_lopo"].quantile(0.25),
            "pct_inside_aoa": 100 * g["inside_aoa"].mean(),
            "median_di_bloc": g["di_bloc"].median(),
            "co2_median": g["CO2"].median(),
        }
    ).reset_index()
    rank["regional_dependence"] = rank["median_di_bloc"] - rank["median_di_lopo"]
    rank = rank.sort_values("median_di_lopo").reset_index(drop=True)
    rank.to_parquet(OUT / "analysis/project_di_ranking.parquet", index=False)

    # robustness rho on per-project median DI
    med = pd.DataFrame({"p": proj, "w": dsp_lopo.train_di, "u": di_unw, "m": di_maha})
    medg = med.groupby("p").median()
    rho_unw = spearmanr(medg["w"], medg["u"]).correlation
    rho_maha = spearmanr(medg["w"], medg["m"]).correlation

    summary = {
        "n_plots": int(len(df)),
        "n_projects": int(n_proj),
        "dropped": DROP_PROJECTS,
        "feature_space": "embonly-64-codec",
        "k_blocs": int(k),
        "bloc_sizes": blocs["bloc_id"].value_counts().sort_index().to_dict(),
        "threshold_cast": dsp_lopo.threshold_cast,
        "p95": dsp_lopo.p95,
        "p99": dsp_lopo.p99,
        "spearman_weighted_vs_unweighted": float(rho_unw),
        "spearman_weighted_vs_mahalanobis": float(rho_maha),
    }
    (OUT / "analysis/thresholds.json").write_text(json.dumps(summary, indent=2))

    print(f"\nAOA threshold (LOPO) = {dsp_lopo.threshold_cast:.3f}")
    print(
        f"Robustness rho: weighted-vs-unweighted {rho_unw:.3f}, "
        f"weighted-vs-Mahalanobis {rho_maha:.3f}"
    )
    print("\nMost interior (lowest DI):")
    print(
        rank.head(5)[["project_name", "biome", "n", "median_di_lopo", "pct_inside_aoa"]].to_string(
            index=False
        )
    )
    print("\nMost frontier (highest DI):")
    print(
        rank.tail(5)[["project_name", "biome", "n", "median_di_lopo", "pct_inside_aoa"]].to_string(
            index=False
        )
    )
    print(f"\nSaved analysis + bloc assignments to {OUT}")


if __name__ == "__main__":
    main()
