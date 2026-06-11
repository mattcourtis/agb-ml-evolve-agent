"""Iteration 1 — Analog Selection Actor (Tier-1, no model training).

Measures each ANEW plot/project's similarity to Ireland in the native-float 64-D AEF
embedding space, defines candidate subsets S0-S4, and computes Tier-1 OOD diagnostics.

seed 42. British English. plt.savefig only. uv run.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_ireland_biomass_regression_20260608"
)
POOL = ROOT / "preprocessing" / "iter1_pool_embeddings.parquet"
FIG = ROOT / "evaluation" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EMB = [f"emb_{i:02d}" for i in range(64)]

# Irish-relevant biomass band (tCO2/acre). Research: rotation-end ~150-376 Mg/ha => x0.6977.
# Young restock down to ~0; rotation-end primary anchor ~262 (376 Mg/ha). Use 0..262 as the
# Irish-relevant span; coverage measured against the high-relevant band [105, 262] (saturation
# onset 150 Mg/ha = ~105 tCO2/acre up to rotation-end 262).
BAND_LO, BAND_HI = 0.0, 262.0  # full Irish-plausible CO2 span
HIGH_LO, HIGH_HI = 105.0, 262.0  # Irish-relevant high band (saturation-onset -> rotation-end)


def load():
    df = pd.read_parquet(POOL)
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def mahalanobis_to_ireland(X_anew, X_ire):
    """Mahalanobis distance of each ANEW plot to the Irish distribution (Ledoit-Wolf cov)."""
    mu = X_ire.mean(axis=0)
    lw = LedoitWolf().fit(X_ire)
    VI = lw.precision_
    d = X_anew - mu
    m2 = np.einsum("ij,jk,ik->i", d, VI, d)
    m2 = np.clip(m2, 0, None)
    return np.sqrt(m2)


def mahalanobis_self(X, mu, VI):
    d = X - mu
    m2 = np.clip(np.einsum("ij,jk,ik->i", d, VI, d), 0, None)
    return np.sqrt(m2)


def rbf_mmd2(X, Y, gamma):
    """Unbiased-ish biased MMD^2 with RBF kernel."""

    def k(A, B):
        aa = (A * A).sum(1)[:, None]
        bb = (B * B).sum(1)[None, :]
        d2 = aa + bb - 2 * A @ B.T
        return np.exp(-gamma * np.clip(d2, 0, None))

    kxx = k(X, X).mean()
    kyy = k(Y, Y).mean()
    kxy = k(X, Y).mean()
    return float(kxx + kyy - 2 * kxy)


def energy_distance(X, Y):
    def pdist_mean(A, B):
        aa = (A * A).sum(1)[:, None]
        bb = (B * B).sum(1)[None, :]
        d2 = np.clip(aa + bb - 2 * A @ B.T, 0, None)
        return np.sqrt(d2).mean()

    return float(2 * pdist_mean(X, Y) - pdist_mean(X, X) - pdist_mean(Y, Y))


def median_heuristic_gamma(X, Y, cap=2000):
    Z = np.vstack([X, Y])
    if len(Z) > cap:
        idx = rng.choice(len(Z), cap, replace=False)
        Z = Z[idx]
    aa = (Z * Z).sum(1)[:, None]
    d2 = aa + aa.T - 2 * Z @ Z.T
    d2 = d2[np.triu_indices(len(Z), 1)]
    med = np.median(np.sqrt(np.clip(d2, 0, None)))
    return 1.0 / (2 * med**2)


def main():
    df = load()
    ire = df[df.source == "ireland"].copy()
    anew = df[df.source == "anew"].copy()
    Xire = ire[EMB].to_numpy(float)
    Xanew = anew[EMB].to_numpy(float)
    print(f"ireland={len(ire)} anew={len(anew)} projects={anew.project_name.nunique()}")

    # ---- 1a. Domain classifier Ireland-likeness (USA vs Ireland) on FULL pool ----
    y = (df.source == "ireland").astype(int).to_numpy()
    X = df[EMB].to_numpy(float)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    auc_full = roc_auc_score(y, proba)
    print(f"[1a] full-pool USA-vs-Ireland AUC = {auc_full:.6f}")
    df["p_ireland"] = proba
    p_anew = df.loc[df.source == "anew", "p_ireland"].to_numpy()

    # ---- 1b. Mahalanobis to Irish distribution ----
    mahal_to_ire = mahalanobis_to_ireland(Xanew, Xire)

    # ---- 1c. kNN distance to Irish manifold ----
    k = 5
    nn = NearestNeighbors(n_neighbors=k).fit(Xire)
    knn_d, _ = nn.kneighbors(Xanew)
    knn_mean = knn_d.mean(axis=1)

    # assemble per-plot scores
    scores = anew[["row_id", "project_name", "CO2", "survey_year", "ECO_NAME", "BIOME_NAME"]].copy()
    scores["p_ireland"] = p_anew
    scores["mahal_to_ireland"] = mahal_to_ire
    scores["knn_dist_to_ireland"] = knn_mean

    # ---- 1d. Per-project distributional distance (MMD + energy) ----
    gamma = median_heuristic_gamma(Xanew, Xire)
    proj_rows = []
    for proj, g in anew.groupby("project_name"):
        Xp = g[EMB].to_numpy(float)
        mmd2 = rbf_mmd2(Xp, Xire, gamma)
        ed = energy_distance(Xp, Xire)
        proj_rows.append(
            dict(
                project_name=proj,
                n_plots=len(g),
                mmd2=mmd2,
                energy=ed,
                mean_p_ireland=float(
                    g.index.map(lambda i: df.loc[i, "p_ireland"]).to_series().mean()
                ),
                mean_mahal=float(mahal_to_ire[anew.project_name.values == proj].mean()),
                mean_knn=float(knn_mean[anew.project_name.values == proj].mean()),
                co2_min=float(g.CO2.min()),
                co2_med=float(g.CO2.median()),
                co2_max=float(g.CO2.max()),
                eco=g.ECO_NAME.mode().iloc[0] if len(g.ECO_NAME.mode()) else "NA",
            )
        )
    proj = pd.DataFrame(proj_rows)
    # ranks: lower mmd/energy/mahal/knn = more similar; higher p_ireland = more similar
    proj["rank_mmd"] = proj["mmd2"].rank()
    proj["rank_energy"] = proj["energy"].rank()
    proj["rank_mahal"] = proj["mean_mahal"].rank()
    proj["rank_knn"] = proj["mean_knn"].rank()
    proj["rank_p"] = (-proj["mean_p_ireland"]).rank()
    proj["mean_rank"] = proj[["rank_mmd", "rank_energy", "rank_mahal", "rank_knn", "rank_p"]].mean(
        1
    )
    proj = proj.sort_values("mmd2").reset_index(drop=True)

    # rank correlations among measures (project-level)
    rc = {}
    for a in ["rank_mmd", "rank_energy", "rank_mahal", "rank_knn", "rank_p"]:
        for b in ["rank_mmd", "rank_energy", "rank_mahal", "rank_knn", "rank_p"]:
            if a < b:
                rho = spearmanr(proj[a], proj[b]).correlation
                rc[f"{a}|{b}"] = round(float(rho), 3)
    print("[1d] project-level rank correlations:")
    for kk, vv in rc.items():
        print(f"   {kk}: {vv}")

    # ============================================================
    # 2. CANDIDATE SUBSETS
    # ============================================================
    anew = anew.reset_index(drop=True)
    # align score arrays to anew order
    s = scores.set_index("row_id")
    anew["p_ireland"] = anew["row_id"].map(s["p_ireland"])
    anew["mahal_to_ireland"] = anew["row_id"].map(s["mahal_to_ireland"])
    anew["knn_dist_to_ireland"] = anew["row_id"].map(s["knn_dist_to_ireland"])

    subsets = {}

    # S0 full CONUS
    subsets["S0_full"] = anew["row_id"].tolist()

    # S1 climate+ecoregion heuristic: maritime New England-Acadian + Pacific/Cascades/Alaskan conifer
    s1_ecos = {
        "New England-Acadian forests",
        "Eastern Cascades forests",
        "Central-Southern Cascades Forests",
        "Northern Pacific Alaskan coastal forests",
        "Interior Yukon-Alaska alpine tundra",  # Doyon
        "Blue Mountains forests",  # LongviewRanch Pacific NW interior conifer
    }
    s1_projects_hint = {"HighCascades", "RainierGateway", "Kootznoowoo", "Doyon", "LongviewRanch"}
    s1_mask = anew.ECO_NAME.isin(s1_ecos) | anew.project_name.isin(s1_projects_hint)
    subsets["S1_climate_ecoregion"] = anew.loc[s1_mask, "row_id"].tolist()
    s1_projs = sorted(anew.loc[s1_mask, "project_name"].unique())

    # S2 project-level embedding-nearest: top-k projects by MMD to Ireland.
    # Choose k so biomass coverage of the Irish high band [105,262] is retained AND >=5 projects
    # remain for LOPO. Grow k from the MMD ranking until co2_max spans HIGH_HI; min k=8.
    proj_by_mmd = proj.sort_values("mmd2").reset_index(drop=True)
    k = 8
    while k < len(proj_by_mmd):
        top = proj_by_mmd.head(k)
        if top.co2_max.max() >= HIGH_HI and top.co2_med.max() >= HIGH_LO:
            break
        k += 1
    s2_projs = proj_by_mmd.head(k)["project_name"].tolist()
    subsets["S2_project_nearest"] = anew.loc[anew.project_name.isin(s2_projs), "row_id"].tolist()

    # S3 plot-level embedding-nearest, COVERAGE-CONSTRAINED.
    # Rule: rank ANEW plots by Ireland-likeness (low Mahalanobis-to-Ireland, primary; tie-break
    # high p_ireland). Take the most-similar plots, but ENFORCE that within each decile of the
    # Irish-relevant band [0,262] we keep at least the top `q` most-similar plots so the biomass
    # range is not truncated. Target overall fraction ~30% of pool.
    anew_s3 = anew.copy()

    # composite similarity rank (lower = more similar): blend mahal + knn (z-scored)
    def z(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    anew_s3["sim_rank"] = (
        z(anew_s3["mahal_to_ireland"]) + z(anew_s3["knn_dist_to_ireland"]) - z(anew_s3["p_ireland"])
    )
    band = anew_s3[(anew_s3.CO2 >= BAND_LO) & (anew_s3.CO2 <= BAND_HI)].copy()
    band["decile"] = pd.qcut(band.CO2, 10, labels=False, duplicates="drop")
    frac_per_decile = 0.30  # keep most-similar 30% within each Irish-relevant decile
    keep_ids = []
    for d, gd in band.groupby("decile"):
        gd = gd.sort_values("sim_rank")
        n_keep = max(1, int(np.ceil(frac_per_decile * len(gd))))
        keep_ids.extend(gd.head(n_keep)["row_id"].tolist())
    subsets["S3_plot_coverage"] = sorted(set(keep_ids))

    # S4 importance weights = density-ratio Ireland-likeness over full pool (no hard cut).
    # w_i = p_ireland / (1 - p_ireland), clipped, normalised to mean 1 across the pool.
    p = np.clip(anew["p_ireland"].to_numpy(), 1e-6, 1 - 1e-6)
    w = p / (1 - p)
    w = np.clip(w, np.quantile(w, 0.0), np.quantile(w, 0.99))  # clip top 1% to tame heavy tail
    w = w / w.mean()
    anew["importance_weight"] = w
    scores = scores.merge(anew[["row_id", "importance_weight"]], on="row_id", how="left")

    # subset summary
    def subset_summary(ids):
        g = anew[anew.row_id.isin(ids)]
        co2 = g.CO2
        # coverage of high band [105,262]
        in_hi = (co2 >= HIGH_LO) & (co2 <= HIGH_HI)
        # fraction of the high band spanned: by deciles of [105,262]
        edges = np.linspace(HIGH_LO, HIGH_HI, 11)
        spanned = sum(((co2 >= edges[i]) & (co2 < edges[i + 1])).any() for i in range(10)) / 10.0
        return dict(
            n_plots=int(len(g)),
            n_projects=int(g.project_name.nunique()),
            co2_min=float(co2.min()),
            co2_med=float(co2.median()),
            co2_max=float(co2.max()),
            high_band_decile_coverage=round(spanned, 3),
            frac_plots_in_high_band=round(float(in_hi.mean()), 3),
        )

    subset_meta = {name: subset_summary(ids) for name, ids in subsets.items()}
    subset_meta["S2_project_nearest"]["k_projects"] = k
    print("\n[2] subset sizes / coverage:")
    for nm, mt in subset_meta.items():
        print(f"   {nm}: {mt}")
    print(f"   S1 projects ({len(s1_projs)}): {s1_projs}")
    print(f"   S2 top-{k} projects: {s2_projs}")

    # ============================================================
    # 3. TIER-1 OOD DIAGNOSTICS (Ireland vs each subset)
    # ============================================================
    tier1 = {}
    for name in ["S0_full", "S1_climate_ecoregion", "S2_project_nearest", "S3_plot_coverage"]:
        ids = subsets[name]
        sub = anew[anew.row_id.isin(ids)]
        Xs = sub[EMB].to_numpy(float)

        # (a) Mahalanobis: Irish distances vs subset's OWN 99th-pct radius
        mu = Xs.mean(axis=0)
        lw = LedoitWolf().fit(Xs)
        VI = lw.precision_
        d_self = mahalanobis_self(Xs, mu, VI)
        r99 = float(np.quantile(d_self, 0.99))
        d_ire = mahalanobis_self(Xire, mu, VI)
        frac_beyond = float((d_ire > r99).mean())

        # (b) domain classifier AUC subset-vs-Ireland (5-fold)
        Xd = np.vstack([Xs, Xire])
        yd = np.r_[np.zeros(len(Xs)), np.ones(len(Xire))]
        clf2 = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        pr = cross_val_predict(clf2, Xd, yd, cv=skf2, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(yd, pr))

        # (c) biomass coverage of Irish high band
        cov = subset_meta[name]["high_band_decile_coverage"]

        tier1[name] = dict(
            n_plots=int(len(sub)),
            n_projects=int(sub.project_name.nunique()),
            subset_radius_99pct=round(r99, 4),
            ireland_mahal_min=round(float(d_ire.min()), 4),
            ireland_mahal_median=round(float(np.median(d_ire)), 4),
            ireland_mahal_max=round(float(d_ire.max()), 4),
            frac_ireland_beyond_99pct=round(frac_beyond, 4),
            domain_auc_subset_vs_ireland=round(auc, 6),
            high_band_decile_coverage=cov,
            co2_max=round(float(sub.CO2.max()), 2),
        )
        print(
            f"\n[3] {name}: beyond99={frac_beyond:.3f} AUC={auc:.4f} cov={cov} n={len(sub)} proj={sub.project_name.nunique()}"
        )

    # ---- figures: PCA overlap Ireland vs each subset ----
    pca = PCA(n_components=2, random_state=SEED).fit(Xanew)
    ire_pca = pca.transform(Xire)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, name in zip(
        axes, ["S0_full", "S1_climate_ecoregion", "S2_project_nearest", "S3_plot_coverage"]
    ):
        sub = anew[anew.row_id.isin(subsets[name])]
        sp = pca.transform(sub[EMB].to_numpy(float))
        ax.scatter(
            sp[:, 0], sp[:, 1], s=4, alpha=0.25, c="tab:blue", label=f"{name} (n={len(sub)})"
        )
        ax.scatter(
            ire_pca[:, 0], ire_pca[:, 1], s=14, alpha=0.8, c="tab:red", label="Ireland (141)"
        )
        ax.set_title(name)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("PCA overlap: Ireland vs candidate subsets (native-float 64-D AEF)")
    fig.tight_layout()
    fig.savefig(FIG / "iter1_pca_overlap_subsets.png", dpi=120)
    plt.close(fig)

    # UMAP if available
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=SEED)
        allX = np.vstack([Xanew, Xire])
        emb2 = reducer.fit_transform(allX)
        ua = emb2[: len(Xanew)]
        ui = emb2[len(Xanew) :]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        anew_idx = {rid: i for i, rid in enumerate(anew.row_id.values)}
        for ax, name in zip(
            axes, ["S0_full", "S1_climate_ecoregion", "S2_project_nearest", "S3_plot_coverage"]
        ):
            idxs = [anew_idx[r] for r in subsets[name]]
            ax.scatter(ua[idxs, 0], ua[idxs, 1], s=4, alpha=0.25, c="tab:blue", label=name)
            ax.scatter(ui[:, 0], ui[:, 1], s=14, alpha=0.8, c="tab:red", label="Ireland")
            ax.set_title(name)
            ax.legend(fontsize=8)
        fig.suptitle("UMAP overlap: Ireland vs candidate subsets")
        fig.tight_layout()
        fig.savefig(FIG / "iter1_umap_overlap_subsets.png", dpi=120)
        plt.close(fig)
        umap_ok = True
    except Exception as e:
        print(f"[fig] UMAP skipped: {e}")
        umap_ok = False

    # project similarity bar (top 15 by MMD)
    fig, ax = plt.subplots(figsize=(10, 7))
    topp = proj.sort_values("mmd2").head(15)
    ax.barh(topp.project_name[::-1], topp.mmd2[::-1], color="tab:green")
    ax.set_xlabel("MMD^2 to Ireland (lower = more similar)")
    ax.set_title("Top-15 ANEW projects most similar to Ireland (RBF-MMD, native-float AEF)")
    fig.tight_layout()
    fig.savefig(FIG / "iter1_project_mmd_ranking.png", dpi=120)
    plt.close(fig)

    # ============================================================
    # WRITE OUTPUTS
    # ============================================================
    # similarity scores parquet
    scores_out = scores.rename(columns={"row_id": "anew_row_id"})
    scores_out.to_parquet(ROOT / "preprocessing" / "iter1_similarity_scores.parquet", index=False)

    # subset membership json (project lists + plot row_ids)
    membership = {
        "seed": SEED,
        "irish_band_tco2_acre": {"full": [BAND_LO, BAND_HI], "high_relevant": [HIGH_LO, HIGH_HI]},
        "subsets": {},
    }
    for name, ids in subsets.items():
        g = anew[anew.row_id.isin(ids)]
        membership["subsets"][name] = {
            "n_plots": int(len(ids)),
            "n_projects": int(g.project_name.nunique()),
            "projects": sorted(g.project_name.unique().tolist()),
            "plot_row_ids": [int(x) for x in ids],
        }
    membership["subsets"]["S2_project_nearest"]["k"] = int(k)
    membership["S4_importance_weights_note"] = (
        "Per-plot density-ratio weights saved in iter1_similarity_scores.parquet "
        "(importance_weight col; mean 1, top-1% clipped). Full pool, no hard cut."
    )
    with open(ROOT / "configs" / "iter1_subset_membership.json", "w") as f:
        json.dump(membership, f, indent=2)

    # tier1 OOD yaml
    import yaml

    yout = {
        "meta": {
            "seed": SEED,
            "space": "native-float 64-D AEF (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL)",
            "n_ireland": int(len(ire)),
            "n_anew": int(len(anew)),
            "irish_high_band_tco2_acre": [HIGH_LO, HIGH_HI],
            "baseline_iter0": {
                "frac_ireland_beyond_99pct_training_radius": 1.0,
                "domain_auc_usa_vs_ireland": 0.999998,
            },
            "full_pool_domain_auc": round(float(auc_full), 6),
        },
        "project_rank_correlations_spearman": rc,
        "tier1_ood_by_subset": tier1,
        "figures": [
            "evaluation/figures/iter1_pca_overlap_subsets.png",
            "evaluation/figures/iter1_project_mmd_ranking.png",
        ]
        + (["evaluation/figures/iter1_umap_overlap_subsets.png"] if umap_ok else []),
    }

    def to_native(o):
        if isinstance(o, dict):
            return {k: to_native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_native(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    with open(ROOT / "evaluation" / "iter1_tier1_ood.yaml", "w") as f:
        yaml.safe_dump(to_native(yout), f, sort_keys=False)

    # save project ranking table for the md
    proj.to_csv(ROOT / "preprocessing" / "_iter1_project_ranking.csv", index=False)

    print("\n=== WROTE OUTPUTS ===")
    print("similarity_scores, subset_membership.json, iter1_tier1_ood.yaml, figures")
    return df, anew, ire, scores, proj, subset_meta, tier1, auc_full, rc, k, s1_projs, s2_projs


if __name__ == "__main__":
    out = main()
