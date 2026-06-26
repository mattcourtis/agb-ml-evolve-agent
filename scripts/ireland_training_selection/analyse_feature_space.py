"""
PART 1 — Ireland-anchored feature-space + GT analysis.

Ranks every ANEW project by how close its AlphaEarth embeddings sit to IRELAND's, so we can
later pick whole projects to train an Ireland-bound model. Ireland is the *reference cloud*:
we fit an importance-weighted CAST DI space on Ireland's 141 plots (leave-one-out self-DI for
the threshold) and score each ANEW project's plots against it. Lower DI = looks more like Ireland.

Everything runs EMB-ONLY (64-dim) — the only feature space all 52 ANEW projects AND Ireland
share (codec). Cross-checks the CAST-DI ranking against Mahalanobis-DI and a weighted-centroid
distance so the ordering is not metric-dependent. Screens biomass (CO2) for erroneous projects
(auto-drops Quinte; flags any others) — Ireland has no CO2 ground-truth, so CO2 only bounds
plausibility, it does not match Ireland.

Outputs:
  data-space  …/agb_ireland_training_selection_20260626/analysis/project_ranking.parquet
  report      experiments/agb_ireland_training_selection_20260626/analysis/feature_space.md
  figures     experiments/agb_ireland_training_selection_20260626/figures/{pca,di_bar,co2_box}.png

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/ireland_training_selection/analyse_feature_space.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402
import di as di_mod  # noqa: E402

EMB = common.EMB
EXP = common.REPO / "experiments/agb_ireland_training_selection_20260626"
FIG_DIR = EXP / "figures"
ANALYSIS_DIR = EXP / "analysis"
DATA_OUT = common.DATASPACE / "agb_ireland_training_selection_20260626/analysis"
IRE_FEATURES = (
    common.DATASPACE
    / "agb_ireland_biomass_regression_20260608/preprocessing/ireland_features.parquet"
)


def load_ireland_emb() -> np.ndarray:
    ire = pd.read_parquet(IRE_FEATURES)
    X = ire[EMB].astype(float).to_numpy()
    return X[np.isfinite(X).all(1)]


def fit_ireland_di(X_ire: np.ndarray) -> di_mod.DISpace:
    """CAST DI space anchored on Ireland; threshold from leave-one-out self-DI."""
    w = common.gain_weights("embonly")
    mu, sd = X_ire.mean(0), X_ire.std(0) + 1e-9
    Zt = ((X_ire - mu) / sd) * w
    dbar = di_mod._mean_pairwise(Zt)

    # leave-one-out self-DI: nearest OTHER Ireland point (2-NN, drop self at distance 0)
    nn2 = NearestNeighbors(n_neighbors=2).fit(Zt)
    d, _ = nn2.kneighbors(Zt, n_neighbors=2)
    self_di = d[:, 1] / dbar

    q75, q25 = np.percentile(self_di, [75, 25])
    thr = float(q75 + 1.5 * (q75 - q25))
    nn_all = NearestNeighbors(n_neighbors=1).fit(Zt)
    return di_mod.DISpace(
        features=EMB,
        mu=mu,
        sd=sd,
        w=w,
        dbar=dbar,
        threshold_cast=thr,
        p95=float(np.percentile(self_di, 95)),
        p99=float(np.percentile(self_di, 99)),
        train_di=self_di,
        _nn=nn_all,
        _zt=Zt,
    )


def rank_projects(dsp: di_mod.DISpace, X_ire: np.ndarray) -> pd.DataFrame:
    """Per-project DI-to-Ireland + Mahalanobis + centroid distance, plus CO2/biome."""
    canon = common.load_canonical()
    # Mahalanobis reference is Ireland; query is each project.
    ire_mu = X_ire.mean(0)
    ire_cov_inv = np.linalg.pinv(np.cov(X_ire, rowvar=False))

    def maha(X):
        diff = X - ire_mu
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, ire_cov_inv, diff))

    ire_centroid = dsp.transform(X_ire).mean(0)

    rows = []
    for proj, g in canon.groupby("project_name"):
        X = g[EMB].astype(float).to_numpy()
        ok = np.isfinite(X).all(1)
        X = X[ok]
        di = dsp.di(X)
        cdist = np.linalg.norm(dsp.transform(X).mean(0) - ire_centroid)
        rows.append(
            {
                "project_name": proj,
                "biome": g["BIOME_NAME"].iloc[0],
                "n": int(ok.sum()),
                "median_di_to_ireland": float(np.median(di)),
                "iqr_di": float(np.percentile(di, 75) - np.percentile(di, 25)),
                "pct_inside_ireland_aoa": float(100 * (di <= dsp.threshold_cast).mean()),
                "median_maha_to_ireland": float(np.median(maha(X))),
                "centroid_dist_to_ireland": float(cdist),
                "co2_median": float(g["CO2"].median()),
                "co2_max": float(g["CO2"].max()),
                "co2_min": float(g["CO2"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("median_di_to_ireland").reset_index(drop=True)


def screen_co2(rank: pd.DataFrame) -> pd.DataFrame:
    """Flag erroneous/implausible projects on biomass. Always flag Quinte explicitly."""
    med = rank["co2_median"]
    mad = float(np.median(np.abs(med - med.median()))) or 1.0
    rob_z = (med - med.median()) / (1.4826 * mad)
    rank = rank.copy()
    rank["co2_robust_z"] = rob_z
    # implausible if median biomass is a strong upper outlier vs the cohort, OR the named bad project
    rank["co2_flag"] = (rob_z > 3.5) | (rank["project_name"] == "Quinte")
    return rank


def fig_pca(dsp, X_ire, rank, cohort, path):
    """PCA of all projects + Ireland in the Ireland-anchored transform space."""
    canon = common.load_canonical()
    Zall, labels = [], []
    for proj, g in canon.groupby("project_name"):
        X = g[EMB].astype(float).to_numpy()
        X = X[np.isfinite(X).all(1)]
        Zall.append(dsp.transform(X))
        labels += [proj] * len(X)
    Zire = dsp.transform(X_ire)
    Z = np.vstack(Zall + [Zire])
    labels = np.array(labels + ["Ireland"] * len(Zire))

    Zc = Z - Z.mean(0)
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt[:2].T
    var = (S**2 / (S**2).sum())[:2] * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    cohort_set = set(cohort)
    far = ~np.isin(labels, list(cohort_set) + ["Ireland"])
    near = np.isin(labels, list(cohort_set))
    ire = labels == "Ireland"
    ax.scatter(P[far, 0], P[far, 1], s=5, alpha=0.2, c="#bbb", label="other ANEW projects")
    ax.scatter(
        P[near, 0],
        P[near, 1],
        s=10,
        alpha=0.5,
        c="#1f77b4",
        label=f"core (closest, n={len(cohort)})",
    )
    ax.scatter(P[ire, 0], P[ire, 1], s=40, alpha=0.85, c="#d62728", marker="X", label="Ireland")
    ax.set_xlabel(f"PC1 ({var[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.0f}%)")
    ax.set_title(
        "Ireland-anchored embedding PCA: ANEW projects vs Ireland\n(closest cohort highlighted)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_di_bar(dsp, rank, path):
    biomes = sorted(rank["biome"].unique())
    cmap = dict(zip(biomes, plt.cm.tab10(np.linspace(0, 1, len(biomes)))))
    fig, ax = plt.subplots(figsize=(9, 12))
    y = range(len(rank))
    colors = [cmap[b] for b in rank["biome"]]
    ax.barh(list(y), rank["median_di_to_ireland"], color=colors)
    ax.axvline(dsp.threshold_cast, ls="--", c="k")
    ax.text(dsp.threshold_cast, len(rank) - 1, " Ireland AOA thr", fontsize=8, va="top")
    labels = [
        f"{p} (n={n}){' ⚑' if f else ''}"
        for p, n, f in zip(rank["project_name"], rank["n"], rank["co2_flag"])
    ]
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("median CAST DI to Ireland (emb-only, codec)")
    ax.set_title(
        "ANEW projects ranked by feature-space closeness to Ireland\n(⚑ = biomass-flagged)"
    )
    import matplotlib.patches as mp

    ax.legend(
        handles=[mp.Patch(color=c, label=b) for b, c in cmap.items()], fontsize=7, loc="lower right"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_co2_box(rank, path):
    canon = common.load_canonical()
    order = rank["project_name"].tolist()
    data = [canon.loc[canon.project_name == p, "CO2"].dropna().to_numpy() for p in order]
    fig, ax = plt.subplots(figsize=(9, 12))
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True)
    flagged = set(rank.loc[rank["co2_flag"], "project_name"])
    for patch, p in zip(bp["boxes"], order):
        patch.set_facecolor("#d62728" if p in flagged else "#1f77b4")
        patch.set_alpha(0.6)
    ax.set_yticklabels(order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("CO2 (tCO2/acre)")
    ax.set_title(
        "Per-project biomass distribution (ordered by DI to Ireland)\nred = flagged erroneous/implausible"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    for d in (FIG_DIR, ANALYSIS_DIR, DATA_OUT):
        d.mkdir(parents=True, exist_ok=True)

    X_ire = load_ireland_emb()
    dsp = fit_ireland_di(X_ire)
    rank = screen_co2(rank_projects(dsp, X_ire))

    # metric-robustness: Spearman rank agreement
    rho_maha = spearmanr(rank["median_di_to_ireland"], rank["median_maha_to_ireland"]).correlation
    rho_cent = spearmanr(rank["median_di_to_ireland"], rank["centroid_dist_to_ireland"]).correlation

    # the closest cohort for the PCA highlight: gap-defined core (Part 2 owns the formal cut).
    # contiguous top while each step still clears a 0.10 DI gap (the "flat shelf" threshold).
    di_sorted = rank.loc[~rank["co2_flag"], "median_di_to_ireland"].to_numpy()
    kc = 1
    while kc < len(di_sorted) and (di_sorted[kc] - di_sorted[kc - 1]) >= 0.10:
        kc += 1
    cohort = rank.loc[~rank["co2_flag"], "project_name"].head(kc).tolist()

    rank.to_parquet(DATA_OUT / "project_ranking.parquet", index=False)
    fig_pca(dsp, X_ire, rank, cohort, FIG_DIR / "pca_ireland_vs_anew.png")
    fig_di_bar(dsp, rank, FIG_DIR / "di_to_ireland_bar.png")
    fig_co2_box(rank, FIG_DIR / "co2_distribution_box.png")

    flagged = rank.loc[rank["co2_flag"], ["project_name", "co2_median", "co2_max", "co2_robust_z"]]
    closest = rank.loc[~rank["co2_flag"]].head(15)
    cols = [
        "project_name",
        "biome",
        "n",
        "median_di_to_ireland",
        "pct_inside_ireland_aoa",
        "co2_median",
    ]

    md = []
    md.append("# Part 1 — Ireland-anchored feature-space + GT analysis\n")
    md.append(
        f"Ireland reference cloud: {len(X_ire)} plots (emb-only, codec). CAST AOA threshold "
        f"(Ireland leave-one-out self-DI) = **{dsp.threshold_cast:.3f}**; dbar = {dsp.dbar:.3f}.\n"
    )
    md.append("## Metric robustness\n")
    md.append(
        f"Spearman rank agreement of CAST-DI ordering with cross-checks: "
        f"Mahalanobis ρ = **{rho_maha:.3f}**, weighted-centroid ρ = **{rho_cent:.3f}** "
        "(high → the ranking is not an artefact of one metric).\n"
    )
    md.append("## Closest projects to Ireland (excluding biomass-flagged)\n")
    md.append(closest[cols].to_markdown(index=False, floatfmt=".2f") + "\n")
    md.append("## Biomass screen (erroneous / implausible)\n")
    md.append(
        "Robust-z on per-project median CO2 (flag if z > 3.5 or named Quinte). "
        "Ireland has no CO2 GT, so this only bounds plausibility.\n"
    )
    md.append(flagged.to_markdown(index=False, floatfmt=".2f") + "\n")
    md.append(
        f"\n**Auto-dropped:** Quinte (Canada) — median CO2 {rank.loc[rank.project_name == 'Quinte', 'co2_median'].iloc[0]:.0f}, "
        f"max {rank.loc[rank.project_name == 'Quinte', 'co2_max'].iloc[0]:.0f} tCO2/acre.\n"
    )
    ire_floor = rank["median_di_to_ireland"].min()
    md.append("## Honest read on achievable closeness\n")
    md.append(
        f"Closest project median DI to Ireland = **{ire_floor:.2f}** vs Ireland's own AOA threshold "
        f"{dsp.threshold_cast:.2f}. {'Even the closest projects sit beyond Ireland self-similarity — this is near-extrapolation, not in-domain training.' if ire_floor > dsp.threshold_cast else 'Some projects fall within Ireland self-similarity.'}\n"
    )
    md.append(
        "\nFigures: `figures/pca_ireland_vs_anew.png`, `di_to_ireland_bar.png`, `co2_distribution_box.png`.\n"
    )
    (ANALYSIS_DIR / "feature_space.md").write_text("\n".join(md))

    print(
        f"Ireland AOA thr={dsp.threshold_cast:.3f} | Spearman maha {rho_maha:.2f} centroid {rho_cent:.2f}"
    )
    print(f"Closest project DI={ire_floor:.2f}; flagged: {flagged['project_name'].tolist()}")
    print("Top 8 closest (non-flagged):")
    print(closest[cols].head(8).to_string(index=False))
    print(f"\nSaved ranking -> {DATA_OUT / 'project_ranking.parquet'}")
    print(f"Saved report  -> {ANALYSIS_DIR / 'feature_space.md'}")


if __name__ == "__main__":
    main()
