"""
ANEW GT applicability — plot-level AOA maps and DI-ranking figures.

Renders five figures from the artifacts written by compute_di_folds.py (run that first):
  aoa_national_map.png    all plots in lon/lat, coloured by LOPO DI, inside/outside-AOA marker
  di_ranking_bar.png      median LOPO DI per project, coloured by bloc, AOA threshold line
  di_box_by_project.png   per-project DI boxplots ordered by median
  pca_gt_space.png        PCA of the 64-dim weighted space, blocs coloured, AOA threshold ring
  lopo_vs_bloc_scatter.png  per-project LOPO DI vs bloc DI (regional dependence)

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/anew_gt_applicability/make_maps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402
import di as di_mod  # noqa: E402

OUT = common.DATASPACE / "agb_anew_gt_applicability_20260626"
FIG_DIR = common.REPO / "experiments/agb_anew_gt_applicability_20260626/figures"


def _bloc_colours(bloc_ids):
    uniq = sorted(set(bloc_ids))
    palette = plt.get_cmap("tab10", max(len(uniq), 3))
    return {b: palette(i) for i, b in enumerate(uniq)}


def national_map(plots: pd.DataFrame, thr: float) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    inside, outside = plots[plots["inside_aoa"]], plots[~plots["inside_aoa"]]
    vmax = float(np.percentile(plots["di_lopo"], 98))
    for sub, mk, lab in [(inside, "o", "inside AOA"), (outside, "x", "outside AOA")]:
        sc = ax.scatter(
            sub["lon"],
            sub["lat"],
            c=sub["di_lopo"],
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            s=10,
            marker=mk,
            alpha=0.6,
            label=lab,
        )
    fig.colorbar(sc, ax=ax, label="LOPO CAST DI (emb-only)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(
        f"ANEW GT self-referential AOA — {100 * plots['inside_aoa'].mean():.0f}% "
        f"of plots inside AOA (thr={thr:.2f})"
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "aoa_national_map.png", dpi=120)
    plt.close(fig)


def ranking_bar(rank: pd.DataFrame, thr: float) -> None:
    colours = _bloc_colours(rank["bloc_id"])
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(range(len(rank)), rank["median_di_lopo"], color=[colours[b] for b in rank["bloc_id"]])
    ax.axvline(thr, ls="--", c="k")
    ax.text(thr, 1, " AOA thr", rotation=90, va="bottom")
    ax.set_yticks(range(len(rank)))
    ax.set_yticklabels(
        [f"{p} (n={n})" for p, n in zip(rank["project_name"], rank["n"])], fontsize=7
    )
    ax.set_xlabel("median LOPO CAST DI (emb-only, codec)")
    ax.set_title("Per-project dissimilarity ranking (interior -> frontier)")
    ax.set_ylim(-0.5, len(rank) - 0.5)
    ax.legend(
        handles=[mp.Patch(color=c, label=f"bloc {b}") for b, c in colours.items()],
        loc="lower right",
        title="spatial bloc",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "di_ranking_bar.png", dpi=120)
    plt.close(fig)


def box_by_project(plots: pd.DataFrame, rank: pd.DataFrame, thr: float) -> None:
    order = rank["project_name"].tolist()
    data = [plots.loc[plots["project_name"] == p, "di_lopo"].to_numpy() for p in order]
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.boxplot(data, vert=False, showfliers=False, widths=0.7)
    ax.axvline(thr, ls="--", c="k")
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("LOPO CAST DI (emb-only, codec)")
    ax.set_title("Per-project DI distribution (ordered by median)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "di_box_by_project.png", dpi=120)
    plt.close(fig)


def pca_gt_space(plots: pd.DataFrame) -> None:
    """PCA of the weighted emb space; colour by bloc. Reuses the di.fit transform."""
    df = common.load_canonical()
    df = df[df["project_name"].isin(plots["project_name"].unique())].reset_index(drop=True)
    X = df[common.EMB].astype(float).to_numpy()
    okm = np.isfinite(X).all(1)
    X, df = X[okm], df[okm].reset_index(drop=True)
    proj = df["project_name"].to_numpy()
    dsp = di_mod.fit(X, proj, common.EMB, common.gain_weights("embonly"))
    Z = dsp.transform(X)
    Zc = Z - Z.mean(0)
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt[:2].T
    var = (S**2 / (S**2).sum())[:2] * 100

    bloc_of = dict(zip(plots["project_name"], plots["bloc_id"]))
    bloc = np.array([bloc_of.get(p, -1) for p in proj])
    colours = _bloc_colours([b for b in bloc if b >= 0])
    fig, ax = plt.subplots(figsize=(9, 7))
    for b, c in colours.items():
        m = bloc == b
        ax.scatter(P[m, 0], P[m, 1], s=6, alpha=0.4, color=c, label=f"bloc {b}")
    ax.set_xlabel(f"PC1 ({var[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.0f}%)")
    ax.set_title("Weighted emb-only PCA of the ANEW GT space (by spatial bloc)")
    ax.legend(title="spatial bloc", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_gt_space.png", dpi=120)
    plt.close(fig)


def lopo_vs_bloc(rank: pd.DataFrame) -> None:
    colours = _bloc_colours(rank["bloc_id"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        rank["median_di_lopo"],
        rank["median_di_bloc"],
        c=[colours[b] for b in rank["bloc_id"]],
        s=40,
    )
    lim = [0, max(rank["median_di_bloc"].max(), rank["median_di_lopo"].max()) * 1.05]
    ax.plot(lim, lim, ls="--", c="#999", label="y = x (no regional dependence)")
    for _, r in rank.iterrows():
        ax.annotate(
            r["project_name"], (r["median_di_lopo"], r["median_di_bloc"]), fontsize=6, alpha=0.7
        )
    ax.set_xlabel("median LOPO DI (leave one project out)")
    ax.set_ylabel("median bloc DI (leave whole region out)")
    ax.set_title("Regional dependence: DI lift when the whole bloc is removed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lopo_vs_bloc_scatter.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plots = pd.read_parquet(OUT / "analysis/plot_level_di.parquet")
    rank = pd.read_parquet(OUT / "analysis/project_di_ranking.parquet")
    thr = json.loads((OUT / "analysis/thresholds.json").read_text())["threshold_cast"]

    national_map(plots, thr)
    ranking_bar(rank, thr)
    box_by_project(plots, rank, thr)
    pca_gt_space(plots)
    lopo_vs_bloc(rank)
    print(f"Saved 5 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
