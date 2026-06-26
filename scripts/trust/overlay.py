"""
OVERLAY — reusable codec-space applicability visuals for any prediction area.

overlay_prediction_area(X_codec, label) renders the two-panel view from the earlier
exploration, but now in the correct training-codec space: (left) PCA scatter of the query
points over the training cloud; (right) DI histogram vs the training self-DI with the AOA
threshold. Drop in any expansion area's codec embeddings to get an instant applicability read.

Also regenerates the all-projects figures (the codec, model-comparable replacements for the
earlier raw-space fig7/fig8).

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/trust/overlay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import aoa as aoa_mod  # noqa: E402
import common  # noqa: E402

EMB = common.EMB
FIG_DIR = common.REPO / "experiments/agb_trust_aoa_20260626/figures"


def _pca_fit(Z: np.ndarray):
    Zc = Z - Z.mean(0)
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:2], Z.mean(0), (S**2 / (S**2).sum())[:2] * 100


def overlay_prediction_area(X_codec: np.ndarray, label: str, dsp=None, ax_pca=None, ax_hist=None):
    """DI + 2-panel applicability figure for a prediction area in codec space."""
    dsp = dsp or aoa_mod.load_di_space("embonly")
    tr = common.load_full_training()
    Xtr = tr[EMB].astype(float).to_numpy()
    Ztr, Zq = dsp.transform(Xtr), dsp.transform(X_codec)
    V, mu, var = _pca_fit(Ztr)
    Ptr, Pq = (Ztr - mu) @ V.T, (Zq - mu) @ V.T
    di_q = dsp.di(X_codec)

    own = (ax_pca is None) or (ax_hist is None)
    if own:
        fig, (ax_pca, ax_hist) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax_pca.scatter(Ptr[:, 0], Ptr[:, 1], s=5, alpha=0.25, c="#999", label="training (23)")
    ax_pca.scatter(Pq[:, 0], Pq[:, 1], s=18, alpha=0.6, c="#d62728", marker="X", label=label)
    ax_pca.set_xlabel(f"PC1 ({var[0]:.0f}%)")
    ax_pca.set_ylabel(f"PC2 ({var[1]:.0f}%)")
    ax_pca.legend()
    ax_pca.set_title(f"Codec embedding PCA: {label} vs training")

    ax_hist.hist(
        dsp.train_di, bins=40, density=True, alpha=0.5, color="#999", label="training self-DI"
    )
    ax_hist.hist(di_q, bins=30, density=True, alpha=0.6, color="#d62728", label=f"{label} DI")
    ax_hist.axvline(dsp.threshold_cast, ls="--", c="k")
    ax_hist.text(dsp.threshold_cast, ax_hist.get_ylim()[1] * 0.9, " AOA thr", fontsize=9)
    pct_in = 100 * (di_q <= dsp.threshold_cast).mean()
    ax_hist.set_title(f"DI distribution — {pct_in:.0f}% inside AOA")
    ax_hist.set_xlabel("emb-only CAST DI")
    ax_hist.legend(fontsize=8)
    if own:
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig_overlay_{label.lower().replace(' ', '_')}.png", dpi=110)
        plt.close(fig)
    return di_q


def all_projects_codec() -> None:
    """Codec-space replacements for the raw-space fig7 (PCA) and fig8 (per-project DI)."""
    dsp = aoa_mod.load_di_space("embonly")
    canon = common.load_canonical()
    ire = pd.read_parquet(
        common.DATASPACE
        / "agb_ireland_biomass_regression_20260608/preprocessing/ireland_features.parquet"
    )

    Xtr = common.load_full_training()[EMB].astype(float).to_numpy()
    Ztr = dsp.transform(Xtr)
    V, mu, var = _pca_fit(Ztr)

    def proj_pts(df):
        X = df[EMB].astype(float).to_numpy()
        ok = np.isfinite(X).all(1)
        return ((dsp.transform(X[ok]) - mu) @ V.T), X[ok]

    fig, ax = plt.subplots(figsize=(9, 7))
    groups = [
        ("modelled-23", canon[canon["modelled"]], "#1f77b4", 5, 0.25, "o"),
        ("unused-29", canon[~canon["modelled"]], "#ff7f0e", 9, 0.35, "o"),
        ("Ireland", ire, "#d62728", 30, 0.8, "X"),
    ]
    for name, df, c, s, al, mk in groups:
        P, _ = proj_pts(df)
        ax.scatter(
            P[:, 0],
            P[:, 1],
            s=s,
            alpha=al,
            c=c,
            marker=mk,
            label=f"{name} (n={len(P)})",
            zorder=5 if name == "Ireland" else 1,
        )
    ax.set_xlabel(f"PC1 ({var[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.0f}%)")
    ax.set_title(
        "CODEC-space embedding PCA: all 52 ANEW projects + Ireland\n(model-comparable; cf. raw-space fig7)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_codec_allprojects_pca.png", dpi=110)
    plt.close(fig)

    # fig8: per-project median DI bar (reuse the AOA report)
    rep = aoa_mod.per_project_report("embonly")
    cmap = {"modelled-23": "#1f77b4", "unused-29": "#ff7f0e", "Ireland": "#d62728"}
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(range(len(rep)), rep["median_di"], color=[cmap[s] for s in rep["src"]])
    ax.axvline(dsp.threshold_cast, ls="--", c="k")
    ax.text(dsp.threshold_cast, 1, " AOA thr", rotation=90, va="bottom")
    ax.set_yticks(range(len(rep)))
    ax.set_yticklabels([f"{p} (n={n})" for p, n in zip(rep["project_name"], rep["n"])], fontsize=7)
    ax.set_xlabel("median CAST DI (emb-only, codec)")
    ax.set_title("CODEC-space per-project DI vs AOA threshold (cf. raw-space fig8)")
    ax.set_ylim(-0.5, len(rep) - 0.5)
    import matplotlib.patches as mp

    ax.legend(handles=[mp.Patch(color=v, label=k) for k, v in cmap.items()], loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_codec_perproject_di.png", dpi=110)
    plt.close(fig)
    print(f"Saved fig7_codec_allprojects_pca.png, fig8_codec_perproject_di.png to {FIG_DIR}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_projects_codec()
