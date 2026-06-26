"""
UNCERTAINTY SURFACE — expected error as a monotonic function of DI.

Fits expected_RMSE = f(DI) via isotonic regression on DI bins. The key design choice:
the 23 training projects are all in-domain (DI ≤ ~1.0), so a curve fit on them alone is
flat and useless for expansion. We therefore calibrate in EMB-ONLY space over the full
DI range by combining two out-of-fold sources:
  - 23 modelled projects: leave-one-project-out OOF residuals (emb-only refit) — low DI.
  - 29 unused labelled projects: emb-only deployed model predictions (OOF by construction,
    the model never saw them) — high DI, the expansion/OOD regime.

Both are predictions on data the (fold) model did not train on, so they are comparable.
The full-feature LOPO curve is also recorded as an in-domain reference.

Flags the calibration limit (max DI with labelled support); beyond it expected error is a
lower bound, not a measurement.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/trust/uncertainty.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

warnings.filterwarnings("ignore", message="X does not have valid feature names")
OUT = common.TRUST_OUT / "trust"
FIG_DIR = common.REPO / "experiments/agb_trust_aoa_20260626/figures"
EMB = common.EMB
N_BINS = 12
PARAMS = dict(
    num_leaves=31, learning_rate=0.05, min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1
)


def fit_curve(di: np.ndarray, resid: np.ndarray, n_bins: int = N_BINS):
    bins = pd.qcut(di, n_bins, labels=False, duplicates="drop")
    rows = []
    for b in np.unique(bins):
        m = bins == b
        rows.append(
            {
                "di_mid": float(np.median(di[m])),
                "rmse": float(np.sqrt(np.mean(resid[m] ** 2))),
                "n": int(m.sum()),
            }
        )
    tab = pd.DataFrame(rows)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(tab["di_mid"], tab["rmse"], sample_weight=tab["n"])
    return tab, iso


def emb_only_oof() -> pd.DataFrame:
    """Per-plot (di, resid) in emb-only space: 23-project LOPO OOF + 29 unused direct."""
    import di as di_mod  # local import to avoid shadowing the `di` array name elsewhere

    # 23 modelled: leave-one-project-out OOF, finite emb rows only
    tr = common.load_full_training()
    Xtr = tr[EMB].astype(float).to_numpy()
    finite = np.isfinite(Xtr).all(1)
    tr, Xtr = tr[finite].reset_index(drop=True), Xtr[finite]
    ytr = tr["target"].to_numpy()
    proj = tr["project_name"].to_numpy()
    # fit DI space on these exact rows so train_di (fold-aware, leave-project-out) aligns
    dsp = di_mod.fit(Xtr, proj, EMB, common.gain_weights("embonly"))
    oof = np.full(len(ytr), np.nan)
    for p in np.unique(proj):
        te = proj == p
        m = lgb.LGBMRegressor(n_estimators=143, **PARAMS).fit(Xtr[~te], ytr[~te])
        oof[te] = m.predict(Xtr[te])
    # fold-aware DI for training points (NOT dsp.di, which would find self at distance 0)
    modelled = pd.DataFrame({"di": dsp.train_di, "resid": oof - ytr, "src": "modelled-LOPO"})

    # 29 unused: deployed emb-only model (OOF by construction)
    canon = common.load_canonical()
    un = canon[(~canon["modelled"]) & canon["CO2"].notna()].copy()
    Xun = un[EMB].astype(float).to_numpy()
    ok = np.isfinite(Xun).all(1)
    Xun, un = Xun[ok], un[ok]
    booster = lgb.Booster(model_file=str(common.MODELS / "inference_model_embonly.txt"))
    unused = pd.DataFrame(
        {
            "di": dsp.di(Xun),
            "resid": booster.predict(Xun) - un["CO2"].to_numpy(),
            "src": "unused-direct",
        }
    )
    out = pd.concat([modelled, unused], ignore_index=True)
    out["aoa_threshold"] = dsp.threshold_cast
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = emb_only_oof()
    df = df[np.isfinite(df["di"]) & np.isfinite(df["resid"])]
    thr = float(df["aoa_threshold"].iloc[0])
    di = df["di"].to_numpy()
    # labelled-support limit: highest DI decile edge that still has reasonable n
    di_cal = float(np.percentile(di, 99))

    tab, iso = fit_curve(di, df["resid"].to_numpy())
    np.savez(
        OUT / "error_curve_embonly.npz",
        x=iso.X_thresholds_,
        y=iso.y_thresholds_,
        di_max=float(di.max()),
        di_cal=di_cal,
        threshold_cast=thr,
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for src, c in [("modelled-LOPO", "#1f77b4"), ("unused-direct", "#ff7f0e")]:
        s = df[df.src == src]
        ax.scatter(s["di"], np.abs(s["resid"]), s=6, alpha=0.2, c=c, label=f"{src} |resid|")
    grid = np.linspace(di.min(), di.max(), 200)
    ax.plot(grid, iso.predict(grid), c="k", lw=2.5, label="expected RMSE = f(DI)")
    ax.scatter(tab["di_mid"], tab["rmse"], s=tab["n"] / 5, c="k", zorder=5)
    ax.axvline(thr, ls="--", c="green")
    ax.text(thr, ax.get_ylim()[1] * 0.92, " AOA thr", fontsize=9, color="green")
    ax.set_xlabel("emb-only CAST DI")
    ax.set_ylabel("error (tCO2/acre)")
    ax.set_title(
        "Uncertainty surface (emb-only): expected error vs DI\n"
        "calibrated 23-project LOPO + 29 unused labelled projects"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_error_vs_di.png", dpi=110)
    plt.close(fig)

    probe = [thr, 0.8, 1.0, 1.2]
    summary = {
        "aoa_threshold_cast": thr,
        "calibration_di_p99": di_cal,
        "di_max_labelled": float(di.max()),
        "bins": tab.to_dict("records"),
        "expected_rmse_at": {f"DI={d}": float(iso.predict([d])[0]) for d in probe},
    }
    (OUT / "uncertainty_curve.json").write_text(json.dumps(summary, indent=2))
    print(f"AOA thr DI={thr:.3f}; labelled calibration to DI≈{di_cal:.2f} (max {di.max():.2f})")
    print("Expected RMSE (tCO2/acre) by DI:")
    for d in probe:
        print(f"  DI={d:.2f} -> {iso.predict([d])[0]:.1f}")
    print(f"Saved {FIG_DIR / 'fig_error_vs_di.png'} and uncertainty_curve.json")


if __name__ == "__main__":
    main()
