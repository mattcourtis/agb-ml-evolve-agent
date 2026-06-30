"""
Trust layer for the emb-only ANEW model: AOA threshold + DI->expected-error curve.

Every prediction the shipped model makes can then carry: its DI, an inside/outside-AOA flag,
and an expected RMSE. Two pieces:
  - AOA threshold: refit di.fit() on the 51-project emb cloud (confirm it matches the
    self-referential threshold already in the GT-applicability run, ~0.558).
  - DI->error curve: isotonic expected-RMSE = f(DI), fit on the SHIPPED model's LOPO OOF
    residuals only (all 51 projects are in-fold, so LOPO spans the full DI range; bloc/biome
    residuals are far-transfer extrapolation and are NOT pooled in).

Reuses scripts/trust/uncertainty.py::fit_curve and scripts/trust/di.py::fit.

Outputs (data-space): trust/di_space_anew51.npz, trust/error_curve.npz, trust/thresholds.json;
figure experiments/.../figures/error_vs_di.png.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/anew_emb_model/trust_fit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402
import di as di_mod  # noqa: E402
import uncertainty as unc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as data_mod  # noqa: E402

EXP = common.DATASPACE / "agb_anew_emb_weighted_20260630"
TRUST = EXP / "trust"
FIG_DIR = common.REPO / "experiments/agb_anew_emb_weighted_20260630/figures"


def main() -> None:
    TRUST.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = data_mod.load_eligible()
    X = df[common.EMB].astype(float).to_numpy()

    # 1. AOA threshold refit on the 51-project cloud; confirm it matches the applicability run
    dsp = di_mod.fit(X, df["project_name"].to_numpy(), common.EMB, common.gain_weights("embonly"))
    thr_applic = data_mod.aoa_threshold()
    assert abs(dsp.threshold_cast - thr_applic) < 1e-6, (
        f"threshold drift: refit {dsp.threshold_cast:.4f} vs applic {thr_applic:.4f}"
    )
    np.savez(
        TRUST / "di_space_anew51.npz",
        features=np.array(common.EMB),
        mu=dsp.mu,
        sd=dsp.sd,
        w=dsp.w,
        dbar=dsp.dbar,
        threshold_cast=dsp.threshold_cast,
        p95=dsp.p95,
        p99=dsp.p99,
    )

    # 2. DI->error curve from the shipped (S0) model's LOPO OOF residuals only
    oof = pd.read_parquet(EXP / "cv/oof_S0.parquet")
    di = oof["di_lopo"].to_numpy()
    resid = (oof["oof_S0"] - oof["CO2"]).to_numpy()
    fin = np.isfinite(di) & np.isfinite(resid)
    di, resid = di[fin], resid[fin]
    tab, iso = unc.fit_curve(di, resid)
    di_cal = float(np.percentile(di, 99))
    np.savez(
        TRUST / "error_curve.npz",
        x=iso.X_thresholds_,
        y=iso.y_thresholds_,
        di_max=float(di.max()),
        di_cal=di_cal,
        threshold_cast=dsp.threshold_cast,
    )

    # interior-vs-frontier sanity header (expected RMSE at each tier's median DI)
    oof_f = oof[fin].reset_index(drop=True)
    header = {}
    for tier in ["interior", "regional_frontier", "self_standing_frontier"]:
        m = oof_f["tier"] == tier
        if m.any():
            md = float(oof_f.loc[m, "di_lopo"].median())
            header[tier] = {
                "median_di": round(md, 3),
                "expected_rmse": round(float(iso.predict([md])[0]), 1),
            }

    probe = [dsp.threshold_cast, 0.5, 0.7, 0.8]
    summary = {
        "aoa_threshold_cast": dsp.threshold_cast,
        "calibration_di_p99": di_cal,
        "di_max": float(di.max()),
        "expected_rmse_at": {f"DI={d:.2f}": round(float(iso.predict([d])[0]), 1) for d in probe},
        "tier_header": header,
        "note": "curve fit on 51-project LOPO OOF (S0); beyond DI p99 it is a lower bound.",
    }
    (TRUST / "thresholds.json").write_text(json.dumps(summary, indent=2))

    # figure
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(di, np.abs(resid), s=5, alpha=0.15, c="#1f77b4", label="LOPO |resid|")
    grid = np.linspace(di.min(), di.max(), 200)
    ax.plot(grid, iso.predict(grid), c="k", lw=2.5, label="expected RMSE = f(DI)")
    ax.scatter(tab["di_mid"], tab["rmse"], s=tab["n"] / 8, c="k", zorder=5)
    ax.axvline(dsp.threshold_cast, ls="--", c="green")
    ax.text(dsp.threshold_cast, ax.get_ylim()[1] * 0.92, " AOA thr", fontsize=9, color="green")
    ax.set_xlabel("emb-only CAST DI (51-project self-referential)")
    ax.set_ylabel("error (tCO2/acre)")
    ax.set_title("ANEW emb-only model — uncertainty surface (expected error vs DI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_vs_di.png", dpi=120)
    plt.close(fig)

    print(f"AOA threshold = {dsp.threshold_cast:.3f} (matches applicability run)")
    print(f"calibration to DI p99 = {di_cal:.2f} (max {di.max():.2f})")
    print("expected RMSE by tier (median DI):")
    for t, h in header.items():
        print(f"  {t:24s} DI {h['median_di']:.3f} -> {h['expected_rmse']:.1f} tCO2/acre")
    print(f"Saved trust bundle to {TRUST} and figure to {FIG_DIR}")


if __name__ == "__main__":
    main()
