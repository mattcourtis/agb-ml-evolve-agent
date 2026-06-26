"""
VALIDATION GATE — does DI rank-order true error?

The hard gate from the plan: before any uncertainty surface or guardrail ships, the
DI must demonstrably predict error. We have a clean test: the 29 unused ANEW projects
are LABELLED and lie outside the training domain, and the emb-only deployed model
(inference_model_embonly.txt) can score them from embeddings alone. So we compute, per
project, the emb-only CAST DI and the model's true error, and check they rank-order.

Two checks:
  (a) cross-biome: Spearman(per-project median DI, per-project RMSE) over the 29 unused.
  (b) in-domain reference: per-region DI vs known LOPO R² (wv worst), reproducing fig3/fig8.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/trust/validate_di_error.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import aoa as aoa_mod  # noqa: E402
import common  # noqa: E402

EMB = common.EMB
FIG_DIR = common.REPO / "experiments/agb_trust_aoa_20260626/figures"
OUT = common.TRUST_OUT / "trust"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    dsp = aoa_mod.load_di_space("embonly")
    booster = lgb.Booster(model_file=str(common.MODELS / "inference_model_embonly.txt"))

    canon = common.load_canonical()
    canon = canon[canon["CO2"].notna()].copy()
    X = canon[EMB].astype(float).to_numpy()
    ok = np.isfinite(X).all(1)
    canon, X = canon[ok].reset_index(drop=True), X[ok]
    canon["pred"] = booster.predict(X)
    canon["di"] = dsp.di(X)
    canon["err"] = canon["pred"] - canon["CO2"]

    # per-project error vs DI (separate modelled vs unused)
    pp = (
        canon.assign(grp=np.where(canon["modelled"], "modelled-23", "unused-29"))
        .groupby(["grp", "project_name"])
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "median_di": g["di"].median(),
                    "rmse": float(np.sqrt((g["err"] ** 2).mean())),
                    "bias": float(g["err"].mean()),
                    "co2_mean": float(g["CO2"].mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    unused = pp[pp.grp == "unused-29"]
    rho_u, p_u = spearmanr(unused["median_di"], unused["rmse"])
    rho_all, p_all = spearmanr(pp["median_di"], pp["rmse"])

    # (b) in-domain per-region DI (reference)
    reg = (
        canon[canon["modelled"] & canon["region"].notna()]
        .groupby("region")
        .apply(
            lambda g: pd.Series(
                {"median_di": g["di"].median(), "rmse": float(np.sqrt((g["err"] ** 2).mean()))}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    known_lopo_r2 = {"wv": 0.157, "mw": 0.415, "ne": 0.476}
    reg["known_lopo_r2"] = reg["region"].map(known_lopo_r2)

    gate_pass = bool(rho_u > 0 and p_u < 0.05)
    summary = {
        "cross_biome_spearman_di_vs_rmse": {"rho": float(rho_u), "p": float(p_u), "n": len(unused)},
        "all52_spearman_di_vs_rmse": {"rho": float(rho_all), "p": float(p_all), "n": len(pp)},
        "per_region": reg.to_dict("records"),
        "GATE_PASS": gate_pass,
    }
    (OUT / "validation_gate.json").write_text(json.dumps(summary, indent=2))
    pp.to_parquet(OUT / "per_project_di_error.parquet", index=False)

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for grp, c in [("modelled-23", "#1f77b4"), ("unused-29", "#ff7f0e")]:
        s = pp[pp.grp == grp]
        ax1.scatter(s["median_di"], s["rmse"], c=c, s=40, alpha=0.8, label=grp)
    ax1.axvline(dsp.threshold_cast, ls="--", c="k", lw=1)
    ax1.text(dsp.threshold_cast, ax1.get_ylim()[1] * 0.95, " AOA thr", fontsize=9)
    ax1.set_xlabel("per-project median DI (emb-only CAST)")
    ax1.set_ylabel("per-project RMSE (tCO2/acre)")
    ax1.set_title(f"DI rank-orders error\nunused-29 Spearman ρ={rho_u:.2f} (p={p_u:.1e})")
    ax1.legend()
    ax2.bar(reg["region"], reg["median_di"], color=["#d62728", "#1f77b4", "#2ca02c"])
    for _, r in reg.iterrows():
        ax2.text(r["region"], r["median_di"], f"  R²={r['known_lopo_r2']:.2f}", fontsize=9)
    ax2.set_ylabel("median DI")
    ax2.set_title("In-domain reference: per-region DI vs known LOPO R²\n(wv worst = highest DI)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_di_vs_error.png", dpi=110)
    plt.close(fig)

    print(f"GATE {'PASS' if gate_pass else 'FAIL'}")
    print(f"  cross-biome (unused-29): Spearman ρ={rho_u:.3f}, p={p_u:.2e}, n={len(unused)}")
    print(f"  all 52 projects:         Spearman ρ={rho_all:.3f}, p={p_all:.2e}")
    print("  per-region DI vs known LOPO R²:")
    for _, r in reg.iterrows():
        print(f"    {r['region']}: median DI {r['median_di']:.3f}  (known R²={r['known_lopo_r2']})")
    print(f"\nSaved {FIG_DIR / 'fig_di_vs_error.png'} and validation_gate.json")


if __name__ == "__main__":
    main()
