"""
AOA — Area of Applicability on top of the CAST DI space.

Provides the applicability API (load a fitted DI space; classify query points as
inside/outside AOA with a continuous DI) and a per-project AOA report over all 52 ANEW
projects + Ireland, in codec space. This is the model-comparable replacement for the
earlier raw-space per-project DI figure.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/trust/aoa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import di as di_mod  # noqa: E402

IRE_FEATURES = (
    common.DATASPACE
    / "agb_ireland_biomass_regression_20260608/preprocessing/ireland_features.parquet"
)


def load_di_space(space: str) -> di_mod.DISpace:
    """Rebuild a fitted DI space from the deployed training cloud (deterministic)."""
    tr = common.load_full_training()
    features = common.SPACES[space][0]
    X = tr[features].astype(float).to_numpy()
    w = common.gain_weights(space)
    return di_mod.fit(X, tr["project_name"].to_numpy(), features, w)


def per_project_report(space: str = "embonly") -> pd.DataFrame:
    """DI + %-inside-AOA per ANEW project and Ireland, in codec space."""
    dsp = load_di_space(space)
    feats = dsp.features
    canon = common.load_canonical()
    frames = [canon.assign(_src=np.where(canon["modelled"], "modelled-23", "unused-29"))]

    # Ireland (already codec); fill any missing co-features with model defaults for 'full'
    ire = pd.read_parquet(IRE_FEATURES)
    for f in feats:
        if f not in ire.columns:
            ire[f] = 100.0 if f == "dstx_pre_ysd" else 0.0
    ire = ire.assign(project_name="Ireland", _src="Ireland", CO2=np.nan)
    frames.append(ire)

    rows = []
    for src, grp in pd.concat(frames, ignore_index=True).groupby("_src"):
        for proj, g in grp.groupby("project_name"):
            X = g[feats].astype(float).to_numpy()
            ok = np.isfinite(X).all(1)
            d = dsp.di(X[ok])
            rows.append(
                {
                    "project_name": proj,
                    "src": src,
                    "n": int(ok.sum()),
                    "median_di": float(np.median(d)),
                    "pct_inside_aoa": float(100 * (d <= dsp.threshold_cast).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values("median_di")


def main() -> None:
    out = common.TRUST_OUT / "trust"
    out.mkdir(parents=True, exist_ok=True)
    rep = per_project_report("embonly")
    rep.to_parquet(out / "per_project_aoa_embonly.parquet", index=False)

    dsp = load_di_space("embonly")
    print(f"emb-only CAST threshold = {dsp.threshold_cast:.3f}\n")
    print("Most in-domain (lowest DI):")
    print(rep.head(5).to_string(index=False))
    print("\nMost out-of-domain (highest DI):")
    print(rep.tail(8).to_string(index=False))
    ire = rep[rep.project_name == "Ireland"].iloc[0]
    n_unused_out = int(((rep.src == "unused-29") & (rep.pct_inside_aoa < 50)).sum())
    print(f"\nIreland: median DI {ire.median_di:.2f}, {ire.pct_inside_aoa:.0f}% inside AOA")
    print(f"Unused-29 projects mostly outside AOA (<50% inside): {n_unused_out}/29")
    print(f"\nSaved {out / 'per_project_aoa_embonly.parquet'}")


if __name__ == "__main__":
    main()
