"""
GUARDRAILS — turn DI + the uncertainty curve into shippable trust layers.

Reusable API: given a prediction feature array (codec space) and a fitted DI space +
isotonic error curve, returns per-point DI, AOA mask, and expected RMSE, plus an
aggregate trust header (% area inside AOA, DI distribution). Out-of-AOA points should be
suppressed or flagged and excluded from aggregations.

Demo (no GEE): applies the emb-only guardrail to the real Bayfield embedding stack
(bayfield_emb_30m.npy, codec). Bayfield is in-sample (mw bloc) so it should come back
≈fully inside the AOA — the sanity check from the plan.

Integration point for inference scripts (scripts/infer_bayfield.py,
scripts/per_pixel_inference.py): after building the per-pixel feature stack `X` (codec),
call `bands = apply(X, dsp, curve)` and write bands["di"], bands["aoa_mask"],
bands["expected_rmse"] as sidecar raster bands; emit `trust_header(...)` into the run log.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/trust/guardrails.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import aoa as aoa_mod  # noqa: E402
import common  # noqa: E402

OUT = common.TRUST_OUT / "trust"
BAYFIELD_EMB = (
    common.REPO / "experiments/agb_usa_biomass_regression_20260529/predictions/bayfield_emb_30m.npy"
)


def load_error_curve(space: str = "embonly"):
    """Return a callable di -> expected RMSE from the saved isotonic knots."""
    z = np.load(OUT / f"error_curve_{space}.npz")
    x, y = z["x"], z["y"]
    return lambda di: np.interp(di, x, y), float(z["di_cal"]) if "di_cal" in z else float(x.max())


def apply(X: np.ndarray, dsp, curve_fn, fast: bool = True) -> dict:
    di = dsp.di_fast(X) if fast else dsp.di(X)
    return {
        "di": di,
        "aoa_mask": di <= dsp.threshold_cast,  # True = inside (trustworthy)
        "expected_rmse": curve_fn(di),
    }


def trust_header(di: np.ndarray, threshold: float, di_cal: float) -> dict:
    inside = di <= threshold
    return {
        "n": int(len(di)),
        "pct_inside_aoa": float(100 * inside.mean()),
        "di_median": float(np.median(di)),
        "di_iqr": [float(np.percentile(di, 25)), float(np.percentile(di, 75))],
        "di_max": float(di.max()),
        "pct_beyond_calibration": float(100 * (di > di_cal).mean()),
    }


def demo_canonical() -> None:
    """Exercise guardrails on clean codec data: representative in-/out-of-AOA projects.

    Treats each project as a 'prediction area' and emits its trust header. Validates that
    the guardrail discriminates in-domain (≈inside AOA) from OOD (≈outside) on real,
    correctly-encoded embeddings — the no-GEE substitute for a wall-to-wall raster.
    """
    dsp = aoa_mod.load_di_space("embonly")
    curve_fn, di_cal = load_error_curve("embonly")
    canon = common.load_canonical()
    ire = pd.read_parquet(
        common.DATASPACE
        / "agb_ireland_biomass_regression_20260608/preprocessing/ireland_features.parquet"
    )
    areas = {
        "BayfieldCounty (in-sample)": canon[canon.project_name == "BayfieldCounty"],
        "NortheastKingdom (unused, in-AOA)": canon[canon.project_name == "NortheastKingdom"],
        "HighCascades (unused, PNW OOD)": canon[canon.project_name == "HighCascades"],
        "Ireland (cross-continent OOD)": ire,
    }
    headers = {}
    for name, df in areas.items():
        X = df[common.EMB].astype(float).to_numpy()
        X = X[np.isfinite(X).all(1)]
        b = apply(X, dsp, curve_fn, fast=False)
        h = trust_header(b["di"], dsp.threshold_cast, di_cal)
        h["expected_rmse_median"] = float(np.median(b["expected_rmse"]))
        headers[name] = h
        print(
            f"{name:38s} inside-AOA {h['pct_inside_aoa']:5.1f}%  median DI {h['di_median']:.2f}  "
            f"exp.RMSE {h['expected_rmse_median']:.0f}"
        )
    (OUT / "guardrail_demo_headers.json").write_text(json.dumps(headers, indent=2))
    print(f"\nSaved {OUT / 'guardrail_demo_headers.json'}")


def flag_bayfield_stack() -> None:
    """Record (do not ship as valid) the band mismatch in the existing Bayfield emb stack."""
    if not BAYFIELD_EMB.exists():
        return
    st = np.load(BAYFIELD_EMB)
    nz = (np.abs(st).reshape(st.shape[0], -1) > 0).mean(1)
    bad = [int(i) for i in range(st.shape[0]) if nz[i] < 0.1]
    flag = {
        "artifact": str(BAYFIELD_EMB),
        "issue": "per-band non-zero fractions do not match the training codec; several bands "
        "are near-all-zero, so per-pixel DI on this stack is invalid until rebuilt.",
        "near_zero_bands": bad,
        "n_near_zero_bands": len(bad),
        "action": "rebuild the Bayfield embedding stack in verified codec space (needs GEE; "
        "out of current no-GEE scope) before computing per-pixel guardrails on it.",
    }
    (OUT / "bayfield_stack_flag.json").write_text(json.dumps(flag, indent=2))
    print(
        f"\n[FLAG] Bayfield emb stack has {len(bad)} near-zero bands {bad} — "
        f"per-pixel guardrail deferred (see {OUT / 'bayfield_stack_flag.json'})."
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    demo_canonical()
    flag_bayfield_stack()
