"""
Shared data layer for the emb-only ANEW model experiment.

Loads the eligible ground truth (51 projects, Quinte dropped), attaches the fold-aware DI /
bloc / AOA labels from the GT-applicability experiment (row-aligned with the eligible
canonical), freezes the DI tiers used for per-tier scoring, and builds the S0-S4 sample-weight
vectors. Importable; run directly for a data audit.

All in emb-only training-codec space (the only space available for all 51 projects). Target is
the canonical `CO2` column (tCO2/acre, raw) -- NOT the 23-project `target` column.

Run (audit):
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/anew_emb_model/data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402

APPLIC = common.DATASPACE / "agb_anew_gt_applicability_20260626"
PLOT_LEVEL_DI = APPLIC / "analysis/plot_level_di.parquet"
DROP_PROJECTS = ["Quinte"]

# frozen frontier sets from the GT-applicability report (regional_dependence axis)
REGIONAL_FRONTIER = [
    "LouisianaLowlands",
    "Apalachicola",
    "Soterra",
    "HighCascades",
    "LongviewRanch",
]
SELF_STANDING_FRONTIER = ["Kootznoowoo", "RainierGateway", "Doyon"]

N_BLOCS = 4  # K from bloc_assignments (sizes 18/12/5/16; bloc 2 == the 5 frontier projects)


def aoa_threshold() -> float:
    """The self-referential AOA threshold computed on the 51-project cloud."""
    import json

    return json.loads((APPLIC / "analysis/thresholds.json").read_text())["threshold_cast"]


def load_eligible() -> pd.DataFrame:
    """Eligible GT (51 projects) with emb + CO2 + DI/bloc/AOA labels + frozen tiers.

    plot_level_di.parquet is row-aligned with the Quinte-dropped canonical; we assert that
    alignment on project_name + CO2 rather than trusting position silently.
    """
    canon = common.load_canonical()
    df = canon[~canon["project_name"].isin(DROP_PROJECTS)].reset_index(drop=True)

    pld = pd.read_parquet(PLOT_LEVEL_DI)
    assert len(pld) == len(df), f"row mismatch: canonical {len(df)} vs plot_level_di {len(pld)}"
    assert (pld["project_name"].to_numpy() == df["project_name"].to_numpy()).all(), (
        "project misalign"
    )
    assert np.allclose(pld["CO2"].to_numpy(), df["CO2"].to_numpy()), "CO2 misalign"

    for col in ["bloc_id", "di_lopo", "di_bloc", "inside_aoa"]:
        df[col] = pld[col].to_numpy()

    # biome group: collapse Tundra + Boreal (both are the single project Doyon)
    df["biome_grp"] = df["BIOME_NAME"].replace(
        {"Tundra": "HighLatitude", "Boreal Forests/Taiga": "HighLatitude"}
    )

    # per-plot DI tier (interior gated on the AOA flag; frontier buckets by project set)
    def tier(row):
        if row["inside_aoa"]:
            return "interior"
        if row["project_name"] in SELF_STANDING_FRONTIER:
            return "self_standing_frontier"
        if row["project_name"] in REGIONAL_FRONTIER:
            return "regional_frontier"
        return "outside_other"

    df["tier"] = df.apply(tier, axis=1)
    return df


def n_eff(w: np.ndarray) -> float:
    """Effective sample size (Kish): (sum w)^2 / sum(w^2)."""
    return float(w.sum() ** 2 / (w**2).sum())


def build_weights(df: pd.DataFrame, scheme: str, alpha: float = 1.0, cap: float = 5.0):
    """Sample-weight vector for a scheme, renormalised to mean 1. Returns (w, info).

    S0 unweighted; S1 per-bloc inverse density; S2/S3 capped-DI (alpha 0.5/1.0); S4 = S1*S2.
    """
    n = len(df)
    di = df["di_lopo"].to_numpy()
    med = np.median(di)

    if scheme == "S0":
        w = np.ones(n)
    elif scheme == "S1":
        bloc = df["bloc_id"].to_numpy()
        counts = pd.Series(bloc).value_counts().to_dict()
        w = np.array([(n / N_BLOCS) / counts[b] for b in bloc])
    elif scheme in ("S2", "S3"):
        a = 0.5 if scheme == "S2" else 1.0
        w = np.clip((di / med) ** a, 1.0, cap)
    elif scheme == "S4":
        w_s1, _ = build_weights(df, "S1")
        w_s2, _ = build_weights(df, "S2")
        w = w_s1 * w_s2
    else:
        raise ValueError(f"unknown scheme {scheme}")

    w = w * (n / w.sum())  # renormalise to mean 1
    info = {
        "scheme": scheme,
        "n_eff": n_eff(w),
        "n_eff_frac": n_eff(w) / n,
        "max_ratio": float(w.max() / w.min()),
    }
    return w, info


def main() -> None:
    df = load_eligible()
    thr = aoa_threshold()
    print(f"{len(df)} plots, {df.project_name.nunique()} projects, AOA threshold = {thr:.3f}")
    print(
        f"CO2: min {df.CO2.min():.1f} median {df.CO2.median():.1f} "
        f"mean {df.CO2.mean():.1f} max {df.CO2.max():.1f} | zeros {int((df.CO2 == 0).sum())}"
    )
    print("\ntier counts:")
    print(df["tier"].value_counts().to_string())
    print("\nbiome_grp counts:")
    print(df["biome_grp"].value_counts().to_string())
    print("\nweight schemes (n_eff frac of N, max/min ratio):")
    for s in ["S0", "S1", "S2", "S3", "S4"]:
        _, info = build_weights(df, s)
        print(f"  {s}: n_eff {info['n_eff_frac']:.3f}N  max/min {info['max_ratio']:.1f}")


if __name__ == "__main__":
    main()
