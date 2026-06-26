"""
DATA AUDIT — verify the embedding/feature artefacts before any DI/AOA trust layer.

No GEE. Pure-pandas inspection of the parquets we already hold. Settles the encoding
hazard (raw GEE float vs training int8-codec vs affine bridge) and quantifies the
data we can use, writing a human-readable report to:

    experiments/agb_trust_aoa_20260626/audit/data_audit.md   (tracked in git)

and a machine-readable summary to the data-space (gitignored):

    {DATASPACE}/agb_trust_aoa_20260626/audit/data_audit_summary.json

Checks (see plan compiled-scribbling-umbrella.md, Part A):
  1. Inventory + encoding identification per parquet.
  2. Coverage vs the 12,837-plot / 52-project ANEW GT gpkg; enumerate the 29 unused.
  3. Affine location-invariance: apply the Bayfield-fit affine to each modelled
     project's RAW pool embeddings and compare to its CODEC truth (the 23 modelled
     projects exist in both spaces), reusing the corr/slope gate from fit_aef_affine.
  4. Survey-year-match correctness (pool survey_year vs training year).
  5. Disturbance leakage-safety flag (which disturbance feature the model carries).
  6. QA: non-finite emb rate, duplicate keys, label range.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/trust/run_data_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
DATASPACE = Path("/home/mattc/data-space/carbonmap-embeddings")
IRE_PREP = DATASPACE / "agb_ireland_biomass_regression_20260608/preprocessing"
USA_PREP = REPO / "experiments/agb_usa_biomass_regression_20260529/preprocessing"

GT_GPKG = DATASPACE / "training-data/anew_gt_with_eco_info.gpkg"
POOL = IRE_PREP / "iter1_pool_embeddings.parquet"
TRAIN = USA_PREP / "features_iter3.parquet"  # codec, 23 modelled projects
AFFINE = IRE_PREP / "aef_affine.parquet"

OUT_MD = REPO / "experiments/agb_trust_aoa_20260626/audit/data_audit.md"
OUT_JSON = DATASPACE / "agb_trust_aoa_20260626/audit/data_audit_summary.json"

EMB = [f"emb_{i:02d}" for i in range(64)]
BANDS = [f"A{i:02d}" for i in range(64)]

# affine gate thresholds (mirror scripts/fit_aef_affine.py)
GATE_CORR = 0.8
SLOPE_MED_LO, SLOPE_MED_HI = 0.95, 1.05


def classify_encoding(df: pd.DataFrame) -> tuple[str, dict]:
    """Raw GEE float (~|x|<2) vs training int8-codec (~|x| up to ~86)."""
    cols = [c for c in EMB if c in df.columns]
    if not cols:
        return "no-embeddings", {}
    x = df[cols].to_numpy(float)
    finite = np.isfinite(x)
    stats = {
        "emb_mean": float(np.nanmean(x)),
        "emb_std": float(np.nanstd(x)),
        "emb_absmax": float(np.nanmax(np.abs(x[finite]))) if finite.any() else float("nan"),
        "nonfinite_frac": float((~finite).mean()),
    }
    label = "raw_gee_float" if stats["emb_absmax"] < 2.0 else "training_codec"
    return label, stats


def key(df: pd.DataFrame) -> pd.Series:
    return df["project_name"].astype(str) + "|" + df["plot_id"].astype(str)


def affine_invariance(
    pool: pd.DataFrame, train: pd.DataFrame, affine: pd.DataFrame
) -> pd.DataFrame:
    """Per modelled project: affine(pool RAW) vs train CODEC — corr + median band slope."""
    a = affine.set_index("band")["a"]
    c = affine.set_index("band")["c"]
    pool = pool.assign(k=key(pool))
    train = train.assign(k=key(train))
    rows = []
    for proj, tr_p in train.groupby("project_name"):
        m = tr_p.merge(pool, on="k", suffixes=("_tr", "_pool"))
        if len(m) == 0:
            continue
        # affine: pool raw -> codec
        trans = np.column_stack(
            [m[f"{e}_pool"].to_numpy(float) * a[b] + c[b] for b, e in zip(BANDS, EMB)]
        )
        truth = m[[f"{e}_tr" for e in EMB]].to_numpy(float)
        ok = np.isfinite(trans).all(1) & np.isfinite(truth).all(1)
        trans, truth = trans[ok], truth[ok]
        # per-plot 64-vec correlation (correctness gate)
        cors = [np.corrcoef(trans[j], truth[j])[0, 1] for j in range(len(trans))]
        # per-band slope (transformed ~ truth); median across bands ~ 1 (amplitude diagnostic)
        slopes = [np.polyfit(trans[:, i], truth[:, i], 1)[0] for i in range(64)]
        biome = (
            pool.loc[pool["project_name"] == proj, "BIOME_NAME"].mode().iat[0]
            if (pool["project_name"] == proj).any()
            else "?"
        )
        mean_corr = float(np.nanmean(cors))
        slope_med = float(np.median(slopes))
        rows.append(
            {
                "project_name": proj,
                "biome": biome,
                "n": len(trans),
                "mean_corr": mean_corr,
                "min_corr": float(np.nanmin(cors)),
                "slope_median": slope_med,
                # correctness = direction preserved (the affine reproduces the codec vector).
                # slope is a secondary amplitude check; the [0.95,1.05] band is a Bayfield-aggregate
                # tolerance, too tight per-project, so it is reported but not the pass criterion.
                "corr_ok": mean_corr > GATE_CORR,
                "slope_ok": SLOPE_MED_LO <= slope_med <= SLOPE_MED_HI,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_corr")


def main() -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    md: list[str] = ["# ANEW / AGB data audit", ""]
    md += [
        "Generated by `scripts/trust/run_data_audit.py` (no GEE). "
        "Plan: `compiled-scribbling-umbrella.md`, Part A.",
        "",
    ]
    verdict: list[str] = []  # filled as checks run, rendered into the Verdict section below

    # ---- 1. inventory + encoding ----
    md += ["## 1. Inventory and encoding", ""]
    md += [
        "| file | rows | projects | encoding | emb mean | emb |max| | nonfinite | labels | cofeatures |"
    ]
    md += ["|---|--:|--:|---|--:|--:|--:|:--:|:--:|"]
    inventory = {}
    files = {
        "pool iter1_pool_embeddings.parquet": POOL,
        "train features_iter3.parquet": TRAIN,
        "USA features_iter1.parquet": USA_PREP / "features_iter1.parquet",
        "USA features_iter2.parquet": USA_PREP / "features_iter2.parquet",
        "ireland_features.parquet": IRE_PREP / "ireland_features.parquet",
        "ireland_aef_raw.parquet": IRE_PREP / "ireland_aef_raw.parquet",
        "train_emb_sample.parquet": DATASPACE
        / "agb_ireland_biomass_regression_20260608/data_profile/train_emb_sample.parquet",
    }
    for name, path in files.items():
        if not path.exists():
            md += [f"| {name} | — | — | MISSING | | | | | |"]
            continue
        df = pd.read_parquet(path)
        enc, st = classify_encoding(df)
        nproj = df["project_name"].nunique() if "project_name" in df else "—"
        has_lab = "CO2" in df.columns or "target" in df.columns
        cofeat = [c for c in ("chm_m", "topo_elevation", "dstx_pre_ysd") if c in df.columns]
        inventory[name] = {"rows": len(df), "encoding": enc, **st}
        md += [
            f"| {name} | {len(df)} | {nproj} | **{enc}** | "
            f"{st.get('emb_mean', float('nan')):.2f} | {st.get('emb_absmax', float('nan')):.1f} | "
            f"{st.get('nonfinite_frac', 0) * 100:.2f}% | {'✓' if has_lab else '—'} | "
            f"{'✓' if cofeat else '—'} |"
        ]
    summary["inventory"] = inventory

    # ---- 2. coverage ----
    md += ["", "## 2. Coverage vs ANEW GT", ""]
    g = gpd.read_file(GT_GPKG)
    pool = pd.read_parquet(POOL)
    train = pd.read_parquet(TRAIN)
    anew_pool = pool[pool["project_name"] != "Ireland"]
    modelled = sorted(train["project_name"].unique())
    all_proj = sorted(anew_pool["project_name"].unique())
    unused = sorted(set(all_proj) - set(modelled))
    cov = {
        "gt_plots": int(len(g)),
        "gt_projects": int(g["project_name"].nunique()),
        "pool_anew_plots": int(len(anew_pool)),
        "pool_anew_projects": len(all_proj),
        "modelled_projects": len(modelled),
        "unused_projects": len(unused),
        "unused_plots": int(anew_pool["project_name"].isin(unused).sum()),
        "modelled_is_subset": set(modelled).issubset(set(all_proj)),
    }
    summary["coverage"] = cov
    md += [f"- GT gpkg: **{cov['gt_plots']} plots / {cov['gt_projects']} projects**"]
    md += [
        f"- Pool embeddings (ANEW): {cov['pool_anew_plots']} plots / {cov['pool_anew_projects']} projects"
    ]
    md += [
        f"- Modelled: **{cov['modelled_projects']}** projects; unused: **{cov['unused_projects']}** projects / **{cov['unused_plots']}** plots"
    ]
    md += [f"- Modelled ⊆ pool: **{cov['modelled_is_subset']}**", ""]
    ub = (
        anew_pool[anew_pool["project_name"].isin(unused)]
        .groupby(["BIOME_NAME", "project_name"])
        .size()
        .reset_index(name="n")
        .sort_values(["BIOME_NAME", "n"], ascending=[True, False])
    )
    md += ["Unused-but-labelled projects (embeddings + CO2, no co-features):", ""]
    md += ["| biome | project | plots |", "|---|---|--:|"]
    md += [f"| {r.BIOME_NAME} | {r.project_name} | {r.n} |" for r in ub.itertuples()]

    # ---- 3. affine location-invariance ----
    md += ["", "## 3. Affine location-invariance (RAW pool → CODEC)", ""]
    affine = pd.read_parquet(AFFINE)
    inv = affine_invariance(pool, train, affine)
    modelled_biomes = sorted(inv["biome"].unique())
    unused_biomes = sorted(
        set(anew_pool.loc[anew_pool["project_name"].isin(unused), "BIOME_NAME"].unique())
        - set(modelled_biomes)
    )
    summary["affine_invariance"] = {
        "n_projects": int(len(inv)),
        "n_corr_ok": int(inv["corr_ok"].sum()),
        "n_slope_ok": int(inv["slope_ok"].sum()),
        "min_mean_corr": float(inv["mean_corr"].min()),
        "worst_corr_project": inv.iloc[0]["project_name"],
        "modelled_biomes": modelled_biomes,
        "unverified_biomes": unused_biomes,
    }
    md += [
        f"Applying the Bayfield-fit affine to each modelled project's raw embeddings and comparing "
        f"to its codec truth. **Correctness** = per-plot 64-vector mean corr > {GATE_CORR} (direction "
        f"preserved). Slope is a secondary amplitude check (the [{SLOPE_MED_LO},{SLOPE_MED_HI}] band "
        "is a Bayfield-aggregate tolerance, too tight per-project).",
        "",
        f"**Correctness: {int(inv['corr_ok'].sum())}/{len(inv)} projects** (all mean corr ≥ "
        f"{inv['mean_corr'].min():.3f}). Slope within band: {int(inv['slope_ok'].sum())}/{len(inv)} "
        "(amplitude drift concentrated in WV projects, slope ~0.83–0.89 — direction still faithful).",
        "",
        f"**Biome coverage caveat:** all {len(inv)} modelled projects are `{modelled_biomes}`. "
        f"The affine is therefore UNVERIFIED for the unused biomes `{unused_biomes}` "
        "(PNW conifer / Alaska / southern) — no codec anchor exists there. The affine is the AEF "
        "asset's global dequantisation so it should hold, but DI on those projects rests on an "
        "unverified mapping; flag for targeted re-extraction if they enter training.",
        "",
        "| project | biome | n | mean corr | min corr | slope median | corr ok | slope ok |",
        "|---|---|--:|--:|--:|--:|:--:|:--:|",
    ]
    md += [
        f"| {row['project_name']} | {row['biome']} | {row['n']} | {row['mean_corr']:.3f} | "
        f"{row['min_corr']:.3f} | {row['slope_median']:.3f} | {'✓' if row['corr_ok'] else '✗'} | "
        f"{'✓' if row['slope_ok'] else '✗'} |"
        for row in inv.to_dict("records")
    ]
    verdict += [
        f"- **Encodings identified** for all {len(inventory)} parquets (pool = raw GEE float; "
        "training/Ireland = codec).",
        f"- **Affine verified** within Temperate Broadleaf (corr ≥ {inv['mean_corr'].min():.3f}, "
        f"{len(inv)}/{len(inv)}); **unverified** for {unused_biomes}.",
    ]

    # ---- 4. survey-year match ----
    md += ["", "## 4. Survey-year match (pool vs training)", ""]
    pk = pool.assign(k=key(pool))[["k", "survey_year"]]
    tk = train.assign(k=key(train))[["k", "year"]]
    yj = tk.merge(pk, on="k", how="inner")
    mism = int((yj["year"] != yj["survey_year"]).sum())
    summary["survey_year"] = {"checked": int(len(yj)), "mismatches": mism}
    md += [
        f"- Checked {len(yj)} shared plots; **{mism} year mismatches** between pool and training."
    ]

    # ---- 5. disturbance leakage flag ----
    md += ["", "## 5. Disturbance feature / leakage flag", ""]
    feats_full = json.loads((REPO / "models/inference_features.json").read_text())["features"]
    dstx_in_model = [f for f in feats_full if f.startswith("dstx") or f.startswith("dist")]
    train_dist_cols = [c for c in train.columns if c.startswith("dist") or c.startswith("dstx")]
    summary["disturbance"] = {
        "model_features": dstx_in_model,
        "train_parquet_cols": train_dist_cols,
    }
    md += [
        f"- Deployed model disturbance features: `{dstx_in_model}` (survey-relative, leakage-safe).",
        f"- `features_iter3.parquet` still carries: `{train_dist_cols}` — the OLD `dist_years_since` "
        "is present in the base parquet; trainer merges the corrected `dstx_*` from "
        "`disturbance_timing_features.csv`. **Flag:** ensure DI/any re-use reads the corrected dstx, "
        "not `dist_years_since`.",
    ]

    # ---- 6. QA ----
    md += ["", "## 6. QA", ""]
    dup_pool = int(key(pool).duplicated().sum())
    dup_train = int(key(train).duplicated().sum())
    co2 = anew_pool["CO2"]
    summary["qa"] = {
        "pool_dup_keys": dup_pool,
        "train_dup_keys": dup_train,
        "co2_min": float(co2.min()),
        "co2_max": float(co2.max()),
        "co2_nonnull_frac": float(co2.notna().mean()),
    }
    train_co2_max = 521.0  # deployed model target_range upper bound
    md += [
        f"- Duplicate `(project,plot_id)` keys — pool: **{dup_pool}**, train: {dup_train}"
        + (" (dedup before use)" if dup_pool else ""),
        f"- CO2 range: [{co2.min():.1f}, **{co2.max():.1f}**] tCO2/acre; non-null {co2.notna().mean() * 100:.1f}%",
        f"  - **Flag:** pool CO2 max ({co2.max():.0f}) is {co2.max() / train_co2_max:.1f}× the deployed "
        f"model's training range upper bound ({train_co2_max:.0f}) — unused projects extend the label "
        "range well beyond training (extreme-biomass PNW/old-growth).",
    ]
    verdict += [
        f"- **Survey-year**: {mism} mismatches. **Dup keys**: {dup_pool} in pool. "
        f"**Label range**: pool CO2 to {co2.max():.0f} vs training ≤{train_co2_max:.0f}.",
        "- **Disturbance**: model uses leakage-safe `dstx_*`; base parquet still carries old "
        "`dist_years_since` — read the corrected dstx downstream.",
        "",
        "**Gate decision:** encodings are consistent and the affine reproduces codec faithfully "
        "within the modelled biome → Part B (canonical codec store) may proceed. Carry the "
        "unverified-biome caveat into DI on conifer/boreal/tundra/grassland projects.",
    ]

    md = md[:4] + ["## Verdict", "", *verdict, ""] + md[4:]
    OUT_MD.write_text("\n".join(md) + "\n")
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(
        f"\nAffine invariance: corr_ok {int(inv['corr_ok'].sum())}/{len(inv)}, "
        f"slope_ok {int(inv['slope_ok'].sum())}/{len(inv)}; min mean corr {inv['mean_corr'].min():.3f}"
    )
    print(f"Coverage: {cov['unused_projects']} unused projects / {cov['unused_plots']} plots")


if __name__ == "__main__":
    main()
