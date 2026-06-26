# Selecting ANEW Projects to Train an Ireland AGB Model (agb_ireland_training_selection_20260626)

Recommends which whole ANEW projects to use as training data for a model destined for **Ireland**,
by ranking projects on embedding feature-space closeness to Ireland, screening out erroneous
biomass, and designing project-intact CV folds. **No model is trained this pass** (training
strategies are the next phase, per the user). Plan: `plans/ireland-training-selection.md`.

Code: `scripts/ireland_training_selection/`. Reuses the `scripts/trust/` DI module. Data outputs
(gitignored) under `/home/mattc/data-space/carbonmap-embeddings/agb_ireland_training_selection_20260626/`.

## Method (locked decisions)

- **Selection metric: Ireland-anchored CAST DI.** The importance-weighted CAST DI space is fit on
  Ireland's 141 plots as the *reference cloud* (leave-one-out self-DI sets the AOA threshold);
  every ANEW project is scored against it. Lower DI = looks more like Ireland.
- **Space: emb-only (64-dim)** — the only feature space all 52 ANEW projects *and* Ireland share
  (codec). Ireland has no `chm_m`/topo and **no CO₂ ground-truth**.
- **Cut: relative top-K** at the knee of the DI curve (no absolute "inside AOA" test, since Ireland
  is out-of-domain to all US projects), subject to anti-overfit floors.

## Part 1 — Feature-space + GT analysis

- **Ranking is metric-robust.** Spearman agreement of the CAST-DI ordering with cross-checks:
  weighted-centroid distance ρ = 0.95, Mahalanobis ρ = 0.77. The ordering is not a single-metric
  artefact.
- **Closest projects are oceanic conifer + maritime broadleaf** — climatically sensible for
  Ireland's Sitka-spruce plantations and Atlantic broadleaf: Kootznoowoo (SE Alaska conifer, DI
  2.25), RainierGateway (PNW conifer), FundyBay (Bay of Fundy broadleaf), LongviewRanch (PNW
  conifer). The 23 deployed-model training projects (Great Lakes / Appalachia) rank lower.
- **Honest read:** the closest project sits at median DI **2.25** vs Ireland's own AOA threshold
  **0.77** — *every* ANEW project is 0% inside Ireland's domain. Selection is **near-extrapolation,
  not in-domain training**; it picks the least-bad analogues, and that limitation must travel with
  any Ireland model trained on this data.
- **Biomass screen:** robust-z on per-project median CO₂ flags **Quinte** (Canada) — median 311,
  max 1262 tCO₂/acre, ~2× the cohort — and it is auto-dropped. No other project trips the rule.

Report: `analysis/feature_space.md`. Figures: `figures/{pca_ireland_vs_anew,di_to_ireland_bar,co2_distribution_box}.png`.

## Part 2 — Assembly + CV folds

The DI ranking has only ~4 genuinely-closer projects, then a **flat shelf**: ~40 projects of all
biomes sit at DI 3.10–3.97, equidistant from Ireland (gaps < 0.10, smaller than within-project
spread). So selection is **gap-based**, not a fixed top-K or plot-floor; biome plays no role
(grassland Soterra ranks above three broadleaf projects — distance, not biome, drives the cut).

- **Recommended set — `core`: 4 projects, 787 plots, 2 biomes.** The gap-defined closest cluster:
  Kootznoowoo, RainierGateway, FundyBay, LongviewRanch (oceanic conifer + Bay-of-Fundy broadleaf).
  Core DI 2.25–3.06 sits clearly below the shelf (3.10–3.97). This is the best Ireland analogue;
  its risk is breadth — 787 plots, mostly conifer — which the bake-off measures.
- **Nested alternatives (defined, not trained) for the next-phase bake-off:** `extended` (9 proj,
  2,072 plots — core padded down the shelf to anti-overfit floors, to test whether extra
  plots/variation beat staying closest) and `all_minus_err` (51 eligible, baseline).
- **CV folds (whole projects intact):** core uses **leave-one-project-out** (its natural scheme,
  `core_lopo_fold`); extended uses a **5-way grouped spatial K-fold** (KMeans on project centroids
  in EPSG:5070, `extended_spatial_fold`). Composition reported in `analysis/feature_space.md`.
- **Manifest:** `preprocessing/selected_projects.parquet` — one row per project with `in_core` /
  `in_extended` / `in_all_minus_err`, `core_lopo_fold`, `extended_spatial_fold`, DI-to-Ireland,
  CO₂ stats, biome. Provenance in `preprocessing/{feature_schema.json,data_version.txt}` +
  `final/DATA_STORE.md` (encoding=codec, cv_partition_key=project_name).

## Verification

- Metric robustness reported (centroid ρ 0.95 / Mahalanobis ρ 0.77).
- Quinte excluded and justified by CO₂ stats; asserted present in `erroneous_excluded`.
- Core size within bounds (3–8) and its LOPO folds are 1:1 with projects; asserted.
- Every `extended` project carries a valid spatial fold (≥0); no project spans two folds; asserted.

## Limitations / next steps

- **Near-extrapolation, not in-domain.** Even the best cohort is far outside Ireland's feature
  domain (DI ≈ 2.3+ vs 0.77). A model trained on it should ship with DI/AOA guardrails
  (see `agb_trust_aoa_20260626`) flagging Ireland predictions as extrapolation.
- **No CO₂ matching to Ireland** — Ireland has no ground-truth, so biomass was only screened for
  plausibility, not matched. Acquiring Irish field plots would let us validate, not just select.
- **emb-only** — selection ignores topo/CHM/dstx because the 29 unused projects and Ireland lack
  them; a richer selection needs GEE co-feature extraction (deferred).
- **Next phase:** train + compare the three candidate sets on the designed folds (LOPO RMSE +
  Ireland AOA-coverage) to pick a winner — this pass deliberately stops at selection + fold design.
