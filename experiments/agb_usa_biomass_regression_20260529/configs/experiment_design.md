# Experiment Design — agb_usa biomass regression (iteration 0)

## Objective

Reproduce the `joint_v2` embeddings-only LightGBM baseline (R²=0.42) through the
biomass-ml-agent-evolve orchestrator, exercising the cross-repo invocation contract and
populating the biomass-specific evaluation matrix. This is a wiring validation, not a
modelling improvement.

## Task framing

- **Task type:** biomass_regression (plot-level).
- **Target:** `CO2` — standing-stock tCO₂/acre (ANEW gpkg column). `Annual_CO2` deferred.
- **Spatial unit:** 1/24-acre circular field plot (~14.7 m radius).
- **Features:** 64-dim AlphaEarth Foundation (AEF) embeddings (10 m, annual), reused from the prior
  joint_v2 extraction. PALSAR-2 ablated out (+0.02 R², within noise) — not in scope.
- **Pool:** 4,636 plots, 23 projects, years 2022+2023 pooled.

## Split design

- **Strategy:** spatial holdout — leave-one-project-out (LOPO) on `project_name`.
- **random_split_used:** false. Every metric is on a project unseen at training time.
- This is also the **external/new-project holdout**: because each project is held out exactly
  once, the LOPO aggregate is the realistic new-project generalisation estimate. No separate
  random or temporal split is reported.

## Acceptance metrics (drive the evaluation-matrix gates)

The experiment **claims target-level discrimination and new-project generalisation**, so the
matrix must populate all of:

- `r2`, `rmse`, `mae`, `bias` (aggregate)
- `calibration`
- `per_quintile_bias` (Q1..Q5, all required)
- `predicted_range_discrimination` (range-collapse proxy)
- `per_ecoregion_r2` (pool spans >1 ecoregion → required, by `ECO_NAME`)
- `error_by_region` (wv / mw / ne blocs)
- `external_holdout_r2` (new-project claim → required; = LOPO aggregate R²)

## Thresholds

| metric | realistic | stretch | iter-0 acceptance |
|---|---|---|---|
| R² | ≥ 0.40 | ≥ 0.55 | 0.42 ± 0.03 |
| RMSE | ≤ 60 | ≤ 45 | ≈ 57 |
| MAE | — | — | ≈ 41 |
| \|bias\| | ≤ 5 | — | ≈ 0.5 |
| predicted_range_discrimination | — | ≥ 0.6 | < 1.0 expected (feature ceiling) |
| per_quintile_bias \|max\| | ≤ 30 | — | range-compression signature expected |

## Models

- **Simple baseline (required):** mean-predictor and ridge-on-PC20 (LOPO) — embeddings linear
  floor, see `models/baseline_registry.md`.
- **Production candidate:** `lightgbm_emb64` — reproduction target.

## Stop condition for iteration 0

Reproduction within ±0.03 of R²=0.42 → ACCEPT and advance to iteration 1 (GEDI feature
test). Failure to reproduce → ESCALATE (wiring fault).

## Out of scope

GEDI integration, feature re-extraction, hyperparameter changes, sibling-repo code edits,
and any of the five ruled-out levers (see `research/deep_research.md`).
