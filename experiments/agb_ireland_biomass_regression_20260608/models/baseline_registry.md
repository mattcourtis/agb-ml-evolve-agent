# Baseline registry

Authored by Orchestrator (collapsed formality stage); consistent with ACCEPTED experiment_design.md.

In a zero-shot transfer + model-vs-model comparison with NO ground truth, "baselines" are the
reference points the embdstx head is judged against, not trained competitors.

## B0 — Deep Biomass reference (primary comparator / directional lower bound)
- Source: Deep Biomass aggregated CSV, per-Location AGB.
- Quantity: 2020–2024 mean Mg/ha (= mean of cell/Area_Ha), and 2024-only Mg/ha.
- Converted to tCO2/acre via x0.6977 (preprocessing/db_reference.parquet).
- Portfolio: 2020–2024 mean 39.19 Mg/ha -> 27.35 tCO2/acre; 2024-only 45.71 -> 31.89.
- Role: known under-estimator; expectation H1 our_pred >= DB, gap widening in the high-biomass band.

## B1 — Structural-covariate sanity reference (no-model rank check)
- Not a numeric predictor; uses the Dasos covariates (PlantingYe->age, Hdom, YC, MainSp) to sanity-check
  that BOTH our_pred and DB rank-track stand structure (older/taller/higher-YC -> more biomass).
- Provides the falsifiable structural-consistency test (H3) in the absence of truth.

## No trained baseline
- No new model is trained in this pass (transfer mode). A naive mean predictor is uninformative here
  because the target of comparison is itself a model, not truth. If the conditional analog-subset
  retrain is later triggered (improvement stage), the full-CONUS head (S3) becomes the trained control.

status: ACCEPTED (orchestrator-authored formality; reference baselines only, no training)
