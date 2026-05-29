# Evaluation

## Purpose
Evaluate against the user target or benchmark-derived default.

## Regression metrics (primary)
- MAE
- RMSE
- R²
- bias (mean signed residual)
- calibration (residual vs. predicted bins, or reliability curve)
- **per_quintile_bias** — mean signed residual per true-target quintile (Q1..Q5). Decisive diagnostic for whether a regressor is collapsing the dynamic range. Required for any regression task that claims to discriminate target levels.
- **predicted_range_discrimination** — `(predicted_Q5_mean − predicted_Q1_mean) / (true_Q5_mean − true_Q1_mean)`. Numerical proxy for the "model collapses the dynamic range" failure mode. 1.0 = perfect range coverage; 0.0 = model predicts the same value for all quintiles.
- **per_ecoregion_r2** — R² reported separately per ecoregion when the training pool spans more than one ecoregion. Aggregate R² alone hides ecoregion-specific failures.
- **external_holdout_r2** — R² on a project / area held out from the training pool entirely. Required when the experiment claims generalisation to "new" projects or ecoregions. This is the realistic "new project" expectation.
- error_by_region
- error_by_year
- error_by_target_quintile (sliced residuals; usually reported alongside per_quintile_bias)

## Segmentation metrics (for biomass_segmentation)
- pixel MAE / RMSE / R²
- per-bin bias (binned by true biomass)
- area-aggregated bias at hex / polygon scale
- coverage metrics (fraction of valid pixels per tile)

## Change-detection metrics
- pixel MAE / RMSE on Δbiomass
- change-class confusion matrix when discretised
- temporal-window robustness

## Required matrix fields
- threshold
- current_score
- gap
- pass_fail
- suspected_failure_source
- recommended_upstream_fix

## Holdout requirements
- report spatial holdout separately
- report temporal holdout separately when years are pooled
- report external holdout separately when the task claims new-project generalisation
- do not claim production-grade performance from random splits alone

## Required artifact
`evaluation/evaluation_matrix.yaml`

## Required upstream ACCEPTED inputs
- `research/deep_research.md` (benchmark range, default/stretch thresholds)
- `configs/experiment_design.md` (acceptance metrics list)
- `configs/split_strategy.yaml` (random_split_used, holdout proof)
- `reports/training_run.md` (scores, checkpoint reference)

## Reproducibility footer (required)
Same schema as `database_preprocessing.md`: input_artefact_sha256, libraries, seed, command_or_entrypoint, timestamp_utc.

## Critic addendum
Reject if:
- metrics do not match the task type;
- any metric appearing in `configs/experiment_design.md` acceptance list has a null `threshold` in the matrix;
- `split_summary.random_split_used` or `split_summary.leakage_checks_passed` disagrees with the corresponding values in the ACCEPTED `configs/split_strategy.yaml`;
- a `current_score` is reported from a random-only split for a task that requires spatial or temporal generalisation;
- spatial-holdout and temporal-holdout scores are not reported separately when the task requires either;
- `per_quintile_bias` is absent or only partially populated (Q1..Q5 all required) for a regression task whose `experiment_design.md` lists a generalisation claim;
- `predicted_range_discrimination` is absent for a regression task whose `experiment_design.md` claims target-level discrimination;
- the training pool spans more than one ecoregion but `per_ecoregion_r2` is empty or aggregate-only;
- the experiment_design claims new-project generalisation but `external_holdout_r2.current_score` is null;
- the reproducibility footer is missing.
