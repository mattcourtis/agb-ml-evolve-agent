# Error Analysis

## Purpose
Explain why performance is weak or fragile.

## Required lenses
- by region / ecoregion
- by target quintile (Q1..Q5) — required for any regression task
- by canopy cover class (sparse / open / closed)
- by forest type (hardwood / softwood / mixed, where determinable)
- by stand age (where the label set carries it)
- by terrain slope (DEM-derived; bin by quintile of slope)
- by year (when training pool pools multiple years)
- by calibration bucket
- by plot-footprint coverage (where the embedding tile coverage is incomplete)
- by feature missingness
- by GEDI footprint density at the plot (record as "N/A — embeddings-only iteration" until the GEDI extractor is integrated; required from the first GEDI iteration onward)

## Required outputs
- `error_analysis/error_analysis.md`
- `error_analysis/failure_slices.csv` — one row per slice with `{slice_name, slice_value, n, mae, rmse, r2, bias}`
- For regression tasks: `error_analysis/quintile_diagnostics.csv` with `{quintile, true_mean, predicted_mean, n_plots}` proving the predicted-range-discrimination calculation.

## Root-cause taxonomy
- experimental design issue
- data quality issue
- label issue
- leakage or split issue
- preprocessing issue
- feature insufficiency
- model mismatch
- hyperparameter issue
- training instability
- evaluation mismatch
- benchmark mismatch

## Reproducibility footer (required)
Same schema as `database_preprocessing.md`: input_artefact_sha256, libraries, seed, command_or_entrypoint, timestamp_utc.

## Critic addendum
Reject if:
- analysis names symptoms but not a likely earliest upstream cause from the root-cause taxonomy above;
- no row from `references/improvement_loop.md`'s error-analysis-to-stage mapping table is cited (or no explicit justification is given for why no row fits);
- `quintile_diagnostics.csv` is missing for a regression task whose evaluation matrix reports `per_quintile_bias` or `predicted_range_discrimination`;
- the reproducibility footer is missing.
