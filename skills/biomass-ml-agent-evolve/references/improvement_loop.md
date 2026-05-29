# Improvement Loop

## Purpose
Convert evaluation gaps into the smallest valid rerun.

## Earliest-stage rule
Always fix the earliest stage plausibly responsible for the gap.

## Rerun suffix examples
- research -> everything downstream
- design -> preprocess onward
- split -> training onward
- preprocess (new feature source, e.g. GEDI) -> split (if partition assumptions changed) -> train -> evaluate
- training -> training and evaluation only
- evaluation mismatch -> evaluation only

## Error-analysis to earliest-weak-stage mapping

Use this table to translate an error-analysis finding into the earliest stage to repair and the rerun boundary. The Improvement Planner Actor must cite the matched row.

| Error-analysis finding | Earliest weak stage | Rerun from |
|---|---|---|
| Q1 over-prediction & Q5 under-prediction (predicted range collapsed; isotonic calibration fails to fix it) | Feature insufficiency | `research -> preprocess -> train -> evaluate` (must add a new feature source — e.g., GEDI canopy height — not re-tune existing pipeline) |
| Strong random split, weak spatial holdout | Split design / model generalisation | `split -> baseline -> candidate training -> evaluate` |
| Stable on training ecoregions, collapses on external holdout | Experimental design / training-pool coverage | `experimental design -> preprocess -> train -> evaluate` (broaden pool or declare narrower deployment) |
| Calibration poor despite good RMSE | Evaluation / training objective | `training -> evaluate` |
| Suspiciously high score on plots in same project as training | Leakage or split issue | `split -> all downstream reruns` |
| Unstable across years when pool is multi-year | Experimental design / temporal alignment | `experimental design -> preprocess -> train -> evaluate` |
| Weak baseline and weak complex models | Research, labels, or preprocessing | earliest confirmed of `research`, `data profile`, or `preprocess` |
| Pixel-level metrics good, hex-aggregated bias high (segmentation) | Evaluation design or aggregation logic | `evaluation` or `preprocess -> evaluate` depending on root cause |

## Stop conditions
- target met
- benchmark met
- max iterations reached
- budget exhausted
- data insufficient
- user escalation required

## Iteration cap rule

The improvement loop's per-experiment iteration ceiling is `experiment_config.yaml :: training.max_full_iterations`. If that value exceeds the budget tier's iteration cap from `references/model_selection.md` (Small=3, Medium=4, Large=6), the Orchestrator MUST verify `experiment_config.yaml :: training.allow_extended_iterations: true` is present and that the compute budget still permits it. The Critic REJECTs an `improvement_plan.md` whose proposed iteration index exceeds the tier cap unless that flag is true.

## Required output
`reports/improvement_plan.md` with:
- observed gap (metric, current score, threshold, gap)
- suspected stage — must be one of the stage names in the IMPLEMENTATION_PLAN.md artefact registry
- evidence — non-empty list of `evidence_artifacts:` (path with line or row anchor, e.g. `error_analysis/failure_slices.csv:rows 12-40`, or `error_analysis/quintile_diagnostics.csv` for feature-insufficiency claims)
- mapped row from the table above (or explicit justification if no row fits)
- proposed fix
- rerun boundary — starting stage name + ordered downstream suffix
- `expected_cost:` structured block with:
  - `compute_hours: <float>` — total expected compute time across reruns
  - `wall_time_hours: <float>` — calendar time until the rerun completes
  - `gpu_class: <string|null>` — e.g., `T4`, `A100`, `none`
- For feature-insufficiency proposals: a `new_feature_source:` block declaring the source (e.g., `GEDI L4A footprint AGBD`), the integration path (Earth Engine vs. LP DAAC direct), the expected extraction cost in hours, and the cited source-registry row.

## Critic addendum
Reject if:
- the rerun boundary starts downstream of any plausible upstream cause not yet ruled out;
- `evidence_artifacts` is empty or paths do not resolve;
- the suspected_stage is not in the registry vocabulary;
- the error-analysis root cause sits in `experimental design issue | data quality issue | label issue | leakage or split issue | preprocessing issue | feature insufficiency` and the rerun boundary does not begin at or upstream of that stage;
- the proposal claims feature insufficiency but does not declare a `new_feature_source:` block;
- the proposal proposes hyperparameter tuning, loss-function changes (e.g., Huber, log-target), or post-hoc calibration (e.g., isotonic, Platt) as a fix for Q1/Q5 quintile-bias collapse — this pattern is the canonical feature-insufficiency signal, and the prior AGB investigation (`agb-modelling-context.md` §Controls) showed each of these levers individually fails to lift the ceiling; the canonical fix is a structural feature addition (e.g., GEDI canopy height);
- `expected_cost` is missing any of `compute_hours`, `wall_time_hours`, or `gpu_class`, or `compute_hours`/`wall_time_hours` are not numeric;
- the proposed iteration index exceeds the budget tier's cap in `references/model_selection.md` while `experiment_config.yaml :: training.allow_extended_iterations` is not `true`.
