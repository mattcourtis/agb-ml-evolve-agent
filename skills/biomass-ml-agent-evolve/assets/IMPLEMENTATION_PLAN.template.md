# IMPLEMENTATION_PLAN

## Experiment header

- experiment_id: {experiment_id}
- created_at: {created_at}
- subject: {subject}
- geography: {geography}
- task: {task}
- target_variable: {target_variable}
- spatial_resolution: {spatial_resolution}
- temporal_horizon: {temporal_horizon}
- evaluation_metrics: {evaluation_metrics}
- performance_threshold: {performance_threshold}
- output_dir: {output_dir}
- compute_budget: {compute_budget}
- runtime_budget: {runtime_budget}
- status: NOT_STARTED

## Task restatement

{task_restatement}

## Benchmark anchor

- user_target_provided: {yes_no}
- benchmark_range: TBD
- realistic_default_target: TBD
- stretch_target: TBD
- benchmark_notes: TBD

## Artifact registry

| stage | actor | artifact | status | attempt | critic | last_update | notes |
|---|---|---|---|---:|---|---|---|
| research | Research Actor | research/deep_research.md | NOT_STARTED | 0 | Research Critic | {created_at} | |
| data_profile | Database Profiling Actor | data_profile/database_profile.md | NOT_STARTED | 0 | Data Profile Critic | {created_at} | |
| experiment_design | Experimental Design Actor | configs/experiment_design.md | NOT_STARTED | 0 | Experiment Design Critic | {created_at} | |
| preprocessing | Preprocessing Actor | preprocessing/preprocessing_spec.md | NOT_STARTED | 0 | Preprocessing Critic | {created_at} | |
| split_design | Split Design Actor | configs/split_strategy.yaml | NOT_STARTED | 0 | Split Design Critic | {created_at} | |
| baselines | Baseline Model Actor | models/baseline_registry.md | NOT_STARTED | 0 | Baseline Critic | {created_at} | |
| model_selection | Model Selection Actor | configs/model_candidates.yaml | NOT_STARTED | 0 | Model Selection Critic | {created_at} | |
| training | Training Actor | reports/training_run.md | NOT_STARTED | 0 | Training Critic | {created_at} | |
| evaluation | Evaluation Actor | evaluation/evaluation_matrix.yaml | NOT_STARTED | 0 | Evaluation Critic | {created_at} | |
| error_analysis | Error Analysis Actor | error_analysis/error_analysis.md | NOT_STARTED | 0 | Error Analysis Critic | {created_at} | |
| improvement_plan | Improvement Planner Actor | reports/improvement_plan.md | NOT_STARTED | 0 | Improvement Critic | {created_at} | |
| model_saving | Model Saving Actor | final/run_summary.md | NOT_STARTED | 0 | Model Saving Critic | {created_at} | |
| final_report | Final Report Actor | final/experiment_report.md | NOT_STARTED | 0 | Final Report Critic | {created_at} | |

## Iteration ledger

| iteration | trigger | weakest_stage | rerun_from | target_gap_before | target_gap_after | decision |
|---:|---|---|---|---|---|---|
| 0 | initial run | TBD | bootstrap | TBD | TBD | TBD |

## Current blockers

- none

## Decisions log

### {created_at}
- Created experiment.
- Waiting for first-turn restatement and bootstrap.

## Example accepted entry

### 2026-05-21T10:42:00Z
- stage: data_profile
- actor: Database Profiling Actor
- artifact: data_profile/database_profile.md
- status: ACCEPTED
- attempt: 1
- critic: Data Profile Critic
- decision_summary: schema, label coverage, CRS, missingness, and candidate leakage surfaces documented
- next_stage: experiment_design
