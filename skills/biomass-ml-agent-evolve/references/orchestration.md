# Orchestration

## Purpose
Define the operating system of the skill.

## Orchestrator responsibilities
- Resolve task and unknowns
- Decide start point and rerun point
- Spawn Actors and Critics
- Update IMPLEMENTATION_PLAN.md after every state change
- Enforce artefact naming and directory conventions
- Assemble final package

## Actor registry

The complete registry is persisted in each experiment's `IMPLEMENTATION_PLAN.md`. The canonical mapping is:

| Actor | Primary artefact | Format | Primary reference | Critic | Retry limit |
|---|---|---|---|---|---:|
| Research Actor | `research/deep_research.md` | Markdown | `references/deep_research.md` | Research Critic | 3 |
| Database Profiling Actor | `data_profile/database_profile.md` | Markdown | `references/database_preprocessing.md` | Data Profile Critic | 3 |
| Experimental Design Actor | `configs/experiment_design.md` | Markdown | `references/experimental_design.md` | Experiment Design Critic | 3 |
| Preprocessing Actor | `preprocessing/preprocessing_spec.md` | Markdown | `references/database_preprocessing.md` | Preprocessing Critic | 3 |
| Split Design Actor | `configs/split_strategy.yaml` | YAML | `references/database_preprocessing.md` | Split Design Critic | 3 |
| Baseline Model Actor | `models/baseline_registry.md` | Markdown | `references/model_selection.md` | Baseline Critic | 3 |
| Model Selection Actor | `configs/model_candidates.yaml` | YAML | `references/model_selection.md` | Model Selection Critic | 3 |
| Training Actor | `reports/training_run.md` | Markdown | `references/training.md` | Training Critic | 3 |
| Evaluation Actor | `evaluation/evaluation_matrix.yaml` | YAML | `references/evaluation.md` | Evaluation Critic | 3 |
| Error Analysis Actor | `error_analysis/error_analysis.md` | Markdown | `references/error_analysis.md` | Error Analysis Critic | 3 |
| Improvement Planner Actor | `reports/improvement_plan.md` | Markdown | `references/improvement_loop.md` | Improvement Critic | 3 |
| Model Saving Actor | `final/run_summary.md` (primary; required secondary outputs listed in `references/model_saving.md`) | Markdown | `references/model_saving.md` | Model Saving Critic | 3 |
| Final Report Actor | `final/experiment_report.md` | Markdown | `references/orchestration.md` | Final Report Critic | 3 |

Upstream dependencies are implicit in the order above: every Actor requires all previously-ACCEPTED artefacts in the table. Retry limit is the per-stage default (initial attempt plus 2 revisions); after 3 the stage escalates as BLOCKED.

## Standard Actor prompt

```text
SYSTEM
You are the {ACTOR_NAME} for biomass-ml-agent-evolve. You perform one atomic stage only.

READ FIRST
- {experiment_dir}/IMPLEMENTATION_PLAN.md
- {reference_file}
- All required upstream accepted artefacts listed for your stage

TASK
- Produce the requested stage artefact only.
- Follow the acceptance gates in {reference_file}.
- Be faithful to source data and accepted research.
- Record assumptions explicitly.
- Record reproducibility details: inputs, versions, seeds, commands, file hashes where available.
- Do not edit unrelated files.
- If required input is missing or invalid, write a blocker section in the artefact and stop.

WRITE
- Artefact path: {artifact_path}
- Format: {artifact_format}

RETURN ONLY
PATH: {artifact_path}
SUMMARY: <2-4 sentences, no extra commentary>
```

Stage-specific addenda live in the actor registry rather than being duplicated in this prompt.

## Standard Critic prompt

```text
SYSTEM
You are the {ACTOR_NAME} Critic for biomass-ml-agent-evolve. Review one artefact only.

READ
- {artifact_path}
- {reference_file}
- Relevant upstream accepted artefacts
- Current evaluation targets from {experiment_dir}/IMPLEMENTATION_PLAN.md

CHECK
- Correctness
- Completeness
- Reproducibility
- Leakage risk
- Metric validity
- Faithfulness to source data/research
- Compliance with {reference_file}

DECISION
- ACCEPT if the artefact is safe for downstream use.
- REJECT if any issue would invalidate or materially weaken later stages.

RETURN
STATUS: ACCEPT | REJECT
REASONS:
- ...
REQUIRED_REVISIONS:
- ...
```

## Validation states
- NOT_STARTED
- IN_PROGRESS
- ACCEPTED
- REJECTED
- BLOCKED
- ESCALATED

## Plan update rules
Every stage update must record:
- timestamp
- stage
- owner
- attempt number
- status
- artifact path
- blocker or rejection reason
- next action

`reports/bootstrap_summary.md` is produced once by `scripts/bootstrap_experiment.sh` and is the Orchestrator's read-on-resume artefact: it carries `experiment_id`, `created_at`, declared `subject`/`task`/`geography`, and detected integration env-var presence. The Orchestrator must read it before any stage on a resumed run.

## Retry and escalation policy

| Situation | Action |
|---|---|
| Actor draft rejected once | Revise the same artefact in place and resubmit to Critic. |
| Actor draft rejected twice | Orchestrator intervenes, rewrites the brief more tightly, and allows one final attempt. |
| Third rejection | Escalate stage as BLOCKED; the pipeline may not continue past that dependency. |
| Benchmark target appears unrealistic | Research Actor documents benchmark evidence; Orchestrator proposes a revised threshold or escalates to user. |
| Missing labels or invalid target variable | Escalate immediately; no modelling proceeds. |
| Leakage detected | Invalidate downstream stage results from split design onwards and rerun from the earliest contaminated point. |
| Budget exhausted mid-loop | Save best ACCEPTED state, record unmet gaps, and assemble a partial but reproducible final package. |

## Assembly rules
The Orchestrator may assemble only from ACCEPTED artifacts.

## Final QA gate
Block assembly unless every path below resolves and is non-empty:

- `IMPLEMENTATION_PLAN.md` (status table: every stage ACCEPTED or explicitly waived)
- `research/deep_research.md`
- `data_profile/database_profile.md`
- `configs/experiment_design.md`
- `preprocessing/preprocessing_spec.md`
- `preprocessing/feature_schema.json`
- `preprocessing/data_version.txt`
- `configs/split_strategy.yaml`
- `models/baseline_registry.md`
- `configs/model_candidates.yaml`
- `configs/training_config.yaml`
- `reports/training_run.md`
- `checkpoints/best.ckpt`
- `checkpoints/metrics_history.csv`
- `evaluation/evaluation_matrix.yaml` (no null thresholds for metrics named in `configs/experiment_design.md` acceptance list)
- `error_analysis/error_analysis.md`
- `final/model/` (non-empty)
- `final/preprocessing_pipeline/` (non-empty)
- `final/evaluation_matrix.yaml`
- `final/model_card.md`
- `final/data_card.md`
- `final/run_summary.md`
- `final/experiment_report.md`
- `final/environment.lock` (or `final/environment.txt`)
- `final/git_snapshot.txt` (commit SHA or code-snapshot pointer; the literal string `not_available` is acceptable only if the run is recorded as non-versioned in `run_summary.md`)

Additional gates:
- evaluation matrix complete and consistent with the ACCEPTED `configs/split_strategy.yaml` (random_split_used, leakage_checks_passed match)
- final model traceable to config and checkpoint (paths recorded in `run_summary.md`)
- final report cites only ACCEPTED artefact evidence
