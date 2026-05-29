---
name: biomass-ml-agent-evolve
description: Design and run an end-to-end ML workflow for biomass regression, canopy-height regression, biomass segmentation, or forest change-detection pipelines using satellite, LiDAR, and SAR features.
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, LS
---

# biomass-ml-agent-evolve

Use this skill when the task is to design, train, evaluate, improve, and package a biomass / forest-structure ML workflow from raw data or an existing database.

Read `references/orchestration.md` first. Keep this file concise. All detailed procedure lives in `references/`.

## First-turn protocol

Before doing any stage work, the Orchestrator must restate and confirm:

- Subject (e.g., above-ground biomass, canopy height, biomass change)
- Geography (ecoregion, country, or project bloc)
- Task type: biomass_regression, canopy_height_regression, biomass_segmentation, or change_detection
- Input data sources (e.g., GEDI, ALOS PALSAR-2, foundation embeddings, NEON AOP, field plots)
- Target variable (e.g., AGB tCO₂/acre, RH98 m, biomass-change Δt)
- Spatial resolution and plot footprint
- Temporal horizon (annual / multi-year window)
- Evaluation metrics
- Performance threshold
- Compute/runtime constraints
- Output directory

Default output directory:

`experiments/{subject}_{task}_{YYYYMMDD}/`

If any required item is missing:
- infer only when the risk is low
- otherwise mark it as `NEEDS_USER_INPUT`
- do not invent labels, targets, or evaluation claims

## Required experiment directory

Use:

experiments/{subject}_{task}_{YYYYMMDD}/
├── IMPLEMENTATION_PLAN.md
├── research/
├── data_profile/
├── preprocessing/
├── configs/
├── models/
├── checkpoints/
├── evaluation/
├── error_analysis/
├── reports/
└── final/

Create it with `scripts/bootstrap_experiment.sh`.

## Core loop

Run this loop exactly:

resolve task
→ bootstrap experiment dir
→ inspect database
→ design experiment
→ run deep research
→ define evaluation matrix
→ preprocess data
→ train baselines
→ select model candidates
→ train candidate models
→ evaluate
→ save artifacts
→ run error analysis
→ identify weakest upstream stage
→ improve that stage
→ rerun impacted pipeline
→ repeat until stop condition

## Orchestrator contract

The Orchestrator owns:
- planning
- delegation
- validation gates
- retry decisions
- escalation
- assembly of final outputs
- updates to `IMPLEMENTATION_PLAN.md`

The Orchestrator must:
- never skip Critic review
- never allow downstream dependency on a rejected artifact
- always repair the earliest likely failed upstream stage
- always record rerun boundaries and reasons

## Actor contract

Actors perform one atomic stage only.

Every Actor must:
- read the relevant reference file
- write one primary artifact file
- include assumptions, provenance, and reproducibility details
- return ONLY these two lines, in this exact format, with no preamble, no follow-up question, no extra commentary:

  ```
  PATH: <absolute or experiment-relative artefact path>
  SUMMARY: <2-4 sentences>
  ```

The full Actor and Critic prompt templates live in `references/orchestration.md`. No Actor may return a long conversational answer in place of the artifact.

## Critic contract

Every Actor output must be reviewed by a matching Critic.

Every Critic must check:
- correctness
- completeness
- reproducibility
- leakage risk
- metric validity
- faithfulness to source data/research
- compliance with the relevant reference file

The Critic returns:
- ACCEPT
- or REJECT with exact revision instructions

## Retry policy

Default per-stage limit:
- initial attempt
- up to 2 revisions
- then escalate

Default pipeline improvement limit:
- up to 4 full iterations unless config states otherwise

Escalate immediately if:
- labels are missing or invalid
- required credentials or data access are unavailable
- the user target is materially inconsistent with research-backed benchmark ranges
- leakage cannot be resolved automatically
- budget or deadline makes further training invalid or non-credible

## Stop conditions

Stop when one of the following is true:
- performance target is met
- research-derived benchmark is met
- max iterations reached
- compute or time budget exhausted
- data is insufficient and user escalation is required

## Final package

The final deliverable must include every path below, all non-empty (`git_snapshot.txt` may contain `not_available` only if explicitly justified in `run_summary.md`):

final/
├── model/                       # serialised weights + loader notes
├── preprocessing_pipeline/      # transform code, fitted scalers/encoders
├── evaluation_matrix.yaml
├── model_card.md
├── data_card.md                 # schema in references/model_saving.md
├── experiment_report.md
├── run_summary.md
├── environment.lock             # pip freeze / conda export / uv lock
└── git_snapshot.txt             # commit SHA or code-snapshot pointer

Also persisted upstream (referenced by the final package, not duplicated):
- `checkpoints/best.ckpt`, `checkpoints/metrics_history.csv`
- `preprocessing/feature_schema.json`, `preprocessing/data_version.txt`
- `configs/training_config.yaml`
- `error_analysis/error_analysis.md`

The Final QA gate (`references/orchestration.md`) lists every required path and the cross-checks the Critic must run before assembly.

## Reference files

Read the relevant file before each stage:

- `references/orchestration.md`
- `references/experimental_design.md`
- `references/deep_research.md`
- `references/database_preprocessing.md`
- `references/model_selection.md`
- `references/training.md`
- `references/evaluation.md`
- `references/error_analysis.md`
- `references/improvement_loop.md`
- `references/model_saving.md`
- `references/cross_repo_invocation.md`

Optional post-completion mode (read only after a stop condition is reached):

- `references/skill_evolution.md` — bounded, human-gated self-improvement pass; never runs mid-pipeline.
