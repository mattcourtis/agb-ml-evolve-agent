# agb-ml-agent-evolve

A Claude skill that runs an end-to-end ML workflow for biomass regression, canopy-height regression, biomass segmentation, and forest change-detection pipelines using satellite, LiDAR, and SAR features.

The skill is implemented under `skills/biomass-ml-agent-evolve/` and follows the orchestrator/actor/critic pattern documented in `agb-evolve-review.md`.

## Layout

```
skills/biomass-ml-agent-evolve/
├── SKILL.md
├── references/
│   ├── orchestration.md
│   ├── experimental_design.md
│   ├── deep_research.md
│   ├── database_preprocessing.md
│   ├── model_selection.md
│   ├── training.md
│   ├── evaluation.md
│   ├── error_analysis.md
│   ├── improvement_loop.md
│   ├── model_saving.md
│   ├── source_registry.md
│   ├── cross_repo_invocation.md
│   └── skill_evolution.md
├── assets/
│   ├── IMPLEMENTATION_PLAN.template.md
│   ├── evaluation_matrix.template.yaml
│   ├── experiment_config.template.yaml
│   ├── model_card.template.md
│   ├── data_card.template.md
│   ├── integrations.env.example
│   └── skill_evolution_ledger.template.md
└── scripts/
    ├── bootstrap_experiment.sh
    ├── doctor.sh
    └── validate_skill.sh
```

## How to use

1. From the project root, bootstrap a new experiment:

   ```bash
   ./skills/biomass-ml-agent-evolve/scripts/bootstrap_experiment.sh agb_usa biomass_regression "CONUS forest"
   ```

   Optional flags: `--force` archives an existing same-day experiment dir and recreates it cleanly. The `task` argument must be one of `biomass_regression | canopy_height_regression | biomass_segmentation | change_detection`. The `subject` argument is slugified (lowercased, non-alphanumerics → `_`).

   This creates `experiments/agb_usa_biomass_regression_YYYYMMDD/` with the full directory tree, the `IMPLEMENTATION_PLAN.md`, a seeded `configs/experiment_config.yaml`, a seeded `evaluation/evaluation_matrix.yaml`, draft card files at `reports/model_card.md` and `reports/data_card.md`, a copy of `integrations.env.example`, and a `reports/bootstrap_summary.md`. The script aborts (exit 4) if any `{placeholder}` token remains in the rendered files.

2. Edit the first-run inputs that cannot be inferred safely:

   - `configs/experiment_config.yaml`
   - `IMPLEMENTATION_PLAN.md`
   - `configs/integrations.env.example` or matching environment variables

3. Open the skill in your Claude client and follow the protocol in `skills/biomass-ml-agent-evolve/SKILL.md`. The Orchestrator restates the task, then runs Actors and Critics stage by stage until a stop condition is reached.

4. After an experiment hits a stop condition, the optional Skill Evolution pass (`references/skill_evolution.md`) can propose conservative, evidence-cited edits to the skill itself. Edits are gated by a Critic and explicit user approval.

## Core principles

- Orchestrator owns planning, validation, retries, escalation, and final assembly.
- Each Actor writes exactly one artefact and returns only `PATH:` + `SUMMARY:`.
- Every Actor output passes through a Critic before downstream work depends on it.
- Status, retries, rerun boundaries, and decisions are persisted in `IMPLEMENTATION_PLAN.md`.
- Improvements always start at the earliest plausible upstream failure stage.

## Integrations (optional)

- Google Earth Engine — set `GCP_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`, `EE_SERVICE_ACCOUNT`.
- Hugging Face Hub — set `HF_TOKEN`.
- AWS S3 — set `AWS_PROFILE`, `AWS_REGION`, `S3_BUCKET`.

The bootstrap script does not enforce these — it writes `configs/integrations.env.example` for the user to fill in.

## Product checks

Run the contract validator after editing skill files:

```bash
./skills/biomass-ml-agent-evolve/scripts/validate_skill.sh
```

Run the bootstrap smoke test after editing templates or bootstrap logic:

```bash
bash tests/bootstrap_smoke.sh
```

Run the readiness doctor before a real experiment:

```bash
./skills/biomass-ml-agent-evolve/scripts/doctor.sh
```

## Reference

The design rationale, porting decisions, and gap analysis live in `agb-evolve-review.md`.
