# Skill Evolution Ledger

Append one row per evolution pass. Never edit historical rows.

## Header

- skill: biomass-ml-agent-evolve
- created_at: {created_at}
- max_passes_per_month: 3
- max_proposals_per_pass: 5

## Pass log

| pass_id | timestamp_utc | experiment_id | trigger | proposals_count | accepted_by_critic | approved_by_user | applied | proposal_artifact | notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| (example) | 2026-01-01T00:00:00Z | agb_usa_biomass_regression_20260101 | target_met | 2 | 2 | 1 | 1 | experiments/agb_usa_biomass_regression_20260101/reports/skill_evolution_proposal.md | first pass |

## Applied edits log

| applied_at_utc | pass_id | target_file | edit_summary | reverted | revert_reason |
|---|---|---|---|---|---|
| (example) | (example) | references/database_preprocessing.md | tightened leakage checklist | no | |

## Rejected proposals log

| timestamp_utc | pass_id | proposal_summary | rejection_reason |
|---|---|---|---|
