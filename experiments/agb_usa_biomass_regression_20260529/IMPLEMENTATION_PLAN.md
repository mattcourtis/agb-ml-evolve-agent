# IMPLEMENTATION_PLAN

## Experiment header

- experiment_id: agb_usa_biomass_regression_20260529
- created_at: 2026-05-29T09:03:38Z
- subject: agb_usa
- geography: CONUS forest joint v2 (WV Appalachia + Upper Midwest WI/MN + Northeast Maine bloc, 23 projects)
- task: biomass_regression
- target_variable: CO2 (tCO2/acre standing stock; ANEW gpkg column)
- spatial_resolution: 10 m features, ~14.7 m plot footprint (1/24 acre)
- temporal_horizon: annual, 2022 + 2023 pooled
- evaluation_metrics: r2, rmse, mae, bias, per_quintile_bias, predicted_range_discrimination, per_ecoregion_r2
- performance_threshold: realistic R² ≥ 0.40 / stretch R² ≥ 0.55; iteration-0 must reproduce joint_v2 R²=0.42 within ±0.03
- output_dir: ./experiments/agb_usa_biomass_regression_20260529
- compute_budget: Small (LightGBM, minutes per fold; project-LOPO over 23 projects)
- runtime_budget: 1 working day per iteration (8 hours)
- status: ITERATION_0_COMPLETE (wiring validation reproduced R²=0.4182; ready for iteration 1)

## Task restatement

Reproduce the existing `tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/` baseline (R²=0.42, RMSE=57 tCO₂/acre, MAE=41, 4,636 plots, 23 projects) inside the `biomass-ml-agent-evolve` orchestrator. Iteration 0 is a **wiring validation** — the model code already exists in `tf-deep-landcover/src/agb/`; the agent must invoke it via the cross-repo contract (`references/cross_repo_invocation.md`), capture both repos' SHAs in `final/git_snapshot.txt`, populate `evaluation/evaluation_matrix.yaml` with the new biomass-specific metrics (per_quintile_bias, predicted_range_discrimination, per_ecoregion_r2), and assemble the `final/` bundle. If iteration 0 reproduces R²≈0.42, the agent advances to iteration 1, which is feature-insufficiency hypothesis testing (GEDI integration; route deferred to Research Actor).

## Benchmark anchor

- user_target_provided: yes (joint_v2 baseline R²=0.42 / RMSE=57 / MAE=41)
- benchmark_range: realistic R² ≥ 0.40 / stretch R² ≥ 0.55 — see `references/deep_research.md` default-threshold anchor table for CONUS forest AGB plot-LOPO
- realistic_default_target: R² ≥ 0.40, RMSE ≤ 60 tCO₂/acre, |bias| ≤ 5
- stretch_target: R² ≥ 0.55, RMSE ≤ 45 tCO₂/acre, predicted_range_discrimination ≥ 0.6
- benchmark_notes: Iteration 0 acceptance is reproducing the existing baseline within ±0.03 R². The published `tf-deep-landcover/docs/runs/agb_usa.md` investigation already showed isotonic calibration, Huber loss, log-target, and footprint-weighted sampling each fail to lift the embeddings-only ceiling — `references/improvement_loop.md` Critic addendum forbids the agent from re-litigating those levers. Iteration 1's anticipated lever is GEDI canopy height; access route (GEE-asset mirror of the sugar pipeline vs. LP-DAAC `earthaccess`) is deferred to the Research Actor.

## Artifact registry

| stage | actor | artifact | status | attempt | critic | last_update | notes |
|---|---|---|---|---:|---|---|---|
| research | Research Actor (Orchestrator) | research/deep_research.md | ACCEPTED | 1 | Research Critic | 2026-05-29T09:36:00Z | benchmark + ruled-out levers fixed |
| data_profile | Database Profiling Actor (Orchestrator) | data_profile/database_profile.md | ACCEPTED | 1 | Data Profile Critic | 2026-05-29T09:36:00Z | 4636/4646 modelled; 4 ecoregions |
| experiment_design | Experimental Design Actor (Orchestrator) | configs/experiment_design.md | ACCEPTED | 1 | Experiment Design Critic | 2026-05-29T09:36:00Z | acceptance metrics + LOPO split |
| preprocessing | Preprocessing Actor (Orchestrator) | preprocessing/preprocessing_spec.md | ACCEPTED | 1 | Preprocessing Critic | 2026-05-29T09:36:00Z | reuse features.parquet (no re-extraction) |
| split_design | Split Design Actor (Orchestrator) | configs/split_strategy.yaml | ACCEPTED | 1 | Split Design Critic | 2026-05-29T09:36:00Z | project_name LOPO, no random split |
| baselines | Baseline Model Actor (Orchestrator) | models/baseline_registry.md | ACCEPTED | 1 | Baseline Critic | 2026-05-29T09:36:00Z | mean R²=-0.01, ridge-PC20 R²=0.37 |
| model_selection | Model Selection Actor (Orchestrator) | configs/model_candidates.yaml | ACCEPTED | 1 | Model Selection Critic | 2026-05-29T09:36:00Z | lightgbm_emb64 fixed (reproduction) |
| training | Training Actor (cross-repo) | reports/training_run.md | ACCEPTED | 1 | Training Critic | 2026-05-29T09:35:00Z | R²=0.4182 bit-identical to baseline |
| evaluation | Evaluation Actor (Orchestrator) | evaluation/evaluation_matrix.yaml | ACCEPTED | 1 | Evaluation Critic (subagent ACCEPT) | 2026-05-29T09:37:00Z | full biomass matrix, no nulls |
| error_analysis | Error Analysis Actor (Orchestrator) | error_analysis/error_analysis.md | ACCEPTED | 1 | Error Analysis Critic | 2026-05-29T09:38:00Z | range compression; weakest stage = features |
| improvement_plan | Improvement Planner Actor | reports/improvement_plan.md | WAIVED | 0 | Improvement Critic | 2026-05-29T09:39:00Z | iteration-1 trigger captured in error_analysis + eval matrix; no separate plan needed for wiring run |
| model_saving | Model Saving Actor (Orchestrator) | final/run_summary.md | ACCEPTED | 1 | Model Saving/Cross-Repo Critic (subagent ACCEPT) | 2026-05-29T09:39:00Z | final bundle assembled; both SHAs |
| final_report | Final Report Actor (Orchestrator) | final/experiment_report.md | ACCEPTED | 1 | Final QA gate Critic (subagent ACCEPT after plan update) | 2026-05-29T09:40:00Z | cites accepted artefacts only |

## Iteration ledger

| iteration | trigger | weakest_stage | rerun_from | target_gap_before | target_gap_after | decision |
|---:|---|---|---|---|---|---|
| 0 | wiring validation — reproduce joint_v2 baseline inside the new orchestrator | N/A — reproducing existing baseline | bootstrap → (research → data_profile → preprocess → split → baselines → model_selection → training → evaluate → save) | R²=0 (no model yet); target R²≈0.42 | R²=0.4182 (Δ −0.0018 vs 0.42; bit-identical to baseline) | PROCEEDED — reproduction confirmed; advance to iteration 1 (GEDI feature test) |
| 1 (anticipated) | feature ceiling rediscovered (Q1/Q5 quintile-bias collapse confirmed) | feature insufficiency | research → preprocess → train → evaluate (new feature source) | R²≈0.42 (from iter 0) | target R² ≥ 0.55 with predicted_range_discrimination ≥ 0.6 | Research Actor's iteration-1 first deliverable: GEDI access feasibility note comparing GEE-asset (mirror sugar pipeline) vs LP-DAAC (`earthaccess`) routes — cost, latency, coverage. Skill must not pre-commit to a route. |

## Current blockers

- None for iteration 0 (complete). **Iteration-1 preflight:** if the GEDI route lands on the GEE-asset path, GCP_PROJECT / EE_PROJECT / GOOGLE_APPLICATION_CREDENTIALS must be set and the `earthengine` CLI installed (all currently unset). The LP-DAAC `earthaccess` route would need an Earthdata login instead. Route choice deferred to the iteration-1 Research Actor.

## Decisions log

### 2026-05-29T09:03:38Z
- Created experiment.

### 2026-05-29T09:30:00Z
- First-turn restatement captured: subject `agb_usa`, geography `CONUS forest joint v2 (WV+Midwest+NE, 23 projects)`, task `biomass_regression`, target `CO2` (tCO₂/acre standing stock), pool 4,636 plots from prior `tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/`.
- Performance threshold locked: iteration-0 must reproduce R²=0.42 ± 0.03; realistic R² ≥ 0.40 / stretch R² ≥ 0.55.
- Iteration 0 pre-registered as a **wiring validation** against the existing tf-deep-landcover/src/agb baseline. Iteration 1 pre-registered as **feature insufficiency → GEDI integration** (route deferred to Research Actor in iteration 1).
- Cross-repo invocation contract loaded from `references/cross_repo_invocation.md`. Actors must invoke `tf-deep-landcover/src/agb/extract_features_batched.py` and `tf-deep-landcover/src/agb/train_agb_lgbm.py`, never reimplement. `final/git_snapshot.txt` must capture both repos' SHAs.
- Status: NOT_STARTED → ready for Orchestrator to delegate the Research Actor.

### 2026-05-29T09:40:00Z — iteration 0 COMPLETE
- All stages ACCEPTED (improvement_plan WAIVED — iteration-1 trigger is captured in `error_analysis/error_analysis.md` and the eval-matrix `overall_decision`; no separate plan needed for a wiring run).
- **Reproduction confirmed bit-identically:** R²=0.4182, RMSE=56.58, MAE=41.49, bias=+0.50, n=4636, 23 project-LOPO folds — equal to the prior `joint_v2` baseline `metrics.json` and within ±0.03 of the 0.42 target.
- Training invoked via cross-repo `tf-deep-landcover/src/agb/train_agb_lgbm` @ e8c70584...; never reimplemented. Features reused (no re-extraction). Breakdown metrics computed from baseline `oof.parquet` (justified exact by the bit-identical retrain).
- Biomass diagnostics confirm the documented **feature ceiling**: per_quintile_bias Q1 +35.6 → Q5 −72.1; predicted_range_discrimination 0.468 (WV 0.19); weakest ecoregion Appalachian R²=0.157. These FAIL the stretch thresholds by design — not a wiring/tuning fault.
- Load-bearing artefacts independently reviewed by Critic subagents: Evaluation ACCEPT, Model-Saving/Cross-Repo ACCEPT, Final QA gate ACCEPT (after this plan-status update; the gate correctly rejected the pre-update state).
- `final/git_snapshot.txt` records both SHAs (crop ea0551b…, tf-deep-landcover e8c70584…); iteration-0 artefacts are uncommitted in the working tree (origin is a third-party fork — no push per project policy).
- **Decision:** PROCEED to iteration 1 — add GEDI canopy height; rerun boundary research → preprocess → train → evaluate. GEDI access route (GEE-asset vs LP-DAAC `earthaccess`) deferred to the iteration-1 Research Actor.

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
