# agb-ml-agent-evolve — implementation record

Source design: `agb-evolve-review.md`
Skill target: `skills/biomass-ml-agent-evolve/`

## Skill build (2026-05-28/29)

- [x] Scaffold from `crop-ml-agent-evolve` (verbatim copy baseline)
- [x] Tailor all 13 reference files to biomass/forest domain
  - [x] orchestration.md — added cross-repo invocation section
  - [x] experimental_design.md — biomass task taxonomy; GEDI/feature-ceiling framing
  - [x] deep_research.md — CONUS forest AGB default-threshold anchor table
  - [x] database_preprocessing.md — leakage controls generic; no crop-calendar alignment
  - [x] model_selection.md — biomass regression / canopy-height ladder
  - [x] training.md — generic; reproducibility and checkpointing unchanged
  - [x] evaluation.md — added per_quintile_bias, predicted_range_discrimination, per_ecoregion_r2
  - [x] error_analysis.md — biomass lenses (canopy cover, forest type, stand age, GEDI density)
  - [x] improvement_loop.md — feature-ceiling veto; GEDI as next lever
  - [x] model_saving.md — generic; final bundle schema unchanged
  - [x] source_registry.md — GEDI L2A/L4A/L4B, NEON AOP, ESA CCI Biomass, PALSAR-2, COPDEM + full access routes, auth, licences
  - [x] cross_repo_invocation.md — new; codifies how skill invokes `tf-deep-landcover/` entry points and captures dual SHA
  - [x] skill_evolution.md — carried over unchanged
- [x] Fix AlphaEarth Foundation (AEF) naming throughout (was incorrectly "OlmoEarth AEF")
- [x] Add COPDEM GLO-30 to preferred topography sources (GEE `COPERNICUS/DEM/GLO30`)
- [x] Add `pyproject.toml` with ruff dev dependency

## Iteration 0 — wiring validation (2026-05-29, COMPLETE)

Experiment: `experiments/agb_usa_biomass_regression_20260529/`

- **Result:** R²=0.4182, RMSE=56.58, MAE=41.49, bias=+0.50, n=4636, 23 project-LOPO folds
- **Verdict:** bit-identical reproduction of the prior `joint_v2` baseline (within ±0.03 tolerance) — wiring confirmed
- **Feature ceiling confirmed:** per_quintile_bias Q1 +35.6 → Q5 −71.1; predicted_range_discrimination 0.468 (WV 0.19)
- All stages ACCEPTED by Critic subagents; final bundle assembled at `experiments/agb_usa_biomass_regression_20260529/final/`

## Iteration 1 — GEDI feature integration (pending)

- **Trigger:** feature ceiling rediscovered (Q1/Q5 collapse); optical embeddings cannot resolve vertical structure
- **Lever:** GEDI canopy height as added feature (spaceborne LiDAR measures what optical sensors cannot)
- **Target:** R² ≥ 0.55, predicted_range_discrimination ≥ 0.6
- **First deliverable:** Research Actor access-feasibility note — GEE asset (`LARSE/GEDI/GEDI02_A_002_MONTHLY`) vs LP DAAC `earthaccess` route; cost, latency, CONUS coverage
- **Preflight blocker:** `GCP_PROJECT` / `GOOGLE_APPLICATION_CREDENTIALS` unset if GEE route chosen; NASA Earthdata login needed if LP DAAC route chosen

## Risks to monitor

- GEDI coverage gaps in steep terrain (WV Appalachia) may limit the canopy-height signal for the weakest ecoregion (R²=0.157 in iteration 0).
- The `skill_evolution_ledger.template.md` is intentionally append-only — do not mutate historical rows.
- The Final QA gate enumerates required artefact paths. If new artefacts are added, update the gate list in `references/orchestration.md` or assembly will (correctly) block.
