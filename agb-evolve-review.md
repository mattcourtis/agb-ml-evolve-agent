# AGB Modelling × crop-ml-agent-evolve — Setup Review

Date: 2026-05-28
Author: Claude (Opus 4.7) on behalf of mcourtis
Scope: assess what is needed to apply this repo's `crop-ml-agent-evolve` skill to the USA above-ground biomass (AGB) regression challenge described in `../tf-deep-landcover/docs/runs/agb-modelling-context.md`. The goal of the agentic loop is to **explore alternative datasets and methods** for AGB modelling and converge on improvements past the current R² ≈ 0.42 plot-level ceiling.

## 1. Executive summary

The skill's orchestrator → actor → critic loop is structurally a good fit for AGB: it already enforces leakage-safe spatial holdouts, benchmark-anchored thresholds, reproducibility footers, an error-analysis → earliest-weak-stage mapping, and a human-gated `skill_evolution` pass. The yield-forecasting branch is also tabular-regression flavoured (RF / XGBoost / LightGBM / temporal NN ladder), which mirrors the existing AGB LightGBM pipeline.

However, the skill is **crop-centric by contract**. Five concrete gaps block a clean reuse:

1. **Task vocabulary** — `bootstrap_experiment.sh` hard-rejects anything other than `detection|classification|acreage|yield_forecast|combined`. AGB is forest-biomass regression; closest fit is `yield_forecast` but it isn't yield.
2. **Source registry** has no LiDAR, biomass-product, or foundation-embedding sources (no GEDI, ESA CCI Biomass, ALOS PALSAR-2, ICESat-2, NEON AOP, OlmoEarth/AEF, Presto, SRTM/Copernicus DEM, HLS).
3. **Model ladder** lacks multimodal-fusion options (optical embedding + LiDAR height + SAR) and foundation-model-embedding stacking, which the AGB doc identifies as the next lever.
4. **Evaluation matrix** has yield-style RMSE/MAE/R²/bias/calibration buckets but does not surface the **per-quintile bias** and **rank-discrimination ratio** that the AGB investigation already proved are the decisive diagnostics (isotonic calibration failure is the headline finding).
5. **Cross-repo working model** is undefined — the skill bootstraps an experiment directory inside *this* repo, but the data, code, and prior artefacts live in `../tf-deep-landcover/`. A convention is needed for where the agent reads vs. writes vs. invokes code.

Verdict: **feasible after ~1 day of skill-side adaptation work** (forking/extending the skill with a `regression` task type, adding biomass-relevant sources, and a small set of AGB-specific actors/critic addenda). All adaptations are additive, fit the existing extension pattern used by `references/skill_evolution.md`, and would survive the existing `validate_skill.sh` / `bootstrap_smoke.sh` gates.

## 2. Where AGB fits the framework, and where it doesn't

| Concern | Fit | Note |
|---|---|---|
| Orchestrator / Actor / Critic loop | Yes | Stage list (`research → data_profile → experimental_design → preprocessing → split → baselines → model_selection → training → evaluation → error_analysis → improvement → save → report`) maps 1:1 onto the AGB workflow already used informally. |
| Spatial holdout enforcement | Yes | `references/database_preprocessing.md` requires `split_audit.csv` proving zero partition-key intersection. AGB project-leave-one-out maps to `holdout_proof: project_name`. |
| Reproducibility footers | Yes | AGB pipeline already records git SHA, library versions, seeds; trivial to comply. |
| `task` vocabulary | **No** | Bootstrap blocks `regression` / `biomass` / `carbon`. Either widen the enum or piggyback on `yield_forecast` and override the language in the rendered files. |
| `crop` field as primary identifier | Partial | Slug works syntactically (`agb_usa`, `agb_wv_appalachia`); semantically misleading. Rename `crop` → `subject` in templates (additive: keep `crop` for back-compat). |
| Source registry | **No** | Crop-centric — no LiDAR, no biomass products, no DEM, no large-EO foundation embeddings. |
| Model ladder | Partial | Yield ladder includes LGBM; missing multimodal-fusion and foundation-embedding stacks. |
| Default-threshold anchors | **No** | Only maize-acreage and soybean-yield are tabulated. AGB needs a CONUS forest-biomass anchor row (e.g., GEDI L4B vs. NFI: R² ~0.45–0.62 county, ~0.30–0.50 plot LOPO; ESA CCI Biomass v5 systematic bias literature). |
| Evaluation metrics | Partial | Yield set (RMSE / MAE / R² / bias / calibration) is correct; missing per-quintile bias, predicted-range discrimination ratio, and per-ecoregion holdout scores. |
| Error-analysis lenses | Partial | Has `by_region`, `by_year`, `by_yield_quantile` — needs `by_biomass_quintile`, `by_canopy_cover_class`, `by_forest_type`, `by_stand_age`. |
| Skill-evolution mode | Yes | Can be used after iteration 1 to capture GEDI hypothesis and quintile-bias lens as standing additions. |

## 3. Required framework modifications (must-do before launch)

These changes are minimal, additive, and follow the lessons captured in `tasks/lessons.md` (no architectural drift, no gate relaxation).

### 3.1 Extend `task` enum in the bootstrap script

`skills/crop-ml-agent-evolve/scripts/bootstrap_experiment.sh:45-51` currently:

```bash
case "$task" in
  detection|classification|acreage|yield_forecast|combined) ;;
```

Add `biomass_regression` (or generic `regression`). Update the help text and the corresponding lists in:

- `references/experimental_design.md` "Task taxonomy"
- `references/model_selection.md` (add a regression ladder)
- `references/evaluation.md` (regression metrics already present under `yield:` — reuse and rename or alias)
- `assets/experiment_config.template.yaml` (no change needed; `task_type` is a free string)
- `assets/IMPLEMENTATION_PLAN.template.md` (no change needed; `{task}` is templated)

### 3.2 Add a `references/biomass_sources.md` extension (or augment `source_registry.md`)

Required new rows in the source hierarchy:

| Layer | Source | Use |
|---|---|---|
| LiDAR canopy height & biomass | NASA GEDI L2A (RH metrics), L4A (footprint AGBD), L4B (gridded AGBD) — via LP DAAC, MAAP, GEE | The identified next lever. Co-feature or co-target. |
| Spaceborne biomass products | ESA CCI Biomass v5 (100 m), JPL/Global Forest Watch products | Cross-checking, label augmentation in low-supervision regions. |
| Aerial LiDAR | NEON AOP discrete-return LiDAR (CONUS 30 m biomass, 1 m CHM) | High-quality co-supervision over NEON sites. |
| SAR | JAXA ALOS PALSAR-2 / -4 mosaics (GEE); Sentinel-1 GRD (GEE) | Optional fusion input; AGB doc has already shown PALSAR-2 alone is +0.02 R² noise — but worth retesting in combination with GEDI. |
| Foundation embeddings | OlmoEarth AEF (Source Coop COG); AlphaEarth / Presto / Prithvi / Clay / SatMAE / GeoFM | Embedding-stacking experiments. |
| Topography | Copernicus DEM 30 m; SRTM v3; ASTER GDEM | Stand-conditioning covariate. |
| Disturbance / cover | Hansen Global Forest Change; NLCD; LCMAP; GLAD ARD | Mask, stratify, condition. |
| Ground truth | ANEW field plots (project-internal); USFS FIA plot data (CONUS); NEON woody veg; FOS plot networks | Auxiliary supervision and bias diagnostics. |
| Climate covariates | TerraClimate; CHIRPS; PRISM | Ecoregion conditioning. |

Critic rule: each cited source must record `{name, url_or_doi, access_date, license, spatial_resolution, temporal_coverage}` — matches the existing schema.

### 3.3 Add a regression / biomass model ladder to `references/model_selection.md`

Append (do not replace existing ladders):

```
## Biomass regression ladder
- linear / ridge / lasso baseline on top-K PCs of embeddings
- random forest
- LightGBM / XGBoost  ← current AGB production
- LightGBM + GEDI-RH features
- multimodal fusion (optical embedding + GEDI + SAR + DEM) via concatenation or attention head
- temporal stacking (multi-year AEF + GEDI deltas)
- foundation-embedding ensemble (AEF + Presto + Clay) → meta-learner
- co-target training (LightGBM/NN with GEDI L4A as auxiliary target)
- end-to-end CNN/U-Net regressor over imagery (only at Large tier)
```

Each entry must declare expected runtime tier, data dependency (e.g., requires GEDI extraction), and known risks.

### 3.4 Tighten `references/evaluation.md` for AGB

Add to the yield-metrics block (or a new `regression:` block):

- `per_quintile_bias` — mean signed residual per true-target quintile (Q1..Q5). Decisive diagnostic per AGB doc.
- `predicted_range_discrimination` — `(predicted_Q5_mean − predicted_Q1_mean) / (true_Q5_mean − true_Q1_mean)`. Numerical proxy for the "the model collapses the dynamic range" failure mode.
- `external_holdout_r2` — required when the experiment claims generalisation to "new project". (Sylvania-style realistic-expectation slot.)
- `per_ecoregion_r2` — required when training pool spans more than one ecoregion.

Critic addendum: reject the evaluation matrix if `per_quintile_bias` is absent for any regression task whose `experiment_design.md` lists a generalisation claim.

### 3.5 Extend error-analysis lenses

`references/error_analysis.md` lenses today include `by_yield_quantile`. Add explicit lenses:

- `by_biomass_quintile` (or generalise to `by_target_quintile`)
- `by_canopy_cover_class` (sparse / open / closed)
- `by_forest_type` (hardwood / softwood / mixed)
- `by_stand_age` if available
- `by_terrain_slope` (DEM-derived)

Append a row to the error-analysis → stage mapping table in `references/improvement_loop.md`:

| Q1 over-prediction & Q5 under-prediction (feature ceiling) | feature insufficiency | research → preprocess → train → evaluate (with a new feature source) |

This is the *exact* lesson the AGB doc has already learned. Encoding it in the framework prevents an agent rediscovering it.

### 3.6 Default-threshold anchor row

In `references/deep_research.md` add to the anchor table:

| CONUS forest AGB regression at field-plot scale, project-LOPO | R² ≥ 0.40, RMSE ≤ 60 tCO₂/acre, |Bias| ≤ 5 | R² ≥ 0.55, RMSE ≤ 45 tCO₂/acre, predicted_range_discrimination ≥ 0.6 | Internal pilot (this repo's AGB doc) + GEDI L4B literature |

Record both the user-supplied target (if any) and the iteration cap that triggers escalation. The current AGB pipeline establishes R² = 0.42 as the "current best" — agents should not be allowed to *regress* below this without an explicit user-approved diagnostic excursion.

## 4. Optional / recommended modifications

- **Rename `crop` → `subject` in user-facing surfaces.** Keep `crop` aliased for back-compat (template rendering is one-line; both can render simultaneously). Lower priority than 3.1–3.6.
- **Add `--non_crop` flag to `bootstrap_experiment.sh`** that swaps `crop_calendar_aligned` for `annual_composite` in `experiment_config.template.yaml :: preprocessing.temporal_alignment`. AGB does not have a crop calendar.
- **Pre-register the GEDI hypothesis** in iteration 0 of the IMPLEMENTATION_PLAN, with the rerun boundary already declared (`research → preprocess → train → evaluate`). This forces the Research Actor to start by validating GEDI access rather than re-litigating the embedding-only ceiling.
- **Doctor checks** for GEDI / Source Coop access:
  - `aws s3 ls` against `s3://us-west-2.opendata.source.coop/<aef-bucket>/` (anonymous read)
  - `earthengine` CLI auth + project access to LP DAAC GEDI collections
  - Read access to the EC2 path `/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg` (currently the ground-truth file)

## 5. AGB-specific configuration sketch

For the first run, an agent should produce a populated `configs/experiment_config.yaml` resembling the following (filled values, no `TBD` left when bootstrap finishes — the post-render `{placeholder}` grep would catch any miss):

```yaml
experiment:
  experiment_id: "agb_usa_biomass_regression_20260528"
  seed: 42
  subject: "above_ground_biomass"   # new field; crop kept for compat
  crop: "agb_usa"                   # slug retained
  geography: "CONUS forest projects (WV Appalachia, Upper Midwest, NE Maine bloc)"
  task_type: "biomass_regression"

task:
  target_variable: "CO2_tCO2_per_acre"  # locked column from anew_gt
  spatial_resolution: "10 m features; ~14.7 m plot footprint"
  temporal_horizon: "annual 2022–2023"
  spatial_unit: "field plot (1/24 acre)"
  inference_unit: "plot, optional wall-to-wall 10 m raster"

data:
  raw_sources:
    - "/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg"
  preferred_sources:
    embeddings: ["OlmoEarth AEF (Source Coop COG)"]
    lidar:      ["GEDI L2A RH metrics", "GEDI L4A footprint AGBD"]
    sar:        ["JAXA PALSAR-2 mosaic (GEE)"]
    optical:    ["Sentinel-2 SR (GEE)"]
    topography: ["Copernicus DEM 30 m"]
    aux:        ["Hansen GFC", "NLCD"]
    labels:     ["ANEW field plots", "USFS FIA (auxiliary)"]
  gee: { enabled: true, cloud_project: "${GCP_PROJECT}" }
  huggingface: { enabled: true, repo_org_or_user: "treefera" }
  aws: { enabled: true, profile: "${AWS_PROFILE}", region: "us-west-2", s3_bucket: "${S3_BUCKET}" }

preprocessing:
  cloud_masking: "n/a (annual embeddings already gap-filled)"
  temporal_alignment: "annual_composite"   # not crop_calendar_aligned
  resampling_policy: "bilinear at plot centre, 3x3 mean pool"
  normalization: "fit_on_train_only"
  leakage_checks_required: true
  data_versioning_required: true

splits:
  strategy: "spatial_holdout"
  spatial_holdout_unit: "project_name"
  temporal_holdout_rule: null              # cross-year held inside project
  validation_fraction: 0.0                 # LOPO uses no static val pool
  test_fraction: "leave_one_project_out"

models:
  require_simple_baseline: true
  baseline_candidates:
    - "ridge_on_pc20_embedding"
  advanced_candidates:
    - "lightgbm_emb64"               # current production
    - "lightgbm_emb64_plus_gedi_rh"  # iteration 1 hypothesis
    - "lightgbm_emb64_plus_gedi_plus_dem"
    - "fusion_attention_emb_gedi_sar"

training:
  budget_tier: "Small"   # LGBM training is minutes per fold
  max_full_iterations: 4
  allow_extended_iterations: false
  per_actor_retry_limit: 3
  early_stopping: true

evaluation:
  metrics:
    - rmse
    - mae
    - r2
    - bias
    - per_quintile_bias
    - predicted_range_discrimination
    - per_ecoregion_r2
    - external_holdout_r2
  thresholds:
    r2: 0.55
    rmse: 45
    predicted_range_discrimination: 0.6
    per_quintile_bias_abs_max: 30
  benchmark_range:
    realistic: { r2: 0.40, rmse: 60 }
    stretch:   { r2: 0.55, rmse: 45 }

finalisation:
  save_model_card: true
  save_data_card: true
  push_to_huggingface: false
  mirror_to_s3: true
```

## 6. Cross-repo working model

The agent runs from `/home/mattc/code/crop-ml-agent-evolve/`, but the data and model code live in `/home/mattc/code/tf-deep-landcover/`. A clean convention:

- **Experiment metadata, plans, evaluation matrices, model cards, data cards, reports** → `crop-ml-agent-evolve/experiments/agb_usa_biomass_regression_<date>/` (the canonical skill output).
- **Feature extraction, training, inference code** → continues to live under `tf-deep-landcover/src/agb/`. The skill's actors *invoke* this code through `subprocess` / `uv run`, never reimplement it.
- **Checkpoints, parquet feature tables** → `tf-deep-landcover/experiments/agb/<pool>_<variant>/` (existing pattern), and the skill records absolute paths + SHA256 in `checkpoints/` and `preprocessing/data_version.txt`.
- **Git snapshot** → `final/git_snapshot.txt` must capture *both* repo SHAs (`crop-ml-agent-evolve` for the experiment, `tf-deep-landcover` for the model code). Extend the Model Saving Actor's checklist to record both.
- **Code edits to `tf-deep-landcover/`** are out of scope for the orchestrator's actors except for new feature-extractor scripts (e.g., `src/agb/extract_gedi_at_points.py`). Those are written by an **Implementation Actor** (an extension not present in the base skill — see §7).

This is the principal architectural delta from a same-repo experiment. The base skill assumes the experiment dir is self-contained; AGB requires it to be a thin metadata layer wrapping an external code tree.

## 7. Suggested actor extensions for AGB

The base actor registry covers every stage *except* writing model code. For AGB the next iteration must extract GEDI features — i.e. produce new Python in `tf-deep-landcover/src/agb/`. Two options:

1. **Treat code-writing as a Preprocessing Actor sub-task.** Have it write `preprocessing/preprocessing_spec.md` which *references* the new script's path and commit SHA in the sibling repo, and have the Critic verify the script exists, runs on a 10-plot smoke sample, and emits the declared schema. Low ceremony, fits existing contract.
2. **Add an `Implementation Actor` stage** between Preprocessing and Training, with a dedicated Critic that checks (a) the script lives in the declared path in the sibling repo, (b) imports/dependencies match `environment.lock`, (c) a 10-row smoke run produces the schema in `feature_schema.json`. Higher ceremony, more explicit, easier to audit later.

Recommendation: **Option 1** for the first run (keeps the actor registry stable; the smoke-test gate is already supported via the existing reproducibility footer), promote to Option 2 via `skill_evolution.md` if it proves brittle.

## 8. First-iteration orchestrator restatement (draft)

The Orchestrator's first-turn protocol expects these confirmations. A minimal AGB restatement should be:

- subject: above-ground biomass (forest)
- geography: CONUS forest projects (WV Appalachia + Upper Midwest + NE Maine bloc; held-out test on Sylvania, MI)
- task type: **biomass_regression** (regression of standing-stock tCO₂/acre at field-plot resolution)
- input data sources: ANEW field plots (gpkg) + OlmoEarth AEF embeddings + GEDI L2A/L4A (new); auxiliary PALSAR-2, DEM
- target variable: `CO2` (tCO₂/acre standing stock), locked; `Annual_CO2` deferred
- spatial resolution: 10 m features, plot footprint ~14.7 m
- temporal horizon: annual 2022 + 2023
- evaluation metrics: R², RMSE, MAE, bias, per-quintile bias, predicted-range discrimination, per-ecoregion R², external holdout R²
- performance threshold: realistic R² ≥ 0.55 / stretch R² ≥ 0.65; per-quintile |bias| ≤ 30 tCO₂/acre
- compute budget: Small (LGBM minutes-per-fold) or Medium if multimodal-fusion candidates are admitted
- runtime budget: 1 working day per iteration
- output directory: `experiments/agb_usa_biomass_regression_20260528/`

The Research Actor's first task is **not** literature framing — that was done in `agb-modelling-context.md`. It is **validating GEDI access and integration design**, since that is the gating dependency for every subsequent stage.

## 9. Risks

- **Re-discovery instead of progress.** Without the per-quintile-bias and predicted-range-discrimination metrics encoded in the evaluation matrix, the agent may re-litigate the loss/sampling/calibration excursions that the AGB doc has already ruled out. Mitigation: §3.4 + pre-registered iteration ledger row.
- **Source-registry drift.** The current registry leans on USDA CDL / WorldCereal / FAOSTAT — none of which apply to forest biomass. If the Research Actor reads the registry verbatim it will mis-anchor sources. Mitigation: §3.2.
- **Cross-repo split confusion.** A plain bootstrap writes only into `crop-ml-agent-evolve/`. If the agent forgets to capture the `tf-deep-landcover` SHA + script paths in `final/git_snapshot.txt` and `preprocessing/data_version.txt`, the run is non-reproducible. Mitigation: §6 dual-SHA rule.
- **Budget-tier mis-classification.** Multimodal-fusion candidates with full-imagery U-Net regressors are Medium/Large tier. The default Small tier in `experiment_config.template.yaml` would silently exclude them. Mitigation: the AGB config (§5) declares tier explicitly; the Critic must reject any candidate that exceeds the declared tier rather than allow silent upgrade.
- **Label-quality drift.** ANEW plots have ~10 m GPS error; 1/24-acre footprint ~14.7 m radius. A naive Critic may flag this as a leakage risk; it is a structural label-noise floor instead. Mitigation: document in `data_card.md` under "Known limitations and biases" and again in `experiment_design.md` design-risk register.
- **Holdout claim inflation.** The framework's evaluation Critic already rejects random-split claims for tasks requiring generalisation. Reinforce that AGB's claim of generalisation is "to a project the model never saw" — encoded as `holdout_proof: project_name` with `split_audit.csv` proving zero project intersection.

## 10. Recommended next steps

1. **Author the AGB extension files** in this repo, additively, mirroring `references/skill_evolution.md`'s "extension only" pattern:
   - `skills/crop-ml-agent-evolve/references/biomass_sources.md`
   - Append regression ladder to `references/model_selection.md`
   - Append per-quintile + range-discrimination metrics to `references/evaluation.md`
   - Append biomass anchor row to `references/deep_research.md` and lens additions to `references/error_analysis.md`
   - Add the feature-ceiling row to `references/improvement_loop.md`'s mapping table
2. **Patch `bootstrap_experiment.sh`** to accept `biomass_regression` and pass `bootstrap_smoke.sh` with the new case.
3. **Add doctor checks** for Source Coop anonymous read, GEDI / LP DAAC GEE access, and the ANEW gpkg path.
4. **Run `bootstrap_experiment.sh agb_usa biomass_regression "CONUS forest" --force`** and edit the produced `experiment_config.yaml` to match §5.
5. **Pre-register iteration 0** in the IMPLEMENTATION_PLAN ledger row with: trigger = "initial baseline lift", weakest_stage = "feature insufficiency", rerun_from = "research → preprocess → train → evaluate", expected fix = GEDI integration.
6. **Run the first iteration** — the Research Actor's deliverable is a `deep_research.md` whose primary finding is GEDI L2A/L4A access feasibility (cost, latency, coverage over the three ecoregions), not yet-another biomass literature scan.
7. **After iteration 1 closes**, invoke the `skill_evolution` pass with explicit user approval, so the per-quintile-bias lens, the regression ladder, and the biomass sources become permanent skill additions rather than one-off forks.

## 11. Open questions for the user

These are decisions only mcourtis can make and are best clarified before launching the first iteration:

- **Single repo or split?** Keep code in `tf-deep-landcover/` and metadata in `crop-ml-agent-evolve/experiments/` (§6 recommendation), or migrate the AGB code path into this repo? The former preserves the existing pipeline; the latter simplifies the agent contract.
- **Skill scope.** Adapt the existing `crop-ml-agent-evolve` skill with regression as a first-class task type, or fork to a new `geo-ml-agent-evolve` skill where `subject` (not `crop`) is the noun? The latter is cleaner long-term but doubles the surface area to maintain.
- **Iteration cap.** Default Small-tier cap is 3, Medium is 4. The AGB doc has already burned five informal iterations on the embedding-only ceiling. Recommend declaring `max_full_iterations: 4` plus `allow_extended_iterations: true` only if a multimodal-fusion experiment is admitted.
- **External-holdout policy.** Should `external_holdout_r2` (Sylvania-style) be a *hard* acceptance gate or a *reported* metric? The AGB doc treats it as a reported expectation (R² = 0.25 on Sylvania). The framework's evaluation gate currently only enforces spatial+temporal holdouts; promoting external holdout to a gate would tighten the standard but may stall early iterations.

## Appendix A — Files touched by the recommended adaptation

Additive changes only; nothing deleted, no acceptance gate relaxed, in line with `references/skill_evolution.md`'s forbidden-changes list.

| File | Change |
|---|---|
| `skills/crop-ml-agent-evolve/scripts/bootstrap_experiment.sh` | Add `biomass_regression` to task enum; help text update |
| `skills/crop-ml-agent-evolve/references/source_registry.md` | Add biomass / LiDAR / foundation-embedding rows (or new `biomass_sources.md`) |
| `skills/crop-ml-agent-evolve/references/experimental_design.md` | Add `biomass_regression` to task taxonomy |
| `skills/crop-ml-agent-evolve/references/model_selection.md` | Append biomass-regression ladder section |
| `skills/crop-ml-agent-evolve/references/evaluation.md` | Add per_quintile_bias, predicted_range_discrimination, external_holdout_r2 |
| `skills/crop-ml-agent-evolve/references/error_analysis.md` | Add biomass-quintile, canopy-cover, forest-type, stand-age lenses |
| `skills/crop-ml-agent-evolve/references/improvement_loop.md` | Add feature-insufficiency row to mapping table |
| `skills/crop-ml-agent-evolve/references/deep_research.md` | Add CONUS forest AGB anchor row to threshold table |
| `skills/crop-ml-agent-evolve/scripts/doctor.sh` | Add checks for Source Coop, GEDI/LP DAAC, ANEW gpkg path |
| `tests/bootstrap_smoke.sh` | Add a `biomass_regression` happy-path case |

## Appendix B — Files left untouched

These should *not* be modified for AGB; they are the skill's structural contract and changing them would constitute architectural drift:

- `skills/crop-ml-agent-evolve/SKILL.md` (Orchestrator / Actor / Critic contract; first-turn protocol)
- `references/orchestration.md` (retry policy, Final QA gate, validation states)
- `references/database_preprocessing.md` (leakage controls, split audit requirement, reproducibility footer)
- `references/training.md` (deterministic seeds, checkpoint lineage)
- `references/model_saving.md` (final package schema)
- `references/skill_evolution.md` (the evolution-mode contract itself)
- `assets/IMPLEMENTATION_PLAN.template.md` (ledger schema)
