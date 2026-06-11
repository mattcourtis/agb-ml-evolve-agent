# IMPLEMENTATION_PLAN

## Experiment header

- experiment_id: agb_ireland_biomass_regression_20260608
- created_at: 2026-06-08T08:04:30Z
- subject: agb_ireland
- geography: Ireland - Dasos forestry portfolio (Sitka-dominant plantation)
- task: biomass_regression (MODE: zero-shot transfer + model-vs-model comparison — NOT train-from-scratch)
- target_variable: AGB as CO2 standing stock, tCO2/acre (model output). Reference = Deep Biomass AGB Mg/ha x 0.6977 -> tCO2/acre. Both AGB-only.
- spatial_resolution: 10 m AEF embeddings, area-averaged over stand polygons
- spatial_unit: 141 Deep-Biomass "Locations" (dissolved from 1053 Dasos sub-compartments)
- temporal_horizon: survey-year prediction (2023-2024 per SurveyDate); DB reference = mean of 2020-2024
- evaluation_metrics: per-Location signed delta distribution; per-quintile signed bias (Q1..Q5); PRD; covariate cuts (age/Hdom/YC/MainSp/Thinned); OOD Mahalanobis fraction + USA-vs-Ireland domain-classifier AUC
- performance_threshold: NONE (no ground truth). Success = full bias characterisation vs Deep Biomass + directional/structural-consistency checks (our >= DB esp. high-biomass; pattern tracks structural covariates)
- output_dir: ./experiments/agb_ireland_biomass_regression_20260608
- compute_budget: local CPU + GEE (Earth Engine verified OK, default Initialize)
- runtime_budget: Small
- status: iteration 0 COMPLETE (RETRAIN_WARRANTED); iteration 1 COMPLETE -> ESCALATE (no US analog exists; next step is a DATA dependency: in-region labels +/- SAR/CHM lever, not a modelling step)

## Task restatement

Apply the existing, already-trained AGB regression head `models/inference_model_embdstx.txt`
(67 features = 64 AEF embeddings + 3 survey-relative Hansen disturbance-timing features;
target tCO2/acre; training range [0, 520.95]) to 141 Irish forestry Locations and characterise
how its predictions differ from the Deep Biomass model (a known under-estimator, used as a
directional reference, NOT ground truth). There is no ground truth; therefore no accuracy claim
is made and no model is trained from scratch in the first pass. Conversion: Deep Biomass native
unit Mg/ha -> tCO2/acre via x0.6977 (= 0.47 IPCC carbon fraction x 3.667 CO2/C x 0.4047 ha/acre).
Both Deep Biomass and the ANEW CO2 column are AGB-only, so pools match.

Approved source plan: `plans/ireland-agb-test-v1.md`.

### Central technical risk (must be resolved in data_profile + preprocessing)
The repo sources AEF embeddings from LOCAL int8 tiles (Bayfield: `agb_usa_pilot_midwest/embeddings_annual/2023`)
read raw int8->float WITHOUT dequantization (`infer_bayfield.py:119-150`). Ireland has NO local
tiles. The preprocessing stage must build a GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` polygon
extraction path AND reproduce the exact training encoding. The ENCODING-CONSISTENCY GATE
(corr>0.8 of Irish embeddings vs the training-parquet encoding at known overlap, or a documented
equivalent) is the hard validation and MUST pass before any prediction is trusted.

## Benchmark anchor

- user_target_provided: no (no GT; comparison is to an external model)
- benchmark_range: Irish Sitka spruce stand AGB typically ranges ~0 (recent clearfell/replant) to
  ~300-500+ Mg/ha at rotation end; portfolio mean (Deep Biomass 2024) = 44.6 Mg/ha. To be confirmed
  by Research Actor against literature.
- realistic_default_target: N/A (no accuracy target)
- stretch_target: N/A
- benchmark_notes: success is directional + structural-consistency, not error vs truth.

## Stage mode adaptations (transfer/inference)

- split_design: WAIVED — no training and no GT in first pass. Artifact `configs/split_strategy.yaml`
  records "zero-shot transfer; reference is external model not GT; no train/val/test split".
- baselines: REFRAMED — the "baseline" is the Deep Biomass reference itself (lower bound) plus a
  mean/structural-covariate sanity predictor; no new model trained.
- model_selection: REFRAMED — selects the pre-trained `inference_model_embdstx.txt`; no candidate
  training. Records why embdstx (matches embeddings+disturbance feature set) over the 73-feat head.
- training: REFRAMED — "inference run": load head, predict per Location; `reports/training_run.md`
  documents the inference run, encoding-gate result, and prediction summary.
- improvement_plan: decides whether the conditional analog-subset retrain (plan Section F) is warranted.

## Artifact registry

| stage | actor | artifact | status | attempt | critic | last_update | notes |
|---|---|---|---|---:|---|---|---|
| research | Research Actor | research/deep_research.md | ACCEPTED | 2 | Research Critic | 2026-06-08 | Sitka rotation-end AGB ~150-376 Mg/ha (Black 2009); most stands above ~150-200 Mg/ha saturation onset; AEF weak cross-region transfer = central risk; DB 44.6 Mg/ha mean = strong under-estimate |
| data_profile | Database Profiling Actor | data_profile/database_profile.md | ACCEPTED | 1 | Data Profile Critic | 2026-06-08 | 141/141 crosswalk (direct+underscore->slash); DB 2020-24 mean 39.2 Mg/ha; Sitka 60%; ENCODING GATE FEASIBLE via per-band affine (GEE corr 0.957->held-out R2 0.85), codec is global |
| experiment_design | Experimental Design Actor | configs/experiment_design.md | ACCEPTED | 1 | Experiment Design Critic | 2026-06-08 | H1 our>=DB, H2 widening high-biomass gap, H3 covariate rank-tracking; reuse compute_biomass_metrics quintile-bias(48-52)+PRD(53-55) on DB-magnitude; drop GT funcs; threshold-free decision rule |
| preprocessing | Preprocessing Actor | preprocessing/preprocessing_spec.md | ACCEPTED | 2 | Preprocessing Critic | 2026-06-08 | GATE PASSED: held-out(287/122) corr 0.986, slope median 1.006; production affine refit on full 409; 141/141 features (67, exact order, 0 NaN), 17 pre-2017 fallback; embdstx pred min26.7/mean91.6/max138.4 vs DB 27.4 |
| split_design | Orchestrator (fast-track) | configs/split_strategy.yaml | ACCEPTED | 1 | self (vs ACCEPTED upstream) | 2026-06-08 | WAIVED: zero-shot, no GT, no training; DB is external reference not labels |
| baselines | Orchestrator (fast-track) | models/baseline_registry.md | ACCEPTED | 1 | self | 2026-06-08 | B0 DB reference (lower bound); B1 structural-covariate rank check; no trained baseline |
| model_selection | Orchestrator (fast-track) | configs/model_candidates.yaml | ACCEPTED | 1 | self | 2026-06-08 | embdstx selected (matches feature set); full73/embonly not selected |
| training | Evaluation Actor (folded) | reports/training_run.md | ACCEPTED | 1 | Evaluation Critic | 2026-06-08 | inference-only; pred min26.7/mean91.6/median100.3/max138.4; 0% > training max |
| evaluation | Evaluation Actor | evaluation/evaluation_matrix.yaml | ACCEPTED | 1 | Evaluation Critic | 2026-06-08 | H1 SUPPORTED(98.6% pred>=DB, 3.35x); H2 NOT(non-monotone Q5-lowest, PRD 0.77); H3 MOSTLY(rho_pred age0.55/Hdom0.56 vs DB 0.11/0.20); OOD SEVERE(AUC~1.0,100% beyond radius); 79% >80 tco2; decision RETRAIN_WARRANTED. Critic re-ran=bit-identical |
| error_analysis | Error Analysis Actor | error_analysis/error_analysis.md | ACCEPTED | 1 | Error Analysis Critic | 2026-06-08 | H2 = COMPARISON ARTEFACT not saturation (top-quintile pred still tracks age rho0.57, only 26.6% of training range, no plateau); |delta| NOT growing w/ OOD dist (rho-0.21); weakest stage=model_selection (no vertical-structure lever). Critic reproduced all |
| improvement_plan | Orchestrator (fast-track) | reports/improvement_plan.md | ACCEPTED | 1 | self (follows ACCEPTED error_analysis) | 2026-06-08 | analog-subset retrain + structural lever; re-cut H2 on age/Hdom; no calibration |
| model_saving | Model Saving+Final Actor | final/run_summary.md | ACCEPTED | 1 | Final Report Critic | 2026-06-08 | final/ assembled; training-only paths (best.ckpt/metrics_history/training_config) justified N/A inference-only |
| final_report | Model Saving+Final Actor | final/experiment_report.md | ACCEPTED | 1 | Final Report Critic | 2026-06-08 | QA gate PASS; eval_matrix byte-identical; all numbers faithful; no placeholders; real git SHA; remote PRIVATE |

## Iteration ledger

| iteration | trigger | weakest_stage | rerun_from | target_gap_before | target_gap_after | decision |
|---:|---|---|---|---|---|---|
| 0 | initial transfer run | model_selection (optical-only embdstx; no vertical-structure lever; no maritime/plantation analogs in pool) | n/a (no GT) | n/a | n/a | COMPLETE -> RETRAIN_WARRANTED: escalate analog-subset retrain (improvement_plan) as iteration 1 |
| 1 | RETRAIN_WARRANTED (iter0 severe OOD) | training pool / model_selection | model_selection | OOD: 100% beyond 99th-pct radius, domain AUC ~1.0 | unchanged (no analog) | COMPLETE -> ESCALATE: full-pool (12,837-plot) embedding analog selection proves NO US subset reduces Irish OOD (best S3 still 100% beyond, 0/141 within internal-NN). Pivot to in-region data + SAR/CHM. reports/iter1_decision.md |

## Iteration 1 — analog-subset selection (artifact registry)

Reuses ACCEPTED iter0 upstream (research, data_profile, experiment_design, encoding gate, evaluation harness). Encoding decision: re-extract ALL candidate + Irish embeddings from the GEE float asset natively (one space, no affine) — required for clean cross-project distance comparisons.

| stage | actor | artifact | status | attempt | critic | notes |
|---|---|---|---|---:|---|---|
| iter1_design | Orchestrator | configs/iter1_analog_selection_design.md | ACCEPTED | 1 | self | two-objective (Ireland-similarity + biomass coverage); S0-S4; Tier-1 OOD diagnostics then Tier-2 dual eval |
| iter1_extract | Extraction Actor | preprocessing/iter1_pool_embeddings.parquet | ACCEPTED | 2 | Extraction Critic | 12,978 rows (12,837 ANEW + 141 Ireland), native GEE float, L2~1.0 both, 0 missing, spot-checks bit-identical to GEE. Attempt-1 REJECT (script not saved); attempt-2 wrote scripts/extract_pool_embeddings.py, dry resume reassembles identical (maxabsdiff 0) |
| iter1_select | Analog Selection Actor | configs/iter1_analog_subsets.md | ACCEPTED | 1 | Selection Critic | NO subset reduces OOD: best (S3) still 100% beyond radius, AUC 0.99999; non-circular check 0/141 Irish within subset internal-NN; coverage holds 10/10 deciles; S4 weights degenerate. Critic reproduced + non-circular metric |
| iter1_train_eval | Training+Eval Actor | evaluation/iter1_subset_eval.md | SKIPPED | 0 | n/a | SKIPPED by design ESCALATE rule (Tier-1 shows no usable analog; tree head cannot calibrate fully-OOD inputs) |
| iter1_decision | Orchestrator | reports/iter1_decision.md | ACCEPTED | 1 | self (follows ACCEPTED iter1_select) | ESCALATE: no US analog exists; pivot to in-region labelled data + SAR/CHM lever; retain S2/S3 as ordering prior |
| forest_mask | Forest-Mask Actor | final/ireland_forest_mask.md | ACCEPTED | 1 | Forest-Mask Critic (re-verified, identity to 7e-15) | DW trees>=0.5 (growing-season median, year-aligned) per-pixel -> non-forest=0, re-aggregate. FLOOR now 0 (was 16/30.5); 3yr-mean 91.6->81.5 tCO2/acre, DB ratio 3.42x->3.05x; 4 stands ->0, 5 now <DB. Young stands drop w/ forest_fraction (Moyne 44->5.7 ff0.08). Caveats: DW false-neg (Cashel/Benmore missed clearfell) + false-pos (Cummeen Upper age21 wrongly zeroed). Masked cols+4x141 GeoTIFFs+VRT+dual-scale figure; unmasked preserved. Fixes structural zero NOT in-domain stocked-young floor |
| yearmatched_3yr | Year-Matched Actor | final/ireland_yearmatched_comparison.md | ACCEPTED | 1 | Year-Matched Critic (re-verified) | 423/423 (141x3), 0 fail, no clamp. our vs DB tCO2/acre: 2022 87.5/25.9=3.37x, 2023 95.5/22.6=4.23x, 2024 91.8/31.9=2.88x, 3yr-mean 91.6/26.8=3.42x; H1 ~99-100%. Growth ours +4.9% / DB +23%. Cross-check 73 coincident-yr stands max abs diff 0.000. 3.42x confirms earlier 3.25x not a year artefact. Outputs: ireland_agb_yearmatched.{csv,parquet,gpkg} + per-year GeoTIFFs(141x3)+VRTs + figure |
| full_pixel_inference | Pixel-Inference Actor | final/ireland_pixel_inference.md | ACCEPTED | 1 | Pixel-Inference Critic (re-verified) | 141/141, 593,754 px, 0 fail. mean(f) portfolio 88.79 (aw 90.27) tCO2/acre, DB ratio 3.25x, H1 140/141. gap vs polygon-mean median -2.1%, range -28.7..+77.4%, 93/141 >±5%. Outputs: final/ireland_agb_pixel.{csv,parquet,gpkg EPSG:2157} + 141 per-stand 2-band GeoTIFFs + _index.vrt. Refines levels; OOD verdict unchanged |
| support_sensitivity | Support-Test Actor | evaluation/support_sensitivity.md | ACCEPTED | 1 | Support-Test Critic (re-ran 2 stands live) | 18 stands / ~96k px. Consistency holds (pixel-mean==polygon-mean, f(mean)==reported pred) -> gap = pure non-linearity. gap mean(f)-f(mean): median -3.0%, mean|gap| 10.9%, range -21.7..+40.1%, 7/18 beyond +/-10%; driven by within-stand pred dispersion (rho +0.54) NOT n_subcpt. Polygon-mean net slightly OVER-reads. Recommend full-141 pixel re-run; does NOT overturn OOD-driven RETRAIN_WARRANTED |

## Current blockers

- none yet (Earth Engine init verified OK)

## Decisions log

### 2026-06-08T08:04:30Z
- Created experiment.

### 2026-06-08 (Orchestrator first-turn)
- Confirmed parameters from approved plan `plans/ireland-agb-test-v1.md`.
- Set MODE = zero-shot transfer + model-vs-model comparison (no training, no GT in first pass).
- Recorded central technical risk: no local Ireland AEF tiles -> must build+validate GEE AEF extraction encoding.
- Verified Earth Engine initialises (default).
- Starting Research Actor.

### 2026-06-08 research ACCEPTED (attempt 2)
- stage: research; actor: Research Actor; artifact: research/deep_research.md; status: ACCEPTED; attempt: 2; critic: Research Critic
- attempt 1 REJECTED: misattributed [S1] citation (Wellock 2014 ash vs Black 2009 Sitka), tCO2/acre arithmetic contradiction (246 vs 262), unverified 2026 cites. All fixed in attempt 2; critic spot-checked DOI 10.1093/forestry/cpp005.
- decision_summary: Irish Sitka rotation-end AGB envelope anchored ~150-376 Mg/ha; most stands above optical-saturation onset; AEF cross-region transfer is the central risk; DB mean strongly under-reads.
- next_stage: data_profile

### 2026-06-08 data_profile ACCEPTED (attempt 1)
- stage: data_profile; actor: Database Profiling Actor; artifact: data_profile/database_profile.md; status: ACCEPTED; attempt: 1; critic: Data Profile Critic (independently reproduced + cross-validated)
- decision_summary: all sources profiled with executed code. 141/141 crosswalk resolves (124 direct + 17 underscore->slash; plan's suffix-split was wrong); dissolved area within 0.32% of CSV. DB cells = total tonnes = Mg/ha x Area; 2020-24 mean density 39.2 Mg/ha (noisy single years). Sitka SS ~60%. Training parquet emb = raw int8-range floats (NOT dequantized), per infer_bayfield.py read path. GEE SATELLITE_EMBEDDING/V1/ANNUAL = 64 A00-A63 doubles, 2017-2025 (covers 2023/24). ENCODING GATE: FEASIBLE via per-band AFFINE (GEE-float -> training int8-codec space); same embedding signal, different per-band quantisation; codec is global so transferable.
- carry-over requirements pushed to PREPROCESSING: (a) re-gate validates GEE->codec consistency at Bayfield, NOT that US-fitted slopes are Ireland-correct (Ireland has no overlap plots); (b) Irish extraction must use reduceRegions(mean) over polygons (area support matching training), not point sampling; (c) fit affine on FULL valid Bayfield plot set (>=~50), require post-transform per-band slope~=1 AND bounded intercept before trusting predictions.
- next_stage: experiment_design

### 2026-06-08 experiment_design ACCEPTED (attempt 1)
- stage: experiment_design; actor: Experimental Design Actor; artifact: configs/experiment_design.md; status: ACCEPTED; attempt: 1; critic: Experiment Design Critic (verified all script/line refs + metric-reuse).
- decision_summary: no-GT design fixed. Hypotheses H1 (our>=DB), H2 (gap widens in high-biomass band), H3 (rank-tracks age/Hdom/YC/MainSp), falsifiable without thresholds. Conversion x0.6977; polygon-mean support; survey-year AEF vs DB 2020-24 mean (+2024-only sensitivity). Encoding gate = hard go/no-go. Reuse compute_biomass_metrics quintile-bias+PRD on DB-magnitude quintiles; drop GT-dependent funcs (_agg, per_ecoregion_r2, error_by_region, prd_by_region, external_holdout_r2). Threshold-free decision rule: credible / retrain-warranted / halt.
- next_stage: preprocessing (heavy: GEE extraction + encoding gate)

### 2026-06-08 preprocessing ACCEPTED (attempt 2)
- stage: preprocessing; actor: Preprocessing Actor; artifact: preprocessing/preprocessing_spec.md; status: ACCEPTED; attempt: 2; critic: Preprocessing Critic (independently reproduced gate + checked leakage/scale/order/DB-ref).
- ENCODING GATE PASSED to upstream contract (corr>0.8 AND slope~=1 AND bounded intercept): held-out (287 train / 122 val) mean corr 0.986, post-affine per-band slope median 1.006 (98% in [0.8,1.2]), median |intercept|/sigma 0.085. No leakage (fit on train, validated on disjoint held-out).
- attempt-1 ACCEPTED w/ required doc fixes: (1) affine had been applied as train-only-287 while spec claimed full-409 -> fixed via standard production refit on FULL 409 (held-out result preserved as gate evidence; train-only saved to aef_affine_gate_train287.parquet); (2) softened overstated "pure noise" wording (honest: per-band RMSE ~31% of band-sigma, central slope ~1).
- OUTPUTS: preprocessing/ireland_features.parquet (141 x 67, exact inference_features_embdstx.json order, 0 NaN, emb on int8-codec scale); preprocessing/db_reference.parquet (DB 2020-24 mean 39.19 Mg/ha -> 27.35 tCO2/acre; 2024-only 45.71 -> 31.89; factor 0.6977); fitted affine; feature_schema.json; data_version.txt; dissolved gpkg. 17 Locations pre-2017 AEF fallback (clamped 2017), limitation noted.
- FIRST SIGNAL (smoke): embdstx head min 26.7 / mean 91.6 / max 138.4 tCO2/acre vs DB mean 27.4 -> our model reads ~3.3x higher, directionally consistent with H1 (DB under-estimation) pending formal evaluation.
- next_stage: split_design (WAIVED) -> baselines -> model_selection -> training(inference) -> evaluation (core deliverable)

### 2026-06-08 fast-track formalities + evaluation
- User chose FAST-TRACK: split_design/baselines/model_selection authored by Orchestrator (ACCEPTED, consistent w/ upstream); training folded into evaluation.
- evaluation ACCEPTED (attempt 1; Evaluation Critic re-ran run_bias_characterisation.py -> bit-identical, recomputed all headline numbers, confirmed no-GT integrity + method correctness + figures).
  - NOTE: Evaluation Actor hit an API connection drop AFTER completing all computation (predictions, matrix, _results.json, 5 figures, scripts); Orchestrator wrote the narrative bias_characterisation.md + reports/training_run.md from the saved results; Critic then verified.
- KEY RESULT: head reads 3.35x DB (91.6 vs 27.3 tco2/acre); structurally sensible (tracks age/height far better than DB) but SEVERE OOD (entirely outside US training manifold) and no demonstrated saturation-resistance (H2). Decision: RETRAIN_WARRANTED.
- next_stage: error_analysis -> improvement_plan -> final_report
