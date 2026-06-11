---
language:
- en
license: "Internal / Treefera proprietary"
tags:
- biomass
- forest-structure
- remote-sensing
- regression
- alphaearth-embeddings
- zero-shot-transfer
library_name: "lightgbm"
pipeline_tag: "tabular-regression"
---

# Model Card — embdstx head, Ireland zero-shot transfer

## Model Details

- Model name: `inference_model_embdstx` (LightGBM Booster, 73 trees)
- Experiment ID: `agb_ireland_biomass_regression_20260608`
- Subject: above-ground biomass / standing-carbon regression (`agb_ireland`)
- Geography: Ireland — Dasos forestry portfolio (Sitka-dominant plantation), 141 dissolved Locations
- Task type: biomass_regression (**zero-shot transfer + model-vs-model**, NO ground truth)
- Primary modality: AlphaEarth (AEF) optical-derived embeddings + Hansen disturbance timing
- Final artifact path: `final/model/inference_model_embdstx.txt`
- Best checkpoint path: **N/A — inference only; no model was trained.** No `checkpoints/best.ckpt`
  exists (would be fabrication). The applied artefact IS the pre-trained head text file above.
- Training framework: LightGBM (pre-trained upstream on US ANEW pool; not retrained here)
- Version / git commit: `b6d219ac5090543f58480d9df30e6a16acb35003` (branch `main`)

## Intended Use

### Primary use
Directional / structural screening of standing carbon (tCO₂/acre) for the Irish Dasos plantation
portfolio, as a **model-vs-model** comparison against the Deep Biomass (DB) product. Trustworthy
output is the **ranking / structural response** (the head rank-tracks stand age ρ=0.55 and dominant
height ρ=0.56), NOT calibrated absolute levels.

### Out-of-scope use
- Reporting calibrated absolute tCO₂/acre for Ireland: the entire portfolio is deep OOD (100% beyond
  the 99th-pct training Mahalanobis radius; domain-classifier AUC ≈ 1.0).
- Any new region without re-passing the encoding gate (see Preprocessing).
- Treating DB agreement/disagreement as accuracy — DB is a directional **lower bound**, not truth.

## Data

### Training data
NOT trained here. The pre-trained head was fit upstream on **4636 US ANEW plots** (incl. Bayfield,
in-sample), target CO₂ standing stock tCO₂/acre, range `[0, 520.95]`. No maritime-temperate or
high-biomass-plantation analogs in that pool — the root cause of the Irish OOD.

### Validation and test data
**N/A — no train/val/test split** (`configs/split_strategy.yaml` = `none_zero_shot_transfer`,
fractions 0/0/0). Evaluation partition = all 141 Irish Locations, compared per-Location vs DB.
There is NO Irish ground truth.

### Data limitations
No field-measured Irish labels exist. DB is a satellite-inferred reference model (known
under-estimator), used only as a directional lower bound. 17/141 Locations used a pre-2017 AEF
fallback (survey 2015/16 clamped to 2017) — shown to cause no detectable divergence distortion.

## Preprocessing

- Cloud masking: handled upstream in the AlphaEarth annual composite (no per-scene masking here).
- Temporal alignment: AEF mosaic sampled at each Location's area-weighted-mode survey year, clamped
  to AEF coverage [2017, 2025]; Hansen disturbance features are survey-relative (leakage-safe).
- CRS and resampling: dissolve in EPSG:2157, sample in EPSG:4326; `reduceRegions(mean, scale=10)`.
- Normalisation: **per-band affine** GEE AEF (A00..A63) → training int8 codec
  (`emb_j = a_j·A{j} + c_j`); production affine fit on all 409 Bayfield plots after a held-out gate.
- **Encoding gate (HARD precondition): PASS** — held-out (122 plots) mean corr 0.986, post-affine
  per-band slope median 1.006, 98% of bands in [0.8,1.2], intercept median 0.085·band-σ.
- Leakage controls: head training pool (US) geographically disjoint from Ireland; affine fit only on
  US Bayfield, uses no Irish target; no Irish label used anywhere.

## Model

- Architecture: LightGBM gradient-boosted regression, 73 trees.
- Inputs: 67 features = 64 affine-mapped AEF embeddings + 3 Hansen disturbance-timing features
  (`dstx_pre_ysd`, `dstx_pre_loss_5yr`, `dstx_loss_frac_buf`); exact order in
  `final/model/inference_features_embdstx.json`.
- Outputs: CO₂ standing stock, tCO₂/acre.
- Loss: upstream regression objective (not retrained here).
- Key hyperparameters: 73 trees; no vertical-structure lever (optical-AEF only, no CHM/SAR).

## Training

- Seed: 42 (used only for the held-out encoding-gate split and evaluation; predict is deterministic).
- Hardware / Runtime / Early stopping / Checkpoint policy: **N/A — inference only; no training run.**
  `metrics_history.csv` and `training_config.yaml` do not apply and were not produced.

## Evaluation

### Main metrics
NO accuracy metrics (no GT). Threshold-free, divergence-vs-DB characterisation
(`final/evaluation_matrix.yaml`):
- Pred distribution (141 Locations): min 26.7 / mean 91.6 / median 100.3 / max 138.4 tCO₂/acre.
- Portfolio pred 91.6 vs DB 27.35 → **ratio 3.35×**.
- **H1 directional dominance — SUPPORTED**: pred ≥ DB for 98.6% of Locations.
- **H2 saturation-resistance (DB-quintile) — NOT_SUPPORTED** (but see Error Analysis: this is a
  comparison artefact of DB-self-quintiling, NOT the head saturating).
- **H3 covariate rank-tracking — MOSTLY_SUPPORTED**: pred vs age ρ=0.553, vs Hdom ρ=0.556 (both
  p<1e-6), far exceeding DB (ρ=0.11 / 0.20); YC null for both.
- **OOD — SEVERE_DOMAIN_SHIFT**: Mahalanobis min 27.8 (1.9× the 14.79 training 99th-pct radius),
  100% beyond it; domain-classifier AUC 0.999998.

### Robustness slices
- by region/ecoregion: single portfolio (maritime-temperate plantation), entirely OOD vs US pool.
- by DB quintile (Q1..Q5): signed bias 63.9 / 74.6 / 64.2 / 61.9 / 56.9 (flat-to-declining, Q5 lowest).
- by forest type: Sitka SS (n=137) Δ=64.0; broadleaf/other (n=3) Δ=84.9.
- by age (yr): 0-10 Δ=32.4; 10-20 Δ=68.7; 20-30 Δ=76.3; 30-40 Δ=81.2 (Δ rises with structure).
- by year: 2024-only sensitivity — H1 still 95.7%; PRD lower (0.46).

## Error Analysis

Dominant pattern: a coherent **level offset** (pred ~3.35× DB) reflecting DB under-reading, NOT OOD
noise — |Δ| vs Mahalanobis ρ=−0.21 (divergence does not grow with OOD distance). The H2 "failure" is
a measurement artefact: quintiles cut on DB magnitude mechanically maximise DB's own spread (DB Q1→Q5
+31 vs pred +24), and DB magnitude is a poor biomass proxy. Within the top quintile the head still
rank-tracks age (ρ=0.57) and height — no plateau (pred max 138.4 = 26.6% of training range). Earliest
responsible stage: **model selection** (optical-only head, no structural lever; US-only training pool).

## Risks and Limitations

- Absolute tCO₂/acre levels are **deep extrapolation** — do not present as calibrated.
- The head **cannot prove saturation resistance** (no independent structural signal); it can only
  rule out the co-saturation failure mode.
- Encoding gate proves codec fidelity at Bayfield (in-sample), NOT Irish transfer accuracy.
- No post-hoc calibration is valid (no Irish truth; calibrating to DB would import DB's bias).

## Reproducibility

- Config path: `configs/experiment_design.md`, `configs/experiment_config.yaml`,
  `configs/split_strategy.yaml`, `configs/model_candidates.yaml`
- Feature schema path: `final/preprocessing_pipeline/feature_schema.json`
- Environment record path: `final/environment.lock`
- Command or entrypoint:
  `uv run python experiments/.../evaluation/run_bias_characterisation.py` (seed 42, deterministic predict)
- Data version: `final/preprocessing_pipeline/data_version.txt`
  (sha256[:16]: model `681d939258695f76`, features_json `9f80f0dfe17fdb3c`)
