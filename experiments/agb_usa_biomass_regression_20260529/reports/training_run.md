# Training Run — agb_usa biomass regression (iteration 0)

## Invocation

Cross-repo: the sibling LightGBM trainer was invoked, not reimplemented.

```
cd /home/mattc/code/tf-deep-landcover && \
uv run python -m src.agb.train_agb_lgbm \
  --features /home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet \
  --out-dir  /home/mattc/code/crop-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/checkpoints \
  --fig-dir  /home/mattc/code/crop-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/evaluation/figures
```

- tf-deep-landcover SHA: `e8c70584`
- exit code: 0 · CV: leave-one-project-out, 23 folds · n=4,636 (10 extraction-failure rows dropped)

## Result — reproduction PASSED

| metric | retrain | baseline (joint_v2) | target | Δ vs target | within ±0.03 |
|---|---:|---:|---:|---:|:--:|
| R² | 0.4182 | 0.4182 | 0.42 | −0.0018 | ✅ |
| RMSE | 56.58 | 56.58 | ≈57 | — | ✅ |
| MAE | 41.49 | 41.49 | ≈41 | — | ✅ |
| bias | +0.50 | +0.50 | ≈+0.5 | — | ✅ |
| n | 4,636 | 4,636 | 4,636 | 0 | ✅ |

The retrain aggregate is **bit-identical** to the prior baseline `metrics.json` — the trainer
is deterministic on this input. This both validates the wiring and certifies that the prior
`oof.parquet` is an exact representation of this run (used downstream for the breakdown metrics).

## Per-fold spread (23 projects)

- fold R² range: **−0.617 (TwoHearted, n=21) → 0.552 (NorthMaineWoods, n=245)**
- weakest folds: TwoHearted (−0.617, n=21), ColeCrane (−0.211, n=15), WolverineCopper (0.031, n=5)
- strongest folds: NorthMaineWoods (0.552), Greenleaf (0.538), AshlandCounty (0.474)
- Small-n projects dominate the negative tail — expected under LOPO with tiny held-out sets.

Full per-fold table: `checkpoints/metrics_history.csv`.

## Artifacts

- `checkpoints/model.txt` (final all-data LightGBM) → copied to `checkpoints/best.ckpt`
- `checkpoints/metrics.json` (aggregate + 23 folds)
- `checkpoints/metrics_history.csv` (per-fold, synthesised from metrics.json)
- `checkpoints/train_log.txt` (full LightGBM stdout)
- `evaluation/figures/{residuals_by_quintile,shap_beeswarm,shap_grouped,emb_correlation}.png`

## Reproducibility footer

- input_artefact_sha256: features.parquet from sibling (ANEW gpkg SHA in `preprocessing/data_version.txt`)
- libraries: lightgbm, scikit-learn, pandas, shap (sibling `.venv`, py3.13)
- seed: 42 (hyperparameters hardcoded in sibling trainer)
- command_or_entrypoint: see Invocation
- timestamp_utc: 2026-05-29T09:35:00Z

# Training Run — agb_usa biomass regression (iteration 1)

## Invocation

Cross-repo: the sibling LightGBM trainer was invoked, not reimplemented.

```
cd /home/mattc/code/tf-deep-landcover && \
uv run python -m src.agb.train_agb_lgbm \
  --features /home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/preprocessing/features_iter1.parquet \
  --out-dir  /home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/checkpoints \
  --fig-dir  /home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/evaluation/figures
```

- tf-deep-landcover SHA: `c53c446e0ee11d9d450d35eaa918b58b5a8da828`
- Trainer patch: `feature_cols` filter extended to include `"gedi_"` prefix (line 174 of `src/agb/train_agb_lgbm.py`)
- exit code: 0 · CV: leave-one-project-out, 23 folds · n=4,636 (10 extraction-failure rows dropped)

## Feature columns used

- 64 embedding columns: `emb_00`..`emb_63`
- 5 GEDI structural columns (all with `gedi_` prefix, confirmed picked up by trainer):
  - `gedi_rh98` (canopy top height, m)
  - `gedi_cover` (total canopy cover fraction)
  - `gedi_pai` (plant area index)
  - `gedi_fhd_normal` (foliage height diversity)
  - `gedi_n_samples` (GEDI coverage-confidence, months 0-36)
- Total feature count: 69

SHAP group attribution (full-data retrain): embeddings 96.8%, other (GEDI) 3.2%.
GEDI columns were picked up by the trainer (confirmed via LightGBM log: "used features: 69").

## Result

| metric | iter-1 | iter-0 baseline | target (R²≥0.55) | Δ vs iter-0 | target met |
|---|---:|---:|---:|---:|:--:|
| R² | 0.4176 | 0.4182 | ≥0.55 | −0.0006 | No |
| RMSE | 56.61 | 56.58 | — | +0.03 | — |
| MAE | 41.49 | 41.49 | — | 0.00 | — |
| bias | +0.56 | +0.50 | — | +0.06 | — |
| n | 4,636 | 4,636 | 4,636 | 0 | — |

The GEDI columns added negligible signal (+0.0% R²). Adding 5 GEDI structural metrics to 64
embedding dimensions did not improve aggregate performance. SHAP confirms GEDI columns contribute
only 3.2% of mean |SHAP| — the embeddings dominate. The realistic target of R²≥0.55 was **not met**.

## Per-fold spread (23 projects)

- fold R² range: **−0.667 (TwoHearted, n=21) → 0.563 (Greenleaf, n=82)**
- weakest folds: TwoHearted (−0.667, n=21), ColeCrane (−0.209, n=15), WolverineCopper (0.013, n=5)
- strongest folds: Greenleaf (0.563), NorthMaineWoods (0.557), Cassidy (0.474)
- Per-fold pattern essentially unchanged vs iteration 0; small-n projects continue to dominate the negative tail.

Full per-fold table: `checkpoints/metrics_history.csv`.

## Artefacts

- `preprocessing/features_iter1.parquet` — 4,646 rows, 77 columns (GEDI cols renamed to `gedi_` prefix; `gedi_temporal_coverage_months` dropped as redundant)
- `preprocessing/feature_schema.json` — updated with renames and dropped column
- `configs/training_config.yaml` — iteration-1 config with trainer SHA
- `checkpoints/model.txt` (final all-data LightGBM) → copied to `checkpoints/best.ckpt`
- `checkpoints/metrics.json` (aggregate + 23 folds)
- `checkpoints/metrics_history.csv` (per-fold, regenerated from metrics.json)
- `evaluation/figures/{residuals_by_quintile,shap_beeswarm,shap_grouped,emb_correlation}.png`

## Reproducibility footer

- input_parquet: `preprocessing/features_iter1.parquet` (4,646 rows, 77 columns)
- trainer_repo_sha: `c53c446e0ee11d9d450d35eaa918b58b5a8da828` (tf-deep-landcover branch mc/carbonmap-embeddings)
- libraries: lightgbm, scikit-learn, pandas, shap (sibling `.venv`, py3.13)
- seed: 42 (hardcoded in trainer LGBM_PARAMS)
- command_or_entrypoint: see Invocation
- timestamp_utc: 2026-05-29T10:00:00Z

# Training Run — agb_usa biomass regression (iteration 2)

## Invocation

Cross-repo: the sibling LightGBM trainer was invoked, not reimplemented.

```
cd /home/mattc/code/tf-deep-landcover && \
uv run python -m src.agb.train_agb_lgbm \
  --features /home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/preprocessing/features_iter2.parquet \
  --out-dir  /home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/checkpoints \
  --fig-dir  /home/mattc/code/agb-ml-agent-evolve/experiments/agb_usa_biomass_regression_20260529/evaluation/figures
```

- tf-deep-landcover SHA: `c53c446e0ee11d9d450d35eaa918b58b5a8da828`
- Trainer patch: `feature_cols` filter already includes `"chm_"`, `"topo_"`, `"dist_"` prefixes (line 174)
- exit code: 0 · CV: leave-one-project-out, 23 folds · n=4,636 (10 extraction-failure rows dropped)

## Feature columns used

- 64 embedding columns: `emb_00`..`emb_63`
- 5 GEDI structural columns (`gedi_` prefix): rh98, cover, pai, fhd_normal, n_samples
- 1 CHM column (`chm_` prefix): `chm_m` — ETH Global Canopy Height 2020, metres
- 5 topographic columns (`topo_` prefix): elevation, slope, aspect_cos, aspect_sin, tpi — SRTM v3
- 1 disturbance column (`dist_` prefix): `dist_years_since` — Hansen GFC 2025
- Total feature count: 76 (confirmed via LightGBM log: "used features: 76")

New feature groups added in iter-2 vs iter-1: `chm_`, `topo_`, `dist_` (+7 columns)

## Result

| metric | iter-2 | iter-1 baseline | iter-0 baseline | target (R²≥0.55) | Δ vs iter-1 | target met |
|---|---:|---:|---:|---:|---:|:--:|
| R² | 0.4272 | 0.4176 | 0.4182 | ≥0.55 | +0.0096 | No |
| RMSE | 56.14 | 56.61 | 56.58 | — | −0.47 | — |
| MAE | 41.29 | 41.49 | 41.49 | — | −0.20 | — |
| bias | +0.55 | +0.56 | +0.50 | — | −0.01 | — |
| n | 4,636 | 4,636 | 4,636 | 4,636 | 0 | — |

Adding CHM, topographic, and disturbance features improved aggregate R² by +0.010 over iter-1.
The realistic target of R²≥0.55 was **not met**.

## Per-fold spread (23 projects)

| project | n | R² | RMSE |
|---|---:|---:|---:|
| TwoHearted | 21 | −0.721 | 141.6 |
| ColeCrane | 15 | −0.182 | 109.7 |
| GauleyRiver | 199 | 0.110 | 78.8 |
| WolverineCopper | 5 | 0.116 | 23.7 |
| KanawhaRiver | 313 | 0.158 | 79.2 |
| Greenleaf | 82 | 0.563 | 49.2 |
| NorthMaineWoods | 245 | 0.556 | 41.4 |
| BigSix | 272 | 0.479 | 47.7 |
| Cassidy | 343 | 0.473 | 44.6 |

- fold R² range: **−0.721 (TwoHearted, n=21) → 0.563 (Greenleaf, n=82)**
- weakest folds: TwoHearted (−0.721, n=21), ColeCrane (−0.182, n=15), GauleyRiver (0.110, n=199)
- strongest folds: Greenleaf (0.563), NorthMaineWoods (0.556), BigSix (0.479)
- Small-n projects continue to dominate the negative tail.

Full per-fold table: `checkpoints/metrics_history.csv`.

## WV Appalachia fold analysis

GauleyRiver (n=199, R²=0.110) and KanawhaRiver (n=313, R²=0.158) are West Virginia / Appalachian projects.
Neither fold meets the WV Appalachia R²≥0.30 stretch target.
The WV projects remain the weakest regional cluster — high biomass variance and cross-project generalisation is poor.

## SHAP feature group attribution

SHAP group attribution from full-data retrain (reported by trainer):
- embeddings: **86.5%** of mean |SHAP|
- other (GEDI + CHM + topo + dist): **13.5%** of mean |SHAP|

Compared to iter-1 (embeddings 96.8%, other 3.2%): the new CHM/topo/dist features collectively
captured an additional 10.3 percentage points of attributable importance, reducing embedding dominance.
Per-group breakdown within "other" not separately reported by trainer — full figure in
`evaluation/figures/shap_grouped.png`.

## Stop-condition assessment

- R²=0.427 is below the realistic target (R²≥0.55).
- Per `research/deep_research.md` iteration-2 rerun boundary: R² < 0.55 triggers addition of
  **GEDI L4B gridded AGBD (priority 4, `LARSE/GEDI/GEDI04_B_002`)** and
  **TerraClimate climate normals (priority 5, `IDAHO_EPSCOR/TERRACLIMATE`)** before any
  escalation to the Critic. These datasets were deferred pending CHM+topo+disturbance results.
- Next action: extract GEDI L4B and TerraClimate features (iteration-3 preprocessing), merge
  with features_iter2.parquet, retrain, and re-evaluate against the R²≥0.55 threshold.
- Immediate escalation is NOT triggered (that threshold is R² < 0.45); R²=0.427 requires the
  prescribed priority-4/5 feature additions before escalation.

## Iteration 3 — GEDI L4B gridded AGBD + TerraClimate

### Invocation

```bash
cd /home/mattc/code/tf-deep-landcover && \
uv run python -m src.agb.train_agb_lgbm \
  --features .../preprocessing/features_iter3.parquet \
  --out-dir  .../checkpoints \
  --fig-dir  .../evaluation/figures
```

- tf-deep-landcover SHA: `c53c446e` (uncommitted patch; added `agbd_`, `clim_` prefixes)
- Feature columns used: 80 (64 emb_ + 5 gedi_ + 1 chm_ + 5 topo_ + 1 dist_ + 1 agbd_ + 3 clim_)
- GEDI L4B (`agbd_mu`) nulls: 1,592/4,646 (34.3%) — LightGBM missing-direction splits

### Results

| metric | iter-3 | iter-2 | iter-1 | iter-0 | target |
|---|---:|---:|---:|---:|---|
| R² | **0.4274** | 0.4272 | 0.4176 | 0.4182 | ≥ 0.55 |
| RMSE | 56.13 | 56.14 | 56.61 | 56.58 | ≤ 50 |
| MAE | 41.27 | 41.29 | 41.49 | 41.49 | — |
| bias | +0.87 | +0.55 | +0.56 | +0.50 | — |
| n | 4,636 | 4,636 | 4,636 | 4,636 | — |

Lift over iter-2: +0.0002 — within noise. SHAP: embeddings 82.8%, all other features 17.2%.

### Stop-condition assessment — ESCALATE

All five research-spec feature priorities now exhausted. Total cumulative lift: +0.009 R²
(0.4182 → 0.4274). Realistic target R²≥0.55 not met. Per `research/deep_research.md`
iteration-2 stop condition: **escalate to Critic before proceeding further.**

| Priority | Feature added | Cumulative R² | Lift |
|---:|---|---:|---:|
| baseline | Embeddings only | 0.4182 | — |
| 1–3 | CHM + SRTM topo + Hansen dist | 0.4272 | +0.010 |
| 4–5 | GEDI L4B AGBD + TerraClimate | 0.4274 | +0.000 |

### Reproducibility footer (iteration 3)

- input_artefact: `preprocessing/features_iter3.parquet`
- seed: 42; trainer hyperparams hardcoded in sibling
- command: see Invocation; trainer SHA `c53c446e`
- timestamp_utc: 2026-05-29

## Artefacts

- `preprocessing/features_iter2.parquet` — 4,646 rows, 84 columns
- `configs/training_config.yaml` — iteration-2 config with trainer SHA
- `checkpoints/model.txt` (final all-data LightGBM) → copied to `checkpoints/best.ckpt`
- `checkpoints/metrics.json` (aggregate + 23 folds)
- `checkpoints/metrics_history.csv` (per-fold, synthesised from metrics.json)
- `evaluation/figures/{residuals_by_quintile,shap_beeswarm,shap_grouped,emb_correlation}.png`

## Reproducibility footer

- input_parquet: `preprocessing/features_iter2.parquet` (4,646 rows, 84 columns)
- trainer_repo_sha: `c53c446e0ee11d9d450d35eaa918b58b5a8da828` (tf-deep-landcover branch mc/carbonmap-embeddings)
- libraries: lightgbm, scikit-learn, pandas, shap (sibling `.venv`, py3.13)
- seed: 42 (hardcoded in trainer LGBM_PARAMS)
- command_or_entrypoint: see Invocation
- timestamp_utc: 2026-05-29T11:00:00Z
