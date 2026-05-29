# Experiment Report — agb_usa biomass regression, iteration 0

## Purpose

Iteration 0 is a **wiring validation** of the `biomass-ml-agent-evolve` orchestrator: drive the
existing `tf-deep-landcover` joint_v2 LightGBM baseline (R²=0.42) end-to-end through the
Actor/Critic contract — invoking the sibling model code via the cross-repo invocation contract,
capturing both repository SHAs, populating the biomass-specific evaluation matrix, and
assembling a reproducible `final/` bundle. Success = reproduce R²=0.42 within ±0.03.

## Result

**Reproduced bit-identically.** Out-of-fold, project-LOPO over 23 projects (n=4,636):

| metric | iteration 0 | baseline joint_v2 | target | status |
|---|---:|---:|---:|:--:|
| R² | 0.4182 | 0.4182 | 0.42 ± 0.03 | ✅ |
| RMSE | 56.58 | 56.58 | ≈57 | ✅ |
| MAE | 41.49 | 41.49 | ≈41 | ✅ |
| bias | +0.50 | +0.50 | ≈+0.5 | ✅ |

The retrain aggregate is byte-for-byte equal to the prior baseline — the trainer is
deterministic on this input, which both validates the wiring and certifies the reused OOF
predictions as an exact representation (basis for the breakdown metrics).

## Method

- **Features:** reused `experiments/agb/usa_v1_pilot_joint_v2/features.parquet` (64-dim
  AlphaEarth Foundation (AEF) embeddings; 4,646 rows, 4,636 after dropping 10 extraction failures).
- **Model:** `tf-deep-landcover/src/agb/train_agb_lgbm` (LightGBM, hardcoded hyperparams),
  invoked — not reimplemented — per the cross-repo contract.
- **Split:** leave-one-project-out on `project_name` (23 folds); no random split.
- **Baselines:** mean-predictor R²=−0.01, ridge-on-PC20 R²=0.37 → LightGBM adds +0.045 R² over
  the linear floor; the model class is not the bottleneck.

## Diagnostics — the feature ceiling

The headline error mode is **dynamic-range compression**, confirmed on the reproduced run:

- per_quintile_bias: Q1 **+35.6** (over-predict low) → Q5 **−72.1** (under-predict high).
- predicted_range_discrimination = **0.468** (WV 0.19, MW 0.47, NE 0.49) — under half the true
  spread; the WV value matches the research anchor's 0.21.
- Worst ecoregion: Appalachian mixed mesophytic (WV) R²=0.157.

This is a **feature deficit**, not a tuning fault: optical embeddings cannot resolve vertical
structure (a 50 vs 250 tCO₂/acre closed-canopy hardwood stand looks identical from above). Five
tuning levers (more plots, footprint sampling, Huber loss, log-target, isotonic calibration)
were previously falsified and were not re-run (see `research/deep_research.md`).

## Decision

- **Iteration-0 acceptance met** (R²=0.42 reproduced; realistic benchmark R²≥0.40 met).
- **Advance to iteration 1:** add **GEDI canopy height**. Rerun boundary research → preprocess →
  train → evaluate with the new feature source. The GEDI access route (GEE-asset vs LP-DAAC
  `earthaccess`) is deferred to the iteration-1 Research Actor; iteration 0 does not pre-commit.

## Evidence index (accepted artefacts only)

- `reports/training_run.md`, `checkpoints/{metrics.json,metrics_history.csv,best.ckpt}`
- `evaluation/evaluation_matrix.yaml`, `evaluation/biomass_metrics.json`,
  `evaluation/figures/residuals_by_quintile.png`
- `models/baseline_registry.md`, `error_analysis/error_analysis.md`
- `final/` bundle (model, preprocessing_pipeline, cards, environment.lock, git_snapshot.txt)
- reproducibility: `preprocessing/data_version.txt` (gpkg SHA256 + tf-deep-landcover SHA)
