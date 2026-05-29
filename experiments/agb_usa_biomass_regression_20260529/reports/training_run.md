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
