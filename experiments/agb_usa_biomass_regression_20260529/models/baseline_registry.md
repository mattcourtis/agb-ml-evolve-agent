# Baseline Registry — agb_usa biomass regression (iteration 0)

All baselines use the same leave-one-project-out (LOPO) partition as the production model
(23 folds, n=4,636), so numbers are directly comparable. Script:
`models/compute_baselines.py` → `models/baseline_metrics.json`.

| rung | model | R² | RMSE | MAE | bias | notes |
|---|---|---:|---:|---:|---:|---|
| floor | mean-predictor (train-fold mean) | −0.015 | 74.7 | 58.2 | +0.15 | trivial "no skill" floor; ~R²=0 by construction |
| linear | ridge-on-PC20 embeddings | 0.373 | 58.7 | 44.0 | +1.44 | embeddings linear floor (20-PC PCA + Ridge) |
| **production** | **LightGBM emb64 (sibling trainer)** | **0.418** | **56.6** | **41.5** | **+0.50** | reproduction target — see `reports/training_run.md` |

## Reading

- The mean-predictor confirms the target carries real variance the features must explain
  (RMSE 74.7 = the standard deviation floor).
- Ridge-on-PC20 already reaches R²=0.37 from the embeddings alone — most of the signal is
  linear. LightGBM adds **+0.045 R²** over the linear floor (non-linear interactions), which is
  the entire production gain. This is consistent with the diagnosed feature ceiling: the model
  class is not the bottleneck; the embeddings are.

## Provenance

- features: `/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet`
  (tf-deep-landcover `e8c70584`)
- baseline script: `models/compute_baselines.py` (sklearn LeaveOneGroupOut; seed 42)
- production model: invoked via `src.agb.train_agb_lgbm` — not reimplemented here.
