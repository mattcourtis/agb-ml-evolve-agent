# Disturbance-Timing LOPO Experiment

Baseline `features_iter3.parquet` (4636 plots), LightGBM 23-project LOPO. dstx predictive features: ['dstx_pre_loss_5yr', 'dstx_pre_ysd', 'dstx_loss_frac_buf', 'dstx_lt_mag']. Cleaned configs drop 99 post-survey-contaminated plots.

| config | n | R² | RMSE | MAE | bias | Q1 | Q2 | Q3 | Q4 | Q5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1_baseline | 4636 | 0.4274 | 56.13 | 41.27 | +0.9 | +36.0 | +32.1 | +18.0 | -9.5 | -72.2 |
| 2_baseline+dstx | 4636 | 0.4355 | 55.73 | 40.90 | +1.1 | +34.4 | +31.0 | +18.2 | -8.3 | -69.6 |
| 3_baseline_clean | 4537 | 0.4312 | 55.81 | 40.94 | +0.7 | +35.3 | +31.8 | +17.5 | -9.4 | -71.8 |
| 4_clean+dstx | 4537 | 0.4404 | 55.36 | 40.63 | +0.9 | +34.6 | +30.6 | +17.4 | -8.6 | -69.5 |

**Read:** success = Q1 bias shrinks toward 0 (config 2 vs 1) without harming Q5 or overall R²; config 3 lifts R² if contamination is material.

## Within-Q1 correlation of dstx predictors with baseline residual

(+resid = over-prediction; a strong −corr means the feature flags the plots the baseline over-predicts.)

| feature | corr (Q1 only) | corr (all) |
| --- | --- | --- |
| dstx_pre_loss_5yr | -0.251 | +0.008 |
| dstx_pre_ysd | +0.354 | -0.058 |
| dstx_loss_frac_buf | -0.340 | +0.061 |
| dstx_lt_mag | -0.253 | +0.109 |

