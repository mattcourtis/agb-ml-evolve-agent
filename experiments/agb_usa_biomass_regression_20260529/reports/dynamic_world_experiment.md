# Dynamic World Experiment — observed vs predicted

Baseline = `features_iter3.parquet` with the **disturbance fix** (broken `dist_years_since` replaced by survey-relative `dstx_pre_ysd`), 4636 plots, LightGBM 23-project LOPO. DW = survey-year growing-season buffer-mean probability bands ['dw_trees', 'dw_shrub_and_scrub', 'dw_grass', 'dw_crops', 'dw_bare'].

## 1. `dist_years_since` fix (before → after)

- plots whose value changed: **986** / 4636
- post-survey-harvest plots (n=99): 47 had broken `years_since==0` (a 'just-disturbed' signal on high-biomass plots); after the fix 84 are correctly set to the undisturbed sentinel (100).
- example post-survey plot (plot_id=100.0): broken=0 → fixed=100, target=73 tCO₂/acre (legitimately high).

## 2. LOPO: with vs without Dynamic World

| config | R² | RMSE | MAE | bias | Q1 | Q2 | Q3 | Q4 | Q5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (no DW) | 0.4251 | 56.24 | 41.35 | +0.8 | +36.1 | +32.2 | +17.4 | -9.4 | -72.1 |
| baseline + DW | 0.4256 | 56.22 | 41.36 | +0.9 | +36.9 | +32.9 | +17.1 | -9.7 | -72.4 |

ΔR² from DW = **+0.0005**; ΔQ1 bias = +0.8 (negative = less over-prediction of low-biomass plots).

## 3. Within-Q1 correlation of DW bands with baseline residual

(+resid = over-prediction; −corr ⇒ band flags the over-predicted low plots)

| band | corr (Q1) | corr (all) |
| --- | --- | --- |
| dw_trees | +0.174 | -0.024 |
| dw_shrub_and_scrub | -0.146 | +0.035 |
| dw_grass | -0.204 | +0.030 |
| dw_crops | -0.113 | +0.030 |
| dw_bare | +0.051 | +0.008 |

