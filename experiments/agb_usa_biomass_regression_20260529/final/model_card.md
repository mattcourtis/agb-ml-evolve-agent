# Model Card — agb_usa joint_v2 LightGBM (iteration 0 reproduction)

## Overview

- **Task:** plot-level above-ground biomass regression (CONUS forest).
- **Target:** `CO2` — standing-stock tCO₂/acre.
- **Model:** LightGBM gradient-boosted trees (num_leaves=31, lr=0.05, n_estimators=2000,
  early_stop=50) on 64-dim AlphaEarth Foundation (AEF) embeddings (10 m, annual).
- **Pool:** 4,636 plots, 23 projects, years 2022+2023 (WV Appalachia + Upper Midwest + NE).
- **Validation:** leave-one-project-out (23 folds). Reported metrics are out-of-fold.
- **Provenance:** trained via `tf-deep-landcover/src/agb/train_agb_lgbm` @
  `e8c70584fb1a8705308004fbed123392c8f51654`; orchestrated by
  `crop-ml-agent-evolve` @ `ea0551b...`.

## Performance (out-of-fold, project-LOPO)

| metric | value | realistic threshold | status |
|---|---:|---:|:--:|
| R² | 0.4182 | ≥ 0.40 | ✅ |
| RMSE | 56.58 tCO₂/acre | ≤ 60 | ✅ |
| MAE | 41.49 | — | ✅ |
| bias | +0.50 | \|·\| ≤ 5 | ✅ |
| predicted_range_discrimination | 0.468 | ≥ 0.6 (stretch) | ❌ (feature ceiling) |

Reproduces the joint_v2 baseline R²=0.42 bit-identically (iter-0 acceptance ±0.03 met).

### By region / ecoregion

| region | ecoregion | R² | discrimination | n |
|---|---|---:|---:|---:|
| wv | Appalachian mixed mesophytic | 0.157 | 0.19 | 598 |
| mw | Western Great Lakes | 0.415 | 0.47 | 2,619 |
| ne | New England-Acadian | 0.476 | 0.49 | 1,407 |

## Intended use & limitations

- **Use:** project- / stand-level biomass estimates where aggregate accuracy suffices.
- **Key limitation — dynamic-range compression:** over-predicts low-biomass plots (Q1 +35.6),
  under-predicts high-biomass plots (Q5 −72.1). Not suitable for plot-scale discrimination of
  young vs. mature stands, especially in Appalachian closed-canopy hardwood. This is a feature
  deficit (optical embeddings cannot see vertical structure), not a tuning issue — five tuning
  levers were ruled out (see `research/deep_research.md`).
- **Next step:** add GEDI canopy height (iteration 1).

## Ethical / provenance notes

- Labels: ANEW client field plots (~10 m GPS error, 1/24-acre footprint).
- No personal data. Predictions are environmental estimates with quantified uncertainty
  (RMSE 57) and a documented systematic failure mode (range compression).
