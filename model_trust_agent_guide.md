# Model Trust — Agent Guide

## Definitions
- **DI** — Dissimilarity Index: distance of a prediction pixel from the training set in predictor space
- **LPD** — Local Point Density: KNN density of training points in geographic or feature space
- **AOA** — Area of Applicability: the subset of the prediction domain where DI is below the threshold

## Key Concepts
- Trust is spatially heterogeneous: one metric doesn’t describe a national map.
- AOA is the core safety layer: it formalizes where the model is interpolating vs extrapolating.
- Spatial CV is non-negotiable for credibility when mapping beyond sampled regions.
- DI + LPD + error profiles translate into operational “risk-of-wrong” layers that can ship alongside AGB predictions.

## Workflow

**0. Define domain** — confirm spatial extent (CONUS / full USA / subset) and land cover scope before anything else.

**1. Audit training data** — summarise by ecoregion, elevation, forest type, biomass range. List missing regimes explicitly; these will be high-DI zones.

**2. Validate honestly** — spatial CV (block/hex splits) as primary. Report random CV alongside it.

**3. Compute DI/AOA** — standardise predictors (same transforms as training), compute DI per pixel. AOA threshold = 95th percentile of DI values within the training set. Output: binary AOA mask + continuous DI map.

**4. Build uncertainty surface** — bin predictions by DI quantile, compute RMSE per bin, fit a monotonic curve. Output as an expected-error map layer alongside AGB.

**5. Enforce guardrails** — for out-of-AOA pixels: suppress (null) or label as extrapolation and exclude from aggregations. All aggregated outputs must report % area inside AOA + DI distribution statistics.
