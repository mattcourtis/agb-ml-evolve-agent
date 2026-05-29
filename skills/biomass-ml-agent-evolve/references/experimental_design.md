# Experimental Design

## Purpose
Translate the user task into a testable ML problem.

## Task taxonomy
- biomass_regression (continuous AGB or stock at point / plot / pixel)
- canopy_height_regression (continuous canopy / RH metric at point / plot / pixel)
- biomass_segmentation (wall-to-wall raster regression / dense prediction)
- change_detection (biomass delta or disturbance class between two times)

## Required design decisions
- target variable and units (e.g., tCO₂/acre, t/ha, m, % cover)
- spatial unit and geometry (plot, pixel, polygon, hex cell)
- temporal unit and aggregation window (annual composite, multi-year mean, change interval)
- resolution and CRS
- inference granularity (per-plot, per-pixel 10 m, per-hex)
- deployment format (parquet, GeoTIFF, COG, EE asset)
- acceptance metrics (must include per-quintile bias and predicted-range discrimination for any regression task claiming generalisation — see `references/evaluation.md`)

## Design gates
Reject if:
- target is ambiguous
- spatial or temporal unit mismatches labels
- metric does not match task objective
- generalisation claim is not paired with a spatial-holdout strategy that proves zero partition-key overlap (see `references/database_preprocessing.md`)
- expected ecoregion coverage is broader than the training-pool ecoregion coverage without an external holdout declared

## Required artifact
`configs/experiment_design.md` — single primary artifact produced by the Experimental Design Actor.

## Required artifact sections
- task statement
- assumptions
- target definition (including units and reference: standing stock vs. annual increment vs. change)
- label definition (provenance, plot footprint, GPS error tolerance)
- data scope (ecoregions, years, exclusions)
- train/val/test design summary (must cite the partition key — typically `project_name` for plot data)
- candidate baselines
- budget tier
- ecoregion-coverage statement: training-pool ecoregions vs. claimed inference ecoregions

## Actor addendum
Require a "design-risk register" with top 5 risks.

For biomass regression with optical embeddings alone, pre-register the known feature-ceiling finding from the prior AGB investigation: isotonic post-hoc calibration cannot fix the Q1/Q5 predicted-range collapse (`agb-modelling-context.md` §Controls). The structural fix is canopy-height features (GEDI or NEON). If the embeddings-only baseline rediscovers this ceiling, the iteration-1 lever should be GEDI extraction, not loss-function tuning or calibration layers — see `references/improvement_loop.md` Critic addendum for the explicit veto.

The risk register must include:
- label-noise floor from GPS error vs. plot footprint (e.g., ~10 m error on a ~14.7 m radius plot)
- ecoregion mismatch between training pool and deployment area
- ecoregion-specific failure mode (per-ecoregion R² variance — e.g., WV R²=0.17 vs. Midwest R²=0.42 in the prior investigation)
- feature-ceiling risk (cite `references/improvement_loop.md` mapping table; name the candidate next-feature source from `references/source_registry.md`)

## Critic addendum
Check target leakage, target definition mismatch, and infeasible evaluation claims. Reject if:
- the design claims generalisation to a new project / ecoregion but does not declare an external holdout
- the acceptance-metric list omits `per_quintile_bias` for a regression task that claims to discriminate biomass levels
- the design-risk register has fewer than 5 risks or omits the feature-ceiling risk for biomass regression
