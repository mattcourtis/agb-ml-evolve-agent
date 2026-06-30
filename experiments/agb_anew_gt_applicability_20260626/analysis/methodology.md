# Methodology — self-referential DI & AOA over the ANEW GT space

## Purpose

The trust toolkit (`scripts/trust/`) measures Dissimilarity Index (DI) and Area of
Applicability (AOA) relative to the **deployed 23-project model** — every other ANEW
project is scored only as an outside query. This experiment turns the same lens on the
**ground-truth training data itself**: which of the 51 eligible ANEW projects sit in the
interior of the GT embedding cloud (redundant), which sit on its frontier (unique), and
how much each project's applicability depends on its regional neighbours.

## Why self-referential leave-one-project-out

AOA (Meyer & Pebesma 2021, *Predicting into unknown space*) is defined relative to the
training set of the model being trusted, and an honest DI distribution and threshold
require **fold-aware leave-one-out** distances — a point's distance to itself is zero and
would bias the threshold downward. Applied to "the training data itself", that principle
resolves exactly to **leave-one-project-out (LOPO) over all 51 projects**: each plot's DI
is the nearest-neighbour weighted distance to plots of *other* projects. This is precisely
what `scripts/trust/di.py::fit()` already computes as `train_di` when grouped on
`project_name`.

## Data & feature space

- **Source:** `anew_canonical_codec.parquet` (52 ANEW projects, 12,834 plots, codec space).
- **Eligible set:** 51 projects / 12,636 plots after dropping **Quinte** (label outlier —
  CO₂ median ≈ 2× the cohort, robust-z 5.0; same exclusion as the Ireland selection run).
- **Features:** the 64 codec embeddings (`emb_00..emb_63`). Emb-only is the only space
  available for all 51 projects without new GEE extraction (CHM/topo/dstx exist only for
  the 23 modelled projects + Ireland).
- **Weights:** the deployed emb-only model's gain importance (`gain_weights("embonly")`),
  normalised to mean 1 — keeps this comparable to the existing AOA reports and deterministic.

## Cross-fold design — two folds, one DI implementation

Both folds are the *same* `di.fit(X, groups, features, w)` call with a different `groups`
array; `train_di` is the fold-aware "NN weighted distance to points in *other* groups". No
new DI maths.

1. **Project fold (LOPO)** — `groups = project_name`. `di_lopo` per plot; the AOA boundary
   is `threshold_cast = Q75 + 1.5·IQR` of the LOPO DI distribution (**0.558**).
2. **Regional fold (leave-bloc-out)** — `groups = spatial bloc`. `di_bloc` per plot.
   Because a whole region is removed at once, `di_bloc ≥ di_lopo`; the lift
   `regional_dependence = median(di_bloc) − median(di_lopo)` measures how much a project
   leans on its regional neighbours.

### Spatial blocs

Per-project centroid = mean(lon, lat) reprojected to **EPSG:5070** (Albers, metres);
KMeans (`random_state=42`). K starts at 6 and is reduced until every bloc holds ≥3
projects — here **K = 4** (sizes 18 / 12 / 5 / 16), because the Alaska/PNW projects are
spatially sparse. Bloc 0 = Lake States / Upper Midwest, bloc 1 = Northeast / Maine,
bloc 2 = Alaska + Pacific Northwest (the conifer/tundra frontier), bloc 3 =
Appalachia / Southeast.

## Robustness cross-check

The per-project ranking by median LOPO DI is compared (Spearman ρ) against two alternative
distance definitions, mirroring the Ireland ranking validation:

- **vs unweighted CAST DI:** ρ = **1.00** (the emb-only gain weights are near-uniform, so
  weighting does not reorder projects).
- **vs Mahalanobis DI** over the 64 embeddings (fold-aware): ρ = **0.85**.

High agreement on both shows the ranking is a property of the embedding geometry, not an
artefact of the weighting or the specific metric.

## Outputs

Plot-level (`plot_level_di.parquet`), per-project ranking (`project_di_ranking.parquet`),
thresholds + ρ (`thresholds.json`), and bloc assignments
(`bloc_assignments.parquet`) — all in the data-space (see `final/DATA_STORE.md`). Five
plot-level figures in `figures/`.

## Limitations

- **Emb-only.** No CHM/topo/disturbance — structural separation between similar-spectra
  forests is not captured. A 73-feature pass needs GEE extraction for the 28 unused projects.
- **Plot-level, not wall-to-wall.** Maps colour the plots, not full project extents; the
  raster-over-boundaries map is a documented TODO (`scripts/anew_gt_applicability/extract_aoa_rasters.py`).
- **Affine verified only for Temperate Broadleaf.** The codec mapping was fit on Bayfield;
  conifer/tundra codec values are less certain, which compounds with their frontier status.
