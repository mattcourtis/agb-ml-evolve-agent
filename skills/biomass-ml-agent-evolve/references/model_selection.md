# Model Selection

## Purpose
Choose a baseline-first candidate set appropriate to task and budget.

## Baseline requirement
A simple baseline is mandatory before complex models.

## Biomass regression ladder

1. **Linear / ridge / lasso on top-K PCs of embeddings** — cheapest baseline; reveals whether the embedding carries linear signal at all.
2. **Random forest on raw embeddings** — non-linear baseline; useful sanity check on tree-vs-linear gap.
3. **LightGBM / XGBoost on raw embeddings** — current production class; fast, regularised, handles 64–256-dim feature vectors well.
4. **LightGBM + GEDI RH metrics** — the identified next lever for breaking the optical-embedding feature ceiling. Adds vertical-structure signal that optical embeddings cannot encode.
5. **LightGBM + GEDI + DEM (topography)** — slope/aspect conditioning for stand structure.
6. **Multimodal fusion**: optical embedding + GEDI + SAR (PALSAR-2) + DEM, via concatenation or attention-head meta-learner.
7. **Foundation-embedding ensemble**: stack AEF + Presto + Clay (or comparable) under a tabular meta-learner. Tests whether different pre-training regimes recover different aspects of structure.
8. **Co-target training**: model trained with GEDI L4A as auxiliary target alongside the primary plot target; sparsity of plot supervision augmented by dense L4A supervision.
9. **Temporal stacking**: multi-year AEF + GEDI deltas → captures growth signal where the target reflects annual increment or change.
10. **End-to-end CNN / U-Net regressor** over raw imagery (Large tier only) — most expensive; only when feature engineering ladder has reached an ablation-bounded ceiling.

## Canopy-height regression ladder
Same shape as biomass regression but: GEDI RH98 / NEON CHM as primary target; ridge on embeddings → LightGBM on embeddings + DEM → fusion with SAR if seasonal signal matters. End-to-end CNN regressor is more common at the canopy-height task because GEDI provides dense pseudo-labels.

## Biomass segmentation ladder
- pixel-wise LightGBM baseline on aggregated stack
- U-Net / DeepLabV3+ regression head (replaces softmax with linear / Gaussian-NLL output)
- Swin / ViT segmentation backbone
- multimodal-fusion segmentation (optical + GEDI gridded + SAR + DEM)

## Change-detection ladder
- per-pixel difference baseline (post-target − pre-target embedding distance)
- LightGBM on per-pixel feature deltas
- Siamese CNN
- temporal transformer

## Candidate selection criteria
- data volume (plot count for regression; tile count for segmentation)
- label granularity (plot footprint, GPS error tolerance)
- spatial scale (plot vs. pixel vs. region)
- temporal length (single-year, multi-year, change interval)
- missingness (GEDI footprint sparsity, embedding year coverage)
- budget (tier — see below)
- explainability needs (LightGBM + SHAP is the default explainable option)
- deployment constraint (parquet at plots vs. COG at 10 m vs. EE asset)

## Compute-budget tiers

Pick the tier matching `experiment_config.yaml :: training.budget_tier`. The Critic rejects candidate sets that violate the allowed-model column for the declared tier.

| Budget tier | Expected hardware | Allowed models | Default iteration cap |
|---|---|---|---:|
| Small | laptop / small CPU / modest single GPU | ridge/lasso, RF, XGBoost, LightGBM (incl. fusion via concatenated tabular features), shallow MLP | 3 |
| Medium | single strong GPU | foundation-embedding ensembles, attention-head fusion meta-learner, U-Net / DeepLabV3+ on small tiles, small transformer segmentation backbones | 4 |
| Large | multi-GPU or managed training | end-to-end CNN regressors on raw imagery, large segmentation transformers, broad HPO, full-tile multimodal end-to-end fusion | 6 only if explicitly approved |

## Required artifact
`configs/model_candidates.yaml` with:
- model name
- rationale
- baseline flag
- expected runtime tier
- expected strengths
- known risks
- feature dependencies (declare if the candidate requires GEDI extraction, SAR pre-processing, DEM mosaic, etc., so the upstream Preprocessing Actor can stage them)

## Critic addendum
Reject if:
- the skill skips a simple baseline (ridge/RF) before proposing LightGBM-only or fusion candidates
- a fusion or end-to-end candidate is proposed without first establishing the LightGBM-on-embeddings reference point in the same experiment
- a candidate has a feature dependency (e.g., GEDI) that is not declared in any ACCEPTED upstream artefact
- candidates declared exceed the budget tier's allowed-model column without `experiment_config.yaml :: training.allow_extended_iterations: true`
