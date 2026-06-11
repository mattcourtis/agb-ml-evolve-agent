# Iteration 1 — embedding-space analog-subset selection (experiment design)

## Context
Iteration 0 found the embdstx head transfers to Ireland with severe OOD (100% of 141 Locations beyond
the 99th-pct training Mahalanobis radius; USA-vs-Ireland domain-classifier AUC ≈1.0), so absolute
tCO2/acre levels are extrapolation — decision RETRAIN_WARRANTED. Iteration 1 selects, from the FULL
ANEW pool (52 projects / 12,837 plots, `anew_gt_with_eco_info.gpkg`: project_name, CO2 tCO2/acre, Date,
ECO_NAME/BIOME, points EPSG:4326), the subset whose AEF embedding distribution is most similar to
Ireland's, to train a more Ireland-relevant head. Reuses ACCEPTED iter0 artefacts (research, data_profile,
experiment_design, encoding gate, evaluation harness).

## Objective & hypothesis
Identify the ANEW subset that minimises Irish OOD while (a) staying valid in-domain on US holdout and
(b) preserving coverage of Ireland's biomass band. H: a data-driven embedding-similarity subset cuts the
Irish Mahalanobis fraction-beyond-radius and domain AUC well below full-CONUS, without truncating the
high-biomass range, and beats the climate/ecoregion heuristic on the same OOD metrics.

## Guardrails (binding)
1. ONE ENCODING. Re-extract ALL candidates AND Ireland from the GEE float asset
   `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` natively (no affine), survey-year aligned, plot/polygon mean.
   Mixing local-int8 (existing 4,636) with GEE-affine (other ~8,200 + Ireland) would contaminate
   cross-project distances. Native float for all is methodologically required here.
2. TWO-OBJECTIVE SELECTION. Maximise Ireland-similarity AND retain biomass-range coverage (Ireland
   rotation-end ~150-376 Mg/ha per research). Pure nearest-neighbour selection risks a similar-but-
   biomass-truncated subset that would reproduce the saturation problem.
3. NO TARGET LEAKAGE. Selection uses Irish EMBEDDINGS only (Ireland has no labels) — legitimate
   transductive/domain-adaptation. US labels used only for training/holdout, partitioned by project (LOPO).

## Method — triangulated Ireland-similarity (per-plot + per-project)
- Domain-classifier Ireland-likeness: reuse iter0 USA-vs-Ireland classifier; per-plot P(Ireland-like)
  ≈ density ratio. Primary soft analog score; also yields importance weights.
- Mahalanobis distance to the Irish distribution (Irish mean+covariance). Lower = more analog.
- k-NN distance to the Irish manifold (robust cross-check).
- MMD / energy distance per project vs Ireland (project-level ranking table).
- Report which embedding dims drive similarity; cross-check vs the ~8 dims carrying 68% of iter0 shift.
- PCA/UMAP overlap visual.

## Candidate subsets (pre-registered)
- S0 full-CONUS (negative control / baseline).
- S1 climate+ecoregion heuristic: New England-Acadian (maritime) + Pacific/Cascades/Alaskan conifer.
- S2 project-level embedding-nearest: top-k projects by MMD to Ireland (k keeps biomass coverage + LOPO).
- S3 plot-level embedding-nearest, COVERAGE-CONSTRAINED: most Ireland-like plots subject to retaining
  >=X% of each biomass decile in the Irish-relevant range.
- S4 (optional) importance-weighted full pool: all plots, weighted by Ireland-likeness density ratio.

## Evaluation — two tiers
TIER 1 (cheap, selection quality, no retrain) — recompute iter0 OOD diagnostics, Ireland vs each subset:
- Irish Mahalanobis fraction beyond the subset 99th-pct radius (target: 100% -> much lower).
- Domain-classifier AUC subset-vs-Ireland (target: << 1.0).
- Biomass-range coverage retained (guardrail check); PCA/UMAP overlap.
TIER 2 (payoff, requires training) — train a LightGBM head per subset; dual matrix:
- In-domain: US LOPO within subset (R2/PRD vs locked CONUS baseline R2 0.418 / PRD 0.468).
- Zero-shot Ireland: rerun the iter0 bias-characterisation matrix (H1/H2/H3, OOD, saturation vs DB).
Report a trade-off frontier: OOD reduction vs in-domain R2 vs biomass coverage vs subset size.

## Decision rule (threshold-free)
- ADOPT subset if Tier-1 OOD materially reduced AND Tier-2 in-domain R2 within ~0.05 of CONUS AND
  biomass coverage spans Ireland's band.
- PREFER importance-weighting (S4) if hard subsets sacrifice too much in-domain R2 for the OOD gain.
- ESCALATE "need in-region data / SAR-CHM lever" if even the best subset leaves Ireland far outside the
  manifold (no true US analog exists).

## Sequencing & checkpoint
iter1_extract (heavy GEE) -> iter1_select (Tier-1) -> CHECKPOINT/report selection quality -> iter1_train_eval
(Tier-2, only if Tier-1 shows a usable analog) -> iter1_decision. Stop after Tier-1 if no subset reduces OOD.

## Reproducibility
seed 42; AEF asset pinned V1 ANNUAL; reuse scripts/extract_ireland_aef.py + fit pattern; reuse
evaluation/run_bias_characterisation.py and compute_biomass_metrics.py for Tier-2.
