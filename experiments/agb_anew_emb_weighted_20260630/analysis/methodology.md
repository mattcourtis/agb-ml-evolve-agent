# Methodology — emb-only ANEW AGB model, frontier-aware, trust-gated

## Purpose

Build a new above-ground-biomass (AGB) regression model over the **whole ANEW ground truth**
(51 projects, Quinte dropped) and answer, concretely, *how to think about regional/groups* for
training. Grounded in the prior self-referential DI/AOA analysis
(`experiments/agb_anew_gt_applicability_20260626/`), which showed the GT is one dense
temperate-broadleaf interior plus a small conifer/tundra frontier, with a regional-dependence
axis separating frontiers that collapse when their region is removed from those isolated
under any fold.

## The four roles of "groups"

Conflating these is the usual mistake; this experiment keeps them separate.

1. **CV grouping (honest error).** Plots within a project are spatially autocorrelated, so the
   atomic CV unit is the **whole project**, and we run a ladder of increasingly strict
   `groups`: LOPO (near transfer) → leave-bloc-out (far/regional) → leave-biome-out
   (new-ecology floor). Each answers a different deployment question.
2. **Architecture grouping — deliberately none.** The interior is too redundant to justify
   regional/biome sub-models and the frontier too small to train specialists, so a **single
   global model**.
3. **Weighting grouping.** Per-bloc, capped, sub-linear sample weights to stop the broadleaf
   majority compressing the frontier — *measured, not assumed* (decision rule below).
4. **Interpretation grouping.** Each project tagged by `regional_dependence`: a bad far-transfer
   fold for a regional-frontier project means "no nearby analogue" (benign); for a
   self-standing-frontier project it is intrinsic and only local ground truth fixes it.

## Data & model
- 51 projects / 12,636 plots; 64 codec embeddings (`common.EMB`); target `CO2` (tCO2/acre,
  **raw, uncapped** — eligible max 1010.8, p99 411; capping would bias the high-biomass conifer
  frontier low, so we keep raw and report MAE alongside RMSE).
- Single global LightGBM (`num_leaves=31, lr=0.05, min_child_samples=20`). 265 zero-CO2 plots
  (2.1%) are real young-stand/clearcut zeros and kept; predictions clipped at 0.
- `n_estimators` chosen **once** via a grouped-project holdout (~15% of plots, spans blocs,
  includes RainierGateway) with early stopping → reused across all schemes (random holdout
  would leak within-project redundancy and pick too many trees).

## CV ladder (groups = project / bloc / biome)
Generalised `cv_predict(X, y, groups, make_model, sample_weight)` (after
`scripts/trust/spatial_cv.py`). Bloc 2 **is exactly the 5 frontier projects**, so
leave-bloc-2-out is the true far-transfer test. Folds whose held-out median `di_bloc` exceeds
the AOA threshold are auto-flagged as extrapolation magnitudes, not generalisation. Leave-biome-out
is a directional floor only (Tundra+Boreal collapsed — both are the single project Doyon).

## Learner baseline check (cheap, not a tuned bake-off)
Because the emb-only ceiling is a *signal* limit (embeddings lack vertical structure), one
unweighted LOPO pass compares LightGBM vs untuned XGBoost vs a ridge floor to confirm the
ceiling is feature-driven, not learner-driven.

## Weighting comparison & ship rule
Schemes (all renormalised to mean weight 1): **S0** unweighted; **S1** per-bloc inverse density;
**S2/S3** capped-DI `clip((di/median)^α,1,5)`, α∈{0.5,1.0}; **S4** S1×S2. Scored on **unweighted**
LOPO OOF per-tier RMSE (weighted RMSE is not comparable across schemes). Ship the best frontier
scheme only if **all** hold, else S0:
1. regional-frontier RMSE improves ≥10% rel; 2. interior RMSE degrades ≤3% rel;
3. no regional-frontier range-compression regression; 4. `n_eff ≥ 0.85·N`.
Self-standing-frontier not improving is **not** a veto — it's the documented "needs local GT" boundary.

## Trust layer (ships with the model)
- AOA threshold refit via `di.fit()` on the 51-project cloud (asserted to match the
  applicability run, 0.558).
- DI→expected-RMSE isotonic curve (`uncertainty.fit_curve`) fit on the **shipped model's LOPO
  OOF residuals only** — every project is now in-fold so LOPO spans the full DI range;
  bloc/biome residuals are far-transfer extrapolation and are *not* pooled in.

## Modelling options considered (and why this design)
- **The data limits the approach more than the model choice limits the outcome.** Sample size
  (~12.6k) rules out data-hungry deep nets and rules in gradient-boosted trees; feature content
  (64-dim spectral-temporal embedding, no vertical structure) caps achievable accuracy
  regardless of learner; coverage geometry bounds what's learnable at the frontier.
- **Tabular deep learning** (MLP/FT-Transformer/TabNet) is not expected to beat GBTs at this
  scale, and cannot break a *signal* ceiling. The learner check empirically confirms this
  (LightGBM ≈ XGBoost ≈ ridge).
- **Spatial-imagery DL** (CNN/ViT over embedding patches or raw imagery) is the only genuinely
  different option — but it needs raster re-extraction (the same GEE blocker), is small-sample
  risky, and still doesn't target vertical structure as cheaply as adding CHM/GEDI. Deferred.
- **Native-uncertainty models** (quantile loss, NGBoost, GPs) could complement the DI→error
  curve; deferred as out of scope for this run.

## Limitations
- **Emb-only ceiling** (R²≈0.4, range compression): the model's value is coverage (all 51
  projects, all biomes) + the trust layer, not peak accuracy.
- **Affine verified only for Temperate Broadleaf** — conifer/tundra codec values are less
  certain, compounding their frontier status.
- Final model is a **candidate** in the data-space; not auto-promoted over the deployed model.
