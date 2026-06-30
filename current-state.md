# AGB ANEW model — current state

**Status (2026-06-30):** the current baseline for ANEW above-ground-biomass (AGB) modelling is a
**log1p, emb-only LightGBM over all 51 eligible projects, shipped with a self-referential DI/AOA
trust layer**. This document is the reference point; further discovery should branch from here.

> A self-contained, figure-inclusive version of this report is at **`current-state.html`** (figures
> embedded; open in a browser — no network needed). Repo figures themselves are gitignored.

The model is honest and trust-gated but **emb-only feature-limited** (R²≈0.4, range compression):
its value is *coverage* (all 51 projects, all biomes) + *calibrated applicability* + a *de-biased
low end*, not peak accuracy. The one ceiling-breaking lever — structural (vertical) features —
remains deferred (see "Next discovery").

## Current baseline — what "current" means

| Item | Value |
|---|---|
| Data | 51 ANEW projects, 12,636 plots (Quinte dropped — label outlier) |
| Features | 64 codec embeddings `emb_00..emb_63` (the only space covering all 51 projects; no GEE) |
| Target | `CO2` (tCO2/acre, raw, uncapped; eligible max 1010.8) |
| Model | LightGBM, `num_leaves=31, lr=0.05, min_child_samples=20`, `n_estimators=172` |
| **Target transform** | **log1p** — predict with `clip(expm1(booster.predict(X)), 0)` |
| Weighting | none (S0) — frontier-aware weighting was tested and did not help |
| Trust layer | self-referential CAST DI, AOA threshold **0.558**, isotonic DI→expected-RMSE curve |
| Reference artifact | `models/anew_emb51_log1p_model.txt` (+ `…_log1p_features.json`) — **data-space candidate, current baseline; not auto-promoted** over the deployed model |

## Headline metrics (LOPO, leave-one-project-out)

| Metric | S0 raw-L2 | **log1p (current)** |
|---|---|---|
| RMSE (all) | 73.7 | 77.3 |
| bias, true<100 | +43.0 | **+23.8** |
| RMSE, true<100 | 58.5 | **44.4** |
| zero-detection recall | 0.54 | **0.74** |
| discrimination, true<100 (Spearman) | 0.574 | 0.599 |
| separability AUC, true<50 | 0.874 | 0.881 |
| bias, true>150 (accepted trade) | −65.6 | −87.7 |

Far-transfer (leave-bloc-out) and new-ecology (leave-biome-out) errors are higher and AOA-flagged;
see the transfer ladder in the model experiment report.

## How we got here — three experiments

### 1. GT-space DI/AOA applicability — `experiments/agb_anew_gt_applicability_20260626/`
Self-referential DI/AOA over the 51-project GT. The GT is **one dense temperate-broadleaf
interior** (44 projects, ~100% mutually inside the AOA, highly redundant) plus a small **genuine
frontier** (Kootznoowoo, Doyon, RainierGateway, LongviewRanch, HighCascades — conifer/tundra).
A *regional-dependence* axis separates frontiers that collapse when their whole region is removed
(LouisianaLowlands, Apalachicola, Soterra) from those isolated under any fold (Kootznoowoo,
RainierGateway, Doyon). AOA threshold 0.558; 94% of plots inside. This is what justified a single
global model (interior too redundant to split, frontier too small to specialise) and the
project/bloc/biome CV grouping.

### 2. Emb-only global model + trust layer — `experiments/agb_anew_emb_weighted_20260630/`
- **Learner check:** LightGBM (71.4) ≈ XGBoost (71.9) ≈ ridge (71.6) interior LOPO RMSE — even a
  linear readout matches boosted trees, so the ceiling is **feature-driven, not learner-driven**.
- **Frontier-aware weighting:** S1–S4 all made the frontier *worse*; the 4-gate ship rule
  selected **S0 (unweighted)**. No frontier signal in the embeddings for weighting to exploit.
- **Trust layer:** AOA threshold 0.558 + monotone DI→expected-RMSE curve (interior ~69 →
  frontier ~99 tCO2/acre); every prediction carries a DI, an inside/outside-AOA flag, expected error.

### 3. Low/zero-biomass de-biasing — `…/analysis/low_end_debiasing.md`
The raw model over-predicts true CO2 < 100 by +35..+49 (regression-to-the-mean; 42% of plots).
A bake-off of objective re-aiming variants under the same LOPO CV found:
- **log1p wins** — halves the <100 bias, cuts <100 RMSE 24%, lifts zero-recall 0.54→0.74, slightly
  improves discrimination; accepted, quantified cost is more high-end under-prediction.
- **Two-stage hurdle did not help** — its classifier AUC (~0.88) equals the regressor's implicit
  separability, so low biomass *is* moderately separable but already exploited; structure adds nothing.
- **Calibration cannot de-bias** — post-hoc isotonic is inert and *re-inflates* bias when stacked
  on log1p (it targets the conditional mean, re-imposing the mean-reversion). Re-aiming the
  objective is the right tool; calibrating toward the mean is the wrong one.

## Known ceiling & next discovery

The residual +24 low-end bias and the range compression are the floor reachable on emb-only
features. The single highest-leverage next step is the deferred **GEE extraction of CHM / topo /
disturbance for the 28 non-modelled projects → full-feature model over all 51**, giving the
vertical-structure signal needed to separate biomass *level* (not just rank) within similar-spectra
stands. Native-uncertainty heads (quantile/NGBoost) could later complement the DI→error curve.
Any such work should be evaluated against the metrics above, on the same LOPO/bloc/biome CV.

## References

| Area | Path |
|---|---|
| Trust toolkit (DI/AOA/curve) | `scripts/trust/{di,aoa,uncertainty,guardrails,spatial_cv,common}.py` |
| GT applicability code / docs | `scripts/anew_gt_applicability/`, `experiments/agb_anew_gt_applicability_20260626/` |
| Model + low-end code | `scripts/anew_emb_model/{data,cv_ladder,decide_and_train,trust_fit,make_figs,low_end,finalise_low_end}.py` |
| Model experiment docs | `experiments/agb_anew_emb_weighted_20260630/{analysis,final,figures}/` |
| Data-space outputs (gitignored) | `/home/mattc/data-space/carbonmap-embeddings/agb_anew_emb_weighted_20260630/` (models, cv, trust, low_end) |
| Canonical GT (codec) | `/home/mattc/data-space/carbonmap-embeddings/agb_trust_aoa_20260626/preprocessing/anew_canonical_codec.parquet` |
| Regeneration | see each experiment's `final/DATA_STORE.md` |
