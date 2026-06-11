# Ireland AGB — model-vs-Deep-Biomass bias characterisation

**Mode:** zero-shot transfer of the pre-trained `embdstx` head (64 AEF embeddings + 3 survey-relative
Hansen disturbance features; target tCO₂/acre) to 141 Irish Dasos forestry Locations.
**There is NO ground truth.** Deep Biomass (DB) is a known under-estimator used as a *directional lower
bound*, not truth. All comparisons measure agreement/divergence, never accuracy. DB converted Mg/ha →
tCO₂/acre via ×0.6977 (both AGB-only). Primary reference = DB 2020–2024 mean; 2024-only is the
sensitivity. Encoding gate PASSED (held-out corr 0.986, post-affine slope median 1.006). seed 42.

Numbers below are from `evaluation/evaluation_matrix.yaml` / `_results.json`, produced by
`evaluation/run_bias_characterisation.py`.

## Headline
The transfer is **encoding-valid and structurally sensible but operates in a severe extrapolation
regime**, so absolute values are not yet trustworthy. Decision-rule outcome: **RETRAIN_WARRANTED**
(escalate the conditional analog-subset retrain) — not "credible", not "halt".

- Portfolio: our head **91.6** tCO₂/acre vs DB **27.3** (2020–24 mean) — **3.35×** higher.
- Pred range 26.7–138.4 tCO₂/acre; 0% above the training max (520.95).

## H1 — directional dominance (our_pred ≥ DB): **SUPPORTED**
- 2020–24 mean: signed Δ (pred−DB) mean **+64.3**, median **+72.5**, IQR [49.6, 83.2]; **98.6%** of
  Locations have pred ≥ DB (range −8.3 … +104.8).
- 2024-only sensitivity: Δ mean +59.5; 95.7% pred ≥ DB.
- Interpretation: consistent with DB systematically under-reading this high-biomass Sitka plantation,
  exactly as the research envelope (rotation-end ~150–376 Mg/ha) predicts. Figures:
  `figures/scatter_pred_vs_db.png`, `figures/delta_histogram.png`.

## H2 — saturation resistance (gap widens with biomass band): **NOT SUPPORTED**
- Per-quintile signed bias by DB magnitude (2020–24): Q1 63.9, Q2 74.6, Q3 64.2, Q4 61.9, **Q5 56.9** —
  flat-to-declining, **not** monotone increasing (the Q5 gap is the *smallest*, not largest).
- 2024-only: same non-monotone shape, Q5 lowest (34.5).
- PRD (predicted-range-discrimination over DB-magnitude quintiles): **0.774** (2020–24) / 0.462 (2024).
  Healthy spread, but the level is not directly comparable to the CONUS 0.468 baseline (that denominator
  is true-target spread; here it is DB-magnitude spread — context only).
- Interpretation: our head does **not** pull progressively further above DB in the highest-biomass
  stands. Because both models compress at the top end, we cannot show added saturation resistance from
  this comparison alone. Figure: `figures/quintile_signed_bias.png`.

## H3 — structural-covariate rank-tracking (the GT substitute): **MOSTLY SUPPORTED**
Spearman ρ (our_pred vs structure, with DB for contrast):
- Stand age: ρ_pred **0.553** (p<1e-6) vs ρ_DB 0.113 (p=0.18) → our_pred tracks age far better than DB.
- Hdom: ρ_pred **0.556** (p<1e-6) vs ρ_DB 0.200 (p=0.018) → our_pred tracks height far better than DB.
- Yield class (YC): ρ_pred −0.077 (ns) and ρ_DB 0.031 (ns) → **null for both** (YC is a growth-rate
  potential index, not a standing-stock measure, so a weak link is unsurprising).
- Monotone covariate gradients confirm it: pred rises 55→98→103→110 across age bins 0-10→30-40, and
  55→87→101→108 across Hdom bins; DB stays ~23–31 throughout (essentially flat).
- Species: Sitka SS (n=137) pred 90.9 / DB 26.9; broadleaf/other (n=3) pred 119 / DB 34.
- Interpretation: the head is reading real stand structure (older/taller → more biomass) markedly better
  than DB — the strongest evidence the transfer is not noise. Figures: `figures/pred_db_vs_age.png`,
  `figures/pred_vs_hdom.png`.

## OOD / covariate shift: **SEVERE DOMAIN SHIFT**
- Mahalanobis: training 99th-pct radius 14.79; Irish embeddings min/median/max 27.8 / 32.4 / 38.9 →
  **100%** of Locations lie beyond the 99th-pct training radius.
- Domain classifier (USA 4636 vs Ireland 141, HistGBM, 5-fold): AUC **≈1.0** (0.999998).
- Interpretation: although the embeddings are *encoding-consistent* with training, the Irish
  distribution is entirely outside the US training manifold. Every Irish prediction is extrapolation —
  this is the central caveat and the main driver of the retrain decision.

## Saturation
- Our head: 79.4% of Locations > 80 tCO₂/acre; 28.4% above the ~105 tCO₂/acre saturation-onset
  reference; **0%** above training max; max 138.4. DB: 0% > 80; max 66.9.
- Most Irish stands sit in the band where optical AGB models lose discrimination; the head keeps them
  below its training ceiling, so no clipping artefact, but high-end absolute values are uncertain.

## Decision (threshold-free rule, experiment_design §7)
- encoding_gate PASS · H1 SUPPORTED · H2 NOT SUPPORTED · H3 MOSTLY SUPPORTED · OOD SEVERE.
- **Outcome: RETRAIN_WARRANTED.** The credible-transfer branch requires H1+H2+H3 with non-catastrophic
  OOD. H2 fails and OOD is catastrophic (matches counter-hypothesis C2 — severe domain shift — partly
  offset by genuine H3 structural tracking). Per rule (b): gate-OK but H2 clearly fails → escalate the
  conditional analog-subset (maritime + high-biomass-conifer) retrain to the Improvement Planner.
  Not "halt" (gate passed and structure is tracked); not "fully credible" (extrapolation + no H2).

## Limitations
- No ground truth: H1/H2 measure divergence from an under-estimator, not error. DB's own bias shape
  contaminates the quintile read of H2.
- Severe OOD: absolute tCO₂/acre values are extrapolation beyond the US training manifold; treat the
  *ranking/structure* signal (H3) as more trustworthy than the *levels* (H1 magnitude).
- 17/141 Locations (surveyed 2015–2016) use a pre-2017 AEF fallback (clamped to 2017) — small temporal
  misalignment.
- Bayfield-in-sample only affects encoding-gate interpretation, not the Irish comparison.
- YC null is expected (yield class ≠ standing stock).
