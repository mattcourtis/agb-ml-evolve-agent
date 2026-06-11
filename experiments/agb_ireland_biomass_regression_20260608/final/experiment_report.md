# Experiment Report — Ireland AGB zero-shot transfer (embdstx head)

- experiment_id: `agb_ireland_biomass_regression_20260608`
- mode: **inference-only zero-shot transfer + model-vs-model** (NO model trained, NO ground truth)
- decision: **RETRAIN_WARRANTED**
- git: `b6d219ac5090543f58480d9df30e6a16acb35003` (branch `main`)
- generated: 2026-06-08 by the Model-Saving + Final-Report actor, from ACCEPTED artefacts only.

> Reference-gate note: the skill's `references/` directory
> (`/home/mattc/.claude/skills/biomass-ml-agent-evolve/references/`) is **absent** — every upstream
> actor flagged this. This report proceeds on the conventions in the ACCEPTED `configs/experiment_design.md`
> §7 decision rule and the ACCEPTED evaluation/error-analysis artefacts, and records the absence here.

## 1. Context

There is **no Irish ground truth**. The pre-trained `embdstx` LightGBM head (73 trees, 67 features,
target CO₂ standing stock tCO₂/acre, trained on 4636 US ANEW plots) was applied **zero-shot** to 141
dissolved Irish Dasos Locations (Sitka-dominant maritime-temperate plantation). It is judged in a
**model-vs-model** comparison against the Deep Biomass (DB) product — a known under-estimator used
only as a **directional lower bound**, never as truth. DB density (Mg/ha) → tCO₂/acre via **×0.6977**,
both sides AGB-only. The pre-registered hypotheses (`experiment_design.md §1`): **H1** our pred ≥ DB
(directional dominance); **H2** the gap widens in the high-biomass band (saturation resistance);
**H3** both pred and DB rank-track stand structure (covariate sanity).

## 2. Method

- **Crosswalk + dissolve**: 141/141 Locations resolve; 1,053 sub-compartments → 141 MultiPolygons;
  area-weighted covariates; survey_year = area-weighted mode clamped to [2017, 2025] (17 pre-2017
  fallbacks lifted to 2017). DB reference per Location.
- **ENCODING GATE (HARD precondition) — PASS.** Per-band affine GEE AlphaEarth (A00..A63) → training
  int8 codec (`emb_j = a_j·A{j} + c_j`), fit on 287 Bayfield plots, validated on 122 held-out:
  **mean per-plot corr 0.986** (min 0.951), **post-affine per-band slope median 1.006** (98% of bands
  in [0.8,1.2]), intercept median 0.085·band-σ — all three conditions hold
  (`preprocessing_pipeline/encoding_gate.json`). Production affine then refit on all 409 plots and
  applied to Ireland. The gate validates **encoding fidelity at Bayfield (in-sample), NOT Irish
  transfer accuracy.**
- **Feature assembly**: 141 × 67 (64 affine-mapped AEF + 3 Hansen survey-relative disturbance
  features), 0 NaNs, exact `inference_features_embdstx.json` order.
- **Inference**: deterministic LightGBM predict (seed 42 for the gate split / eval).
- **Evaluation**: threshold-free divergence characterisation vs DB (no accuracy metrics) +
  Mahalanobis / domain-classifier OOD + saturation checks.

## 3. Results

(All numbers from ACCEPTED `final/evaluation_matrix.yaml`.)

**Predictions (141 Locations):** min 26.7 / mean 91.6 / median 100.3 / max 138.4 tCO₂/acre.
Portfolio pred 91.6 vs DB 27.35 → **ratio 3.35×**. 0% above training max (520.95).

- **H1 directional dominance — SUPPORTED.** Pred ≥ DB for **98.6%** of Locations (2020–24 mean;
  95.7% on 2024-only). Mean signed Δ +64.3 tCO₂/acre. Only 2 negative-Δ Locations, both very young /
  near-zero-structure stands where DB's noisy floor exceeds a correctly-low prediction.
  Fig: `evaluation/figures/scatter_pred_vs_db.png`, `evaluation/figures/delta_histogram.png`.

- **H2 saturation-resistance (DB-quintile) — NOT_SUPPORTED.** Per-quintile signed bias is
  flat-to-declining (Q1→Q5: 63.9 / 74.6 / 64.2 / 61.9 / 56.9; Q5 lowest), not widening.
  **Crucial nuance (error_analysis §2): this is a comparison artefact, NOT the head saturating.**
  Quintiles are cut on DB magnitude, so DB's own spread is mechanically maximised across them
  (DB Q1→Q5 +30.8 vs pred +23.8); DB magnitude is also a poor biomass proxy (DB vs age ρ=0.11 ns).
  Within the top quintile the head still rank-tracks age (ρ=0.57) and reaches 138.4 (only 26.6% of
  the training range) — **no plateau.** H2 as designed tests a quantity the truth-free data cannot
  expose; the co-saturation failure mode is ruled out.
  Fig: `evaluation/figures/quintile_signed_bias.png`,
  `error_analysis/figures/topquintile_plateau_check.png`.

- **H3 covariate rank-tracking — MOSTLY_SUPPORTED.** Pred vs age **ρ=0.553**, vs Hdom **ρ=0.556**
  (both p<1e-6), far exceeding DB (ρ=0.11 / 0.20). Δ rises monotonically with age (0-10→30-40:
  32.4→81.2) and height. YC is null for both (ρ_pred −0.077 ns). This is the strongest evidence the
  transfer is signal, not noise. Fig: `evaluation/figures/pred_db_vs_age.png`,
  `evaluation/figures/pred_vs_hdom.png`.

- **OOD — SEVERE_DOMAIN_SHIFT.** Irish Mahalanobis (64-emb, US-train covariance) min **27.8** =
  **1.9× the training 99th-pct radius (14.79)**; **100%** of Locations beyond it. Domain-classifier
  AUC **0.999998**. Reassuringly, divergence does NOT grow with OOD distance (|Δ| vs Mahalanobis
  ρ=−0.21), so the large gap is a coherent **level offset** (DB under-reading + genuine structure),
  not OOD noise. Top 8 of 64 embedding dims carry 68% of the centroid shift (emb_26/50/23/55…) — the
  maritime-temperate-plantation signature with no US analog.
  Fig: `error_analysis/figures/absdelta_vs_mahalanobis.png`,
  `error_analysis/figures/perdim_ood_contribution.png`.

- **Saturation context:** 79.4% of preds exceed the 80 tCO₂/acre optical ceiling; 0% exceed training
  max. Against the research envelope (≈110 tCO₂/acre for a 19-yr Irish Sitka stand) the top values are
  plausible-to-mildly-low, not collapsed.

## 4. Decision — RETRAIN_WARRANTED

Per `experiment_design.md §7`, credible transfer requires **all of H1+H2+H3 with non-catastrophic
OOD**. The gate passed and H1 holds, but **H2 fails** (gap does not widen; though this is a comparison
artefact, not a head failure) and **OOD is catastrophic** (AUC ≈ 1.0, 100% beyond the 99th-pct radius).
This matches counter-hypothesis C2 (severe domain shift) partly offset by genuine structural tracking
(H3). Per design rule (b) — gate-OK but H2 clearly fails — the outcome is **escalate the conditional
analog-subset (maritime + high-biomass conifer) retrain**, NOT halt (gate passed, structure tracked,
predictions coherent) and NOT fully credible (absolute levels are deep extrapolation).

## 5. Limitations

- No ground truth — all divergence is vs an under-estimator; cannot positively prove high-end accuracy.
- Absolute tCO₂/acre levels are deep extrapolation (100% OOD); trust rankings/structure only.
- The head is optical-AEF-only with **no vertical-structure lever** (no CHM/SAR) — it structurally
  cannot prove saturation resistance.
- Encoding gate proves codec fidelity at Bayfield (in-sample), not Irish accuracy; per-band affine is
  not pixel-perfect (reconstruction RMSE ~31% of band-σ).
- No post-hoc calibration is valid (calibrating to DB would import DB's bias).

## 6. Next steps (from ACCEPTED `reports/improvement_plan.md`)

1. **Analog-subset retrain (highest priority).** Rebuild the head on the maritime-temperate +
   high-biomass-conifer subset of the ANEW pool (New England-Acadian + Pacific-coastal/Cascades/
   Alaskan conifer) to pull the Irish manifold inside training support; re-run dual evaluation
   (US LOPO + zero-shot Ireland). Targets the OOD dims emb_26/50/23/55.
2. **Add a non-optical structural lever** (L-band SAR / Sentinel-1) so a future evaluation can
   actually test saturation resistance.
3. **Re-cut H2 on age/Hdom** (independent structure), not DB magnitude — cheap, immediate; on these
   axes the head keeps rising while DB stays flat.
4. **Do NOT apply post-hoc calibration** to absolute levels until an in-range US-holdout-validated
   analog-subset head exists. Fine-tuning on Irish "labels" and wall-to-wall inference are not
   warranted now.

## 7. Artefacts

- Head: `final/model/inference_model_embdstx.txt` + `inference_features_embdstx.json` + `loader_notes.md`
- Pipeline: `final/preprocessing_pipeline/` (affines, schema, gate, data version, README)
- Evaluation matrix: `final/evaluation_matrix.yaml`
- Cards: `final/model_card.md`, `final/data_card.md`
- Traceability + QA self-check: `final/run_summary.md`
- Environment: `final/environment.lock`; git: `final/git_snapshot.txt`
- Figures: `evaluation/figures/*.png`, `error_analysis/figures/*.png`
