# Experiment Design — Ireland AGB Zero-Shot Transfer + Model-vs-Model Comparison

- experiment_id: agb_ireland_biomass_regression_20260608
- stage: experiment_design
- actor: Experimental Design Actor
- generated: 2026-06-08
- mode: zero-shot transfer + model-vs-model comparison (NO ground truth, NO from-scratch training in the first pass)
- upstream status: research ACCEPTED (attempt 2); data_profile ACCEPTED (attempt 1)

> **NOTE — missing acceptance-gate reference.** The prompt's first required input,
> `/home/mattc/.claude/skills/biomass-ml-agent-evolve/references/experimental_design.md`,
> **does not exist** (the whole `.../biomass-ml-agent-evolve/references/` directory is absent —
> the data_profile actor flagged the same for its own reference file). I therefore follow the
> conventions evident in the IMPLEMENTATION_PLAN, the approved plan `plans/ireland-agb-test-v1.md`,
> and the ACCEPTED upstream artefacts, and flag the absence here for the Critic.

---

## 0. TL;DR

We apply the already-trained head `models/inference_model_embdstx.txt` (67 features = 64 AEF
embeddings + 3 survey-relative Hansen disturbance-timing features; target tCO₂/acre; training
range [0, 520.95]) zero-shot to 141 Irish Sitka-dominated forestry Locations and characterise
how its per-Location predictions diverge from Deep Biomass (DB, a known under-estimator used as a
**directional lower bound, not ground truth**). There is **no ground truth and no accuracy
threshold**. Comparison is performed entirely in **tCO₂/acre density space** (DB Mg/ha × 0.6977).
A hard **encoding-consistency gate** (per-band affine GEE→training codec, re-gated at the Bayfield
overlap) is a **go/no-go precondition** before any prediction is trusted. Success is *directional
and structural*: our prediction ≥ DB with the gap growing in the high-biomass band, and the
prediction rank-tracking the independent Dasos structural covariates.

---

## 1. Objective & hypotheses (framed for NO ground truth)

**Objective.** Quantify and explain the signed divergence between our pre-trained embdstx head and
the Deep Biomass reference across 141 Irish Locations, and decide whether the zero-shot transfer is
credible enough to report, or whether the conditional analog-subset retrain (plan Section F) is
warranted. We assert **no accuracy claim** because neither model is anchored to truth; DB is treated
as a directional lower bound per the user's standing characterisation that it under-estimates.

**Primary hypotheses (directional, no truth required).**

- **H1 — Directional dominance.** Per Location, `our_pred ≥ DB_tco2`. Formally, the signed
  per-Location delta `Δ = our_pred − DB_tco2` is predominantly positive.
- **H2 — Saturation resistance / widening gap.** Δ grows monotonically (or near-monotonically)
  across DB-magnitude quintiles Q1→Q5; the absolute and signed gap is largest in the high-biomass
  band. Mechanism: DB compounds optical saturation + range-compression (research §3–4); our head,
  carrying a disturbance-timing age proxy, is expected to under-read *less* severely at the top end.

**Secondary hypotheses (structural consistency — the substitute for GT).**

- **H3 — Covariate rank-tracking.** `our_pred` rank-correlates positively and monotonically with the
  independent Dasos structural covariates: stand age (survey-year minus `PlantingYe`), `Hdom`,
  `YC`, and Sitka vs broadleaf `MainSp`. (Older / taller / higher-YC / Sitka → higher predicted
  biomass.) These covariates are independent of *both* models, so this is a genuine external sanity
  check, not circular (confirmed non-circular by data_profile leakage analysis).

**Counter-hypotheses we must be honest about (failure modes to detect, not assume away).**

- **C1 — Co-saturation.** Our head *also* saturates (cross-region AEF transfer is AEF's documented
  weak point, research §2; optical ceiling ~80 tCO₂/acre / ~115 Mg/ha, research §3), so `our_pred`
  collapses toward DB and the portfolio mean lands near DB's ~31 tCO₂/acre (44.6 Mg/ha) rather than
  the literature-plausible mid/late-rotation band (~70–175 tCO₂/acre). This would *not* support H2.
- **C2 — Domain-shift garbage.** Predictions are dominated by out-of-distribution AEF inputs (high
  Mahalanobis fraction / high domain-classifier AUC) and neither track covariates (H3 fails) nor
  dominate DB sensibly — i.e. transfer is not credible at all.

These hypotheses are evaluated by *pattern*, never by a numeric error threshold.

---

## 2. Units & conversion (fixed, deterministic)

- **Fixed factor:** `tCO₂/acre = (AGB Mg/ha) × 0.6977`, where
  `0.6977 = 0.47 (IPCC carbon fraction) × 3.667 (CO₂/C = 44/12) × 0.4047 (ha→acre)`. Inverse:
  `AGB Mg/ha = tCO₂/acre ÷ 0.6977`. Sanity check: 200 Mg/ha ≈ 140 tCO₂/acre.
- **Comparison space:** all primary deltas and metrics are computed in **tCO₂/acre density space**
  (`our_pred` directly vs `DB_tco2 = DB_Mgha × 0.6977`). This is the most robust comparison because
  it removes area as a confounder.
- **Pool boundary:** both DB and the ANEW training CO₂ column are **AGB-only** (user-confirmed,
  plan §D), so the 0.47 carbon fraction applies to the correct pool and no below-ground term is
  added on either side.
- **Carbon-fraction sensitivity:** the only modelling assumption is the 0.47 fraction. Report a
  ±sensitivity using 0.50 (DB × 0.6977 → DB × 0.6604, ≈ −5.3%); this changes neither the rank of
  Locations nor the sign of Δ, so it cannot overturn H1/H2/H3 — report it as a footnote band only.
- **Portfolio totals (secondary):** for portfolio-level figures only, total tCO₂ per Location =
  `density_tco2_acre × Area_Ha ÷ 0.4047`. Densities remain the primary comparison surface.

---

## 3. Spatial unit & support

- **Unit:** 141 Deep-Biomass **Locations**, formed by dissolving the 1,053 Dasos sub-compartments
  up to `SiteName` (data_profile §B; crosswalk = direct match + `_`→`/` substitution, **all 141
  resolve**; the plan's suffix-split rule is **wrong** and must not be used). Dissolved area agrees
  with CSV `Area_Ha` to within ±0.32% (mean −0.02%), confirming the same aggregation unit.
- **AEF support:** embeddings are **area-averaged over the dissolved Location polygons** via GEE
  `reduceRegions(reducer = ee.Reducer.mean())` (data_profile §D; plan §A). This matches the
  training plot-mean support (the training parquet emb are buffer/area means of the int8 tiles —
  data_profile §C), which is *why* the encoding gate is expected to tighten under polygon-mean
  extraction (point-vs-area mismatch is the residual ~4% in the Bayfield corr).
- **Support-sensitivity check (secondary).** Also run the **plot-scale-then-aggregate** alternative
  (extract at sub-compartment level, predict per sub-cpt, area-weight to Location) and report the
  delta vs the dissolved-Location prediction. This is the USA point-vs-area support test, adapted
  via `scripts/compare_gaussian_vs_pointextract.py`. Report both; flag any Location where the two
  paths disagree by a large fraction.
- **Reprojection:** gpkg EPSG:4326 → Irish ITM **EPSG:2157** for metric area and GEE extraction
  geometry (data_profile §B). The crosswalk in `data_profile/crosswalk_location_to_sitename.csv` is
  the authoritative Location↔SiteName map; reference it, do not re-derive.

---

## 4. Temporal alignment

- **Our prediction (per Location).** AEF year = the **area-dominant `SurveyDate` year** per
  Location (data_profile §B: 2024 = 289 / 2023 = 280 sub-cpts dominate), **clamped to AlphaEarth
  coverage [2017, 2025]** (2017–2025 available on GEE; ~98 pre-2017 sub-cpts fall back to 2017 and
  are **flagged** in the OOD/temporal section). The 3 `dstx_*` co-features use the **same survey
  year** for survey-relative Hansen timing (`build_dist_image(survey_year)`).
- **DB reference (primary).** `DB_ref_Mgha` = per-Location **mean of the 2020–2024** annual
  densities (cell ÷ `Area_Ha`), then × 0.6977 → tCO₂/acre. The 2020–2024 mean damps the large
  single-year DB noise (data_profile §A: median max/min ratio 6.3×, max 239×; spurious cells such
  as Ahalahana 2015 = 2 t).
- **DB reference (sensitivity).** Re-run the full comparison against the **2024-only** DB column and
  report whether H1/H2 conclusions are stable across the two references. Divergence between the
  windows is itself reported as a DB-noise diagnostic, not as a fault of our model.
- **Assumption / confound.** Our prediction is at survey-year; the DB reference is a 2020–2024 mean.
  Where a Location had a disturbance (clearfell/restock) inside that window, the two are not
  contemporaneous — handled by the `dstx_*` survey-relative timing on our side and flagged where
  the dstx loss-fraction is high; it is a confound to surface, not a data error.

---

## 5. Encoding-consistency gate — HARD GO/NO-GO PRECONDITION

This gate is the **first executed step** and a **blocking precondition**: no downstream number is
trusted unless it passes. It is owned by the preprocessing stage; this design fixes its contract.

- **Why.** The repo reads training AEF from LOCAL int8 tiles raw (int8→float, **no dequantize**;
  `infer_bayfield.py:119–150`). Ireland has no local tiles → embeddings come from GEE
  `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (64 bands `A00..A63`, dequantised `double`, range ≈
  [−0.39, +0.38]). The training parquet is on the raw-int8-averaged scale (range ≈ [−86, +86]).
  data_profile §D proved corr(GEE, parquet) = mean 0.957 but the relationship is **per-band affine,
  not a single scalar** (per-band slope 220–623); a global multiplier mis-scales bands by up to ~3×
  and would corrupt the LightGBM absolute-threshold splits.
- **Transform.** Per band j, fit `emb_j(parquet) = a_j · A{j}(GEE) + b_j` on the **full valid
  Bayfield training-plot set (≥ ~50 plots)**, then apply `emb_irish_j = a_j · A{j}_GEE + b_j` to the
  Ireland polygon-mean extraction.
- **Re-gate (the actual acceptance test).** Re-run the correctness gate
  (`correctness_gate`, `infer_bayfield.py:159–188`) on the **transformed** values at the Bayfield
  overlap and require: **(i) mean corr > 0.8** against parquet `emb_*`, **AND (ii) per-band slope
  ≈ 1 after the affine, with bounded intercept.** Both conditions must hold.
- **Scope caveat (must be stated in the report).** The gate validates **GEE→codec encoding
  consistency at Bayfield**, NOT that the US-fitted slopes are *Ireland-correct* — Ireland has no
  overlap plots, and Bayfield is **in-sample** to the head (data_profile leakage note). Passing the
  gate proves feature-encoding fidelity, not transfer accuracy; this is consistent with the
  no-accuracy-claim design.
- **Decision.** Gate FAIL ⇒ **HALT** the experiment; escalate (no prediction is reported). Gate
  PASS ⇒ proceed to inference + the evaluation matrix below.

---

## 6. Evaluation matrix

**Rows (models compared):** a single row — the **embdstx head** (`inference_model_embdstx.txt`),
predicting tCO₂/acre per Location. The "baseline" row is the **DB reference itself** (the lower
bound) plus a structural-covariate sanity predictor (a simple monotone rank check against age/Hdom/
YC, not a trained model). No new model is trained in the first pass.

**Columns (metrics), all in tCO₂/acre density space unless noted:**

| metric | definition | source |
|---|---|---|
| signed per-Location delta distribution | `Δ = our_pred − DB_tco2` per Location; histogram + scatter our-vs-DB with 1:1 line; fraction Δ>0 (H1) | NEW |
| per-quintile signed bias Q1..Q5 | quintiles by **DB magnitude** (not by truth — there is none); mean signed Δ per quintile; expect Δ increasing Q1→Q5 (H2) | ADAPT `compute_biomass_metrics.py` quintile block |
| predicted-range discrimination (PRD) | `(pred_Q5_mean − pred_Q1_mean)/(DB_Q5_mean − DB_Q1_mean)` over the Irish range (quintiles by DB) | ADAPT same PRD formula |
| covariate cuts | Δ and `our_pred` cut by age (survey−`PlantingYe`), `Hdom`, `YC`, `MainSp` (Sitka vs broadleaf), `Thinned`; Spearman rank ρ of `our_pred` vs each (H3) | NEW |
| OOD — Mahalanobis | fraction of Irish Locations beyond the **99th-pct training radius** (Mahalanobis in 64-emb space, training covariance) | NEW |
| OOD — domain classifier | USA-train vs Ireland binary classifier on the 64 emb; report AUC (≈0.5 ⇒ indistinguishable; →1.0 ⇒ severe domain shift) | NEW |
| saturation fraction | fraction of Locations with `our_pred` > **80 tCO₂/acre** (the empirical optical ceiling, research §3) and > **520.95** (training max → extrapolation) | NEW |
| portfolio summary (secondary) | portfolio-mean `our_pred` vs DB mean, and vs literature band (~70–175 tCO₂/acre); total tCO₂ | NEW |

**Reuse vs new — exact functions in `compute_biomass_metrics.py`:**

- **Reused (verbatim formula, re-pointed inputs):**
  - the **per-quintile signed bias** block (`pd.qcut(...,5,...)` → `groupby(...)["residual"].mean()`,
    lines 48–52) — but quintiles are formed on **DB magnitude** and "residual" is the signed delta
    `our_pred − DB_tco2` (no `target`/truth column exists).
  - the **PRD** computation (lines 53–55), with `pred` = `our_pred` and the denominator built from
    DB-magnitude quintile means (substituting DB for the absent true target).
  - the per-decile **calibration** residual idea (lines 95–99) is reused descriptively (residual =
    Δ by predicted decile) to show where the gap concentrates.
- **NOT reused / dropped:** `_agg` (lines 30–37), `per_ecoregion_r2` (lines 57–72), `error_by_region`
  / `prd_by_region` (lines 74–92), `external_holdout_r2` (line 111) — **all require ground-truth
  `target` and US ECO_NAME/region joins** that do not exist for Ireland. `r2`/`rmse`/`mae`/`bias`
  vs truth are **explicitly excluded** (no truth).
- **New (must be written; no equivalent exists):** the signed-delta distribution, the covariate-cut
  module (age/Hdom/YC/MainSp/Thinned + Spearman ρ), the Mahalanobis OOD fraction, the USA-vs-Ireland
  domain-classifier AUC, and the saturation-fraction counts. These were the v0 OOD diagnostics; carry
  them as new code keyed on the 64 emb and the Dasos covariates.

The realised matrix is written by the Evaluation Actor to `evaluation/evaluation_matrix.yaml`; this
section is its specification.

---

## 7. Decision rule (no ground truth → pattern-based, NO thresholds)

Evaluate by the **joint pattern**, never a single number.

**(a) Transfer is CREDIBLE (report the zero-shot result as a characterised bias study) when ALL of:**
1. Encoding gate **PASSES** (§5).
2. **H1** holds — Δ predominantly positive (`our_pred ≥ DB` for most Locations).
3. **H2** holds — signed Δ increases across DB quintiles Q1→Q5 (gap widens at high biomass) and PRD
   is not collapsed (our model spreads its predictions, not flattened toward a constant).
4. **H3** holds — `our_pred` rank-tracks the structural covariates (positive monotone Spearman ρ vs
   age / `Hdom` / `YC`; Sitka ≳ broadleaf), AND OOD is not catastrophic (domain-classifier AUC and
   Mahalanobis fraction are reported but the predictions still behave sensibly).

**(b) The conditional analog-subset retrain IS warranted (escalate to Improvement Planner, plan
§F — maritime+conifer analog subset) when:**
- The gate passes BUT the prediction pattern is wrong — i.e. **C1** (our model co-saturates:
  Δ does not grow with DB magnitude and the portfolio mean collapses toward DB's ~31 tCO₂/acre,
  far below the ~70–175 tCO₂/acre literature band) **OR** **C2** (predictions neither dominate DB
  nor track covariates while OOD diagnostics show severe domain shift). In short: gate-OK but H2 or
  H3 clearly fails → zero-shot is not trustworthy → recommend the analog-subset retrain.

**(c) HALT (no result):** encoding gate FAILS (§5).

No accuracy thresholds are defined or implied anywhere; "monotone", "predominantly", "collapse",
"sensible" are judged from the distributions and reported as such.

---

## 8. Risks & confounds

- **AEF cross-region transfer (CENTRAL).** Head trained on US plots applied to maritime-temperate
  Irish plantation — exactly the cross-region case AEF handles worst (research §2, [S12][S13]).
  Mitigation: OOD diagnostics surface its severity; predictions are framed as bias-characterisation,
  not calibrated estimates.
- **Optical saturation.** Most mid/late-rotation Irish Sitka sits above the ~150 Mg/ha onset
  (research §1, §3); our head may itself under-read the true high tail, just (hypothesised) less
  than DB. The saturation-fraction metric and the disturbance-timing co-features address this; we do
  NOT claim our high-end values are accurate.
- **DB noise.** Large single-year DB volatility (data_profile §A) → the 2020–2024-mean reference
  with a 2024-only sensitivity check is the mitigation.
- **Covariate missingness.** `MainSp` 28%, `PlantingYe` 27%, `Thinned` 34%, `BA_Conifer` 73% missing
  (data_profile §B). Mitigation: area-weighted aggregation to Location partially fills gaps; report
  each covariate cut with its effective n and a "missing" bucket; **drop `BA_Conifer` as a primary
  cut**. Covariate cuts with thin n are reported as suggestive, not conclusive.
- **Bayfield in-sample caveat for the gate.** The gate's Bayfield plots are in the head's training
  set → the gate proves **encoding consistency only**, not transfer accuracy (data_profile leakage
  note; §5 scope caveat). State this explicitly in the report.
- **Temporal non-contemporaneity** between survey-year prediction and the 2020–2024 DB window
  (§4) — flagged via dstx, not corrected.
- **Pre-2017 survey tail** (~98 sub-cpts) uses 2017 AEF fallback — flagged in OOD/temporal.

---

## 9. Reproducibility

- **Seeds.** Any stochastic step (domain-classifier train/test split, Mahalanobis covariance
  subsampling if used) uses a fixed seed **`SEED = 42`**; record it in the evaluation artefact.
  Inference itself (LightGBM `.predict`) is deterministic.
- **Exact inputs (provenance, all ACCEPTED upstream):**
  - DB CSV: `…/dasos-ireland/deepbiomass-model-outputs/Deep Biomass - Aggregated Data & Portfolio Summary.csv`
  - geometry: `…/boundary-files/dasos_fgl_2025ye.gpkg` (layer `fgl_2025ye_`, EPSG:4326 → 2157)
  - crosswalk: `data_profile/crosswalk_location_to_sitename.csv` (authoritative; all 141 resolve)
  - training parquet (encoding source of truth + Mahalanobis covariance + domain-classifier USA
    class): `/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet`
  - GEE asset: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (64 bands `A00..A63`, double, 2017–2025)
  - head: `models/inference_model_embdstx.txt` + `models/inference_features_embdstx.json`
    (67 feats, target tCO₂/acre, range [0, 520.95], 73 estimators; trained on all 4636 plots,
    Bayfield in-sample) — verified present and matching the 64 emb + 3 dstx feature list.
- **Scripts to reuse / adapt (verified to exist in `scripts/`):**
  - `infer_bayfield.py` — `read_embeddings` (int8→float, **no dequantize**, lines 119–150) and
    `correctness_gate` (lines 159–188, `assert mean corr > 0.8`); **parameterise the hardcoded
    `UTM = "EPSG:32615"` and `RES = 30`** (lines 60–61) for Ireland (EPSG:2157).
  - `infer_bayfield_embdstx.py` — embdstx head load + predict path (`main`, line 62) to adapt to
    the Location-level Irish features.
  - `extract_iter2_features.py` — `build_dist_image(survey_year)` (line 153) for survey-relative
    Hansen `dstx_*` co-features, and the `reduceRegions` mean extractor (lines 80–112). Add the
    **polygon** `reduceRegions` path (currently point/centroid-oriented) per plan §A.
  - `export_bayfield_cofeatures.py` — co-feature export; parameterise CRS.
  - `compare_gaussian_vs_pointextract.py` (`main`, line 144) — the point-vs-area support-sensitivity
    check (§3 secondary).
  - `evaluation/compute_biomass_metrics.py` — reuse the quintile-bias and PRD formulas per §6.
- **Environment.** Python 3.13 via `uv run` (never bare python/pip); pandas, geopandas+pyogrio
  (no fiona), rasterio, lightgbm, scikit-learn, earthengine-api (bare `ee.Initialize()` verified OK).
  Plots saved with `plt.savefig` (never `plt.show`).

---

## 10. Assumptions (explicit)

1. **No ground truth; no accuracy threshold.** DB is a directional lower bound, not truth; all
   conclusions are directional/structural (carried from plan + IMPLEMENTATION_PLAN).
2. Unit conversion is the fixed `×0.6977` (0.47 × 3.667 × 0.4047); both sides AGB-only; 0.50-fraction
   sensitivity is a reported band only (does not change rank/sign).
3. Spatial unit = 141 dissolved `SiteName` Locations; AEF support = polygon-mean via
   `reduceRegions(mean)`; crosswalk = direct + `_`→`/` (all 141 resolve, data_profile).
4. Temporal: our pred at area-dominant survey year clamped to [2017, 2025]; DB ref = 2020–2024 mean
   (2024-only sensitivity); pre-2017 tail falls back to 2017 AEF and is flagged.
5. Encoding gate (per-band affine GEE→codec, re-gated at Bayfield: corr>0.8 AND post-affine slope≈1)
   is a hard go/no-go; it validates encoding consistency, NOT Ireland transfer accuracy; Bayfield is
   in-sample to the head.
6. Quintiles for bias/PRD are formed on **DB magnitude** (substituting for the absent true target).
7. Saturation reference points: 80 tCO₂/acre (empirical optical ceiling, research §3) and 520.95
   (training max). Literature plausibility band for the portfolio mean: ~70–175 tCO₂/acre (research).
8. Covariate cuts are reported with effective n and a missing bucket; `BA_Conifer` dropped as a
   primary cut; thin-n cuts are suggestive only.

## Reproducibility footer

- inputs: IMPLEMENTATION_PLAN.md; plans/ireland-agb-test-v1.md; research/deep_research.md (ACCEPTED);
  data_profile/database_profile.md (ACCEPTED) + crosswalk_location_to_sitename.csv;
  models/inference_features_embdstx.json (read); evaluation/compute_biomass_metrics.py (read);
  scripts/{infer_bayfield,infer_bayfield_embdstx,extract_iter2_features,export_bayfield_cofeatures,
  compare_gaussian_vs_pointextract}.py (line refs verified by grep).
- method: design synthesis from ACCEPTED upstream artefacts + source-code inspection. No data run.
- seed: 42 (downstream stochastic steps). libraries: n/a (design doc).
- conducted by: Experimental Design Actor. timestamp_utc: 2026-06-08.
