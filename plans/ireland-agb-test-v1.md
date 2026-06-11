# Plan: Reworking the Ireland AGB Evaluation — Model-vs-Model Comparison (v1)

## Context

The original plan (`plans/ireland-agb-test-v0.md`) assumed we would receive **stand-level ground
truth** from an Irish provider and run a GT-anchored go/no-go (R²/bias vs truth → accept / calibrate /
fine-tune). That assumption is now wrong and the plan must be reworked:

- **There is no ground truth.** What we have is the **output of another model** — "Deep Biomass" — that
  the user says **systematically under-estimates**. It is therefore a *directional lower bound*, not a
  reference we can compute true accuracy against.
- **The comparison goal is full bias characterisation** (user decision): report the complete per-stand
  delta distribution, per-quintile signed bias, and saturation behaviour of *our* model vs Deep Biomass,
  without asserting which is correct.
- **Geometry now exists** (it was the original blocker). The user supplied
  `dasos_fgl_2025ye.gpkg` — and it is far richer than anticipated, carrying per-compartment forestry
  attributes (species, planting year, height, yield class, survey date). These become direct evaluation
  covariates, replacing the old "GT TBC" open items.
- **Effort (user decision): run the existing head first.** Use the already-trained
  `inference_model_embdstx.txt` (embeddings + disturbance-timing) on Ireland and compare to Deep Biomass.
  Only build the analog-subset head if this first pass looks clearly wrong. This collapses most of the
  old plan's Section A/D/F modelling machinery.

Intended outcome: a reproducible Irish inference run with the existing head, aggregated to the same 141
Deep-Biomass "Locations", and a bias-characterisation report (delta distribution, quintile bias,
saturation, cut by the Dasos covariates) that tells us *how* our model differs from the under-estimating
reference and whether the difference is in the expected direction.

## Data inventory (verified)

**Deep Biomass CSV** — `…/dasos-ireland/deepbiomass-model-outputs/Deep Biomass - Aggregated Data &
Portfolio Summary.csv`
- 141 forestry **Locations** + 2 footer rows. Columns: `Location No`, `Location Name`, `Area Ha`, then
  annual values **2013–2024**.
- Deep Biomass's **native output is AGB density in Mg/ha** (= t biomass/ha). The CSV cells are the
  **aggregated total tonnes per Location** (Mg/ha × `Area Ha`) — confirmed by the footer:
  `Total AGB, ton` (2024 = 150,313 t) and `Total AGB, ton/ha` (2024 = 44.6 Mg/ha) over 3,367.63 ha.
  So recover **per-Location Mg/ha = cell ÷ `Area Ha`** (e.g. Aghaderrard West 2013 = 138 t / 10.21 ha =
  13.5 Mg/ha).
- Series is **noisy** (e.g. Ahalahana 2015 = 2 t total — spurious) → motivates the stable-window mean.

**Geometry + covariates** — `…/boundary-files/dasos_fgl_2025ye.gpkg`
- 1,053 **sub-compartment** MultiPolygons, **EPSG:4326**, total area **3,367.06 ha** (matches the CSV's
  3,367.63 ha → same portfolio).
- Join key: CSV `Location Name` → gpkg **`SiteName`** (141 unique). **124/141 match directly**; the
  remainder use a `Group_SiteName` pattern (e.g. `Upper Shannon North_Garvesk`) → split on `_` and match
  the suffix to `SiteName`, with the prefix as a management-group label. Build the crosswalk explicitly
  and assert all 141 resolve before extraction.
- Rich attributes per compartment: `MainSp`/`SecSp` (**627/≈750 = Sitka spruce `SS`** → confirms the
  saturation-prone plantation domain), `PlantingYe` (1986–2026 → **stand age**), `SurveyDate` (2023–2024
  → **AEF temporal alignment**), `YC` (yield class), `Hmean`/`Hdom` (0–28 m → vertical structure),
  `BA_Conifer`, `MgtRegime`/`Thinned`, `GrossArea`/`ProdArea`.

**Model artefact (use this)** — `models/inference_model_embdstx.txt` +
`models/inference_features_embdstx.json`: **67 features** = `emb_00..63` + `dstx_pre_ysd`,
`dstx_pre_loss_5yr`, `dstx_loss_frac_buf`. Target = **CO₂ standing stock, tCO₂/acre**, training range
**[0, 520.95]**. (The 73-feature `inference_model.txt` also needs CHM+topo, which the chosen Ireland
feature set drops — so embdstx is the right head.)

## Feature set (unchanged from v0)

Embeddings + disturbance-timing only — matches the embdstx head exactly: AEF (64-dim int8, annual,
survey-year-aligned) + survey-relative Hansen harvest-timing (`build_dist_image`). CHM/topo/GEDI/climate
dropped as in v0. Saturation is largely unmitigated by design; disturbance-timing + (now) the Dasos age
covariate are the levers for reasoning about it.

## Plan of work

### A. Geometry crosswalk + spatial support
- Build the `Location Name → SiteName` crosswalk (direct + `_`-split fallback); assert all 141 resolve;
  log any unmatched. Dissolve the 1,053 sub-compartments **up to the 141 SiteName Locations** (matching
  Deep Biomass's aggregation unit), carrying area-weighted covariates (age, Hdom, YC) and the dominant
  species per Location.
- Extract embeddings by **area-averaging over the dissolved polygons** (GEE `reduceRegions`,
  `ee.Reducer.mean()`), reprojecting gpkg EPSG:4326 → Irish ITM (**EPSG:2157**). Reuse the int8-no-dequant
  path. Run the **plot-scale-then-aggregate** alternative too (extract at sub-compartment level, predict,
  area-weight to Location) — the original Section C support-sensitivity test on USA via
  `scripts/compare_gaussian_vs_pointextract.py` still applies; report both.

### B. Encoding-consistency gate (run FIRST — unchanged from v0, B4)
- Irish embeddings MUST use the identical **raw int8→float, no `dequantize`** path as
  `infer_bayfield.py:140-186`, and pass the corr>0.8 correctness gate against the training parquet
  encoding. Every downstream number is meaningless otherwise. This is the literal first step.

### C. Run the existing head + temporal alignment
- Parameterise the hardcoded `EPSG:32615`/RES in `infer_bayfield.py` + `export_bayfield_cofeatures.py`
  for Ireland (EPSG:2157). Build the 3 `dstx_*` co-features per Location with survey-relative timing
  (`build_dist_image`, survey year from `SurveyDate`).
- Load `inference_model_embdstx.txt`, predict **tCO₂/acre** per Location. AEF year = each Location's
  **survey year** (2023/2024) for the primary run.

### D. Unit reconciliation (now concrete — single fixed factor)
- Deep Biomass is **AGB Mg/ha**; our head outputs **tCO₂/acre**. Convert DB into our space with the
  fixed factor the user specified: **tCO₂/acre = Mg/ha × 0.6977**, where 0.6977 = 0.47 (IPCC carbon
  fraction) × 3.667 (CO₂/C = 44/12) × 0.4047 (ha→acre). Sanity check: 200 Mg/ha ≈ 140 tCO₂/acre.
- Do the delta in **tCO₂/acre density space** (our prediction directly vs DB×0.6977), the most robust
  comparison; also report total tCO₂ per Location (×`Area Ha`÷0.4047) for portfolio-level figures.
- The conversion is now a single deterministic factor, **not** an envelope (DB is already biomass — no
  volume→biomass / wood-density / BEF step). The only modelling assumption is the 0.47 carbon fraction;
  note its ±sensitivity but it does not change rank or sign.
- **Pool boundary confirmed (user): both are AGB-only.** Deep Biomass is above-ground biomass and the
  ANEW `CO2` column is above-ground only — the pools match, so the 0.47 carbon fraction is applied to the
  correct (above-ground) pool and no below-ground adjustment is needed on either side.

### E. Comparison & bias-characterisation report (the deliverable)
- **Temporal scope (user): mean over a stable recent window.** Reference = **mean of Deep Biomass
  2020–2024 per Location, in Mg/ha** (= mean of cell÷`Area Ha`), then ×0.6977 → tCO₂/acre; damps the
  year-to-year noise. Sensitivity check vs the single 2024 column. Compare against our survey-year
  prediction, all in **tCO₂/acre density space**.
- Metrics (reuse `evaluation/compute_biomass_metrics.py` verbatim where applicable), our-pred vs
  Deep-Biomass:
  - Full **per-Location delta distribution** (signed; histogram + scatter our-vs-DB with 1:1 line).
  - **Per-quintile signed bias (Q1..Q5)** of Deep-Biomass magnitude; expectation: our model reads
    **higher**, growing with biomass, if it resists the saturation/under-estimation DB exhibits.
  - **PRD** (predicted-range-discrimination) of our model across the Irish range.
  - Cut the delta by the **Dasos covariates**: stand age (from `PlantingYe`), `Hdom`, `YC`, `MainSp`
    (Sitka vs broadleaf), `Thinned` — this is the substitute for ground truth: does our model's
    *relative* pattern track the structural covariates sensibly (older/taller/higher-YC → more biomass)?
  - **OOD context:** fraction of Irish Locations beyond the training max and in the >80 tCO₂/acre
    saturation zone; Mahalanobis OOD fraction + USA-vs-Ireland domain-classifier AUC (v0 diagnostics).
- **No GT means no accuracy claim.** Frame conclusions as: (1) directional check (our ≥ DB, esp.
  high-biomass), (2) structural-covariate consistency, (3) saturation/OOD risk flags. Record whether a
  follow-up analog-subset retrain is warranted.

### F. Sequencing
- **Now:** crosswalk + dissolve · encoding gate (B) · parameterise CRS · extract embeddings+dstx per
  Location · run embdstx head · unit reconciliation · bias report. All unblocked — geometry and
  reference data are in hand.
- **Conditional follow-up (only if first pass looks wrong):** the v0 analog-subset (S1 maritime+conifer)
  retrain, now *validatable* against the Dasos structural covariates rather than absent GT.

## Critical files
- `scripts/infer_bayfield.py` — int8-no-dequant path + corr>0.8 gate; hardcoded CRS to parameterise.
- `scripts/export_bayfield_cofeatures.py` — co-feature export; hardcoded CRS.
- `scripts/extract_iter2_features.py` — `build_dist_image` (survey-relative Hansen) to reuse; add
  polygon `reduceRegions` support (currently point-centroid only).
- `scripts/compare_gaussian_vs_pointextract.py` — USA point-vs-area support sensitivity.
- `models/inference_model_embdstx.txt` + `models/inference_features_embdstx.json` — the head to run.
- `experiments/agb_usa_biomass_regression_20260529/evaluation/compute_biomass_metrics.py` — quintile
  bias / PRD, reuse verbatim.
- `…/boundary-files/dasos_fgl_2025ye.gpkg` — geometry + structural covariates.
- `…/deepbiomass-model-outputs/Deep Biomass - … .csv` — comparison reference.

## Verification
- Crosswalk resolves all 141 Locations (assert); dissolved area per Location matches CSV `Area Ha`
  within tolerance.
- Encoding gate (corr>0.8) passes for Irish embeddings before any analysis.
- Per-Location predictions produced for all 141; DB converted via ×0.6977; comparison in tCO₂/acre.
- Bias-characterisation report produced: delta distribution + Q1..Q5 signed bias + PRD + covariate cuts
  + OOD/saturation flags, under both the 2020–2024-mean and 2024-only references.

## Open items
- None — pool boundary resolved (both AGB-only, Section D). Ready to execute.
