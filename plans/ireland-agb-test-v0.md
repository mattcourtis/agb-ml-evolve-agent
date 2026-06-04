# Plan: Evaluating the AGB Embeddings Model on Irish Forestry

## Context

We have a biomass (AGB) regression model built in `experiments/agb_usa_biomass_regression_20260529/`:
a **LightGBM regression head** on top of **AlphaEarth Foundation (AEF) embeddings** (64-dim, 10 m,
annual, *global*), plus global co-features (ETH canopy height, SRTM topo, Hansen disturbance, Dynamic
World forest mask). For the Ireland transfer we deliberately narrow the feature set (see "Feature set"
below). The head was trained on **ANEW** ground truth — currently a 23-project / 4,636-plot
**eastern-US** subset (West Virginia hardwood, Upper Midwest, New England-Acadian). Target = standing
stock in **tCO2/acre**, plot-level (~14.7 m radius). Best CONUS LOPO R² ≈ 0.43. Its documented failure
mode is **optical saturation / dynamic-range compression above ~80 tCO2/acre** (Q5 under-prediction
bias −72 tCO2/acre).

We want to apply this to Irish forestry. The backbone (AEF + co-features) is global and transfers
unchanged; the **domain gap lives entirely in the regression head**. Irish forestry is dominated by
high-biomass **Sitka spruce plantation**, which sits squarely in the saturation zone and has no
even-aged-plantation analog in the eastern-US pool. Irish provider GT is **stand-level** (with stand
**polygons**), versus the head's plot-level point training — a spatial-support mismatch.

**Chosen approach (user):** assemble the *subset of US projects most analogous to Ireland*, train a
head on that subset, and evaluate it on **both** held-out USA and zero-shot Ireland, against the
full-CONUS model as a control. Decide afterwards whether fine-tuning on Irish GT is needed.

**Key discovery that reshapes Q1:** the source file
`/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg` actually holds
**52 projects / 12,837 plots**, including **Pacific-Northwest and Alaskan conifer projects**
(`HighCascades`, `RainierGateway`, `Kootznoowoo`, `Doyon`; Cascades/Pacific coastal ecoregions)
reaching **>1000 tCO2/acre**. The current 23-project model *excludes* these. So while no *plantation*
analog exists, there ARE high-biomass *conifer* analogs available — and including them is the single
biggest lever for attacking the Q5 saturation that would otherwise dominate Irish error.

## Direct answers to the two questions

**Q1 — which ANEW GT is applicable?** Not the deployed eastern subset. Build a maritime-temperate +
high-biomass-conifer subset from the full 52-project pool: **New England-Acadian** (Köppen Cfb-adjacent,
closest climate) plus the **Pacific-coastal / Cascades / Alaskan conifer** projects (closest structure
and the only ones that span Ireland's biomass range). WV Appalachian hardwood is the least relevant.

**Q2 — how to compare USA-model vs Irish-GT use, given stand-level estimates?** Train the analog-subset
head once, evaluate on a dual matrix (held-out USA LOPO **and** zero-shot Ireland) with identical
metrics, and use a pre-registered go/no-go to decide between accept-zero-shot, cheap post-hoc
calibration, or fine-tune/retrain on Irish GT. Handle the stand-level support explicitly (area-average
over polygons vs the plot-point support the head expects — and test the sensitivity on USA first).

## Feature set (Ireland transfer)

**Chosen: embeddings + disturbance-timing.**
- **AEF embeddings (64-dim, annual)** — core; annual product lets us align to each stand's survey year.
- **Dynamic World tree probability** — used as the **forest mask** (`apply_forest_mask.py`, trees ≥ 0.5)
  *and* tested as a continuous **feature**. Caveat: DW tree-prob is optical (Sentinel-2-derived), so
  highly correlated with the embeddings and shares their saturation limit — evaluate with/without it and
  keep it only if it earns its place.
- **Survey-relative Hansen harvest-timing** — disturbance features computed relative to each stand's
  survey year (reuse the survey-relative formulation; prior finding +0.013 R²). For even-aged Irish
  plantations this is the cheapest age/management proxy and the main lever we retain against saturation;
  global, time-varying.

**Dropped, with reasons:**
- **CHM (ETH 2020)** — static single-epoch layer; can't track fast-growing plantations or varying survey
  years; Irish Sitka likely exceeds its eastern-US range (extrapolation). Trade-off acknowledged: this
  removes the only vertical-structure signal, so the model is optical-only and **saturation is largely
  unmitigated** — disturbance-timing is the partial compensation.
- **GEDI lidar** — coverage stops at ~51.6°N; most of Ireland is north of it. Not viable.
- **Climate (TerraClimate)** — near-uniform across a single small country; ~0 within-Ireland signal.
- **SRTM topo** — legitimately static but Ireland is far flatter than Appalachia; marginal value.
  Optional only.

**Deferred option:** L-band SAR (PALSAR-2) / Sentinel-1 is the only non-optical, time-varying, global
structural signal and is cloud-robust — the natural next arm if disturbance-timing alone leaves PRD low
in the high-biomass band. Not in the initial feature set.

**Note on stand age:** if the Irish GT carries planting year / stand age (TBC when GT is shared), it is
the single strongest plantation-biomass predictor — use it as an evaluation covariate and to validate
the Hansen harvest-timing features. If absent, harvest-timing is our only age proxy and cannot be
directly validated.

## Plan of work

### A. Analog subset selection
- **Expand the candidate pool** from the 52-project gpkg (not the 23-project parquet) so conifer/western
  projects are visible. Extract AEF (int8, **no dequant** — see B4) + the chosen co-features (DW
  tree-prob + survey-relative Hansen harvest-timing) for the ~8,200 additional plots, reusing
  `scripts/extract_iter2_features.py` batched GEE extraction and its `build_dist_image` builder (skip
  `build_chm_image`/`build_topo_image` for the feature set; topo only if the optional arm is run).
- **Rank Ireland-vs-each-US-project** on three axes: (1) embedding-space distance (per-project
  Mahalanobis + energy distance/MMD of Irish samples), (2) Köppen match (Ireland = Cfb temperate
  oceanic), (3) ecoregion/structure match via `ECO_NAME` + canopy-height/target-range overlap.
- Define candidate models: **S1** maritime+conifer, **S2** embedding-nearest-k, **S3** full-CONUS
  baseline (control). Report the ranking table transparently.

### B. Input-distribution / covariate-shift diagnostics (run FIRST, cheapest, the real go/no-go gate)
- **B4 (do this literally first): encoding-consistency gate.** `infer_bayfield.py:140-186` casts int8→float
  **without** `aef.py::dequantize` and asserts corr>0.8 against the training parquet. Irish embeddings
  MUST use the identical raw-int8 path or every distance below is meaningless.
- Per-dim KS tests + joint **PCA/UMAP overlap** Ireland vs each subset.
- **OOD detection:** Mahalanobis fraction beyond the 99th-pct training radius; **domain-classifier AUC**
  (USA-vs-Ireland) — ~0.5 = transferable, ~1.0 = severe shift.
- Co-feature ranges for the retained features (DW tree-prob distribution; Hansen harvest-timing /
  years-since-disturbance — check Ireland's clearfell-driven age structure vs the US pool). Also run a
  CHM-range comparison as a *diagnostic only* to confirm the rationale for dropping it (expect Irish
  Sitka above the US range).
- Target-range overlap (after D): fraction of Irish stands in the **>80 tCO2/acre saturation zone** and
  above the subset training max.

### C. Spatial-support reconciliation
- Area-average embeddings over Irish **stand polygons** (`reduceRegions` with polygon geometries) to
  match the stand-level target.
- **Test the support effect on USA first** using `scripts/compare_gaussian_vs_pointextract.py`:
  point vs plot-radius buffer vs stand-sized buffer. If predictions are support-sensitive, prefer
  extracting at the plot scale the head expects and aggregating plot-scale predictions to stand level,
  rather than feeding stand-averaged embeddings into a plot-trained head. Report both for Ireland.

### D. Target / unit reconciliation (blocked on GT)
- Conversion chain with logged assumptions + uncertainty envelope: m³/ha → biomass (Sitka density
  ~0.33–0.42 t/m³) → total AGB (BEF ~1.2–1.4) → carbon (×0.47–0.50) → CO2 (×44/12) → per-acre (÷2.47105).
- Confirm ANEW's `CO2` column boundary (AGB-only vs AG+BG, all-live) and match the Irish quantity to it.
- Report sensitivity of R²/bias to low/central/high conversion — if go/no-go flips, the unit assumption
  is the binding uncertainty, not the model.

### E. Evaluation design — reuse `evaluation/compute_biomass_metrics.py` verbatim
- Metrics per (model × eval set): R², RMSE, MAE, bias, **per-quintile signed bias (Q1..Q5)**,
  **predicted-range-discrimination (PRD)** (CONUS baseline 0.468), per-ecoregion R², calibration-by-decile.
- **Dual matrix:** rows {S1, S2, S3} × columns {USA-LOPO within subset, USA conifer/high-biomass held-out,
  Ireland zero-shot}. Report the delta: does the analog subset beat full-CONUS on Ireland, and at what
  cost to in-domain USA R²?
- Saturation reporting: Q1..Q5 bias table + PRD + binned-bias plot focused on >80 tCO2/acre; state the
  fraction of Irish stands in that zone.
- **Pre-registered go/no-go:** **GO** if Ireland R² within ~0.05 of USA-LOPO and PRD ≥ 0.6 in the Irish
  range and OOD AUC ~0.5; **CALIBRATE** if good rank-correlation but consistent bias (1-D isotonic/linear
  on a small Irish slice); **FINE-TUNE/RETRAIN** (expected) if PRD <0.6 in the high-biomass band or Irish
  targets sit largely above training max.

### F. Sequencing
- **Phase 0 (now, no GT, parallel):** B4 encoding gate · expand 52-project pool · parameterize hardcoded
  `EPSG:32615`/RES in `infer_bayfield.py` + `export_bayfield_cofeatures.py` for Ireland (`EPSG:2157`/ITM) ·
  climate+ecoregion analog ranking · USA support-sensitivity test · train S1/S2 heads + USA-LOPO metrics.
- **Phase 1 (blocked on Irish GT):** unit reconciliation → polygon embedding extraction + encoding gate →
  covariate-shift diagnostics (B1–B3) → zero-shot dual evaluation + saturation report → apply go/no-go.
- **Must precede any full wall-to-wall agents run:** the encoding gate, the expanded pool, and the
  covariate-shift diagnostics. Do not launch wall-to-wall Irish inference until Ireland is confirmed not
  catastrophically OOD and units are reconciled.

## Critical files
- `scripts/infer_bayfield.py` — int8-no-dequant embedding path + correctness gate; hardcoded CRS to parameterize
- `scripts/export_bayfield_cofeatures.py` — wall-to-wall global co-features; hardcoded CRS
- `scripts/extract_iter2_features.py` — point/area co-feature + topo/CHM/Hansen builders to reuse for pool expansion and Ireland
- `scripts/compare_gaussian_vs_pointextract.py` — point-vs-area extraction sensitivity (Section C)
- `experiments/agb_usa_biomass_regression_20260529/evaluation/compute_biomass_metrics.py` — per-quintile bias, PRD, per-ecoregion R²; reuse verbatim
- `scripts/train_inference_model.py` / `train_agb_lgbm.py` — LightGBM head training; adapt for analog-subset heads
- `/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg` — full 52-project pool incl. conifer analogs

## Verification
- Encoding gate: replicate `infer_bayfield.py` correctness gate (corr>0.8) for Irish embeddings before any analysis.
- USA-LOPO metrics for S1/S2 reproduce via `compute_biomass_metrics.py` and are comparable to the locked S3 baseline (`biomass_metrics.json`: R² 0.418, PRD 0.468).
- Covariate-shift report (PCA/UMAP + OOD AUC + co-feature/target overlap) produced before any wall-to-wall run.
- Dual evaluation matrix + saturation report produced; go/no-go decision recorded per subset and per unit-conversion scenario.

## Open items pending the GT data you will share
Units & AGB boundary definition · record count · survey year(s) for AEF temporal alignment · exact polygon geometry/CRS · species/age metadata (would let us validate the analog selection directly).
