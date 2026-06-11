# Database Profile — Ireland AGB Transfer Experiment

- experiment_id: agb_ireland_biomass_regression_20260608
- stage: data_profile
- actor: Database Profiling Actor
- generated: 2026-06-08
- mode: zero-shot transfer + model-vs-model comparison (no GT, no training in first pass)

All claims below are grounded in real code runs (geopandas / pandas / earthengine, all via
`uv run python`). The probe script for Parts A/B/C is committed alongside this artefact:
`data_profile/probe_abc.py`. Part D used inline `uv run python` snippets (reproduced in the
relevant sections). Library context: Python 3.13, pandas (StringDtype-backed CSV read),
geopandas + pyogrio (no fiona in env), rasterio, earthengine-api (bare `ee.Initialize()` OK).

> **NOTE — missing acceptance-gate reference.** The prompt's first required input,
> `/home/mattc/.claude/skills/biomass-ml-agent-evolve/references/database_preprocessing.md`,
> **does not exist** (the entire `.../biomass-ml-agent-evolve/references/` directory is absent).
> I proceeded using the structure/conventions in the IMPLEMENTATION_PLAN and approved plan
> `plans/ireland-agb-test-v1.md`, and flag the absence here for the Critic.

---

## TL;DR — encoding-gate feasibility verdict (Part D, the crux)

**Verdict: FEASIBLE but a non-trivial encoding transform is required and must be solved in
preprocessing.** GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` is structurally consistent with the
training encoding — sampling it at 20–25 Bayfield **training** plots (year 2023) and correlating
the 64 GEE band values against the parquet `emb_00..63` gives **corr mean = 0.957 (min 0.946)**,
comfortably above the corr>0.8 gate. **However the two encodings are NOT on the same scale and
NOT related by a single scalar.** GEE serves dequantised unit-norm-style floats (range ≈
[-0.39, +0.38]); the training parquet holds **raw int8 values, area-averaged, NOT dequantised**
(range ≈ [-86, +86], per-band std 7.8–41.1). The implied GEE→parquet scale is ~×300 *on average*
but the **per-band slope ranges 220–623** (and a direct local-int8-tile vs GEE-float pixel
comparison gives per-band ratios 258–707, corr 0.957). So the LightGBM head — which splits on
**absolute** parquet-scale thresholds — cannot be fed raw GEE floats, nor GEE×127, and a single
global multiplier will mis-scale individual bands by up to ~3×. **The preprocessing stage must
derive and apply a per-band affine map** from GEE-float space to the training (raw-int8-averaged)
space, validated by re-running the corr-and-scale check at the Bayfield overlap, before any Irish
prediction is trusted. (See Part D for the exact transform recommendation.)

---

## A. Deep Biomass CSV

Path: `/home/mattc/data-space/carbonmap-embeddings/dasos-ireland/deepbiomass-model-outputs/Deep Biomass - Aggregated Data & Portfolio Summary.csv`

**Structure.** 143 rows × 15 cols. Columns: `Location No, Location Name, Area Ha, 2013, 2014,
…, 2024` (year columns 2013–2024 = full coverage, 12 years). **141 Location rows** (Location No
1–141, all `Location Name` unique) + **2 footer rows** (Location No blank):

- `Total AGB, ton` — portfolio total tonnes per year (2024 = 150,313 t; 2023 = 101,497 t).
- `Total AGB, ton/ha` — area-weighted density (2024 = 44.6 Mg/ha; 2020 = 39.4).

Footer `Area Ha` = 3,367.63 (portfolio total). Sum of the 141 per-Location `Area Ha` = **3,367.63
ha** — exact match, confirming locations partition the portfolio.

**Parsing gotcha (confirmed).** Some cells (and the footer 2023/2024 totals) are thousands-
separated strings inside quotes (e.g. `"1,074"`, `"101,497"`); a plain `read_csv(thousands=",")`
leaves a mixed string column that breaks `astype(float)`. Coerce with
`pd.to_numeric(s.str.replace(",","").str.strip())`.

**Cell semantics confirmed: cells = total tonnes AGB = Mg/ha × Area_Ha.** Recovering
`Mg/ha = cell ÷ Area_Ha` and area-weighting (`Σtonnes ÷ Σarea`) reproduces the footer `ton/ha`
row to within rounding:

| year | recovered area-wtd Mg/ha | footer ton/ha |
|---|---|---|
| 2020 | 39.4 | 39.4 |
| 2024 | 44.6 | 44.6 |
| 2013 | 19.2 | 19.2 |

Worked example (plan): Aghaderrard West 2013 = 138 t ÷ 10.21 ha = 13.5 Mg/ha. ✔

**Per-Location Mg/ha distribution (cell ÷ Area_Ha).** Mean of per-location density per year:

| year | mean | median | min | max |
|---|---|---|---|---|
| 2013 | 19.1 | 15.1 | 2.9 | 82.6 |
| 2019 | 29.9 | 23.8 | 0.26 | 116.5 |
| 2020 | 38.9 | 31.1 | 0.67 | 136.2 |
| 2021 | 41.9 | 36.9 | 10.8 | 161.2 |
| 2022 | 37.2 | 33.1 | 8.3 | 113.6 |
| 2023 | 32.4 | 25.8 | 7.3 | 106.6 |
| 2024 | 45.7 | 39.7 | 8.4 | 129.3 |

**2020–2024 mean per-Location Mg/ha** (the plan's stable-window reference): n=141, mean **39.19**,
median 36.33, std 16.53, min 13.44, 25% 26.59, 75% 48.35, max **95.82**. (Note: the simple
per-location mean of 39.2 differs from the area-weighted portfolio 44.6 because large Locations
read higher — see Part E note in the plan; both are valid, report the density-delta in tCO₂/acre.)

**Noise / spurious values (confirmed, motivates the stable-window mean).**
- Ahalahana 2015 = **2 t total** → 0.24 Mg/ha (plan's flagged spurious value). ✔
- 2 cells across the matrix are < 5 t total (near-zero, implausible for a stocked stand).
- Year-on-year volatility is large: median(max/min Mg/ha ratio across years) = 6.3, **max = 239×**
  — single-year values are unreliable; the 2020–2024 mean damps this.

---

## B. Dasos gpkg + crosswalk + covariates

Path: `/home/mattc/data-space/carbonmap-embeddings/boundary-files/dasos_fgl_2025ye.gpkg`

- **Layers:** one — `fgl_2025ye_` (geometry type `MultiPolygon Z`).
- **Rows:** 1,053 (= the expected sub-compartment count). ✔
- **CRS:** `EPSG:4326`. ✔ (reproject to Irish ITM **EPSG:2157** for metric area / GEE extraction).
- **Geometry:** all 1,053 `MultiPolygon`, **all valid** (0 invalid).
- **SiteName:** 141 unique → matches the 141 Deep-Biomass Locations.

**Column schema (37 cols) + missingness.** Format `name dtype nmiss`:

```
Manager str 0        Forester str 0       Key str 0            Management str 0
SiteName str 0       SiteCode str 0       Cpt_ID str 0         Sub_Cpt int64 0
GrossArea f64 0      ProdArea f64 0       ProdArea_ f64 0      LandClass str 0
Category str 0       PlantingYe f64 286   PlantDate str 286    MgtRegime str 283
Thinned str 357      PreviousTh str 489   SurveyType str 1     SurveyDate str 10
SoilType str 258     Drainage str 259     YC f64 0             MainSp str 290
MainSpNha int64 0    SecSp str 605        SecSpNha f64 1       Dmean f64 0
Hmean f64 2          Hdom f64 2           BA_Conifer f64 764   ConiferDam str 1010
RoadMeters f64 1     PathMeters f64 0     AquaMeters f64 0     PointFeatu int64 0
geometry geometry 0
```

### Crosswalk Location Name → gpkg SiteName — ALL 141 RESOLVE ✔

The plan's "underscore-split suffix" hypothesis was **wrong about the mechanism**: the gpkg
`SiteName` uses a **slash** separator for grouped sites (e.g. `Moy/Sonnagh`,
`Upper Shannon North/Garvesk`), while the CSV `Location Name` uses an **underscore**
(`Moy_Sonnagh`, `Upper Shannon North_Garvesk`). The correct rule:

1. **Direct** exact match → **124 / 141**.
2. **Underscore→slash** (`nm.replace("_", "/")`) → **17 / 141**.
3. **Resolved: 141 / 141, zero failures.** (Suffix-only split would mis-handle these and leaves 13
   unmatched — do NOT use it.)

Crosswalk written to `data_profile/crosswalk_location_to_sitename.csv` (cols
`Location_Name, SiteName, method`). **Preprocessing must use the underscore→slash rule, not the
plan's suffix-split.**

### Dissolved area per Location vs CSV Area_Ha — agree ✔

Dissolving the 1,053 sub-compartments to 141 SiteName Locations (area computed in EPSG:2157):
total geom area = **3,367.1 ha** vs CSV 3,367.63. Per-Location `diff_pct` (geom−CSV)/CSV:
mean −0.02 %, median −0.03 %, range [−0.16 %, +0.32 %]; **0 Locations exceed ±10 %**. Areas are
effectively identical → same aggregation unit, crosswalk is sound.

### Covariate profiles (evaluation cuts)

- **MainSp** (dominant species): `SS` (Sitka spruce) = **627 / 1,053 (60 %)** — Sitka dominance
  confirmed; far ahead of LP 28, ASH 24, ALDER 20, SYCAM 17. **290 sub-cpts have MainSp missing**
  (28 %) — material for the MainSp cut; impute/flag from Category/LandClass in preprocessing.
- **PlantingYe** → stand age: range **1986–2026**, mean 2005. **286 missing** (27 %). (Values
  >2024 are future/replant scheduling — clamp when deriving survey-year age.)
- **SurveyDate** → AEF year alignment: stored as `dd/mm/yyyy` strings, **10 missing**. Parsed
  range **2015-03 … 2026-10**. Year counts: **2024 = 289, 2023 = 280, 2025 = 280**, 2016 = 97,
  2026 = 29, 2020 = 28, others few. So the survey-year-aligned AEF target is mostly 2023/2024
  (both available on GEE — Part D), but a non-trivial tail surveys 2025/2026 (use 2025 AEF, the
  latest available) and 2015–2020. **Decide per-Location AEF year from the (area-dominant)
  SurveyDate year, clamped to the AlphaEarth coverage [2017, 2025].**
- **YC** (yield class): 0–34, mean 15.3, **0 missing** (cleanest structural covariate).
- **Hmean** 0–26.8 (mean 6.9, 2 missing); **Hdom** 0–28.0 (mean 7.7, 2 missing) — vertical
  structure, nearly complete.
- **BA_Conifer**: 0–76.5, **764 missing (73 %)** — sparse; usable only as a coarse flag.
- **Thinned**: N=506, Y=190, **357 missing (34 %)**. **MgtRegime**: Thin=531, No Thin=238/1
  (`No thin` casing variant — normalise), 283 missing.
- **SecSp**: 605 missing; top JL 213, ADB 130, LP 59, SS 18. **GrossArea** 0.01–48.8 (mean 3.2),
  **ProdArea** 0.0–44.1 (mean 2.3), both complete.

Missingness summary for the cut variables: YC complete; Hmean/Hdom ~complete; MainSp 28 % missing;
PlantingYe 27 % missing; Thinned 34 % missing; BA_Conifer 73 % missing (drop as a primary cut).
Area-weighted aggregation to Location (plan Section A) will partly fill gaps where some
sub-compartments carry the attribute.

---

## C. Training-data encoding reference (defines the gate)

**Canonical training parquet:** `/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet`
(this is the one `infer_bayfield.py` reads as `TRAIN_PARQUET`). No `emb_*` parquet exists under
the repo `preprocessing/` or `…/carbonmap-embeddings/training-data/` (the latter holds only
`anew_gt_with_eco_info.gpkg`). The repo's `experiments/agb_usa_…/preprocessing/features_iter*.parquet`
are downstream derivatives; the tf-deep-landcover parquet is the encoding source of truth.

- **Shape:** 4,646 rows × 72 cols. Non-emb cols: `plot_id, project_name, year, lon, lat, target,
  failure, region`. `year` ∈ {2022: 2284, 2023: 2362}. (No `survey_year` column — use `year`.)
- **emb columns:** **64** (`emb_00`…`emb_63`), **dtype float64**.
- **NaNs:** 640 NaN cells across 10 rows (drop those rows for the gate).
- **Value range (finite):** min **−86.30**, max **+85.72**, mean 2.03, std 39.78. Per-band:
  col-min ∈ [−86.3, −21.8], col-max ∈ [+8.3, +85.7]; per-band **std 7.8–41.1**, per-band mean
  −62.2…+59.0.
- **Are they raw int8 or dequantised?** **Raw int8-range, area-averaged — NOT dequantised.**
  Range sits inside [−128, 127] (int8) and is far outside [−1, 1] (dequant). Only **5.7 % of
  values are exact integers** — the rest are fractional because the training pipeline
  **area-averaged the raw int8 tiles** (the 3×3 `np.nanmean` / buffer mean), yielding non-integer
  values that remain on the int8 magnitude scale.

**How training embeddings were encoded (cross-ref `infer_bayfield.py`).** `read_embeddings`
(lines 119–150) reads the local int8 AEF tiles as uint8 from the VRT, reinterprets `>127 → −256`
to recover signed int8, casts to float, sets −128 → NaN, then `np.nanmean` over 3×3 blocks to
30 m — **no `dequantize`/`/128` step anywhere.** The training parquet was built the same way (raw
int8 → float → spatial mean). `correctness_gate` (lines 159–188) is exactly the contract: sample
the embedding stack at known training plots, correlate against the parquet `emb_*`, and
**`assert mean corr > 0.8`** ("embedding transform mismatch — ABORT (check int8/dequant)").
**This is the gate the Irish extraction must reproduce.**

---

## D. GEE AlphaEarth asset — the central feasibility check

Asset id: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`. Loads under bare `ee.Initialize()`. ✔

- **Collection size:** 97,155 images.
- **Bands:** **64** — `A00`…`A63`. ✔ (note: `A00..A63`, NOT `emb_*`; rename on extraction).
- **dtype:** `double` (PixelType precision=double) — i.e. **dequantised float**, not int8.
- **Value range (sampled):** ≈ **[−0.39, +0.38]** — unit-norm-style floats, NOT the parquet scale.
- **Years available:** **2017–2025** (from `system:time_start`). Over an Ireland bbox
  (`[-10.5,51.4,-5.9,55.4]`): 2017 → 15 images, **2023 → 15, 2024 → 15**. So 2023 and 2024 (the
  bulk of Dasos survey years) are covered; 2025 is the latest available for the 2025/2026 survey
  tail; pre-2017 surveys (2015/2016, ~98 sub-cpts) have **no AEF** and must fall back to 2017.

### THE CENTRAL CHECK — GEE sampled at Bayfield TRAINING plots vs parquet emb

Method (reproducible): load the parquet, take the 409 `BayfieldCounty` plots (all year 2023),
sample 20–25 of them via `ImageCollection.filterBounds(point).filterDate("2023-…").first()
.reduceRegion(ee.Reducer.first(), point, scale=10)` for bands `A00..A63`, and correlate the 64
GEE values against that plot's `emb_00..63`.

**Result:**
- **corr(GEE A00–63, parquet emb_00–63) = mean 0.957, min 0.946, max 0.972.** → **GATE PASSES**
  on structure: the GEE asset is the same underlying AlphaEarth signal as the training encoding.
- **Scale relationship is NOT a clean scalar.** Through-origin fit `parquet ≈ GEE × k` gives
  k mean ≈ 300, median ≈ 300, pooled linear fit `parquet = 301·GEE + 0.07`. BUT **per-band slope
  ranges 220–623** (std ~80). Candidate fixed scales all fit poorly: RMSE(parquet, GEE×127)=24.7,
  ×256=12.8, ×300=11.5 — none is clean, and a single multiplier mis-scales individual bands by
  up to ~3×.
- **Direct local-int8-tile vs GEE-float pixel check (decisive):** reading one local Bayfield int8
  tile (`agb_usa_pilot_midwest/embeddings_annual/2023/*tile_*.tif`, dtype int8, 64 bands,
  EPSG:32615), recovering signed int8, and comparing to the GEE float at the same lon/lat gives
  **corr 0.957**, through-origin scale ≈ 303, but **per-band ratios 258–707**. So the local tiles
  (the literal training source) are **not** `round(GEE_float × constant)` — the two products use
  **different per-band quantisation/normalisation**. They share direction (corr ~0.96) but differ
  in per-band magnitude.

### Can the GEE extraction be made encoding-consistent? — YES, with a per-band affine

A single global multiply (×127 / ×256 / ×300) will **fail** the spirit of the gate: high overall
corr but wrong per-band magnitudes → wrong LightGBM split activations → biased predictions. The
correct, defensible transform is a **per-band affine map fitted on the Bayfield overlap**:

> For each band j, regress `emb_j (parquet, raw-int8-averaged)` on `A{j} (GEE float)` over the
> Bayfield training plots → slope `a_j`, intercept `b_j`. Apply
> `emb_irish_j = a_j · A{j}_GEE_irish + b_j` to the Ireland extraction. Then **re-run the
> correctness gate** (sample-vs-parquet corr at Bayfield) on the transformed values and require
> mean corr > 0.8 **and** per-band slope ≈ 1 after the fit.

Because corr is already 0.96, a per-band affine recovers the magnitude almost exactly; the residual
~4 % unexplained variance is point-vs-area spatial mismatch (parquet emb are buffer/area means;
GEE was point-sampled here) and will shrink once Irish extraction uses `reduceRegions(mean)` over
polygons (matching the training spatial support, plan Section A). **Alternative, only if the GEE
ANNUAL int8 representation can be exported bit-identically to the local tiles** — but the asset
serves `double`, and the local-tile comparison shows they are NOT the same quantisation, so the
per-band-affine route is the recommended, validated path. The hard requirement stands: **the
encoding-consistency gate must pass on the transformed Irish embeddings before any prediction is
trusted.** This is solvable in preprocessing; the experiment is feasible.

---

## Candidate leakage surfaces

- **No GT-based leakage on the head.** The head is pre-trained; Ireland is pure inference. Deep
  Biomass is an **external reference model, not ground truth and not a training input** → it cannot
  leak into the head and is not a leakage surface.
- **Bayfield in-sample caveat (not Ireland leakage, but affects the gate's interpretation):** the
  parquet/Bayfield plots used for the encoding gate ARE in the head's training set
  (`trained_on: all 4636 plots (incl. Bayfield — in-sample)`). That's fine for *encoding*
  validation (we only check feature reproduction, not target accuracy) but means the gate proves
  encoding consistency, NOT transfer accuracy — consistent with the no-accuracy-claim design.
- **Covariate-cut circularity:** the Dasos covariates (YC, Hdom, age) used to *validate* our
  predictions are independent of both models, so the structural-consistency check is sound.
- **Temporal leakage risk:** none for the head (no target), but using a survey-year AEF that
  post-dates a disturbance vs the DB reference window (2020–2024) could create apparent
  divergence — handle in the dstx co-feature timing (plan Section C), not a data issue here.

## Data-access gaps / blockers

- **Acceptance-gate reference file absent** (`…/biomass-ml-agent-evolve/references/
  database_preprocessing.md` and its parent dir) — flagged for the Critic; proceeded on the plan's
  conventions.
- **~98 sub-compartments survey 2015/2016 (pre-AlphaEarth-2017)** → no exact survey-year AEF;
  fall back to 2017 and flag those Locations in the OOD/temporal section.
- **MainSp 28 % / PlantingYe 27 % / Thinned 34 % / BA_Conifer 73 % missing** — covariate cuts
  feasible but with reduced n; aggregate area-weighted to Location to recover coverage.
- **No Irish AEF local tiles** (known central risk) — the GEE per-band-affine extraction path
  above is the resolution; must pass the gate.
- **`fiona` not installed** in the env — use `pyogrio.list_layers` for layer enumeration (used here).

## Provenance / reproducibility

- DB CSV: `…/dasos-ireland/deepbiomass-model-outputs/Deep Biomass - Aggregated Data & Portfolio Summary.csv`
- gpkg: `…/boundary-files/dasos_fgl_2025ye.gpkg` (layer `fgl_2025ye_`, EPSG:4326)
- training parquet: `/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet`
- local int8 tiles (encoding source): `…/agb_usa_pilot_midwest/embeddings_annual/2023/*tile_*.tif` (int8, 64 bands, EPSG:32615)
- GEE asset: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (64 bands A00–A63, double, 2017–2025)
- head: `models/inference_model_embdstx.txt` + `models/inference_features_embdstx.json` (67 feats, target tCO₂/acre, range [0, 520.95])
- probe script (A/B/C): `data_profile/probe_abc.py`; crosswalk: `data_profile/crosswalk_location_to_sitename.csv`;
  train emb sample: `data_profile/train_emb_sample.parquet`
- env: Python 3.13 (`uv run`); pandas, geopandas+pyogrio, rasterio, pyproj, earthengine-api (bare `ee.Initialize()`).

## Key assumptions

1. CSV cells = total tonnes AGB = Mg/ha × Area_Ha (verified against footer to rounding).
2. Crosswalk = direct match OR `_`→`/` substitution; all 141 resolve (verified).
3. Training encoding = raw int8 area-averaged, no dequant (verified: 94 % non-integer, range
   [−86,86], matches `infer_bayfield.py` read path).
4. GEE→training requires a per-band affine fitted on Bayfield overlap, then re-gated (corr>0.8).
5. Per-Location AEF year = area-dominant SurveyDate year, clamped to [2017, 2025].
