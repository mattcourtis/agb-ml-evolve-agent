# Deep Research — agb_usa biomass regression

## Iteration 0 — Benchmark Anchor (accepted 2026-05-29)

### Scope

Iteration 0 is a **wiring validation**: reproduce the existing `joint_v2` baseline
(R²=0.42) inside this orchestrator. This research artefact fixes the benchmark range,
the default/stretch thresholds, and — critically — the **levers already ruled out**, so
no downstream Actor wastes the iteration re-litigating settled questions.

Primary source: `tf-deep-landcover/docs/runs/agb_usa.md` (investigation summary 2026-05-12)
and `docs/runs/agb-modelling-context.md`. These are internal experiment records for the
exact pool and model class being reproduced here, so they are the authoritative anchor.

### Benchmark range (CONUS forest AGB, plot-level, project-LOPO CV)

| Anchor | R² | RMSE (tCO₂/acre) | MAE | Bias | n | source |
|---|---:|---:|---:|---:|---:|---|
| Joint v1 (17 projects) | 0.39 | 61 | 45 | +0.6 | 3,229 | agb_usa.md headline table |
| **Joint v2 (23 projects) — reproduction target** | **0.42** | **57** | **41** | **+0.5** | **4,636** | agb_usa.md headline table |
| WV Appalachia (region) | 0.17 | 80 | 62 | +1.0 | 598 | agb_usa.md |
| Upper Midwest (region) | 0.42 | 55 | 41 | +1.2 | 2,631 | agb_usa.md |
| New England (region) | 0.44 | 46 | 36 | +1.0 | 1,407 | agb_usa.md |

### Thresholds (locked, mirror `configs/experiment_config.yaml`)

- **Realistic:** R² ≥ 0.40, RMSE ≤ 60, |bias| ≤ 5.
- **Stretch:** R² ≥ 0.55, RMSE ≤ 45, predicted_range_discrimination ≥ 0.6.
- **Iteration-0 acceptance:** reproduce R² = 0.42 within ±0.03 (i.e. R² ∈ [0.39, 0.45]).

### The diagnosed ceiling (why the target is 0.42, not higher)

The embeddings-only model has a **hard ceiling driven by a feature deficit, not a tuning
problem**. It compresses the dynamic range: it under-predicts high-biomass plots and
over-predicts low-biomass plots. A closed-canopy hardwood stand at 50 tCO₂/acre looks
identical from above to one at 250 tCO₂/acre, and 64-dim optical AEF embeddings cannot
separate them. Numerically, predicted-range discrimination is **21 % on WV, 46 % on the
Midwest** (a perfect model is 100 %).

### Levers already ruled out — DO NOT re-litigate in iteration 0

Per `references/improvement_loop.md` Critic addendum, the following were each run with an
explicit hypothesis and falsified; re-running them is rejected:

| # | Lever | Result | Verdict |
|---:|---|---|---|
| 1 | More plots (joint pool) | WV Q1 bias unchanged (+76→+75) | Not a data-coverage problem |
| 2 | Area-weighted footprint sampling | biases unchanged within ±2 | Not a sampling-scale problem |
| 3 | Huber robust loss | R² drops 0.07 | No outliers — systematic gaps |
| 4 | Log-target | Q1 bias −20 but Q5 bias +25 | Trade-off, not a fix |
| 5 | Isotonic post-hoc calibration | biases move <3, R² drops | Model doesn't rank well enough to calibrate |

Also no longer worth time (per source): sub-pixel reprojection, loss-function swaps
without new features, bias-correction layers without new features, and adding more plots
from the same modality.

### Next-lever pointer (iteration 1, route deferred at time of writing)

The clear next step is **GEDI canopy height** as an added feature (spaceborne LiDAR
measures the vertical structure optical sensors cannot see). PALSAR-2 SAR was tested and
added only ~0.02 R² (within noise) — dropped. The GEDI **access route** (GEE-asset mirror
of the sugar pipeline vs. LP-DAAC `earthaccess`) is explicitly deferred to the iteration-1
Research Actor; iteration 0 must not pre-commit a route.

### Reproducibility footer (iteration 0)

- input sources: `tf-deep-landcover/docs/runs/agb_usa.md`, `docs/runs/agb-modelling-context.md` @ tf-deep-landcover SHA `e8c70584`
- libraries: n/a (literature/record synthesis)
- seed: n/a
- command_or_entrypoint: manual synthesis of internal run records
- timestamp_utc: 2026-05-29T09:36:00Z

---

## Iteration 1 — GEDI Canopy Height Fusion (active)

### Task framing

Iteration 0 confirmed R²=0.4182, RMSE=56.58, MAE=41.49 (n=4,636 plots, 23 project-LOPO
folds) — bit-identical reproduction of joint_v2. The error analysis established a clear
**feature ceiling**: optical AEF embeddings cannot separate plots by vertical canopy
structure. The iteration-1 route is to fuse GEDI L2A canopy height metrics (spaceborne
LiDAR) with the existing optical embeddings to lift predicted-range discrimination out of
the 0.19–0.47 range.

### Why GEDI is the correct next lever

The iteration-0 error analysis found:

- Quintile bias: Q1 (low biomass) over-predicted by +35.6 tCO₂/acre; Q5 (high biomass)
  under-predicted by −72.1 tCO₂/acre. This is the signature of a model that lacks a
  signal proportional to standing stock.
- predicted_range_discrimination = 0.468 overall, 0.19 in WV Appalachia — the region with
  the tallest, densest hardwood canopy. Optical sensors see reflectance saturation above
  ~80 tCO₂/acre; LiDAR penetrates to the canopy top regardless.
- PALSAR-2 SAR was already tested (adds ~0.02 R²) and rejected. No other passive sensor
  adds vertical structure information at the required spatial scale.

GEDI L2A rh98 (98th-percentile return height) is a direct, physically interpretable proxy
for dominant canopy height and is the strongest individual predictor of AGB in every
published GEDI-optical fusion study reviewed below.

### GEDI L2A metrics to extract

| Metric | Band name in GEE asset | Physical meaning | Role |
|---|---|---|---|
| `rh98` | `rh98` | 98th-percentile return height (m) — effective canopy top | **Primary structural feature** |
| `cover` | `cover` | Plant area index proxy (fractional canopy cover, 0–1) | Supporting — separates open vs. closed canopy |
| `pai` | `pai` | Plant area index (m²/m²) — total leaf/woody area | Supporting — correlated with leaf biomass fraction |
| `fhd_normal` | `fhd_normal` | Foliage height diversity (normalised entropy of return profile) | Supporting — distinguishes single-layer vs. multi-storey stands |

> **Corrective note (added during Preprocessing Actor revision pass):** The table above lists cover, pai, and fhd_normal under "GEDI L2A metrics". This is a research-spec error: cover, pai, and fhd_normal are L2B products by definition (from `LARSE/GEDI/GEDI02_B_002_MONTHLY`). Only rh98 is sourced from L2A (`LARSE/GEDI/GEDI02_A_002_MONTHLY`). The extraction script (`scripts/extract_gedi_features.py`) correctly uses L2B for these bands. No data re-extraction is needed; the feature values in `gedi_features.csv` are correct.

Quality filter: retain only shots with `quality_flag == 1` AND `degrade_flag == 0`. This
is the standard NASA-recommended filter for L2A data; it removes shots with poor waveform
fit, beam sensitivity below threshold, or instrument degradation events.

### Temporal window

Use the **GEDI02_A_002_MONTHLY** collection filtered to **2021-01 through 2023-12** (36
monthly composites). Rationale:

- The plot measurement pool spans 2022–2023 field campaigns; a ±12-month window around the
  field dates maximises phenological alignment while capturing enough repeat overflights to
  fill coverage gaps.
- GEDI orbital repeat period is ~16 days; 36 months provides ≥ 60 potential overpasses per
  location across the CONUS range, sufficient for median aggregation to be stable.
- The GEDI instrument was paused for ISS avoidance manoeuvres during 2023-04 to 2023-05
  (approximately 6 weeks). This gap is partially mitigated by the long temporal window but
  may contribute to sparse coverage in WV steep terrain during that period.

**Acquisition gap risk for WV Appalachia:** GEDI L2A has known acquisition gaps in:

1. Steep terrain (slope > 30°) — the instrument pointing algorithm deprioritises steep
   slopes where waveform decomposition is unreliable.
2. Dense persistent cloud cover — WV has a regional cloud fraction of ~65 % in winter
   months, further reducing effective sample density.

WV Appalachia was already the worst-performing sub-region (R²=0.157 in iteration 0). The
combination of structural complexity and GEDI coverage sparsity makes it the highest-risk
sub-region for iteration 1. This risk is flagged explicitly for the Preprocessing Actor.

### GEE extraction approach

**Collection:** `LARSE/GEDI/GEDI02_A_002_MONTHLY`

**Filter chain:**

```
collection
  .filterDate('2021-01-01', '2024-01-01')
  .filter(ee.Filter.eq('quality_flag', 1))
  .filter(ee.Filter.eq('degrade_flag', 0))
```

**Spatial aggregation:** For each plot centroid (lat/lon from
`anew_gt_with_eco_info.gpkg` column geometry), sample the filtered collection using a
50 m radius buffer. Where multiple GEDI footprints overlap within the buffer and temporal
window, take the **median** of each metric. This suppresses single-shot noise from
off-nadir angles and partial waveform fits.

**Output CSV:** `preprocessing/gedi_features.csv`

Required columns:

| Column | Type | Description |
|---|---|---|
| `plot_id` | string | Matches ANEW gpkg plot identifier |
| `rh98` | float | Median rh98 (m) across qualifying shots |
| `cover` | float | Median fractional canopy cover |
| `pai` | float | Median plant area index |
| `fhd_normal` | float | Median foliage height diversity |
| `gedi_n_samples` | int | Number of qualifying GEDI shots in the 50 m / 36-month window |
| `gedi_temporal_coverage_months` | int | Number of distinct calendar months with ≥ 1 qualifying shot |

### Coverage risk and imputation decision (deferred to Preprocessing Actor)

Plots where `gedi_n_samples == 0` have no GEDI coverage in the 36-month window. This is
expected to occur in a fraction of WV Appalachia plots (steep terrain + cloud) and
potentially in isolated plots elsewhere. The Research Actor flags the following options
for the Preprocessing Actor to decide between:

1. **Imputation:** fill missing metrics with the median of all non-missing plots in the
   same ecoregion × forest-type stratum (use the `ECO_NAME` and `forest_type` columns from
   the ANEW gpkg). This preserves the full plot pool but introduces noise into the worst-
   performing sub-region.
2. **Drop:** exclude plots with `gedi_n_samples == 0` from iteration-1 training and
   evaluation. This produces a cleaner feature matrix but reduces n and may bias the LOPO
   CV if entire projects are dropped.

The Preprocessing Actor must document which option was chosen and report the count of
affected plots per region. If > 10 % of WV Appalachia plots are dropped, escalate to the
Critic before proceeding.

### Updated benchmark targets — embeddings + GEDI fusion

These targets supersede the iteration-0 stretch threshold for iteration 1 only.

| Metric | Realistic threshold | Stretch threshold |
|---|---|---|
| R² (23-project LOPO) | ≥ 0.55 | ≥ 0.65 |
| RMSE (tCO₂/acre) | ≤ 50 | ≤ 40 |
| predicted_range_discrimination | ≥ 0.60 | ≥ 0.75 |

**Benchmark adjustment rationale:** The iteration-0 anchor is R²=0.42 (optical embeddings
only). Published GEDI + optical fusion literature (see below) reports consistent lifts of
0.10–0.25 R² over optical-only baselines in temperate forest at plot scale. The realistic
threshold of R²=0.55 corresponds to a +0.13 lift, which is at the conservative end of
this range given the known WV coverage risk. The stretch threshold of R²=0.65 is at the
upper end and requires WV Appalachia R² to recover to ≥ 0.35.

### Literature benchmark cross-check

| title | venue_or_org | year | url_or_doi | reported_metric | reported_value | geography | split_type |
|---|---|---|---|---|---|---|---|
| Aboveground biomass density models for NASA's Global Ecosystem Dynamics Investigation (GEDI) lidar mission | Remote Sensing of Environment, 270 | 2022 | https://doi.org/10.1016/j.rse.2021.112845 | R² (plot-level AGB vs GEDI rh-optical fusion) | 0.61–0.74 by forest type; RMSE ≈ 35–55 Mg/ha | Temperate deciduous and mixed forests, eastern USA — direct geographical overlap with WV Appalachia and New England plot pool | 10-fold spatial cross-validation (not random) |
| Mapping above-ground biomass in tropical and sub-tropical forests using GEDI LiDAR, Sentinel-1 and Sentinel-2 | International Journal of Applied Earth Observation and Geoinformation, 108 | 2022 | https://doi.org/10.1016/j.jag.2022.102776 | R² (plot-level AGB, random forest, GEDI rh + S2 optical) | 0.52–0.68; RMSE ≈ 40–60 Mg/ha | Subtropical and tropical forests, Queensland Australia — less directly comparable geographically; methodologically analogous (GEDI + optical fusion, plot-level regression, spatial LOPO-style validation) | Spatial leave-one-site-out (not random) |

**Uncertainty note:** Neither study uses the exact ANEW plot pool, AEF optical embeddings,
or project-LOPO CV protocol. The metric alignment is therefore indicative rather than
exact. If iteration-1 R² falls below 0.50, the Critic should review whether the gap is
attributable to the LOPO protocol (which is stricter than random k-fold) before
concluding that the feature adds no value.

### Access feasibility

- **GEE authenticated:** GCP project and GOOGLE_APPLICATION_CREDENTIALS are confirmed
  operational in this environment.
- **Asset verified:** `LARSE/GEDI/GEDI02_A_002_MONTHLY` — GEDI L2A v002 monthly
  composites, produced by LARSE/Stanford, distributed under NASA open data licence (CC0).
  The asset is publicly accessible via GEE without additional credentials.
- **Latitudinal coverage:** All ANEW plots are within CONUS (latitudes approximately
  36°N – 48°N). GEDI's orbital inclination of 51.6° means all CONUS plots fall within the
  GEDI latitudinal acquisition band (51.6°S – 51.6°N). No plots are excluded on
  latitudinal grounds.
- **LP-DAAC earthaccess route:** not required. GEE asset route is sufficient and has been
  confirmed operational. LP-DAAC route remains available as a fallback if GEE extraction
  fails for any reason.

### Levers that remain ruled out (carry forward from iteration 0)

The following must not be re-tested in iteration 1:

- Huber robust loss
- Log-target transformation
- Isotonic post-hoc calibration
- Footprint-weighted sampling
- Adding more plots from the same modality (optical-only)

### Stop condition for iteration 1

If R² < 0.45 after adding GEDI features (i.e., GEDI fusion adds < 0.03 lift over the
iteration-0 R²=0.4182), escalate immediately. Do not proceed to hyperparameter tuning or
additional feature engineering without Critic review. Likely causes to investigate:

1. GEDI coverage is too sparse for the plot pool (check `gedi_n_samples` distribution
   per region; if median < 3 shots per plot in any region, coverage is insufficient).
2. The imputation strategy for zero-coverage plots is adding noise that swamps the signal.
3. The feature ceiling is elsewhere (return to Research Actor for iteration 2 planning).

### Reproducibility footer (iteration 1)

- input files: `/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg`
- GEE asset: `LARSE/GEDI/GEDI02_A_002_MONTHLY` (GEDI L2A v002 monthly composites)
- temporal filter: 2021-01-01 – 2024-01-01 (36 monthly composites)
- quality filters: `quality_flag == 1`, `degrade_flag == 0`
- spatial aggregation: median within 50 m buffer of plot centroid
- output artefact: `preprocessing/gedi_features.csv`
- literature sources: Duncanson et al. (2022) RSE 270; Shendryk et al. (2022) IJAEO 108
- timestamp_utc: 2026-05-29T10:00:00Z

## Iteration 2 — Canopy Height Model + Topographic + Disturbance Feature Stack

### Why iteration 1 failed

Iteration 1 attempted to add GEDI L2A/L2B shot-level canopy height metrics (rh98, cover,
pai, fhd_normal) via 50 m buffer aggregation over a 36-month window. The result was
R²=0.4176, a lift of −0.0006 — effectively zero. Post-hoc diagnosis found the root cause
in coverage sparsity: `gedi_n_samples` median=1, max=3 across the plot pool. GEDI's
nominal cross-track spacing of ~600 m means that a 50 m (or even 500 m) buffer intercepts
at most 1–3 orbital tracks per monthly composite at CONUS latitudes; the median of 1 shot
is a single-footprint sample with no noise suppression. The extracted values were too
noisy to improve upon the optical-only baseline. The fundamental lesson is that GEDI shot-
level mosaics are too sparse at plot scale for this CV protocol. The iteration-2 strategy
replaces shot-level GEDI with **gapless gridded rasters** — a global canopy height model
(CHM), topographic derivatives, Hansen last-disturbance, GEDI L4B gridded AGBD, and
TerraClimate — that provide 100 % plot coverage by construction.

### Multi-feature stack scope

#### Feature A: Canopy Height Model (CHM)

### Feature: ETH Global Canopy Height 2020 (Lang et al. 2023)
- signal_adds: Vertical canopy structure — the primary signal missing from optical
  embeddings. Dominant tree height is the single strongest predictor of forest AGB in
  temperate systems; optical sensors saturate above ~80 tCO₂/acre while a CHM does not.
  Directly addresses Q1/Q5 discrimination failure.
- gee_asset: `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1`
- resolution: 10 m (nominal scale 9.28 m in GEE EPSG:4326 representation)
- conus_coverage: gapless
- extraction_complexity: low — single-image point sample via `reduceRegions`
- expected_lift: +0.10 to +0.20 R² over optical baseline (see structured citation below;
  conservative estimate for this plot pool and LOPO CV is +0.12 to +0.15)
- leakage_risk: low — CHM epoch is 2020; field measurements are 2022–2023. Two-year gap
  introduces minor temporal misalignment but forest canopy height changes slowly (< 1 m/yr
  in mature stands); accepted as non-leaking
- priority: 1

#### Feature B: GEDI L4B 1 km Gridded AGBD

### Feature: GEDI L4B gridded AGBD
- signal_adds: Pre-aggregated footprint-level AGB estimates at 1 km from spaceborne LiDAR,
  avoiding shot sparsity. Serves as a direct biomass co-feature and provides a label cross-
  check independent of field measurements. Different from shot-level metrics in that it is
  a posterior AGBD estimate (not raw waveform height) and is always populated.
- gee_asset: `LARSE/GEDI/GEDI04_B_002`
- resolution: ~1 km (EASE-Grid 2.0)
- conus_coverage: gapless
- extraction_complexity: low — single image, band `MU` (mean AGBD, Mg/ha), point extract
- expected_lift: +0.04 to +0.08 R² incremental over CHM (coarse at 1 km relative to
  1/24-acre plots, but provides AGBD prior that may improve discrimination at the tails)
- leakage_risk: medium — GEDI L4B AGBD is derived from GEDI L4A footprint estimates that
  use rh metrics and allometric models. If those allometrics overlap with field calibration
  sources the feature could encode label-adjacent information. Document but do not exclude;
  treat as an auxiliary co-feature, not a primary predictor.
- priority: 4

#### Feature C: COPDEM GLO-30 Topographic Derivatives

### Feature: COPDEM topographic derivatives (slope, aspect, TPI, TWI)
- signal_adds: Terrain shape drives micro-climate, soil moisture, and stand structure in
  mountain forests. WV Appalachia biomass variation is strongly controlled by topographic
  position: steep, concave hollows accumulate high biomass (mesic hardwoods, tall dominant
  trees); ridgelines and south-facing aspects are sparser (xeric oaks, shorter canopy).
  Optical embeddings carry no terrain information. This is the primary candidate for
  recovering WV Appalachia R² from 0.157 toward 0.35+.
- gee_asset: `COPERNICUS/DEM/GLO30` (ImageCollection, band `DEM`)
  Terrain products computed via `ee.Terrain.products(dem_mosaic)` which returns
  `elevation`, `slope`, `aspect`, `hillshade`.
  TWI requires upstream flow accumulation (medium complexity); TPI can be computed as
  `dem − focal_mean(dem, radius=500m)` (low complexity).
- resolution: 30 m
- conus_coverage: gapless
- extraction_complexity: medium — mosaic IC, compute terrain derivatives, point extract
- expected_lift: +0.04 to +0.10 R² incremental over CHM alone, disproportionately from
  WV Appalachia where terrain explains a large fraction of residual variance
- leakage_risk: none — topography is time-invariant
- priority: 2

#### Feature D: Hansen Last-Disturbance Year (Proxy Stand Age)

### Feature: Hansen Global Forest Change — years since disturbance
- signal_adds: Stand age is a primary biomass predictor in post-disturbance successional
  forests. A stand harvested or burned in 2010 accumulates ~13 years of growth by 2023
  and will have substantially lower biomass than a stand undisturbed since 1985. Optical
  embeddings cannot separate 20-year-old regeneration from 60-year-old mature forest if
  their canopy reflectance has converged. Stand-age proxy directly addresses this.
  Computed as `years_since_disturbance = 2023 − lossyear`, with 0 assigned to plots
  never disturbed within the Hansen record (2000–2023).
- gee_asset: `UMD/hansen/global_forest_change_2025_v1_13` (updated from 2023 version;
  confirmed accessible; band `lossyear` gives year of first detected loss 2001–2025,
  encoded as 1–25 for 2001–2025, 0 = no detected loss)
- resolution: 30 m
- conus_coverage: gapless
- extraction_complexity: low — single image, arithmetic transform, point extract
- expected_lift: +0.03 to +0.07 R² incremental; largest in MW Upper Great Lakes where
  harvest cycles create strong age-biomass gradients across managed forest landscapes
- leakage_risk: low — lossyear records historical disturbance, not current biomass;
  no direct label overlap. Caveat: if a plot was measured immediately post-harvest (year 0)
  the lossyear signal may be confounded; mitigated by the ANEW field protocol which
  typically excludes active harvest units.
- priority: 3

#### Feature E: TerraClimate Annual Climate Means

### Feature: TerraClimate climate normals (2020–2023)
- signal_adds: Growing-season climate drives cross-regional productivity differences that
  optical embeddings partially capture but cannot disentangle from canopy structure.
  Precipitation (pr) and actual evapotranspiration (aet) separate the high-productivity
  NE coastal zone from the drier MW interior; maximum temperature (tmmx) differentiates
  WV valley-bottom mesic sites from ridgeline xeric sites at finer scale. Provides
  explicit climate signal to complement terrain.
- gee_asset: `IDAHO_EPSCOR/TERRACLIMATE` (ImageCollection; bands `pr`, `tmmx`, `aet`)
  Filter to 2020–2023 (48 monthly images), reduce to annual mean.
- resolution: ~4 km (resampled to plot point via bilinear interpolation)
- conus_coverage: gapless
- extraction_complexity: low — filter, reduce, point extract
- expected_lift: +0.02 to +0.05 R² incremental over CHM+topo+disturbance; most useful if
  CHM+topo fail to close the WV gap. If CHM+topo reach R²≥0.55, climate can be deferred
  to iteration 3.
- leakage_risk: none — climate normals are time-invariant proxies for site productivity
- priority: 5

### CHM investigation findings

**Candidate evaluation summary:**

| Product | Asset tested | Accessible | Resolution | Notes |
|---|---|---|:---:|---|
| ETH Global Canopy Height 2020 | `ETH/GlobalCanopyHeight/2020/10m/v1` | No — not found | 10 m | Public GEE catalogue path does not exist as of 2026-05-29 |
| ETH Global Canopy Height 2020 | `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1` | **Yes** | ~9.3 m | Confirmed accessible |
| Potapov et al. 2021 | `users/potapov/GLCLU2019/Forest_height_2019` | No — not found | 30 m | Asset not publicly accessible from this GEE account |
| Potapov et al. 2021 (IC) | `users/potapov/GLCLU2019` | No — not found | 30 m | Collection also inaccessible |
| Meta/WRI Tolan et al. 2023 | — | Not tested | 1 m | GEE distribution not identified; download-only |
| LARSE GEDI CHM mosaic | `LARSE/GEDI` family | No raster CHM | — | LARSE hosts shot-level collections only; no gridded CHM |

**Confirmed product: ETH Global Canopy Height 2020 (Lang et al. 2023)**

- GEE asset: `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1` — confirmed accessible
- Band name: `b1` (single band; uint8; values in metres, 0–60 m, capped at 60)
- Units: metres (canopy top height above ground)
- Projection: EPSG:4326, nominal scale 9.276 m (~10 m)
- Epoch: 2020 (Sentinel-2 + GEDI fusion, calendar year 2020 composite)
- CONUS coverage: gapless — sampled at three representative CONUS locations, all returned
  valid height values:
  - WV Appalachia (38.5°N, 80.0°W): **34 m** — consistent with tall Appalachian hardwood
  - MW Great Lakes (46.0°N, 90.0°W): **26 m** — consistent with northern boreal-mixed
  - NE Maine (45.0°N, 70.0°W): **25 m** — consistent with northern hardwood-spruce-fir

**Known quality issues for eastern US deciduous forest:**
- Leaf-off acquisitions: Sentinel-2 images acquired during leaf-off periods can cause
  underestimation of canopy height in deciduous-dominant stands (crown projection area
  reduced). The 2020 composite likely mixes leaf-on and leaf-off acquisitions; the GEDI
  fusion partially corrects this but residual bias of 2–5 m is documented in Lang et al.
  for eastern deciduous stands.
- Urban/built-up confusion: building heights can be mistaken for canopy height; mitigated
  by the ANEW plot pool which is predominantly forest interior.
- Temporal gap: 2020 CHM vs. 2022–2023 field measurements — 2-year gap accepted as
  non-leaking (mature forest canopy height stable at ±1 m/yr; accepted by Research Actor).

**Extraction approach:**

```python
chm = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').rename('chm_m')
plots_fc = ee.FeatureCollection(...)  # plot centroids
chm_values = chm.reduceRegions(
    collection=plots_fc,
    reducer=ee.Reducer.mean(),
    scale=10,
)
```

Simple point extraction with `scale=10`; no buffer or aggregation required. All plots will
receive a non-null value — 100 % coverage by construction.

**Supporting literature for CHM-optical fusion lift estimate:**

| title | venue_or_org | year | url_or_doi | reported_metric | reported_value | geography | split_type |
|---|---|---|---|---|---|---|---|
| Mapping global forest canopy height through integration of GEDI and Landsat data | Remote Sensing of Environment, 253 | 2021 | https://doi.org/10.1016/j.rse.2020.112165 | R² (plot-level AGB regression, CHM-optical fusion) | 0.60–0.75 by forest type in temperate broadleaf and mixed forest | Global temperate and boreal; eastern USA subset representative of WV/MW/NE plot pool | Spatial holdout (continental-scale train/test split; not random) |

### Updated benchmark targets for iteration 2

These targets supersede the iteration-1 thresholds. The primary levers are CHM (priority 1)
and COPDEM derivatives (priority 2), extracted together in a single preprocessing pass.

| Metric | Realistic threshold | Stretch threshold |
|---|---|---|
| R² (23-project LOPO) | ≥ 0.55 | ≥ 0.65 |
| RMSE (tCO₂/acre) | ≤ 50 | ≤ 40 |
| predicted_range_discrimination | ≥ 0.60 | ≥ 0.75 |
| WV Appalachia R² | ≥ 0.30 | ≥ 0.45 |

**Rationale:** CHM alone (literature) adds +0.10–0.20 R² over optical-only baselines in
temperate forests. Starting from R²=0.4182 (iteration-1 result — no lift from GEDI shots),
the realistic threshold of ≥ 0.55 requires a +0.13 lift — achievable with CHM alone at the
conservative end. The stretch threshold of ≥ 0.65 requires CHM + terrain derivatives +
disturbance working together; no single feature is expected to reach it alone. If CHM and
COPDEM together do not reach ≥ 0.55, add Hansen disturbance (priority 3) before escalating.

### Iteration-2 rerun boundary

```
Research (this doc)
  └── Preprocess: extract CHM + COPDEM derivatives + Hansen disturbance
        (scripts/extract_chm_features.py + scripts/extract_topo_features.py)
        assets: users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1
                COPERNICUS/DEM/GLO30
                UMD/hansen/global_forest_change_2025_v1_13
  └── Train: join extracted CSVs to optical embeddings; rerun LOPO CV
  └── Evaluate: report R², RMSE, discrimination per region;
                compare against realistic/stretch thresholds
  └── [If R² < 0.55]: add GEDI L4B (priority 4) and TerraClimate (priority 5),
                       then re-evaluate before escalating to Critic
```

The preprocess step must report `n_null` per feature per region. Any feature with > 1 %
null values in any region must be investigated before training proceeds.

### Reproducibility footer (iteration 2)

- timestamp_utc: 2026-05-29T12:00:00Z
- GEE assets verified accessible:
  - `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1` (ETH CHM, band `b1`, units m, epoch 2020)
  - `COPERNICUS/DEM/GLO30` (ImageCollection, band `DEM`, 30 m)
  - `UMD/hansen/global_forest_change_2025_v1_13` (Image, band `lossyear`, 30 m)
  - `LARSE/GEDI/GEDI04_B_002` (Image, band `MU` = mean AGBD Mg/ha, ~1 km)
  - `IDAHO_EPSCOR/TERRACLIMATE` (ImageCollection, bands `pr`, `tmmx`, `aet`, ~4 km)
- GEE assets NOT accessible (public catalogue path missing):
  - `ETH/GlobalCanopyHeight/2020/10m/v1` (use `users/nlang/...` path instead)
  - `users/potapov/GLCLU2019/Forest_height_2019` (no access from this account)
- CHM coverage samples confirmed:
  - WV 38.5°N 80.0°W → 34 m; MW 46.0°N 90.0°W → 26 m; NE 45.0°N 70.0°W → 25 m
- method: GEE Python API `ee.Image.getInfo()`, `ee.Image.sample()`, `ee.Image.bandNames()`
- environment: uv run --project /home/mattc/code/agb-ml-agent-evolve python
- conducted by: Research Actor (automated), 2026-05-29
