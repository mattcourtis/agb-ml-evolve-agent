# Investigation Plan: Sentinel-1 SAR Features with Speckle Filtering

**Motivated by:** PALSAR-2 L-band SAR added only ~0.02 R² in the prior tf-deep-landcover
investigation. Sentinel-1 C-band offers higher revisit (6–12 days over CONUS), multi-temporal
averaging for speckle suppression, and seasonal compositing — all of which differ materially
from the single-date PALSAR-2 approach that was tested.

---

## 1. Why Sentinel-1 Might (or Might Not) Help

### The case for it
- **Multi-temporal speckle filtering is free**: 6–12 day revisit means 30–60 images/year
  over CONUS → averaging eliminates speckle without spatial resolution loss. This is
  fundamentally different from single-date PALSAR-2.
- **Seasonal signal**: VH backscatter changes with canopy leaf-on/leaf-off state in
  deciduous forests. The ratio of summer-VH to winter-VH separates deciduous from
  evergreen and encodes phenology information absent from annual composites.
- **Texture and heterogeneity**: temporal standard deviation of VH is a proxy for
  wind-driven canopy motion and stand structural complexity — potentially related to
  stand age and biomass accumulation.
- **C-band penetration vs optical**: C-band penetrates partial cloud and thin canopy,
  adding signal the optical embeddings may miss during leaf-off.

### The case against it
- **C-band saturation**: Sentinel-1 C-band (~5.4 GHz) saturates at forest AGB of
  approximately 50–100 Mg/ha (~30–60 tCO₂/acre) in closed-canopy temperate forest.
  The ANEW dataset spans 0–500+ tCO₂/acre; C-band has zero sensitivity above ~60
  tCO₂/acre. This is worse than L-band PALSAR-2 (saturates ~100–200 Mg/ha).
- **PALSAR-2 was already uninformative**: If L-band SAR (deeper penetration, less
  saturation) added only 0.02 R², it is unlikely that shallower C-band SAR adds more.
- **The embeddings may already capture SAR-correlated signal**: The AEF optical embeddings
  are trained on Sentinel-2, whose spectral bands correlate with canopy structure in
  ways that overlap with SAR backscatter.

### Net assessment
**Seasonal and temporal SAR statistics** (growing-season vs dormant-season difference,
temporal variance) are the most promising angles — they are NOT correlated with optical
embeddings (which are single-date or seasonally composited) and target a different
biophysical signal than raw backscatter magnitude.

---

## 2. Features to Extract

### 2a. Multi-temporal mean backscatter (annual composite)
The foundation. Average over all good-quality acquisitions in 2022–2023 to eliminate
speckle without spatial blurring.

| Column | Band | Composite | Units |
|---|---|---|---|
| `sar_vv_mean` | VV | Annual mean, 2022–2023 | dB |
| `sar_vh_mean` | VH | Annual mean, 2022–2023 | dB |
| `sar_vhvv_ratio` | VH − VV | Annual mean difference | dB |

### 2b. Seasonal statistics (the most novel signal)
Split the year into growing season (May–Oct, leaf-on) and dormant season (Nov–Apr, leaf-off)
composites. The **leaf-on/leaf-off VH difference** encodes deciduous fraction and canopy
volume — absent from annual mean and from optical embeddings.

| Column | Derivation | Why |
|---|---|---|
| `sar_vh_leafon` | Mean VH, May–Oct | Growing season backscatter |
| `sar_vh_leafoff` | Mean VH, Nov–Apr | Dormant season backscatter |
| `sar_vh_seasonal_diff` | leaf-on VH − leaf-off VH | Deciduous signal; related to LAI seasonality |
| `sar_vh_seasonal_ratio` | leaf-on VH / leaf-off VH | Normalised deciduousness |

### 2c. Temporal variance (stand structure proxy)
Standard deviation across all individual acquisitions (not aggregated by season). High
temporal variance = canopy moves in wind = structurally complex forest. Low variance =
dense, stiff canopy (or bare ground).

| Column | Derivation | Why |
|---|---|---|
| `sar_vh_std` | StdDev of VH across all passes, 2022–2023 | Stand structural complexity proxy |
| `sar_vv_std` | StdDev of VV across all passes, 2022–2023 | — |

### 2d. OPTIONAL: Coherence (if available)
Sentinel-1 interferometric coherence (12-day repeat) measures canopy stability. Low
coherence = dense moving canopy; high coherence = bare or static surface. Available via
Copernicus DEM / ASF but NOT directly in GEE as a standard asset. Skip for initial
investigation unless a GEE community asset is available.

---

## 3. Speckle Filtering Strategy

This is the key design decision. Three approaches to compare:

### Option A: Multi-temporal averaging (recommended first)
Average over all acquisitions within the time window. Each pixel gets a mean from
30–60 passes over 2 years. The law of large numbers suppresses speckle without any
spatial blurring — preserves 10 m resolution.

```python
s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterDate('2022-01-01', '2024-01-01')
      .filterBounds(aoi)
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .select(['VH', 'VV']))
      
s1_mean = s1.mean()  # N=30–60 images; speckle ~1/sqrt(N) of single-image speckle
```

**Expected speckle reduction**: With N=40 images, equivalent number of looks ≈ 40× single
image → speckle standard deviation ≈ 1/sqrt(40) ≈ 0.16 of original → ENL ≈ 40.

### Option B: Spatial Lee filter on single composite
Apply adaptive Lee filter (5×5 or 7×7 kernel) to a single-date or short-window
composite. Reduces speckle spatially but blurs resolution. Available in GEE via
`ee.Image.convolve()`.

**Only relevant if multi-temporal approach is not available or if very short time windows
are needed (e.g., change detection).**

### Option C: Refined Lee / IDAN (not in GEE natively)
More sophisticated spatial filters. Require custom implementation or external tooling
(e.g., pyroSAR, SNAP). Higher implementation cost; skip for initial investigation.

**Recommendation: Use Option A (multi-temporal averaging) exclusively for this
experiment.** It is physically optimal for our use case (annual AGB estimation, not
change detection), has no implementation overhead, and preserves spatial resolution.

---

## 4. Plot-Scale Considerations

The buffer/Gaussian analysis applies here too:

- Sentinel-1 native resolution: 10 m (IW mode, range-Doppler terrain corrected)
- ANEW plot radius: ~7.3 m → same sub-pixel scale challenge as CHM
- The multi-temporal averaging already acts as temporal speckle filtering, but
  spatial averaging is still needed for plot-scale extraction

**Recommended extraction:**
Apply a Gaussian kernel (σ=15 m) to the Sentinel-1 composite images before point
extraction — same approach as the CHM (also 10 m). The Gaussian extraction experiment
showed no aggregate benefit for CHM, but for SAR the plot-scale signal is noisier
(speckle residual even after multi-temporal), so spatial smoothing is more justified.

**However** — given the Gaussian extraction experiment result (−0.004 R²), start with
point centroid extraction to establish the baseline signal first. If SAR shows meaningful
lift, then compare Gaussian vs centroid for SAR specifically.

---

## 5. Orbit Direction and Pass Considerations

Sentinel-1 has ascending and descending orbits over CONUS. The backscatter from the
two directions differs due to look-angle effects on terrain. For AGB estimation:

- **Mix ascending + descending** in the multi-temporal average: reduces look-angle
  artefacts and increases effective number of looks
- **Or: separate ascending and descending** as separate features — the geometry
  difference may encode terrain-structure interactions

**Recommendation**: Mix both orbits in the annual composite. Simplest approach; enough
images (N>>30) to average out geometry effects.

---

## 6. Experiment Design

### 6a. Feature extraction script
`scripts/extract_sentinel1_features.py`

Produces: `preprocessing/sentinel1_features.csv` (4,646 rows × 9 columns):
`row_key, sar_vv_mean, sar_vh_mean, sar_vhvv_ratio, sar_vh_leafon, sar_vh_leafoff, sar_vh_seasonal_diff, sar_vh_seasonal_ratio, sar_vh_std`

Extraction: `reduceRegions` at scale=10, `ee.Reducer.mean()` at point centroid.

### 6b. Feature set comparison
Merge with `features_iter3.parquet` (existing best feature set) to produce
`preprocessing/features_sar.parquet` (88 + 9 = 97 cols with `sar_` prefix).

Trainer prefix list: add `"sar_"`.

Run LOPO CV on:
1. `features_iter3.parquet` (reference, R²=0.4274)
2. `features_sar.parquet` (all features + SAR)
3. SAR-only ablation: `sar_*` + `emb_*` only (no other co-features)

### 6c. SHAP attribution
Report SHAP importance of SAR group vs embedding group vs other co-features. This tells
us whether SAR is contributing genuinely independent signal or is correlated with the
optical features.

---

## 7. Key Questions to Answer

| Question | Experiment | Accept threshold |
|---|---|---|
| Does multi-temporal Sentinel-1 add lift over baseline? | Full feature set + SAR vs baseline | ΔR² > +0.005 |
| Is the seasonal VH difference (leaf-on/off) the most informative SAR feature? | SHAP importance ranking within SAR group | `sar_vh_seasonal_diff` SHAP > `sar_vv_mean` SHAP |
| Does temporal variance (sar_vh_std) encode stand structure independently? | Partial correlation with CHM | Low corr with chm_m = independent signal |
| Is SAR signal additive with GEDI-derived CHM or redundant? | SAR-only SHAP in model with + without CHM | If SAR SHAP drops when CHM present → correlated |
| Does C-band saturation limit the Q5 improvement? | Per-quintile bias with/without SAR | If Q5 bias unchanged → C-band saturated |

---

## 8. Expected Outcome and Risk

**Expected best case:** +0.005–0.015 R² from seasonal SAR features. Seasonal VH
difference encodes deciduousness and canopy volume dynamics, which are not captured
by annual optical embeddings or static structural features (CHM, topo).

**Expected likely case:** +0.000–0.005 R², similar to CHM/topo (+0.010). C-band
saturation limits sensitivity at high biomass; signal is noisier than GEDI-based height.

**Risk:** If C-band SAR backscatter is already implicit in the AEF optical embeddings
(optical–SAR correlation in heavily forested CONUS), no lift is expected, consistent
with the PALSAR-2 result (+0.02 R² from L-band).

**Decision rule:** If aggregate LOPO R² lift is ≤ +0.002, SAR features are not worth
the extraction and maintenance overhead. If > +0.005, include in the production feature
set. Focus on seasonal features (`sar_vh_seasonal_diff`) — if even this feature shows
zero SHAP importance, the C-band saturation hypothesis is confirmed and further SAR
investigation is deprioritised.

---

## 9. Dependencies

- GEE `COPERNICUS/S1_GRD` — confirmed accessible (in source registry)
- No new Python packages required
- Estimated extraction time: ~5–10 min (4,646 plots, multi-temporal reduction in GEE)
- Follow-on: if SAR shows lift, consider also investigating Sentinel-1 coherence
  (requires external data or ESA API)

---

*Plan author: Orchestrator | Status: DRAFT — ready for implementation when approved*
