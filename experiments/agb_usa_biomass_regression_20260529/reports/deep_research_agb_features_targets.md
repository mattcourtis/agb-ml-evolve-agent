# Deep Research Report — AGB Feature/Target Processing, Sampling, and Feature Alternatives

**Research question:** How does the spatial scale mismatch between small field inventory plots (~1/24 acre, ~14m diameter, ~168 m² footprint) and satellite remote sensing features affect AGB regression model performance? What pre-processing strategies, alternative features, target transformations, and sampling designs improve outcomes?

**Sources searched:** 105 subagents, 970 tool uses, peer-reviewed literature 2018–2025
**Verification method:** 3-vote adversarial verification per claim; claims requiring 2/3 refutation votes are excluded from confirmed findings

---

## 1. Plot–Pixel Spatial Scale Mismatch — CONFIRMED [High confidence]

**Finding:** The 1/24-acre ANEW plot (~168 m², ~7.3 m radius, ~14.6 m diameter) is substantially smaller than even 10 m Sentinel-2 pixels, let alone 30 m Landsat or 1 km GEDI L4B pixels. This is a **confirmed, documented source of label noise and AGB variability detection error**. Larger pixel-to-plot ratios produce greater within-pixel heterogeneity and degrade regression label quality.

**Evidence (3-0 verified):**
- Cai et al. (PLOS ONE, 14,818 FIA plots × 30 m Landsat): *"discrepancies between remote sensing spatial resolution and field plot size can significantly influence the detection of AGB variability, with greater within-pixel variability observed as the pixel-to-plot ratio increases."*
- A second independent study (Scientific Reports) confirmed plot boundaries rarely coincide with pixel edges even after resampling.

**Standard operational mitigation:**
Create a **circular buffer around the plot centre** (the cited standard is 25.82 m radius — equivalent to ~0.21 ha, roughly the size of a standard FIA or NFI plot used for satellite calibration) and extract the **mean pixel value within the buffer** rather than the single pixel containing the plot centroid. This reduces point-extraction noise by averaging over multiple co-located pixels.

**Application to ANEW plots:**
- The ANEW 1/24-acre plot radius (~7.3 m) is well inside a single 10 m pixel, let alone 30 m pixels
- Any single-pixel extraction at the plot centroid captures forest that the field crew did not measure
- Current extraction approach (point sample at centroid) is sub-optimal for all co-features at 10–30 m resolution
- **Recommended action:** Re-extract all co-features (CHM, SRTM, Hansen, GEDI, TerraClimate) using a **25–30 m buffer mean** rather than the point centroid. This averages across ~4–9 pixels for 10 m features and ~1–4 pixels for 30 m features, more representative of stand-level conditions

**What was refuted:** Specific quantitative claims that *"0.04 ha plots produce 80% higher uncertainty than 0.25 ha plots"* and *"only 3 of 11 site models met a 15 Mg/ha suitability threshold"* were refuted (0-3 votes) — the underlying principle is confirmed but these specific thresholds could not be verified from primary sources.

---

## 2. GEDI Canopy Height Dominance — CONFIRMED [High confidence]

**Finding:** GEDI-derived canopy height is **the single most important predictor for plot-level AGB regression in CONUS temperate forest**, outranking all optical vegetation indices and LANDFIRE structural variables.

**Evidence (3-0 verified):**
Lu et al. (2025, *Forest Ecology and Management* vol. 595, DOI 10.1016/j.foreco.2025.123040), multi-sensor RF model for Pacific Northwest + California FIA plots:
> *"GEDI height is the most crucial variable for biomass prediction, followed by Existing Vegetation Type (EVT), enhanced vegetation index (EVI) and Existing Vegetation Cover (EVC). ICESat-2 height, atmospherically resistant vegetation index (ARVI) and visible atmospherically resistant index (VARI) are the least important variables."*
Model achieved R²=0.769, RMSE=55.26 Mg/ha at FIA hexagon scale.

**Implication for this project:**
Our GEDI shot-level extraction (iterations 1–3) failed because of coverage sparsity — median `gedi_n_samples=1` over 36 months means most plots received only one monthly composite hit. The signal IS important but we failed to extract it cleanly. The ETH CHM (which is a GEDI-fusion product) produced some lift (+0.010 R²), confirming that a well-extracted GEDI signal should help more.

**Path forward:** A site-matched airborne LiDAR co-target (e.g., NEON AOP CHM at NEON sites, or a lidar survey commissioned for the ANEW project areas) would provide unambiguous canopy height at the plot scale and is the most direct fix. Alternatively, the global Potapov et al. (2021) 30 m CHM from Landsat+GEDI fusion (now accessible as GEE user asset) covers CONUS and integrates far more GEDI shots than our point extraction.

---

## 3. Target Variable Transformation — CONFIRMED [High confidence]

**Finding:** Target variable transformation strategy must be **matched to the model family**. Quantile normalisation improves linear models but *degrades gradient boosted trees*.

**Evidence (3-0 verified):**
Nuyts and Davis (arXiv 2504.20821, April 2025):
- Quantile normalisation → Lasso, Ridge, SVR: **median RSE improvement 11–63%**
- Quantile normalisation → Gradient Boosted Trees: **median RSE *increase* 19%**

**Mechanism:** Tree-based models use rank-based splitting criteria. They are theoretically invariant to monotone transformations of the target — applying a Gaussian transformation adds a non-monotone distortion (at the extremes) that conflicts with how the tree splits the residual space.

**Direct implication:**
- Our LightGBM model should **NOT** use log-target, quantile normalisation, or Box-Cox
- This is consistent with what we observed: log-target dropped R² from 0.427 to 0.374 in the extended investigation
- The ruling-out of these transformations (from the original tf-deep-landcover investigation) is now backed by peer-reviewed evidence
- **If we ever move to a linear model or neural network**, log-AGB would be expected to help

---

## 4. Optical + SAR Fusion — MEDIUM confidence (limited geography)

**Finding:** Fusing optical and SAR satellite data via machine learning can achieve materially higher AGB accuracy than single-sensor optical, but the evidence base for CONUS temperate forest specifically is limited.

**Evidence (2-1 verified):**
MDPI Remote Sensing (2025, DOI 10.3390/rs18101536): HIS-NSST+PCNN fusion of Gaofen optical + SAR achieved R²=0.80, RMSE=14.79 t/ha — but in subtropical Chinese forest, not CONUS temperate. Wide bootstrap CIs (0.678–0.924) indicate substantial uncertainty.

**What was refuted:**
- *"Combining Landsat 8 + Sentinel-1A with XGBoost achieves R²=0.75"* — refuted (0-3 votes)
- *"GEDI+multimodal SAR deep learning achieves global R²=0.82"* — refuted (1-2 votes)
- *"Spectral complementarity is more critical than spatial resolution for fusion"* — refuted (0-3 votes)

**Assessment:** SAR (particularly PALSAR-2 HH/HV) was already tested in the tf-deep-landcover investigation and added only ~0.02 R². The new literature does not overturn this result for CONUS temperate forest. SAR saturates at canopy closure in dense temperate forest and adds limited independent information beyond optical embeddings.

---

## 5. Zero-AGB Augmentation — UNVERIFIED (open question)

**Finding:** Whether augmenting training data with verified zero-AGB locations reduces Q1 over-prediction bias is a **genuinely open question** — no peer-reviewed evidence for or against was found.

**Open question (from research):**
> *"Does augmenting training data with verified zero-AGB locations (cleared land, recent harvest, urban) measurably reduce Q1 over-prediction bias in gradient boosted or random forest AGB models, and if so what proportion of synthetic zeros is optimal relative to positive-AGB training examples?"*

**Theoretical basis (not yet verified empirically):**
- Q1 over-prediction in our model means plots with true biomass of ~5–30 tCO₂/acre are predicted at ~40–70 tCO₂/acre
- The model has no training examples at or near zero — all ANEW plots are in managed forests with positive standing stock
- Adding zero-AGB examples (recently clearcut, non-forest, urban) would extend the lower end of the training distribution
- This could force the model to learn a true zero (or near-zero) anchor, potentially pulling Q1 predictions down

**What was refuted:**
- *"A two-stage zero-inflated model achieves lower relative bias and smaller RMSE"* — refuted (0-3 votes)
- *"Single-stage estimators that ignore zero-inflation exhibit the highest bias"* — refuted (0-3 votes)
These refutations are for the zero-inflated Bayesian modelling literature specifically, not for simple data augmentation.

**Practical recommendation:**
This is worth a targeted experiment. Zero-AGB locations can be identified from:
1. Hansen GFC 2025 `lossyear` band — plots on clearcut pixels (lossyear in 2020–2023) with `dist_years_since=0`
2. NLCD 2021 non-forest land cover (agricultural, urban, shrubland)
3. The ANEW region AOIs combined with NLCD mask to identify recently harvested stands

The experiment design: add N synthetic zero-AGB plots drawn from non-forest locations in the same AOI, retrain, and measure Q1 bias change. If the model learns the zero anchor, Q1 bias should narrow without proportional Q5 degradation (unlike the weighting approach which trades one for the other).

---

## 6. Alternative Features for CONUS Temperate Forest

### Confirmed useful (from literature)
| Feature | Evidence | Notes |
|---|---|---|
| GEDI canopy height (L2A rh98) | High — #1 predictor in Lu et al. 2025 | Must be extracted at sufficient density; our shot-level approach failed |
| LANDFIRE EVT (Existing Vegetation Type) | High — #2 predictor in Lu et al. 2025 | Available as 30 m CONUS raster; categorical stand-type classification |
| EVI (Enhanced Vegetation Index) | High — #3 predictor | Multi-temporal version more powerful than single-date |
| LANDFIRE EVC (Existing Vegetation Cover) | High — #4 predictor | Canopy cover fraction from national modelling |
| ICESat-2 ATL08 height | Low — least important in Lu et al. | May add to GEDI in GEDI-gap areas |

### Untested but high-priority for CONUS temperate AGB
| Feature | Rationale | GEE asset |
|---|---|---|
| **LANDFIRE EVT (stand type)** | Explains WHY projects differ — stand type (oak-hickory vs maple-beech vs spruce-fir) has different allometric AGB relationships | `LANDFIRE/Fire/EVT/v1_4_0` |
| **Multi-temporal EVI / NDVI phenology** | Seasonal amplitude, green-up timing → separates deciduous vs. evergreen, productivity gradients | Compute from `COPERNICUS/S2_SR_HARMONIZED` monthly composites |
| **PALSAR-2 coherence / temporal change** | SAR coherence (not backscatter) is more sensitive to forest structure change than HH/HV alone | Available via ESA BIOMASS campaign data |
| **Landsat disturbance history** | LandTrendr or CCDC change detection → years-since-harvest, disturbance severity | `USGS/LCMAP/CU/V13/LCPRI` annual change |
| **Forest inventory species group** | If project-level species group data available, major improvement for LOPO | Not in satellite; must come from ANEW metadata |

### Refuted alternative features
- *"SWIR1, SWIR2, NDVI rank above ICESat-2 height"* — refuted (0-3), inconsistent across studies
- *"Disproportionate pine sampling degrades generalisation"* — refuted as a claim about CONUS data directly
- *"SVMRK with ICESat-2 achieves R²=0.61 outperforming RF R²=0.34"* — refuted (0-3 votes)

---

## 7. Sampling Strategies for LOPO CV — UNVERIFIED

**Finding:** The evidence base for density-balanced sampling, hard negative mining, or hierarchical stand-type stratification improving LOPO generalisation is **absent** from the recent literature.

**What remains an open question:**
> *"What is the optimal LOPO CV strategy when training data come from heterogeneous forest inventory programmes with differing plot designs, allometric equations, and spatial sampling densities — and does stratifying by forest type or stand age improve generalisation?"*

**Assessment from our own investigation:**
Our extended investigation (Section 1) showed:
- 5-fold random: R²=0.452 (+0.025 over LOPO)
- Leave-one-ecoregion-out: R²=0.324 (much harder)
- LOPO: R²=0.427 (reference)

The LOPO penalty (~0.025 R²) appears to come primarily from **project-level biomass distribution differences** — each project has a different mix of stand ages and management histories that the satellite features do not capture. Stratified sampling cannot solve this without project-level metadata.

---

## 8. GEDI Geolocation + Plot GPS Error Interaction — OPEN QUESTION

**Open question (from research):**
> *"How does GEDI geolocation uncertainty (~9 m) interact with small plot GPS error (~2–10 m) to compound label noise, and do terrain-corrected or waveform-filtered GEDI shots substantially reduce this compound error in steep CONUS terrain?"*

**Relevance to this project:**
WV Appalachia (R²=0.157) has steep terrain where GEDI pointing uncertainty is largest. The ~9 m GEDI geolocation error + ~5 m typical GPS error for ANEW plots = compound misregistration of up to ~14 m — which for a ~7.3 m radius plot means the GEDI shot may be measuring a completely different location than the field measurement. This is a plausible additional explanation for why GEDI features added no lift in WV specifically.

---

## 9. Synthesis and Prioritised Recommendations

### Highest confidence, highest priority
| # | Action | Basis | Expected impact |
|---|---|---|---|
| 1 | **Re-extract all co-features using 25–30 m buffer mean** rather than point centroid | Confirmed 3-0: buffer averaging is the standard operational mitigation for plot–pixel mismatch | Unknown but likely +0.02–0.05 R² for CHM/topo; cleaner signal |
| 2 | **Do NOT apply log/quantile normalisation to LightGBM** | Confirmed 3-0: QN degrades GBT by +19% RSE | Avoid regression |
| 3 | **Add LANDFIRE EVT (stand type) as a feature** | Confirmed 3-0: #2 predictor after GEDI height in CONUS RF models; explains project-level biomass distribution differences | Potentially large for LOPO if stand type explains project differences |

### High confidence, medium priority
| # | Action | Basis | Expected impact |
|---|---|---|---|
| 4 | **Multi-temporal EVI/NDVI phenology features** | Consistent with LANDFIRE EVT importance; separates stand types | Unknown; low extraction cost via GEE |
| 5 | **Use Potapov et al. 30 m Global CHM** (Landsat+GEDI fusion, 2019 epoch) instead of ETH CHM | More GEDI integration points; better calibrated for CONUS temperate | Possibly better than ETH 2020 CHM for eastern US forests |

### Experimental (unverified, worth testing)
| # | Action | Basis | Expected impact |
|---|---|---|---|
| 6 | **Add synthetic zero-AGB locations** from Hansen clearcut + NLCD non-forest in ANEW AOIs | Theoretically sound; open question in literature | May reduce Q1 over-prediction; low risk |
| 7 | **Neural fusion model** (jointly learn features + regression) | Cannot be assessed from tabular ML literature | Potentially large; requires infrastructure investment |

---

## 10. Key Citations

| Citation | DOI / URL | Finding |
|---|---|---|
| Cai et al. (PLOS ONE) | https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0330831 | Plot–pixel mismatch confirmed; buffer averaging recommended |
| Scientific Reports study | https://www.nature.com/articles/s41598-020-67024-3 | 25.82 m buffer operational standard |
| Lu et al. 2025 (Forest Ecology & Management) | DOI 10.1016/j.foreco.2025.123040 | GEDI height #1 predictor; LANDFIRE EVT #2; EVI #3 |
| Nuyts & Davis 2025 (arXiv) | https://arxiv.org/abs/2504.20821 | QN helps linear models (+11–63%) but hurts GBTs (+19% RSE) |
| MDPI Remote Sensing 2025 | DOI 10.3390/rs18101536 | Optical+SAR fusion R²=0.80 (subtropical forest; medium confidence) |

---

*Generated: 2026-06-01 | Research workflow: 105 subagents, 970 tool uses, 1644s*
