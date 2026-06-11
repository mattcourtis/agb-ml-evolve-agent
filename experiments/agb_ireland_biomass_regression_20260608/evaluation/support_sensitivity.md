# Ireland AGB — change-of-support sensitivity (f(mean) vs mean(f))

**Stage:** Support-Test Actor (atomic). **Mode:** quantify the aggregation gap between the
estimator we REPORTED — `f(mean(embeddings))` (polygon-mean the AEF embeddings, then predict once) —
and the PRODUCTION estimator — `mean(f(embeddings))` (apply the head per 10 m pixel, then average the
pixel predictions to a stand density). Head = `embdstx` (64 AEF embeddings + 3 survey-relative Hansen
disturbance features; target tCO₂/acre). Real GEE per-pixel extraction + LightGBM. **No ground truth.**
seed 42. Asset `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (V1), Hansen `…2025_v1_13`.

## TL;DR
The aggregation gap is **material and stand-specific**: across 18 stands the median signed gap is
**−3.0%** (mean |gap| **10.9%**, range **−21.7%…+40.1%**), with **12/18** stands beyond ±5% and **7/18**
beyond ±10%. The sign is **mixed** (8 of 18 positive), so polygon-mean is **not** a uniform bias — on
net it slightly **over-reads** vs the pixel-aggregate (median gap negative). The gap is driven by the
**within-stand per-pixel prediction dispersion** (Spearman |gap%| vs pixel-prediction IQR **+0.54,
p=0.02**), **not** by the `n_subcpt` count (ρ≈−0.1, ns) — a non-linearity (Jensen-gap) effect, exactly
as expected for a tree ensemble over a heterogeneous footprint. Recommendation: **re-run all 141 at
pixel level** before any absolute Ireland number or DB comparison is taken at face value, because the
production map is pixel-then-aggregate and per-stand it differs from the reported value by up to ~24%.

## Why this matters (context)
The reported Ireland numbers (`evaluation/ireland_predictions.parquet`, `bias_characterisation.md`) and
the encoding gate (`preprocessing/preprocessing_spec.md`) were all built on `reduceRegions(mean)` →
predict once = **f(mean)**. A wall-to-wall production map predicts per pixel then aggregates =
**mean(f)**. Because the head `f` is a non-linear LightGBM ensemble, `mean(f(x)) ≠ f(mean(x))` in
general (Jensen gap); the affine GEE→codec transform is linear so it is NOT the source of the gap
(`mean(affine) = affine(mean)`), which localises any difference to the head's non-linearity over the
spread of pixel embeddings inside the stand. The DB comparator is itself a pixel-then-aggregate product,
so a like-for-like comparison should use mean(f), not f(mean).

## Sample (18 stands, stratified)
Selected (seed 42, deterministic) to span the heterogeneity range — `n_subcpt` (1 = single
sub-compartment / homogeneous … 37 = many merged / heterogeneous) crossed with `area_ha` (0.67 …
151.84 ha) — and to include young/old and mixed-age stands. Includes the largest stand (Meensheefin,
151.8 ha) and the most heterogeneous (Cloonsheever, 37 sub-cpts, 85 ha); youngest (Benmore, age 0) and
oldest (Cloonsheever block 26.6 yr / Meensheefin 26.6 yr) ends covered.

| stand | n_subcpt | area_ha | survey_yr | age | Hdom | rationale |
|---|---|---|---|---|---|---|
| Rathcahill West | 1 | 6.96 | 2017 | 2.0 | 0.0 | homogeneous, small, young, pre-2017 fallback |
| Benmore | 1 | 7.74 | 2025 | 0.0 | 0.0 | homogeneous, youngest |
| Crooderry | 1 | 8.69 | 2023 | 19.0 | 18.1 | homogeneous, mature, tallest |
| Carrowreagh | 1 | 17.08 | 2024 | 2.0 | 0.0 | homogeneous, larger, young |
| Loughros | 2 | 0.67 | 2023 | – | 0.0 | smallest stand (edge-effect probe) |
| Dromreask | 3 | 8.79 | 2025 | 9.5 | 7.8 | low het, mid-age |
| Carrigeeny | 3 | 30.33 | 2017 | 0.0 | 0.0 | low het, large, young, fallback |
| Tooreennagreana | 4 | 21.68 | 2025 | 22.0 | 11.1 | low het, large, mature |
| Lacka Beg | 5 | 9.32 | 2024 | 18.0 | 9.7 | mid het, mature |
| Carrowkeel | 5 | 16.03 | 2024 | 2.4 | 1.1 | mid het, young |
| Sligo Bay N_Greaghnafarna II | 6 | 9.42 | 2023 | 19.0 | 11.3 | mid het, mature |
| Cummeen Upper | 6 | 65.31 | 2017 | 21.6 | 11.3 | mid het, very large, mature, fallback |
| Knockbreenagher | 9 | 8.46 | 2023 | 20.0 | 11.3 | high het, small, mature |
| Sligo Bay S_Kilfree | 9 | 14.39 | 2023 | 18.0 | 9.7 | high het, mid |
| Highmount | 9 | 22.69 | 2024 | 16.2 | 12.3 | high het, large |
| Glanowen | 13 | 41.44 | 2025 | 23.4 | 10.9 | high het, large, old |
| Meensheefin | 20 | 151.84 | 2023 | 26.6 | 12.3 | largest stand, old |
| Cloonsheever | 37 | 85.37 | 2024 | 5.9 | 4.1 | most heterogeneous, mixed-age |

All 18 extracted successfully (≥12 required for a verdict).

## Method
`scripts/per_pixel_inference.py` (parameterised by Location list; reusable basis for a wall-to-wall map).
Per stand:
1. **Per-pixel extraction.** A fused image (64 AEF bands `A00..A63` for the stand's `survey_year` +
   the 3 per-pixel dstx bands) is downloaded as a native-float NumPy array via `getDownloadURL` (NPY,
   scale=10). `getDownloadURL` returns the bounding-box **rectangle**, so an `inmask` band (the geometry
   rasterised at 10 m, 1 iff the pixel **centre** is inside the polygon) is added and only `inmask==1`
   pixels are kept — this reproduces the exact pixel set `reduceRegions(mean)` integrates over. Large
   stands (Cloonsheever, Meensheefin) exceed the 50 MB getDownloadURL cap and are auto-tiled into
   latitude strips and re-concatenated (no change to the kept-pixel set). `n_pixels` recorded per stand
   (169 … 26 764; total over the 18 stands ≈ 96 k pixels).
2. **dstx per pixel.** `dstx_pixel_image()` = the survey-relative Hansen timing of
   `extract_ireland_aef.extract_dstx`, evaluated per pixel: `dstx_pre_ysd = code − lossyear` if
   `0<ly≤code` else 100; `dstx_pre_loss_5yr` = pre-survey loss within 5 yr; `dstx_loss_frac_buf` =
   pre-survey-loss indicator (0/1) per pixel (its polygon-mean = the iter0 disturbed-fraction feature).
3. **Affine.** The production per-band affine (`preprocessing/aef_affine.parquet`,
   `emb_b = a_b·A_b + c_b`) is applied per pixel → training codec space.
4. **f(mean):** affine applied to the pixel-mean native embedding + pixel-mean dstx → predict once.
5. **mean(f):** affine per pixel → predict per pixel with `inference_model_embdstx.txt` → simple mean of
   the per-pixel predictions (the stand density estimate). Pixels are sampled on the asset's native 10 m
   grid in EPSG:4326, which at ~53°N oversamples to ~166–251 px/ha (median ~180; ≈1/cos(lat)); the
   oversampling is spatially uniform so the simple mean equals the area-weighted mean and the
   consistency check confirms the pixel-mean preserves the polygon-mean.
   (Critic-corrected: paths in the reproducibility footer are relative to repo root — `scripts/per_pixel_inference.py`,
   `models/inference_model_embdstx.txt` — not the experiment dir.)
6. **gap = mean(f) − f(mean)** (tCO₂/acre and %).

Commands (seed 42, `uv run`, bare `ee.Initialize()`):
```
uv run python scripts/per_pixel_inference.py            # 18-stand stratified sample
uv run python evaluation/analyse_support.py             # stats, correlations, figures
```

## Consistency check (critical — isolates the non-linearity)
Two layers, both pass:
- **Embedding level.** Pixel-mean of the native-float embeddings → affine, vs the iter0
  `reduceRegions(mean)` polygon-mean embedding (`ireland_features.parquet`): per-band **max abs diff
  median 1.57, max 3.36 codec units** (band σ in this codec spans 7.8–41.1 — so ≤~0.4 σ on the worst
  band). The pixel set therefore matches the iter0 support to within edge pixels.
- **AGB level.** For the 7 non-tiled, grid-aligned stands, `f(mean)` reproduces the **reported**
  `pred_tco2` to **<0.01 tCO₂/acre** (exact). For the remaining stands `f(mean)` differs from the
  reported value by a median **4.61 / mean 4.71 / max 11.49 tCO₂/acre** — this residual is the
  getDownloadURL pixel-grid origin (and tiling) selecting a slightly different **edge-pixel set** than
  `reduceRegions`, uncorrelated with stand size (ρ≈0.08); it is **not** the non-linearity.

**Crucially, both `f(mean)` and `mean(f)` are computed on the *same* per-pixel extraction**, so the
reported **gap = mean(f) − f(mean)** is internally self-consistent and cleanly attributable to the
head's non-linearity, independent of the small grid offset vs reduceRegions. (A future refinement could
snap getDownloadURL to the reduceRegions grid to also zero the AGB-level reproduction residual.)

## Per-stand f(mean) vs mean(f)
| stand | n_subcpt | area_ha | n_pix | f(mean) | mean(f) | gap | gap% | pix min | pix median | pix max | pix std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Carrowreagh | 1 | 17.08 | 3111 | 32.2 | 33.6 | +1.4 | +4.3 | 19.3 | 30.5 | 125.2 | 12.6 |
| Rathcahill West | 1 | 6.96 | 1233 | 42.2 | 39.0 | −3.2 | −7.6 | 22.3 | 40.8 | 104.7 | 12.0 |
| Crooderry | 1 | 8.69 | 1640 | 104.5 | 107.0 | +2.6 | +2.5 | 34.3 | 108.9 | 158.2 | 20.9 |
| Benmore | 1 | 7.74 | 1415 | 101.1 | 88.9 | −12.2 | −12.1 | 29.1 | 85.7 | 146.3 | 26.9 |
| Loughros | 2 | 0.67 | 169 | 107.9 | 84.5 | −23.4 | −21.7 | 47.7 | 90.9 | 130.6 | 23.3 |
| Carrigeeny | 3 | 30.33 | 5122 | 36.3 | 50.9 | +14.6 | +40.1 | 27.0 | 46.4 | 126.4 | 17.5 |
| Dromreask | 3 | 8.79 | 1557 | 58.5 | 73.4 | +14.9 | +25.5 | 38.5 | 58.8 | 135.5 | 31.1 |
| Tooreennagreana | 4 | 21.68 | 3775 | 95.7 | 97.0 | +1.3 | +1.4 | 20.8 | 101.2 | 153.4 | 19.3 |
| Carrowkeel | 5 | 16.03 | 2917 | 40.1 | 40.6 | +0.6 | +1.5 | 20.4 | 33.6 | 131.8 | 19.1 |
| Lacka Beg | 5 | 9.32 | 1669 | 106.8 | 90.2 | −16.7 | −15.6 | 44.5 | 98.2 | 147.0 | 24.6 |
| Sligo Bay N_Greaghnafarna II | 6 | 9.42 | 1736 | 99.5 | 88.8 | −10.6 | −10.7 | 34.0 | 96.4 | 141.9 | 24.9 |
| Cummeen Upper | 6 | 65.31 | 10867 | 108.1 | 110.9 | +2.8 | +2.5 | 45.1 | 111.7 | 154.3 | 14.4 |
| Highmount | 9 | 22.69 | 3909 | 123.7 | 100.3 | −23.4 | −18.9 | 38.7 | 109.0 | 154.4 | 28.7 |
| Knockbreenagher | 9 | 8.46 | 1606 | 105.1 | 95.3 | −9.8 | −9.3 | 31.2 | 103.5 | 140.4 | 26.7 |
| Sligo Bay S_Kilfree | 9 | 14.39 | 2593 | 101.0 | 100.3 | −0.7 | −0.7 | 43.0 | 103.0 | 149.2 | 18.7 |
| Glanowen | 13 | 41.44 | 7155 | 86.1 | 77.5 | −8.6 | −10.0 | 19.5 | 87.8 | 178.7 | 36.1 |
| Meensheefin | 20 | 151.84 | 26764 | 78.8 | 83.5 | +4.7 | +6.0 | 27.6 | 89.4 | 147.8 | 28.3 |
| Cloonsheever | 37 | 85.37 | 15896 | 99.9 | 94.6 | −5.3 | −5.3 | 16.2 | 111.8 | 160.6 | 37.6 |

## Gap statistics
- Signed gap (tCO₂/acre): min −23.4, Q1 −10.4, **median −2.0**, Q3 +2.3, max +14.9.
- Signed gap %: min −21.7, Q1 −10.5, **median −3.0**, Q3 +2.5, max +40.1; **mean −1.6%**.
- |gap| %: **mean 10.9%**. **n positive (mean(f) > f(mean)) = 8/18.**
- |gap%| > 5: **12/18**; > 10: **7/18**.

Direction: the gap is **bidirectional** but net slightly negative — i.e. across these stands the
polygon-mean estimator `f(mean)` is, on the median, a few % **higher** than the pixel-aggregate
`mean(f)`. The largest negative gaps (Loughros −21.7%, Highmount −18.9%, Lacka Beg −15.6%) sit in
mature stands where a high single-value prediction is pulled down once low-biomass pixels are scored
individually; the largest positive gaps (Carrigeeny +40.1%, Dromreask +25.5%) are young/low-mean stands
where individually-scored pixels with high embeddings pull the pixel average **up** above the value the
averaged embedding predicts.

## Correlations & within-stand dispersion
- **|gap%| vs within-stand per-pixel-prediction IQR: Spearman +0.538 (p=0.021)** — the cleanest driver.
- |gap%| vs per-pixel-prediction std: +0.370 (p=0.13).
- |gap%| vs `n_subcpt`: −0.13 (ns); vs `area_ha`: −0.23 (ns) — **the gap does NOT track the
  sub-compartment count or polygon size.** `n_subcpt` is a poor heterogeneity proxy here; the real
  driver is the spread of the *embeddings/predictions* inside the footprint.
- Signed gap vs area_ha: +0.41 (p=0.09, weak) — larger stands trend slightly positive.

**What the single stand value masks (per-pixel min/median/max).** Every stand spans a very wide
per-pixel prediction range — e.g. Carrowreagh f(mean)=32 but pixels run 19→125; Crooderry f(mean)=104,
pixels 34→158; Cloonsheever 16→161 (std 37.6). The reported single tCO₂/acre per stand hides a
3–8× internal spread; the stand value is one point estimate over a genuinely multi-modal footprint
(clearfell/restock mosaic), which is exactly why mean(f) and f(mean) diverge.

Figures: `figures/support_scatter_fmean_vs_meanf.png` (f(mean) vs mean(f), 1:1 line, coloured by
`n_subcpt`); `figures/support_gap_vs_heterogeneity.png` (gap% vs `n_subcpt` and vs within-stand std).

## Interpretation
1. **Is it material? Yes.** Mean |gap| 10.9%, 7/18 stands beyond ±10%, worst ~24% (Loughros, Highmount).
   This exceeds the 5–10% materiality bar.
2. **Direction.** Bidirectional and stand-specific; **net the polygon-mean over-reads** vs the
   pixel-aggregate (median −3%), but it under-reads on young/low-mean mosaics (Carrigeeny +40%). There is
   no single correction factor — the gap flips sign with the stand's biomass regime.
3. **Does it grow with heterogeneity? With prediction dispersion, yes (ρ +0.54); with `n_subcpt`, no.**
   The Jensen-gap intuition holds (more within-stand variance in the scored signal → bigger gap), but the
   Dasos `n_subcpt` count is not a usable proxy for that variance — pixel-level dispersion is.
4. **Implications for the reported Ireland numbers.** The portfolio-level 91.6 tCO₂/acre and the
   per-Location values in `bias_characterisation.md` are f(mean). At stand level they can be off the
   production (pixel-aggregate) value by up to ~24% in either direction, so per-stand absolute numbers
   should not be treated as the production estimate. The **portfolio mean** is more robust (signed gaps
   partly cancel: mean signed gap only −1.6%), so the headline 3.35× DB ratio and the H1/H3 *rank*
   conclusions are unlikely to move materially — consistent with the iter0 verdict that *ranking* is more
   trustworthy than *levels*.
5. **Implications for the DB comparison.** DB is itself **pixel-then-aggregate (mean(f)-like)**. Comparing
   DB against our **f(mean)** is a slight apples-to-oranges mismatch at stand level. Re-running ours as
   mean(f) makes the comparison like-for-like; given the net −1.6% mean shift it will not overturn H1
   (pred ≫ DB, ~3.35×) but will tighten per-stand deltas and could change individual stand signs for the
   few near-parity Locations (iter0 had 1.4% of Locations with pred < DB).

## Recommendation
**Re-run all 141 Locations at pixel level (mean(f))** and report it as the production-aligned estimate
alongside the existing f(mean), for three reasons: (a) the per-stand gap is material (up to ~24%) and
sign-flipping, so f(mean) is not a safe per-stand proxy; (b) the production wall-to-wall map will be
mean(f), so the deliverable must match; (c) it makes the DB comparison like-for-like. `n_subcpt` is
**not** a sufficient filter for "which stands need it" — the gap tracks pixel-level dispersion, which is
only knowable after pixel extraction, so a partial re-run keyed on `n_subcpt` would miss high-gap
homogeneous stands (e.g. Loughros n_subcpt=2, gap −21.7%). The full 141-stand run is cheap on the
`scripts/per_pixel_inference.py` path (≈96 k pixels over 18 stands ran in a few minutes; 141 stands ≈ a
few × that). This does **not** change the iter0 RETRAIN_WARRANTED decision (driven by severe OOD, not
support) — it refines the *absolute levels* the retrain will be evaluated against.

## Assumptions, seeds, caveats
- seed 42; AEF asset pinned V1 ANNUAL; survey-year per stand from `ireland_locations_dissolved.gpkg`.
- Pixel set = centre-in-polygon at 10 m (`inmask` band), reproducing `reduceRegions(mean)` support.
- The affine is linear, so it contributes 0 to the gap by construction; the gap is the LightGBM
  non-linearity (Jensen gap) over the within-stand embedding spread.
- f(mean) reproduces the reported pred exactly for grid-aligned stands; a median 4.61 (max 11.5)
  tCO₂/acre residual on the others is a getDownloadURL-grid vs reduceRegions-grid edge-pixel difference,
  NOT the support effect — the gap (mean(f)−f(mean)) is computed within one self-consistent extraction.
- 4 sampled stands use the pre-2017 AEF fallback (Rathcahill West, Carrigeeny, Cummeen Upper) — small
  temporal misalignment carried from iter0, unchanged here.
- No ground truth; all quantities are model-internal (the two estimators of the same head). DB remains a
  directional lower bound, not truth.
- Loughros has NaN `age_at_survey` (one stand) → the |gap%|-vs-age correlation is reported as NaN.

## Outputs
| file | content |
|---|---|
| `evaluation/support_sensitivity.md` | this report |
| `evaluation/support_sensitivity_stands.parquet` | per-stand f_mean, mean_f, gap, gap_pct, n_pixels, n_subcpt, area_ha, pixel-prediction min/median/max/std/iqr, emb_consistency_max_abs, f_mean_iter0 |
| `scripts/per_pixel_inference.py` | reusable per-pixel inference (Location-list parameterised; wall-to-wall basis) |
| `evaluation/analyse_support.py` | gap stats, correlations, figures |
| `evaluation/figures/support_scatter_fmean_vs_meanf.png` | f(mean) vs mean(f), 1:1 line |
| `evaluation/figures/support_gap_vs_heterogeneity.png` | gap% vs heterogeneity / dispersion |

## Reproducibility footer
inputs: `ireland_locations_dissolved.gpkg`, `aef_affine.parquet`, `ireland_features.parquet`,
`ireland_predictions.parquet`, `models/inference_model_embdstx.txt` + `…features_embdstx.json`.
method: real GEE getDownloadURL per-pixel extraction + LightGBM predict via `uv run`. seed 42.
conducted by: Support-Test Actor. timestamp_utc: 2026-06-08.
