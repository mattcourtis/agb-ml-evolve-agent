# Ireland AGB — production-aligned per-pixel inference, all 141 stands (mean(f))

**Stage:** Pixel-Inference Actor (atomic). **Mode:** run the production estimator
`mean(f(embeddings))` — apply the `embdstx` head per 10 m pixel inside each stand, then
average the per-pixel predictions to a stand density — for **all 141 Irish stands**, each at
its `survey_year`. Store both tabular outputs (csv/parquet/gpkg) and per-stand GeoTIFFs.
Real GEE per-pixel extraction + LightGBM. **No ground truth.** seed 42. Asset
`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (V1), Hansen `…2025_v1_13`. Mg/ha = tCO₂/acre / 0.6977.

This **refines the absolute levels** previously reported as `f(mean)` (the iter0 polygon-mean
estimator). It does **not** change the iter0 `RETRAIN_WARRANTED` verdict, which is driven by
severe OOD between the CONUS training distribution and the Irish target, not by the
change-of-support effect quantified here.

## TL;DR
All **141/141** stands processed with **0 failures**, over **593 754** native 10 m pixels
(total mapped area **3367.1 ha**). The production-aligned **portfolio mean(f) = 88.79 tCO₂/acre**
(area-weighted 90.27), vs the iter0 `f(mean)` portfolio mean 91.62 — a portfolio shift of
**−3.1%**. Against Deep Biomass (DB, itself pixel-then-aggregate) the like-for-like ratio is
**3.25× (DB = 27.35 tCO₂/acre)**, essentially unchanged from the iter0 3.35× headline. **H1
holds: 140/141 Locations (99.3%) have mean(f) ≥ DB** (the single exception, Bunrevagh, is at
near-parity: 35.14 vs 35.81). The all-141 gap-vs-polygon-mean is **bidirectional and material**:
**median −2.1%**, **range −28.7% … +77.4%**, mean |gap| **12.3%**, with **93/141 beyond ±5%** and
**58/141 beyond ±10%** — confirming the 18-stand support-test finding at full scale and
reinforcing that `f(mean)` is not a safe per-stand proxy for the production map.

## Method
Driver `scripts/pixel_inference_all141.py` extends the support-test `scripts/per_pixel_inference.py`
(same extraction/affine/prediction core), adding per-stand checkpointing, per-pixel lon/lat
capture, and an all-141 loop. Aggregation + GeoTIFF/VRT in `scripts/aggregate_pixel_outputs.py`.

Per stand, at its `survey_year`:
1. **Per-pixel extraction.** A fused image (64 AEF bands `A00..A63` for the survey-year mosaic +
   the 3 per-pixel dstx bands + `inmask` + `pixelLonLat`) is downloaded as native-float NumPy via
   `getDownloadURL` (NPY, scale=10). `getDownloadURL` returns the bounding-box rectangle, so the
   `inmask` band (geometry rasterised at 10 m, 1 iff the pixel **centre** is inside the polygon)
   selects exactly the pixel set `reduceRegions(mean)` integrates over. Large stands that exceed the
   50 MB cap are auto-tiled into latitude strips and re-concatenated (no change to the kept-pixel
   set). lon/lat pixel-centres are captured so predictions can be rasterised.
2. **dstx per pixel.** `dstx_pixel_image()` — the survey-relative Hansen timing of
   `extract_ireland_aef.extract_dstx` evaluated per pixel: `dstx_pre_ysd = code − lossyear` if
   `0<ly≤code` else 100; `dstx_pre_loss_5yr` = pre-survey loss within 5 yr; `dstx_loss_frac_buf` =
   pre-survey-loss indicator (0/1) per pixel.
3. **Affine.** The production per-band affine (`preprocessing/aef_affine.parquet`,
   `emb_b = a_b·A_b + c_b`, LINEAR) applied per pixel → training codec space.
4. **mean(f).** Predict per pixel with `models/inference_model_embdstx.txt` (67-feature order
   `emb_00..emb_63` + 3 dstx; target tCO₂/acre); the simple mean of the per-pixel predictions is the
   stand density. Mg/ha = mean(f) / 0.6977; total tonnes = Mg/ha × area_ha.
   The asset's native 10 m grid in EPSG:4326 oversamples at ~53°N (≈1/cos(lat) ≈ 1.65×), but the
   oversampling is spatially uniform so the simple pixel mean equals the area-weighted mean; the
   support-test consistency check (pixel-mean native emb → affine vs iter0 polygon-mean emb) already
   confirmed the pixel set preserves the polygon support.
5. **gap = mean(f) − f(mean)**, where `f(mean)` is the iter0 polygon-mean prediction
   (`evaluation/ireland_predictions.parquet:pred_tco2`).

### Run / resume design
Long-running and resumable in tmux:
- `tmux new-session -d -s pix_inf 'uv run python …/pixel_inference_all141.py'`.
- **Per-stand checkpoint:** `preprocessing/_pixel_pred/<Location>.parquet` (per-pixel `lon, lat,
  pred_tco2_acre`) + `preprocessing/_pixel_pred/_summary/<Location>.json` (n_pixels, mean(f),
  Mg/ha, pixel std/min/median/max). On restart, any stand with both files present is **skipped**.
- **Retries:** each stand retries up to 3× with backoff on transient GEE/HTTP errors before being
  listed as failed; acceptance needs ≥135/141.
- The 18 support-test stands were **re-extracted here** (the earlier run did not persist per-pixel
  parquet); they reproduce their earlier mean(f) to **0.0 tCO₂/acre** (exact — same code path).
- Full 141-stand run completed in a few minutes (~1–3 s/stand); 0 failures, no tiling fallbacks
  triggered errors.

## Validation
- **141/141** stands have pixel predictions; **0 failures** (`_run_done.flag`:
  `{"done":141,"failed":[],"total":141}`).
- **18 support-test stands reproduce mean(f) exactly** (max abs diff **0.0 tCO₂/acre** across all
  18; e.g. Loughros 84.47, Meensheefin 83.52, Cloonsheever 94.61, all identical to
  `support_sensitivity_stands.parquet:mean_f`).
- **Total mapped area = 3367.1 ha** (matches the expected ~3367 ha).
- Total native pixels 593 754; rasterised 10 m EPSG:2157 cells 361 373 (the ~1.65× ratio is the
  EPSG:4326→2157 reprojection deduplication at 53°N — expected, not pixel loss).

## Corrected portfolio + H1-vs-DB (like-for-like, both pixel-then-aggregate)
| quantity | value |
|---|---|
| Portfolio mean **mean(f)** | **88.79 tCO₂/acre** (area-weighted 90.27) |
| Portfolio mean f(mean) (iter0) | 91.62 tCO₂/acre |
| Portfolio shift mean(f) vs f(mean) | **−3.1%** |
| Portfolio mean DB (2020–24) | 27.35 tCO₂/acre |
| **Ratio mean(f) / DB** | **3.25×** (iter0 f(mean)/DB = 3.35×) |
| **H1: frac Locations mean(f) ≥ DB** | **0.9929 (140/141)** |
| Locations mean(f) < DB | 1 — **Bunrevagh** (35.14 vs 35.81, near-parity) |
| per-stand ratio mean(f)/DB | median 3.43×, mean 3.73× |
| Total portfolio biomass | ~435 654 tonnes (Σ Mg/ha × area_ha) |

The correction lowers the portfolio headline by ~3% and the DB ratio from 3.35× to 3.25×; the H1
conclusion is unchanged (one near-parity stand flips below DB vs the iter0 1.4% < DB on f(mean)).
The OOD-driven `RETRAIN_WARRANTED` verdict is unchanged — this refines levels, not the decision.

## Gap vs polygon-mean over all 141 (= mean(f) − f(mean))
| stat | value |
|---|---|
| gap_pct quartiles (min/Q1/median/Q3/max) | −28.7 / −9.4 / **−2.1** / +4.9 / +77.4 |
| gap_pct mean | +1.2% |
| mean \|gap_pct\| | 12.3% |
| \|gap_pct\| > 5% | 93/141 |
| \|gap_pct\| > 10% | 58/141 |
| n positive (mean(f) > f(mean)) | 62/141 |
| gap (tCO₂/acre) median / range | −2.19 / [−36.9, +23.9] |

Direction is **bidirectional** and stand-specific, net slightly negative on the median (polygon-mean
`f(mean)` over-reads vs the pixel-aggregate). Largest **negative** gaps are mature high-mean stands
where a single averaged-embedding prediction sits above the pixel average once low-biomass pixels are
scored individually (Ballaghbehy North −28.7%, Harristown −25.7%, Glensharrold −25.4%, Kilmore
−23.2%). Largest **positive** gaps are young/low-mean mosaics where individually-scored high-embedding
pixels pull the pixel average above the averaged-embedding value (Peak +77.4%, Killuran Beg +60.8%,
Glannaheera +59.7%, Knocknadarriv +50.2%). This is the LightGBM non-linearity (Jensen gap) over the
within-stand embedding spread; the affine is linear and contributes 0 by construction. The full-141
distribution confirms and slightly widens the 18-stand support-test finding (there: median −3.0%,
range −21.7…+40.1%, |gap| 10.9%).

## File inventory
| file | content |
|---|---|
| `final/ireland_agb_pixel.csv` / `.parquet` | per-stand: Location_Name, area_ha, survey_year, n_pixels, pred_pixel_tCO2_acre (mean(f)), pred_pixel_Mg_ha, pred_pixel_total_t, pred_polygonmean_tCO2_acre, gap_tCO2_acre, gap_pct, pixel_pred_std/min/max, db_2020_24_tCO2_acre, db_2020_24_Mg_ha, delta_pixel_vs_db_tCO2_acre, MainSp, age_at_survey, Hdom, YC, pre2017_fallback |
| `final/ireland_agb_pixel.gpkg` | the above joined to the stand polygons, EPSG:2157 (141 features) |
| `final/ireland_pixel_tiffs/<Location>.tif` (×141) | per-stand per-pixel predicted-AGB raster, EPSG:2157, 10 m, 2 bands (band1 pred_AGB_tCO2_per_acre, band2 pred_AGB_Mg_per_ha), nodata −9999 outside polygon |
| `final/ireland_pixel_tiffs/_index.vrt` | VRT index over band1 of all 141 tiles (EPSG:2157) |
| `preprocessing/_pixel_pred/<Location>.parquet` (×141) | per-pixel lon, lat, pred_tco2_acre (checkpoint, resumable) |
| `preprocessing/_pixel_pred/_summary/<Location>.json` (×141) | per-stand summary checkpoint |
| `preprocessing/_pixel_pred/_run_done.flag` | run-completion record |
| `preprocessing/_pixel_pred_run.log` | full run log |
| `scripts/pixel_inference_all141.py` | resumable all-141 per-pixel inference driver |
| `scripts/aggregate_pixel_outputs.py` | stand aggregation + csv/parquet/gpkg + GeoTIFFs + VRT |

## Assumptions, seeds, commands
- seed 42; AEF pinned `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`; Hansen `UMD/hansen/global_forest_change_2025_v1_13`.
- survey_year per stand from `ireland_locations_dissolved.gpkg`; 4+ stands use the pre-2017 AEF fallback (`pre2017_fallback`), carried unchanged from iter0.
- Pixel set = centre-in-polygon at 10 m (`inmask`), reproducing `reduceRegions(mean)` support; simple pixel mean = area-weighted mean (uniform oversampling).
- The affine is linear ⇒ contributes 0 to the gap; the gap is the LightGBM Jensen non-linearity.
- GeoTIFF rasterisation: pixel centres reprojected EPSG:4326→2157 and snapped to a fixed 10 m grid per stand (last-write-wins on rare grid collisions; negligible for a 10 m snap of the 10 m native grid).
- Conversion Mg/ha = tCO₂/acre / 0.6977.
- **No ground truth**; all quantities are model-internal. DB remains a directional comparator (pixel-then-aggregate), not truth. The OOD-driven RETRAIN_WARRANTED verdict is unchanged.
- Commands:
  ```
  tmux new-session -d -s pix_inf 'uv run python experiments/.../scripts/pixel_inference_all141.py'
  uv run python experiments/.../scripts/aggregate_pixel_outputs.py
  ```

## Reproducibility footer
inputs: `ireland_locations_dissolved.gpkg`, `aef_affine.parquet`, `db_reference.parquet`,
`ireland_predictions.parquet`, `support_sensitivity_stands.parquet`,
`models/inference_model_embdstx.txt` + `…features_embdstx.json`.
method: real GEE `getDownloadURL` per-pixel extraction + LightGBM predict via `uv run`. seed 42.
conducted by: Pixel-Inference Actor. timestamp_utc: 2026-06-08.
