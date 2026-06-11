# Ireland AGB — year-matched comparison vs Deep Biomass (2022 / 2023 / 2024)

**Stage:** Year-Matched Actor (atomic). **Goal:** a temporally- AND estimator-consistent
comparison of our model vs Deep Biomass (DB) for the THREE fixed years 2022, 2023, 2024. Both
sides are fixed to the **same year**; both use the **pixel-then-aggregate** estimator (our
production `mean(f)`; DB's native per-Location annual aggregate). The prior comparison was
temporally confounded (our per-stand `survey_year` 2017–2025 vs a DB 2020–24 fixed mean); this
run removes that confound. All three years lie inside AEF coverage and the DB CSV, so **no
clamping is applied on either side**. Real GEE per-pixel extraction + LightGBM. seed 42.
Asset `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (V1); Hansen `…2025_v1_13`.
Mg/ha = tCO₂/acre / 0.6977; conversion ×0.6977.

## TL;DR
All **141/141 stands × 3 years = 423/423 completed, 0 failures** (593 754 native 10 m pixels per
year, 3367.1 ha). Year-matched portfolio (tCO₂/acre, mean over stands):

| year | our mean(f) | DB | ratio our/DB | H1 (frac our ≥ DB) |
|---|---|---|---|---|
| 2022 | 87.46 | 25.93 | **3.37×** | 0.993 (140/141) |
| 2023 | 95.51 | 22.59 | **4.23×** | 0.993 (140/141) |
| 2024 | 91.76 | 31.89 | **2.88×** | 0.979 (138/141) |
| **2022–24 mean** | **91.58** | **26.77** | **3.42×** | **1.000 (141/141)** |

The **cross-check is exact**: the 73 stands whose accepted single-year `survey_year` ∈ {2022,
2023, 2024} reproduce their `final/ireland_agb_pixel.parquet:pred_pixel_tCO2_acre` to **max abs
diff 0.000 tCO₂/acre** (3 in 2022, 32 in 2023, 38 in 2024) — same year, same method, same code
path. The year-matched 3-yr-mean ratio (**3.42×**) is materially the same order as the earlier
temporally-confounded headline (**3.25×**), confirming the gap is **not** an artefact of mixed
years: it is a genuine, robust level difference driven by the CONUS→Ireland OOD.

## Method
Driver `scripts/pixel_inference_yearmatched.py` (one run per fixed year Y), a thin parameterised
extension of the accepted `scripts/pixel_inference_all141.py` — identical extraction / affine /
prediction core. Per (year Y, stand):
1. **Per-pixel extraction.** Fused image = 64 AEF bands `A00..A63` for the **year-Y mosaic** + 3
   per-pixel dstx bands + `inmask` + lon/lat, downloaded as native-float NumPy via
   `getDownloadURL` (NPY, scale=10). `inmask` (geometry rasterised at 10 m, 1 iff the pixel
   **centre** is inside the polygon) selects exactly the pixel set `reduceRegions(mean)`
   integrates over. Oversized stands auto-tile into latitude strips (no change to kept pixels).
2. **dstx relative to year Y.** `dstx_pixel_image(Y)` uses Y as the reference year (`code = Y −
   2000`): `dstx_pre_ysd = code − lossyear` if `0 < ly ≤ code` else 100; `dstx_pre_loss_5yr` =
   pre-Y loss within 5 yr; `dstx_loss_frac_buf` = pre-Y-loss indicator per pixel. So the
   disturbance timing is **also** year-consistent (not anchored to the per-stand survey year).
3. **Affine.** Production per-band affine (`preprocessing/aef_affine.parquet`, `emb_b = a_b·A_b +
   c_b`, LINEAR) applied per pixel → training codec space.
4. **mean(f).** Predict per pixel with `models/inference_model_embdstx.txt` (67-feature order
   `emb_00..emb_63` + 3 dstx; target tCO₂/acre); the simple mean of the per-pixel predictions is
   the stand density. Mg/ha = mean(f) / 0.6977.

**Deep Biomass year-matched** (`scripts/deepbiomass_yearmatched.py`): the DB CSV holds per-Location
annual TOTAL TONNES. Per year Y: `Mg/ha = tonnes_Y / Area_Ha`, `tCO₂/acre = Mg/ha × 0.6977`; the
2022–24 mean is the simple mean of the three annual tCO₂/acre. The DB CSV `Location Name` column
matches our `Location_Name` directly (141/141 join — the crosswalk's underscore→slash variants are
not needed because the CSV already uses the underscore form), so no crosswalk remap was required.

### Run / resume design
Per-(year, stand) checkpointing, resumable in tmux (one session per year, run in parallel):
`preprocessing/_pixel_pred_yYYYY/<Location>.parquet` (per-pixel lon, lat, pred_tco2_acre) +
`…/_summary/<Location>.json`. On restart any (year, stand) with both files present is skipped; up
to 3 retries with backoff per stand. Each year's full 141-stand run completed in ~1–2 min
(~0.5 s/stand); 0 failures, no tiling errors. Completion flags
`_pixel_pred_yYYYY/_run_done.flag` all record `{"done":141,"failed":[],"total":141}`.

## Validation
- **423/423** (141 stands × 3 years) complete, **0 failures** each year (≥135/141 acceptance met
  for all three).
- **No clamping** — all three years (2022/2023/2024) are inside AEF coverage `[2017,2025]` and the
  DB CSV; neither side was clamped.
- **Cross-check reproduces exactly.** 73 stands with accepted `survey_year` ∈ {2022,2023,2024}:
  year-matched `our_Y` vs `final/ireland_agb_pixel.parquet:pred_pixel_tCO2_acre` →
  **max abs diff 0.000, mean abs diff 0.000 tCO₂/acre** (see `_yearmatched_crosscheck.csv` and the
  diagonal cross-check panel in the figure). This proves the year-matched code path is identical to
  the accepted single-year run.
- Native pixel count identical across years (593 754) — the polygon support (centre-in-polygon at
  10 m) is year-invariant, as expected; only the embedding/dstx values change with Y.

## Year-on-year trajectory (portfolio mean, tCO₂/acre)
| | 2022 | 2023 | 2024 | 22→23 | 23→24 | 22→24 |
|---|---|---|---|---|---|---|
| **Our mean(f)** | 87.46 | 95.51 | 91.76 | +9.2% | −3.9% | **+4.9%** |
| **Deep Biomass** | 25.93 | 22.59 | 31.89 | −12.9% | +41.2% | **+23.0%** |

Both models show net **growth** 2022→2024 at the portfolio level (ours +4.9%, DB +23.0%), but the
**shapes differ**: ours rises then dips slightly (peak 2023), whereas DB dips in 2023 then jumps
sharply in 2024. DB's larger relative swings reflect its much lower absolute base (a ±few tonnes
change is a large % on a ~26 tCO₂/acre mean). The trajectories are **not** anti-correlated in a way
that would suggest one model is simply tracking the other inverted; they are two independent
estimators of the same forests with a persistent ~3–4× level offset.

## Per-stand ratio and H1 detail
| year | per-stand ratio our/DB (median / mean) | H1 exceptions (our < DB) |
|---|---|---|
| 2022 | 3.91× / 4.30× | 1 — Rathcahill West (44.0 vs 47.7) |
| 2023 | 5.20× / 5.64× | 0 |
| 2024 | 3.20× / 3.67× | 2 — Bunrevagh (34.4 vs 55.8), Erne West_Moher (42.3 vs 62.6) |
| 2022–24 mean | 3.64× / 4.06× | 0 (all 141 ≥ DB on the 3-yr mean) |

H1 (our ≥ DB) holds for **≥97.9% of stands every year** and for **100% on the 3-yr mean**. The
handful of per-year exceptions are stands where DB reads relatively high in that single year; they
all clear DB once averaged over the three years.

## How the year-matched gap differs from the earlier 3.25×
The earlier comparison reported **3.25×** (our `mean(f)` at each stand's own survey year vs a DB
2020–24 fixed mean) — a temporally-confounded number (different years on each side, different DB
aggregation window). Fixing both sides to the same year gives **3.37× (2022), 4.23× (2023), 2.88×
(2024)** and a **3.42× 3-yr mean**. The ratio is therefore **stable across the temporal alignment**
(3.25× confounded → 3.42× year-matched-mean; the per-year spread 2.88–4.23× is driven mostly by
DB's own year-to-year volatility, not by our model). The conclusion is unchanged and strengthened:
the ~3–4× level gap is a **real OOD-driven offset**, not an artefact of comparing mismatched years.

## OOD caveat (unchanged)
There is **no ground truth** in this comparison; all our quantities are model-internal and DB is a
directional comparator (itself a model, pixel-then-aggregate), not truth. The dominant driver of
the absolute level gap is the severe out-of-distribution shift between the CONUS training
distribution and the Irish target. This run refines/aligns the *comparison*; it does **not** change
the upstream OOD-driven `RETRAIN_WARRANTED` verdict.

## File inventory
| file | content |
|---|---|
| `final/ireland_agb_yearmatched.csv` / `.parquet` | per stand: Location_Name, area_ha, survey_year, MainSp, age_at_survey, Hdom, YC, n_pixels_2022/23/24, our_2022/2023/2024_tCO2_acre (+ Mg_ha), our_mean_2022_24_tCO2_acre (+ Mg_ha), db_2022/2023/2024_tCO2_acre (+ Mg_ha), db_mean_2022_24_tCO2_acre (+ Mg_ha), delta_2022/2023/2024_tCO2_acre (our−db), delta_mean_tCO2_acre |
| `final/ireland_agb_yearmatched.gpkg` | the above joined to stand polygons, EPSG:2157 (141 features) |
| `final/ireland_pixel_tiffs_2022/<Location>.tif` (×141) + `_index.vrt` | per-stand 2022 per-pixel raster, EPSG:2157, ~10 m, band1 tCO₂/acre, band2 Mg/ha, nodata −9999; per-year VRT over band1 |
| `final/ireland_pixel_tiffs_2023/…` (×141) + `_index.vrt` | as above, year 2023 |
| `final/ireland_pixel_tiffs_2024/…` (×141) + `_index.vrt` | as above, year 2024 |
| `final/figures/ireland_vs_deepbiomass_yearmatched.png` | per-year scatter (our vs DB, 1:1) ×3 + portfolio trajectory 2022→2024 + 3-yr-mean scatter + cross-check diagonal |
| `final/_yearmatched_crosscheck.csv` | the 73-stand reproduction check vs the accepted single-year run |
| `final/_yearmatched_stats.json` | portfolio means, ratios, H1, cross-check stats |
| `preprocessing/_pixel_pred_yYYYY/<Location>.parquet` (×141 ×3) | per-pixel lon, lat, pred_tco2_acre (checkpoint, resumable) |
| `preprocessing/_pixel_pred_yYYYY/_summary/<Location>.json` (×141 ×3) | per-stand summary checkpoint |
| `preprocessing/db_yearmatched.parquet` | per-Location DB 2022/2023/2024 + 2022-24 mean (tCO₂/acre + Mg/ha) |
| `scripts/pixel_inference_yearmatched.py` | resumable year-matched per-pixel inference driver |
| `scripts/deepbiomass_yearmatched.py` | DB per-year recompute from CSV tonnes |
| `scripts/aggregate_yearmatched.py` | stand aggregation + csv/parquet/gpkg + per-year GeoTIFFs/VRT + cross-check + figure |

## Assumptions, seeds, commands
- seed 42; AEF pinned `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`; Hansen `UMD/hansen/global_forest_change_2025_v1_13`.
- Both sides fixed to year Y; dstx reference year = Y (year-consistent disturbance timing). No clamping (all years in [2017,2025]).
- DB: Mg/ha = tonnes_Y / Area_Ha; tCO₂/acre = Mg/ha × 0.6977. DB CSV `Location Name` = our `Location_Name` (141/141 direct join).
- Pixel set = centre-in-polygon at 10 m (`inmask`), reproducing `reduceRegions(mean)` support; simple pixel mean = area-weighted mean (uniform ~1/cos(lat) oversampling).
- The affine is linear ⇒ contributes 0 to any change-of-support gap; per-pixel prediction is the production estimator.
- GeoTIFF rasterisation: pixel centres reprojected EPSG:4326→2157 and snapped to a fixed 10 m grid per stand (last-write-wins on rare collisions; negligible).
- No ground truth; all quantities model-internal; DB is a directional comparator; OOD-driven RETRAIN_WARRANTED verdict unchanged.
- Commands:
  ```
  for Y in 2022 2023 2024; do
    tmux new-session -d -s ym$Y "uv run python experiments/.../scripts/pixel_inference_yearmatched.py --year $Y"
  done
  uv run python experiments/.../scripts/deepbiomass_yearmatched.py
  uv run python experiments/.../scripts/aggregate_yearmatched.py
  ```

## Reproducibility footer
inputs: `ireland_locations_dissolved.gpkg`, `aef_affine.parquet`,
`Deep Biomass - Aggregated Data & Portfolio Summary.csv`, `final/ireland_agb_pixel.parquet`
(accepted single-year, for cross-check), `models/inference_model_embdstx.txt` +
`…features_embdstx.json`.
method: real GEE `getDownloadURL` per-pixel extraction + LightGBM predict via `uv run`. seed 42.
conducted by: Year-Matched Actor. timestamp_utc: 2026-06-08.
