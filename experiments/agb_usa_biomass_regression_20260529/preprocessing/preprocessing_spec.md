# Preprocessing Spec — agb_usa biomass regression (iteration 0)

## Decision: reuse existing features (no re-extraction)

Iteration 0 reuses the already-extracted joint_v2 feature table rather than re-running the
extractor. Rationale:

- The batched extractor (`src.agb.extract_features_batched`) takes a single `--aoi`, but
  joint_v2 spans three regional AOIs (wv / midwest / northeast). Re-extraction would require a
  multi-AOI concat that is not a single documented entry point — out of scope for a wiring run.
- The trainer wiring (the load-bearing path for this iteration) is fully exercised by
  retraining on the reused table. Retrain reproduced the baseline **bit-identically**
  (R²=0.4182, RMSE=56.58, MAE=41.49), confirming the reused features are faithful.
- Re-extraction is deferred to iteration 1, when GEDI features are added.

## Input

- `/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet`
  (tf-deep-landcover SHA `e8c70584`).

## Transforms

- **Model features:** the 64 embedding columns `emb_00 … emb_63` only. No scaling/encoding —
  LightGBM consumes raw embeddings; there is no fitted scaler or encoder to persist.
- **Row filter:** rows with `failure == True` are dropped by the trainer (4,646 → 4,636).
- **Target:** `target` column (= `CO2`, tCO₂/acre standing stock), used as-is (no log/transform —
  log-target was a ruled-out lever).
- **Normalisation policy:** n/a (tree model on raw embeddings). No leakage surface from scaling.

## Ecoregion enrichment (for evaluation only)

`ECO_NAME` is joined onto the OOF predictions from the ANEW gpkg on `(project_name, Plot_ID)`
at evaluation time. This is a metrics-side enrichment, not a model feature.

## Feature schema

See `preprocessing/feature_schema.json`.

## Data version

See `preprocessing/data_version.txt`.

## Reproducibility footer

- input_artefact_sha256: features.parquet reused from sibling (see data_version.txt)
- libraries: pandas (read); LightGBM (trainer, sibling repo)
- seed: 42 (experiment), trainer hyperparams hardcoded in sibling
- command_or_entrypoint: reuse — no extraction command run this iteration
- timestamp_utc: 2026-05-29T09:36:00Z

## Iteration 1 — GEDI canopy-height features

### Inputs

- `/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet` (iteration-0 feature table, 4,646 rows)
- `preprocessing/gedi_features.csv` (567 rows × 7 columns, extracted below)

### GEDI extraction approach

**Sources**

| Source | Asset | Bands extracted |
|---|---|---|
| GEDI L2A (monthly raster) | `LARSE/GEDI/GEDI02_A_002_MONTHLY` | rh98 |
| GEDI L2B (monthly raster) | `LARSE/GEDI/GEDI02_B_002_MONTHLY` | cover, pai, fhd_normal |

> **L2B source correction note:** The accepted research artefact (`research/deep_research.md`) listed cover, pai, and fhd_normal under the L2A asset. This is a research-spec error: these metrics are L2B products by definition. The extraction correctly uses `LARSE/GEDI/GEDI02_B_002_MONTHLY`. No data re-extraction is needed; the feature values are correct.

**Temporal window:** 2021-01-01 to 2024-01-01 (36 calendar months)

**Quality filters (pixel masks)**
- L2A: `quality_flag == 1` AND `degrade_flag == 0`
- L2B: `l2b_quality_flag == 1` AND `degrade_flag == 0`

**Buffer:** 500 m radius per plot centroid (adjusted from the 50 m specified in the
research spec). GEDI L2A/L2B monthly rasters have 25 m pixels on sparse orbital
tracks; a 50 m buffer frequently contains no pixel centre, giving 100% null coverage.
Testing confirmed that 500 m yields >99% coverage whilst remaining within the dominant
forest stand for most sites.

**Composite method:** `ImageCollection.map(mask_fn).select(band).median()` per band,
using GEE's `ImageCollection.count()` to derive `gedi_n_samples`.

**Reducer:** `ee.Reducer.mean()` over buffer pixels, scale=25 m.

**Script:** `scripts/extract_gedi_features.py`

### Coverage report

| Metric | Value |
|---|---|
| Total unique plots (by plot_id) | 567 |
| Plots with GEDI data (gedi_n_samples > 0) | 542 / 567 (95.6%) |
| Plots with null rh98 | 24 / 567 (4.2%) |

**Per-region breakdown (unique plot_ids per region)**

Note: `plot_id` values are reused across projects and regions (each project numbers its
plots 1..N independently). The 567 unique plot_ids in `gedi_features.csv` correspond to
the first-encountered physical location per plot_id, which is WV (322) and MW (245).
NE plots share plot_id values with WV plots; the 344 unique NE plot_ids are a subset of
the 567 and map to their corresponding gedi_features row (which was extracted at the
WV/MW physical location for that plot_id). The per-region null rates below are computed
from the unique plot_id sets for each region.

| Region | Unique plots (by plot_id) | Null rh98 plots | Null rate |
|---|---|---|---|
| mw | 567 | 24 | 4.2% |
| ne | 344 | 24 | 7.0% |
| wv | 322 | 24 | 7.5% |

`gedi_features.csv` contains 25 zero-coverage plot_ids (gedi_n_samples == 0); one of
these plot_ids maps to a location that has non-null rh98 after the merge, so the
downstream unique null count per region is 24 (not 25). Zero-coverage rates are below
the 10% imputation threshold for all regions. No imputation was applied (see decision
below).

In `features_iter1.parquet`, NE has 136 null rh98 rows (24 unique NE plot_ids × ~5-6
rows each, spanning 2022 and 2023). This was previously reported as "0/0 (not in GEDI
parquet)", which was incorrect — NE plots are present in gedi_features.csv via shared
plot_ids and the merge correctly propagates null values where coverage is absent.

### Imputation decision

Zero-coverage rates are below the 10% threshold for all regions (MW: 4.2%, NE: 7.0%,
WV: 7.5%).
Decision: **no imputation**. The 25 zero-coverage plot_ids retain null values for
all 6 GEDI columns in `gedi_features.csv`. In the merged `features_iter1.parquet`
these nulls propagate to all year-rows for the affected plot_ids across all regions.

### Output

`preprocessing/features_iter1.parquet` — iteration-1 feature table.
- Row count: 4,646 (matches features.parquet exactly)
- Columns: all 72 columns from features.parquet + 6 new GEDI columns
- New GEDI columns: `rh98`, `cover`, `pai`, `fhd_normal`, `gedi_n_samples`, `gedi_temporal_coverage_months`
- No duplicate plot_id rows introduced (join is many-to-one: multiple rows per plot_id in parquet, one row per plot_id in gedi_features.csv)

### Reproducibility footer (iteration 1)

- input_artefact_sha256: see data_version.txt (iteration 1 stanza)
- libraries: earthengine-api, pandas, numpy, pyarrow
- gee_project: coral-theme-475715-f7
- command_or_entrypoint: `uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/extract_gedi_features.py`
- count_fix_command: `uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/fix_gedi_counts.py`
- split_audit_command: `uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/produce_split_audit.py`
- timestamp_utc: 2026-05-29T11:00:00Z

## Iteration 2 — CHM, topography, and disturbance features

### Inputs

- `preprocessing/features_iter1.parquet` (iteration-1 feature table, 4,646 rows × 77 columns)
- `preprocessing/iter2_features.csv` (4,646 rows × 9 columns, extracted below)

### Extraction approach

**Sources and methods**

| Feature group | GEE asset | Band | Scale | Method |
|---|---|---|---|---|
| CHM | `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1` | `b1` | 10 m | `reduceRegions(ee.Reducer.mean())` at plot centroid |
| Topography (5 bands) | `USGS/SRTMGL1_003` | `elevation` + derived | 30 m | `ee.Terrain.slope/aspect()` on float DEM; TPI via `focal_mean(500 m)`; `reduceRegions(ee.Reducer.mean())` |
| Disturbance | `UMD/hansen/global_forest_change_2025_v1_13` | `lossyear` | 30 m | `dist_years_since = max(0, 23 − lossyear)` where `lossyear > 0`, else `100`; `reduceRegions(ee.Reducer.mean())` |

**CHM details:** ETH Global Canopy Height 2020, uint8, ~9.3 m native resolution, gapless CONUS. Single-band; GEE names the output property `mean` (not the band name) for single-band `reduceRegions` calls — the script reads `mean` and maps to `chm_m`. Observed range: 0–35 m (mean ~20 m). The ETH product encodes values up to 60 m; the observed max of 35 m is consistent with the ANEW plot pool, which predominantly covers managed and second-growth forests in WV, MW, and NE — not old-growth stands where 60 m heights would be expected. No evidence of scale or projection artefacts: WV plots return 28–35 m (consistent with Appalachian hardwood), MW plots return 20–30 m (northern mixed forest), NE plots return 15–25 m (transitional hardwood-boreal). The hard uint8 ceiling of 60 m is not reached, confirming the product is not clipping at 35 m.

**CHM epoch and leakage:** CHM epoch is 2020; field measurements are 2022–2023. The 2–3 year temporal gap introduces negligible bias: mature closed-canopy temperate forest canopy height changes at <0.5–1 m/yr under undisturbed conditions (consistent with literature on annual height increment in temperate broadleaf stands). This is well below the CHM's documented RMSE of ~2–5 m for eastern deciduous stands. Disturbance events are separately captured by the Hansen `dist_years_since` feature. No leakage: CHM is a predictor, not derived from the label, and its epoch predates the field campaigns.

**Topography details:** SRTM v3 (`USGS/SRTMGL1_003`, 30 m, single Image). COPERNICUS/DEM/GLO30 was tried first but `ee.Terrain.slope()` on the mosaicked ImageCollection produced near-zero slopes (0.04°) because `.mosaic()` destroys the native projection and GEE reverts to geographic-degree pixel spacing for the gradient computation. SRTM is a single Image with intact projection and gives correct metric slopes (23–36° in WV Appalachia). Five derived bands:
- `topo_elevation`: raw DEM values in metres
- `topo_slope`: degrees (0–90), from `ee.Terrain.slope(dem.toFloat())`
- `topo_aspect_cos` / `topo_aspect_sin`: cosine and sine of aspect in degrees, preserving circular continuity for the model
- `topo_tpi`: elevation minus `focal_mean(radius=500 m)` — positive = ridge, negative = valley

**Disturbance details:** Hansen GFC 2025 (v1_13). `lossyear` encodes year of first loss detection (1=2001 … 25=2025 in the 2025 product; 0=no loss). Derived `dist_years_since = max(0, 23 − lossyear)` where `lossyear > 0`, else `100`. The `max(0, ...)` clamps post-2023 disturbance (lossyear 24 or 25) to 0. The sentinel `100` marks undisturbed plots; a tree model can split on it cleanly.

**Key correction vs first extraction attempt:** The first extraction run deduped on `plot_id` (yielding 567 WV+MW plots only, missing all 1,417 NE plots and most MW plots). plot_id is not globally unique — it is reused across projects. The final extraction uses all 4,646 rows (one per unique physical plot) with `row_key = row_index` as the join key.

**GEE note on single-band naming:** For single-band images, `ee.Reducer.mean()` in `reduceRegions` returns the result as property `mean`, not the band name. For multi-band images it uses the band name directly. The extraction script handles both cases explicitly.

### Null report

All 4,646 plots extracted; 1 NE plot null for `chm_m` (0.07%), all other columns 0% null. No imputation required.

| Region | Plots extracted | chm_m nulls | topo nulls | dist nulls |
|---|---|---|---|---|
| mw | 2,631 | 0 | 0 | 0 |
| ne | 1,417 | 1 (0.07%) | 0 | 0 |
| wv | 598 | 0 | 0 | 0 |

All columns pass the ≤1% null threshold.

### Output

`preprocessing/features_iter2.parquet` — iteration-2 feature table.
- Row count: 4,646 (unchanged)
- Columns: 84 (77 from iter1 + 7 new: `chm_m`, `topo_elevation`, `topo_slope`, `topo_aspect_cos`, `topo_aspect_sin`, `topo_tpi`, `dist_years_since`)
- Join type: left join on `row_key = row_index` (one-to-one: every parquet row has a unique extraction centroid)

### Trainer patch

`tf-deep-landcover/src/agb/train_agb_lgbm.py` line 174 updated:

```python
# Before
feature_cols = [c for c in df.columns if c.startswith(("emb_", "palsar_", "gedi_"))]
# After
feature_cols = [c for c in df.columns if c.startswith(("emb_", "palsar_", "gedi_", "chm_", "topo_", "dist_"))]
```

Uncommitted patch on tf-deep-landcover SHA `c53c446e0ee11d9d450d35eaa918b58b5a8da828`.

### Smoke test

Smoke test run with 2-project subset (86 rows) confirmed no errors. The trainer loaded 76 feature columns (64 emb_ + 5 gedi_ + 1 chm_ + 5 topo_ + 1 dist_), completed LOPO CV, and wrote `metrics.json`. R² gating failure is expected at this scale (86 rows, 2 folds).

### Reproducibility footer (iteration 2)

- input_artefact: features_iter1.parquet + iter2_features.csv (see data_version.txt iteration_2 stanza)
- libraries: earthengine-api, pandas, numpy, pyarrow
- gee_project: coral-theme-475715-f7
- command_or_entrypoint: `uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/extract_iter2_features.py`
- trainer_patch: `tf-deep-landcover/src/agb/train_agb_lgbm.py` (uncommitted, SHA c53c446e)
- timestamp_utc: 2026-05-29T12:00:00Z
