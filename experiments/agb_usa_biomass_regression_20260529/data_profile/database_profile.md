# Database Profile — ANEW field plots (joint_v2 pool)

## Source

- **Label/plot source:** `/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg`
  - SHA256: `b0e490b7ceb5bd59acd5965fefbe705ffe6707ab140575c9f9c49709b6a7393d`
  - 12,837 plots / 51 projects total (full CONUS + Alaska + Ontario). CRS EPSG:4326.
- **Feature source (reused, iteration 0):**
  `/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet`
  (64-dim AlphaEarth Foundation (AEF) embeddings, 10 m annual, sampled at plot locations; produced by the
  sibling extractor at tf-deep-landcover SHA `e8c70584`).

## gpkg schema (relevant columns)

| column | type | role |
|---|---|---|
| `project_name` | str | LOPO partition key; component of join key |
| `Plot_ID` | float | per-plot id — **resets per project** (not globally unique) |
| `CO2` | float | **target** — standing stock tCO₂/acre |
| `Annual_CO2` | float | annual increment — deferred (weaker satellite signal) |
| `ECO_NAME` | str | ecoregion name — used for `per_ecoregion_r2` |
| `ECO_ID` | int | ecoregion id |
| `geometry` | Point | EPSG:4326 |

**Join key for ecoregion enrichment:** composite `(project_name, Plot_ID)` — `Plot_ID`
alone is not unique. The oof predictions carry both columns; join is lossless (0 nulls).

## joint_v2 pool (the 23-project subset used here)

- **Feature rows:** 4,646. **Modelled rows:** 4,636 (10 dropped as extraction failures —
  `failure` flag in features.parquet; the trainer drops them, giving n=4,636 matching the
  baseline).
- **Projects:** 23. **Years pooled:** 2022 + 2023.
- **Target range:** `CO2` ∈ [0, 521] tCO₂/acre.
- **Region (bloc) distribution:** Midwest (`mw`) 2,631 · Northeast (`ne`) 1,407 · WV (`wv`) 598.
- **Ecoregion (`ECO_NAME`) distribution (n=4,636):**

| ECO_NAME | n |
|---|---:|
| Western Great Lakes forests | 2,619 |
| New England-Acadian forests | 1,407 |
| Appalachian mixed mesophytic forests | 598 |
| Upper Midwest US forest-savanna transition | 12 |

## Feature schema (features.parquet)

`plot_id, project_name, year, lon, lat, target, failure, region` + `emb_00 … emb_63`
(72 columns). `target` has 0 nulls. Model features = the 64 embedding columns only.

## Leakage assessment

- **No target leakage:** features are satellite embeddings; the target `CO2` is field-measured
  and is not a derived input.
- **Spatial generalisation:** CV is leave-one-project-out on `project_name`, so every reported
  metric is on a project the model never saw — the realistic "new project" expectation. No
  random splitting.
- **Temporal:** 2022/2023 are pooled; `year` is not a model feature here (embeddings are annual).
  No train/test year overlap concern at the plot level because the split is by project, not year.

## Reproducibility footer

- input_artefact_sha256: `b0e490b7ceb5bd59acd5965fefbe705ffe6707ab140575c9f9c49709b6a7393d` (gpkg)
- source URIs: ANEW gpkg (above); features.parquet (above) @ tf-deep-landcover `e8c70584`
- snapshot timestamp_utc: 2026-05-29T09:36:00Z
- label-source revision: ANEW `anew_gt_with_eco_info.gpkg` (with eco_info)
- libraries: geopandas, pandas (read-only profiling)
- seed: n/a
