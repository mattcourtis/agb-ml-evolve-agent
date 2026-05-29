# Data Card — ANEW joint_v2 (agb_usa iteration 0)

## Source

- **Labels:** ANEW client field plots —
  `/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg`
  - SHA256 `b0e490b7ceb5bd59acd5965fefbe705ffe6707ab140575c9f9c49709b6a7393d`
  - 12,837 plots / 51 projects total (CONUS + Alaska + Ontario); CRS EPSG:4326.
- **Features:** 64-dim AlphaEarth Foundation (AEF) embeddings (10 m, annual), Source Coop COG (anonymous),
  sampled at plot locations; reused from
  `tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet` @ `e8c70584...`.

## Subset used (joint_v2)

- **Modelled:** 4,636 plots, 23 projects (4,646 extracted; 10 dropped as extraction failures).
- **Years:** 2022 + 2023 pooled.
- **Target:** `CO2` (standing stock, tCO₂/acre), range [0, 521].
- **Region blocs:** Midwest 2,631 · Northeast 1,407 · WV 598.
- **Ecoregions (`ECO_NAME`):** Western Great Lakes forests 2,619 · New England-Acadian 1,407 ·
  Appalachian mixed mesophytic 598 · Upper Midwest forest-savanna transition 12.

## Plot design

1/24-acre circular plots (~14.7 m radius), rebar centre marker, ~10 m GPS error.

## Schema

`project_name, Plot_ID (resets per project), CO2 (target), Annual_CO2 (deferred), ECO_NAME,
ECO_ID, geometry (Point EPSG:4326)`. Feature parquet adds `plot_id, year, lon, lat, target,
failure, region, emb_00..emb_63`.

## Known issues / exclusions

- `Plot_ID` is **not globally unique** — ecoregion joins use `(project_name, Plot_ID)`.
- v1 exclusions carried implicitly (Quinte Ontario suspected unit issue; Alaska; PNW old-growth).
- `Annual_CO2` deferred — annual increment is a much weaker satellite signal.

## Provenance footer

- input_manifest_sha256: `b0e490b7ceb5bd59acd5965fefbe705ffe6707ab140575c9f9c49709b6a7393d`
- snapshot_timestamp_utc: 2026-05-29T09:36:00Z
- label_source_revision: `anew_gt_with_eco_info.gpkg`
- tf_deep_landcover_sha: `e8c70584fb1a8705308004fbed123392c8f51654`
