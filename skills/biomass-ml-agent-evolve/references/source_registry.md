# Source Registry

## Purpose

Give the Research Actor a stable source hierarchy for biomass and forest-structure ML experiments. This registry is a starting point, not a substitute for task-specific research. The Research Actor must verify current availability, licensing, spatial/temporal coverage, and access requirements before using any source.

## Preferred source hierarchy

### Spaceborne LiDAR — canopy structure

| Source | Resolution / coverage | Preferred access route | Auth | Licence | Notes |
|---|---|---|---|---|---|
| GEDI L2A v002 (RH metrics, footprint ~25 m) | ~25 m footprint, 51.6°S–51.6°N, 2019–present | GEE `LARSE/GEDI/GEDI02_A_002_MONTHLY`; or LP DAAC via `earthaccess` (Python pkg) | GEE project auth; or free NASA Earthdata account for LP DAAC | NASA open data (CC0) | Primary structural feature — the identified lever for breaking the optical-embedding biomass ceiling |
| GEDI L4A v002 (footprint AGBD) | ~25 m footprint, same coverage | GEE `LARSE/GEDI/GEDI04_A_002_MONTHLY`; or LP DAAC via `earthaccess` | Same as L2A | NASA open data (CC0) | Directly comparable label to ANEW CO2 target |
| GEDI L4B v002 (gridded AGBD 1 km) | 1 km, same coverage | GEE `LARSE/GEDI/GEDI04_B_002` | GEE project auth | NASA open data (CC0) | Gridded product — use for benchmark comparison only, not as plot-level features |
| ICESat-2 ATL08 R006 (terrain/canopy heights) | 100 m segments, near-global | LP DAAC via `earthaccess`; NSIDC | Free NASA Earthdata account | NASA open data (CC0) | Gap-fill outside GEDI latitudinal coverage (>51.6°); CONUS fully covered |

### Airborne LiDAR — high-quality co-supervision

| Source | Resolution / coverage | Preferred access route | Auth | Licence | Notes |
|---|---|---|---|---|---|
| NEON AOP CHM (DP3.30015.001) | 1 m, NEON sites only (~80 US sites) | NEON Data Portal API `https://data.neonscience.org/api/v0`; or `neonscience` Python pkg | Anonymous (public) | CC BY 4.0 | Co-target supervision; bias diagnostics at NEON sites |
| NEON AOP Biomass (DP3.30055.001) | 30 m, NEON sites only | Same as above | Anonymous (public) | CC BY 4.0 | Direct biomass co-target; limited to NEON footprints |

### Optical foundation embeddings

| Source | Resolution / coverage | Preferred access route | Auth | Licence | Notes |
|---|---|---|---|---|---|
| AlphaEarth Foundation (AEF) — 64-dim int8 | 10 m, global, 2018–2024 (Source Coop); 2017–2025 (GCS) | Source Coop direct COG: `s3://us-west-2.opendata.source.coop/` (anonymous); code: `src/fetch/embeddings.py --source source-coop` | Anonymous (Source Coop); GCS credentials for 2017/2025 years | Check Source Coop listing; internal / project licence | Primary feature carrier in current AGB production |
| Presto | Variable | Hugging Face Hub | HF token | Apache-2.0 | Ensemble candidate |
| Prithvi | Variable | Hugging Face Hub `ibm-nasa-geospatial/Prithvi-100M` | HF token | Apache-2.0 | Ensemble candidate |
| Clay | Variable | Hugging Face Hub `made-with-ml/clay-v1` | HF token | Apache-2.0 | Ensemble candidate |
| SatMAE / GeoFM | Variable | Hugging Face Hub | HF token | See individual model cards | Ensemble candidate |

### Optical imagery

| Source | Resolution / coverage | GEE asset | Auth | Licence |
|---|---|---|---|---|
| Sentinel-2 Harmonised SR (GEE) | 10–60 m, global, 2017–present | `COPERNICUS/S2_SR_HARMONIZED` | GEE project auth | Copernicus Sentinel Data Terms (free) |
| Landsat 8 Collection 2 SR (GEE) | 30 m, global, 2013–present | `LANDSAT/LC08/C02/T1_L2` | GEE project auth | USGS public domain |
| Landsat 9 Collection 2 SR (GEE) | 30 m, global, 2021–present | `LANDSAT/LC09/C02/T1_L2` | GEE project auth | USGS public domain |
| HLS (Harmonised Landsat/Sentinel-2) | 30 m, global | LP DAAC or GEE | GEE project auth or NASA Earthdata | NASA open data | Useful for temporal consistency across sensors |

### SAR

| Source | Resolution / coverage | GEE asset | Auth | Licence | Notes |
|---|---|---|---|---|---|
| JAXA ALOS PALSAR-2 annual mosaic | 25 m, global | `JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH` | GEE project auth | JAXA non-commercial research | AGB ablation: PALSAR-2 alone ≤+0.02 R²; retest in combination with GEDI |
| Sentinel-1 GRD | 10 m, global, 2014–present | `COPERNICUS/S1_GRD` | GEE project auth | Copernicus Sentinel Data Terms (free) | |

### Topography

| Source | Resolution / coverage | Preferred access route | Auth | Licence | Notes |
|---|---|---|---|---|---|
| **Copernicus DEM GLO-30 (COPDEM)** | **30 m, global** | **GEE `COPERNICUS/DEM/GLO30` (preferred — no download required); AWS S3 `s3://copernicus-dem-30m/` (requester-pays); OpenTopography API (free, rate-limited)** | **GEE project auth; or requester-pays AWS; or anonymous OpenTopography** | **Copernicus DEM Licence (free commercial + non-commercial)** | **Preferred DEM. Slope, aspect, TPI, TWI and other derivatives at Research Actor discretion — compute via `ee.Terrain.products()` in GEE or `richdem`/`pysheds` locally** |
| SRTM v3 (SRTMGL1) | 30 m (~1 arc-sec), 60°S–60°N | GEE `USGS/SRTMGL1_003` | GEE project auth | USGS public domain | Fallback if COPDEM coverage gaps arise |
| ASTER GDEM v3 | 30 m, 83°S–83°N | GEE `NASA/ASTER_GED/AG100_003`; LP DAAC | GEE project auth | NASA/METI (free) | Fallback; known artefacts in low-relief areas |

### Disturbance and cover masks

| Source | Resolution / coverage | GEE asset | Auth | Licence |
|---|---|---|---|---|
| Hansen Global Forest Change 2023 (UMD) | 30 m, global | `UMD/hansen/global_forest_change_2023_v1_11` | GEE project auth | CC BY 4.0 |
| NLCD 2021 | 30 m, CONUS | `USGS/NLCD_RELEASES/2021_REL/NLCD` | GEE project auth | USGS public domain |
| LCMAP v1.3 | 30 m, CONUS | `USGS/LCMAP/CU/V13/LCPRI` | GEE project auth | USGS public domain |
| GLAD ARD | 30 m, global | GEE collection | GEE project auth | Free research use |

### Spaceborne biomass products

| Source | Resolution / coverage | Access route | Auth | Licence | Notes |
|---|---|---|---|---|---|
| ESA CCI Biomass v5 (AGB) | 100 m, global, 2010/2015/2017–2020 | ESA CCI Data Portal `https://climate.esa.int/en/projects/biomass/` | Free ECMWF/Copernicus CDS account | Copernicus Climate Change Service Licence (free) | Use for benchmark cross-validation; do not borrow gridded R² as plot-level target |
| JPL/GFW biomass mosaics | ~30–100 m | Global Forest Watch API / direct download | Anonymous (public) | CC BY 4.0 | Label augmentation in low-supervision regions |

### Ground truth

| Source | Notes | Access route | Auth | Licence |
|---|---|---|---|---|
| ANEW field plots (primary) | 1/24-acre circular, ~14.7 m radius; CONUS; target column `CO2` (tCO₂/acre standing stock) | Project-internal gpkg: `/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg` | Local (no external auth) | Project-internal / client data — do not publish |
| USFS FIA (public CONUS) | National Forest Inventory; plot-level AGB | FIA DataMart `https://apps.fs.usda.gov/fia/datamart/` | Anonymous (public) | USDA public domain |
| NEON Woody Vegetation Structure (DP1.10098.001) | Stem-level surveys at NEON sites | NEON Data Portal API | Anonymous (public) | CC BY 4.0 |
| Forest Observatory System (FOS) | Multi-network global plot data | Contact FOS directly | Collaboration agreement | Varies |

### Climate covariates

| Source | Resolution / coverage | GEE asset | Auth | Licence |
|---|---|---|---|---|
| TerraClimate | ~4 km, global, 1958–present | `IDAHO_EPSCOR/TERRACLIMATE` | GEE project auth | CC BY 4.0 |
| ERA5-Land monthly | ~9 km, global, 1950–present | `ECMWF/ERA5_LAND/MONTHLY_AGGR` (GEE); Copernicus CDS | GEE project auth; or free ECMWF CDS account | Copernicus Climate Change Service Licence |
| PRISM (CONUS only) | ~4 km, CONUS, 1981–present | `OREGONSTATE/PRISM/AN81m` (GEE) | GEE project auth | Free non-commercial |
| CHIRPS | ~5.5 km, 50°S–50°N, 1981–present | `UCSB-CHG/CHIRPS/DAILY` (GEE) | GEE project auth | Public domain |

### Infrastructure

| Source | Access route | Auth |
|---|---|---|
| Hugging Face Hub | `huggingface_hub` Python pkg; `HF_TOKEN` env var | Free HF account |
| AWS S3 (us-west-2) | `boto3` / `s3fs`; `AWS_PROFILE` or `AWS_ACCESS_KEY_ID` | AWS credentials |

## Actor rules

- Prefer official or benchmark-grade sources over informal mirrors.
- Record source URL, access date, licence, spatial resolution, temporal coverage, and known revisions in every artefact that cites a source.
- Do not use a source for labels or targets until the Database Profiling Actor verifies schema, geography, time coverage, and label/target availability.
- Re-anchor benchmark expectations to the exact subject, geography, ecoregion, lead time, split type, and label quality.
- Treat random-split benchmark results as background only when the task requires spatial or temporal generalisation.
- For LiDAR sources, record the exact product version (e.g., GEDI L4A v2.1, ICESat-2 ATL08 R006) — products are revised and earlier versions may have known biases since superseded.
- Foundation-embedding sources must record the exact model checkpoint, training-data window, and known coverage gaps (e.g., AEF year availability, geographic blind spots).
- For GEE sources, verify the asset ID is current before use — asset paths change between GEE catalogue updates.
- COPDEM GLO-30 is the preferred DEM over SRTM — it has better void-filling and higher vertical accuracy, especially in forested terrain.
