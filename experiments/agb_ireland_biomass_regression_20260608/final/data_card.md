---
license: "Internal / Treefera proprietary"
language:
- en
tags:
- biomass
- forest-structure
- remote-sensing
- alphaearth-embeddings
pretty_name: "agb_ireland_biomass_regression_20260608 dataset (inference inputs + reference)"
size_categories:
- "n<1K"
---

# Data Card — Ireland AGB zero-shot transfer

## Dataset summary

The inference dataset for the zero-shot transfer of the `embdstx` head to the Irish Dasos forestry
portfolio: **141 dissolved Deep-Biomass Locations** (Sitka-dominant maritime-temperate plantation),
each with a **67-feature** vector (64 AlphaEarth optical embeddings affine-mapped into the training
codec + 3 Hansen disturbance-timing features) plus area-weighted stand covariates (age, Hdom, YC,
MainSp). Accompanied by a **Deep Biomass (DB) reference** per Location (an external satellite-inferred
model used as a directional **lower bound**, NOT ground truth). Temporal window: survey years
2017–2025 (17 Locations clamped from 2015/16 to the 2017 AEF floor). The data is **AGB-only**
(above-ground), used for inference + model-vs-model comparison only — there is NO training split and
NO Irish ground truth.

## Sources

| source_name | url_or_doi | access_date | license |
|---|---|---|---|
| Deep Biomass aggregated CSV (per-Location AGB) | `…/dasos-ireland/deepbiomass-model-outputs/Deep Biomass - Aggregated Data & Portfolio Summary.csv` (sha256[:16] `8f34cb1eae381395`, 11721 B) | 2026-06-08 | Treefera internal |
| Dasos geometry (sub-compartments) | `…/boundary-files/dasos_fgl_2025ye.gpkg` layer `fgl_2025ye_` (sha256[:16] `b453cec8d320cc10`) | 2026-06-08 | Dasos / Treefera internal |
| AlphaEarth embeddings | GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (A00..A63) | 2026-06-08 | Google EE terms |
| Hansen forest change | GEE `UMD/hansen/global_forest_change_2025_v1_13` (lossyear) | 2026-06-08 | Hansen/GLAD (CC-BY) |
| Training-parquet (encoding-codec reference) | `…/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet` (sha256[:16] `3ff73b956c3043d2`) | 2026-06-08 | Treefera internal |

## Collection window

- start_date: 2017 (AEF coverage floor; 17 Locations clamped from 2015/16)
- end_date: 2025 (AEF coverage ceiling)
- label_source_revision_tag: Deep Biomass aggregated CSV (reference only; not labels); Hansen 2025 v1.13

## Geography

- covered_regions: Ireland — Dasos forestry portfolio, 141 dissolved Locations (1,053 sub-compartments)
- exclusions: none within the portfolio; no field plots exist (no GT)

## Splits

Matches ACCEPTED `configs/split_strategy.yaml` = `none_zero_shot_transfer` (waived).

| split | n_units | partition_key | partition_value_count |
|---|---:|---|---:|
| train | 0 | — (no training) | 0 |
| val   | 0 | — | 0 |
| test  | 141 (all Locations, evaluated vs DB reference) | `Location_Name` | 141 |

## Label generation

**No labels.** There is NO Irish ground truth. The Deep Biomass reference is a satellite-inferred
product (known under-estimator); it is used strictly as a **directional lower bound** for a
model-vs-model comparison, never as a training/eval label. DB values are converted Mg/ha →
tCO₂/acre by **×0.6977** (= 0.47 · 3.667 · 0.4047), both sides AGB-only; DB density Mg/ha = cell
tonnes ÷ `Area_Ha`. Per design rule, GEDI L4A/L4B and ESA CCI Biomass are NOT used as labels here.

## Known limitations and biases

- DB is a known under-estimator; "divergence" is never "error vs truth".
- **Severe domain shift**: 100% of Locations beyond the 99th-pct training Mahalanobis radius.
- **Missingness** (`evaluation_matrix.yaml`): MainSp / PlantingYe / age each 0.0071 (1/141 Location).
- 17/141 Locations use the pre-2017 AEF fallback (survey 2015/16 → 2017); shown to cause no
  detectable divergence distortion.
- Per-band affine encoding is not pixel-perfect (held-out reconstruction RMSE ~31% of band-σ); the
  contract holds on central tendency (corr 0.986, slope median 1.006).

## Intended use

- Primary use: inference inputs for zero-shot transfer + model-vs-model (vs DB) characterisation.
- Out-of-scope use: training/fine-tuning (no labels); calibrated absolute-level reporting (deep OOD).

## Citation

```
Treefera (2026). Ireland AGB zero-shot transfer dataset — agb_ireland_biomass_regression_20260608.
Internal. Deep Biomass reference (directional lower bound, not ground truth).
```

## Versioning

- content_sha256: `aee82d7b17fb357dbbc466c88c0c4e6317742b02951ad28c7068692600e519ad`
  (sha256 of `preprocessing/ireland_features.parquet`, 141×67). Per-input sha256[:16] recorded in
  `final/preprocessing_pipeline/data_version.txt`.
- snapshot_timestamp_utc: 2026-06-08 (extraction_date in `data_version.txt`)
- input_manifest_path: `final/preprocessing_pipeline/data_version.txt`
