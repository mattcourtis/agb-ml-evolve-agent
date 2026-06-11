# Data store — moved outputs

The data outputs for this experiment (rasters, vectors, parquets, per-pixel tiff
tiles, batch intermediates, figures, the HTML report and the model artefacts) are
**not tracked in git** — they live in the shared EFS data-space to keep the repo
lightweight. The repo retains only code, docs, configs and specs.

## Location

```
/home/mattc/data-space/carbonmap-embeddings/agb_ireland_biomass_regression_20260608/
```

Sub-paths mirror this experiment's directory layout exactly — e.g. a report that
references `final/figures/foo.png` maps to
`…/agb_ireland_biomass_regression_20260608/final/figures/foo.png` in the data-space.

Total: ~3,518 files, ~102 MB.

## Layout

| sub-tree | files | contents |
|---|---|---|
| `preprocessing/` | 2,334 | features, embeddings, similarity scores, DW-mask batches (`_dw_mask_*`), pool batches (`_pool_batches`), pixel-pred batches (`_pixel_pred*`) |
| `final/` | 1,164 | predictions raster, gpkg/csv/parquet tables, report HTML, per-pixel tiff tiles (`ireland_pixel_tiffs*`), figures, model artefacts |
| `evaluation/` | 13 | prediction parquets, support-sensitivity, figures |
| `error_analysis/` | 5 | merged diagnostics, figures |
| `data_profile/` | 2 | embedding sample, crosswalk CSV |

## Key deliverables

| artefact | path (under the location above) |
|---|---|
| Standing-stock prediction raster (10 m) | `final/ireland_agb_predictions_stand_10m.tif` |
| Stand-level predictions (vector) | `final/ireland_agb_predictions.gpkg` (+ `.csv`) |
| Per-pixel predictions (vector) | `final/ireland_agb_pixel.gpkg` (+ `.csv`, `.parquet`) |
| Year-matched DeepBiomass comparison | `final/ireland_agb_yearmatched.gpkg` (+ `.csv`, `.parquet`) |
| HTML report | `final/ireland_agb_report.html` |
| Model head (LightGBM, 73 trees) | `final/model/inference_model_embdstx.txt` (+ `inference_features_embdstx.json`) |
| Feature matrix (141×67) | `preprocessing/ireland_features.parquet` |

> Figures referenced relatively in the committed `*.md` reports resolve under the
> data-space location above. To regenerate the in-repo deliverables, re-run the
> `scripts/` and `final/build_report.py` against the inputs noted in
> `final/run_summary.md`.
