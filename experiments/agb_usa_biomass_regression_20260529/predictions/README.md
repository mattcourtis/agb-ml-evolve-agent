# Bayfield County — wall-to-wall AGB inference

`bayfield_agb_30m.tif` — predicted CO₂ standing stock (tCO₂/acre), 30 m, EPSG:32615, nodata −9999.

- **Model:** LightGBM (`models/inference_model.txt`), 73 features =
  64 AEF embeddings + chm + topo×5 + corrected survey-relative dist + dstx disturbance features
  (`dstx_pre_ysd`, `dstx_pre_loss_5yr`, `dstx_loss_frac_buf`, `dstx_lt_mag`). Trained on all 4,636
  ANEW plots.
- **Embeddings:** local int8 AEF tiles (2023), cast to float **without dequantization** to match
  training; aggregated 10 m→30 m by mean.
- **Co-features:** GEE, survey_year=2023, focal reducers (~30 m) matching the training buffer.
- **Grid:** 1894×2863 px.
- **Stats:** AGB min 26.6, mean 129.5, max 262.1 tCO₂/acre.

**Caveats:** Bayfield is a *training* project, so predictions here are partly in-sample.
The model under-predicts high biomass and over-predicts low biomass (known dynamic-range
compression). Open in QGIS; style by the single float band.

## Embeddings-only comparison (low-end floor test)

`bayfield_agb_embonly_30m.tif` — same pipeline, model = 64 embeddings only
(`models/inference_model_embonly.txt`), to test whether the stale ETH-2020 `chm_m` drives the
~30 tCO₂/acre floor.

| map | min | p1 | p5 | median | %<30 |
| --- | --- | --- | --- | --- | --- |
| embeddings-only (64) | 12.4 | 21.6 | 29.5 | 119.3 | 5.2% |
| full (73-feat) | 26.6 | 38.6 | 53.2 | 133.9 | 0.0% |

- **OOF (plot-level) Q1 bias is identical** (+35.5 both; R² 0.418 emb-only vs 0.425 full) — on the
  in-distribution forest plots, dropping CHM does not change the low-end compression.
- **But the wall-to-wall map floor IS lower** without CHM (min 12.4 vs 26.6; 5.2% vs 0% of pixels
  <30). The stale 2020 canopy height was propping up predictions on non-forest / recently-cut
  pixels (absent from the forest-plot training set), reading tall canopy where the ground is now
  cleared and pulling those predictions up.
- **Residual floor is embedding/compression-driven**: emb-only still rarely predicts near 0
  (p1 = 21.6) — the +35.5 Q1 over-prediction persists. A full fix still needs calibration /
  current-epoch CHM / non-forest mask / hurdle model.

Trade-off: emb-only loses ~0.007 R² and the disturbance-aware signal, but gives a lower, more
realistic low-end for visual inspection.

## Non-forest mask

`bayfield_agb_30m_forestmasked.tif` — the full map with non-forest pixels set to 0 tCO₂/acre.
Forest = Dynamic World 2023 growing-season median `trees` probability ≥ 0.5
(`scripts/apply_forest_mask.py`; the mask layer is saved as `bayfield_dw_trees_prob_30m.tif`).

- Zeroed **4.6%** of county pixels (32,834) as non-forest → the map now reaches **0** where
  appropriate (fields, clearings, recent clearcuts DW reads as non-tree, developed/edge).
- Retained forest: min 29.3, mean 132.7, median 136.3 tCO₂/acre.
- The mask fixes the floor for genuinely non-forest land; it does **not** change the residual
  ~29 floor *within* forest (that's embedding compression — needs calibration / current-epoch CHM).
- Threshold (0.5) and base map are tunable: `--base bayfield_agb_embonly_30m.tif` etc.
- Caveat: a binary mask is crude — young/regenerating stands near the threshold may be mis-cut,
  and DW "trees" includes some non-stocked tree cover. DW 2023 matches the map epoch.

### Embeddings-only + forest mask (best low-end)

`bayfield_agb_embonly_30m_forestmasked.tif` — embeddings-only base + the same DW non-forest mask
(reused identical layer). Combines the lowest within-forest floor with zeroed non-forest:

| masked map | overall min | forest min (kept) | mean | %<30 | % zeroed |
| --- | --- | --- | --- | --- | --- |
| full (73-feat) + mask | 0.0 | 29.3 | 126.5 | 4.6% | 4.6% |
| embeddings-only + mask | 0.0 | 14.0 | 108.3 | 7.4% | 4.6% |

Emb-only+mask reaches 0 on non-forest AND lets genuinely-low forest fall to ~14 (vs ~29 for the
full model), so 7.4% of pixels are <30 (4.6% zeroed non-forest + ~2.8% low forest). This is the
most realistic low-end of the set, at the cost of ~0.007 LOPO R² and the disturbance-aware signal.
