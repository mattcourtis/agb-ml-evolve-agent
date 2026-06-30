# Data store — agb_anew_gt_applicability_20260626

Code, docs, and figures live in git. All data artifacts live in the data-space (outside
the repo, not tracked):

`/home/mattc/data-space/carbonmap-embeddings/agb_anew_gt_applicability_20260626/`

| Path | Rows | Description |
|---|---|---|
| `analysis/project_di_ranking.parquet` | 51 | One row per project: biome, bloc_id, n, median_di_lopo, iqr_di_lopo, pct_inside_aoa, median_di_bloc, regional_dependence, co2_median. Sorted by median_di_lopo (the dissimilarity ranking). |
| `analysis/plot_level_di.parquet` | 12,636 | Per-plot: project_name, BIOME_NAME, lon, lat, bloc_id, di_lopo, di_bloc, inside_aoa, CO2. Backs the maps. |
| `analysis/thresholds.json` | — | threshold_cast (0.558), p95/p99, k_blocs, bloc_sizes, Spearman ρ (weighted-vs-unweighted, weighted-vs-Mahalanobis). |
| `preprocessing/bloc_assignments.parquet` | 51 | project_name → bloc_id, with lon/lat and EPSG:5070 x/y centroid. |

## Regeneration

Deterministic from the canonical codec store; no GEE.

```
uv run --project /home/mattc/code/agb-ml-agent-evolve \
    python scripts/anew_gt_applicability/compute_di_folds.py   # analysis + bloc parquets
uv run --project /home/mattc/code/agb-ml-agent-evolve \
    python scripts/anew_gt_applicability/make_maps.py          # 5 figures -> experiments/.../figures/
```

Inputs: `agb_trust_aoa_20260626/preprocessing/anew_canonical_codec.parquet` (canonical GT,
codec space) and `models/inference_model_embonly.txt` (gain weights). Quinte is dropped in
`compute_di_folds.py` (`DROP_PROJECTS`).
