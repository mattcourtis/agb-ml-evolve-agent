# Part 1 — Ireland-anchored feature-space + GT analysis

Ireland reference cloud: 141 plots (emb-only, codec). CAST AOA threshold (Ireland leave-one-out self-DI) = **0.768**; dbar = 10.902.

## Metric robustness

Spearman rank agreement of CAST-DI ordering with cross-checks: Mahalanobis ρ = **0.771**, weighted-centroid ρ = **0.954** (high → the ranking is not an artefact of one metric).

## Closest projects to Ireland (excluding biomass-flagged)

| project_name   | biome                                       |   n |   median_di_to_ireland |   pct_inside_ireland_aoa |   co2_median |
|:---------------|:--------------------------------------------|----:|-----------------------:|-------------------------:|-------------:|
| Kootznoowoo    | Temperate Conifer Forests                   | 166 |                   2.25 |                     0.00 |       137.33 |
| RainierGateway | Temperate Conifer Forests                   | 204 |                   2.57 |                     0.00 |       243.91 |
| FundyBay       | Temperate Broadleaf & Mixed Forests         | 201 |                   2.92 |                     0.00 |       140.54 |
| LongviewRanch  | Temperate Conifer Forests                   | 216 |                   3.06 |                     0.00 |        63.84 |
| CumberlandGap  | Temperate Broadleaf & Mixed Forests         | 310 |                   3.10 |                     0.00 |       125.33 |
| Soterra        | Temperate Grasslands, Savannas & Shrublands | 286 |                   3.10 |                     0.00 |       114.55 |
| WalkerLine     | Temperate Broadleaf & Mixed Forests         | 301 |                   3.13 |                     0.00 |       159.00 |
| ChestnutOak    | Temperate Broadleaf & Mixed Forests         | 244 |                   3.13 |                     0.00 |       161.35 |
| EagleMountain  | Temperate Broadleaf & Mixed Forests         | 144 |                   3.13 |                     0.00 |       137.05 |
| LittleBear     | Temperate Broadleaf & Mixed Forests         | 218 |                   3.14 |                     0.00 |       130.49 |
| KanawhaRiver   | Temperate Broadleaf & Mixed Forests         | 335 |                   3.16 |                     0.00 |       132.98 |
| Bluestone      | Temperate Broadleaf & Mixed Forests         | 249 |                   3.16 |                     0.00 |       164.41 |
| HighCascades   | Temperate Conifer Forests                   | 343 |                   3.16 |                     0.00 |        52.41 |
| ColeCrane      | Temperate Broadleaf & Mixed Forests         | 199 |                   3.16 |                     0.00 |       159.09 |
| GauleyRiver    | Temperate Broadleaf & Mixed Forests         | 199 |                   3.16 |                     0.00 |       150.36 |

## Biomass screen (erroneous / implausible)

Robust-z on per-project median CO2 (flag if z > 3.5 or named Quinte). Ireland has no CO2 GT, so this only bounds plausibility.

| project_name   |   co2_median |   co2_max |   co2_robust_z |
|:---------------|-------------:|----------:|---------------:|
| Quinte         |       311.14 |   1262.10 |           5.00 |


**Auto-dropped:** Quinte (Canada) — median CO2 311, max 1262 tCO2/acre.

## Honest read on achievable closeness

Closest project median DI to Ireland = **2.25** vs Ireland's own AOA threshold 0.77. Even the closest projects sit beyond Ireland self-similarity — this is near-extrapolation, not in-domain training.


Figures: `figures/pca_ireland_vs_anew.png`, `di_to_ireland_bar.png`, `co2_distribution_box.png`.

## Part 2 — Assembly + CV folds

Gap cut (step ≥ 0.1 in median DI) → **core = 4 projects, 787 plots**, 2 biome(s). Core DI 2.25–3.06 sits clearly below the flat shelf (3.10–3.97), where ~47 projects of all biomes are equidistant from Ireland and cannot be told apart.


### Recommended set — core (closest, gap-defined)

| project_name   | biome                               |   n |   median_di_to_ireland |
|:---------------|:------------------------------------|----:|-----------------------:|
| Kootznoowoo    | Temperate Conifer Forests           | 166 |                   2.25 |
| RainierGateway | Temperate Conifer Forests           | 204 |                   2.57 |
| FundyBay       | Temperate Broadleaf & Mixed Forests | 201 |                   2.92 |
| LongviewRanch  | Temperate Conifer Forests           | 216 |                   3.06 |


### Candidate sets for the next-phase bake-off (nested)

- **core** (4): gap-defined closest cluster. Recommended; CV = leave-one-project-out.

- **extended** (9, 2072 plots): core padded down the shelf to anti-overfit floors (≥8 projects, ≥2000 plots). Tests whether extra plots/variation beat staying closest. CV = grouped spatial K-fold.

- **all_minus_err** (51): all eligible projects (baseline).


### Spatial CV folds — extended set (whole projects intact)

|   extended_spatial_fold |   projects |   plots |   co2_med |   co2_min |   co2_max |
|------------------------:|-----------:|--------:|----------:|----------:|----------:|
|                     0.0 |        2.0 |   345.0 |     139.1 |       0.0 |     570.5 |
|                     1.0 |        2.0 |   420.0 |     129.2 |       0.0 |    1010.8 |
|                     2.0 |        3.0 |   855.0 |     145.6 |       0.0 |     702.4 |
|                     3.0 |        1.0 |   166.0 |     137.3 |       6.5 |     590.6 |
|                     4.0 |        1.0 |   286.0 |     114.6 |       0.0 |     467.7 |


Manifest `preprocessing/selected_projects.parquet`: `in_core` / `in_extended` / `in_all_minus_err` + `core_lopo_fold` + `extended_spatial_fold`.

## Part 3 — Candidate-set bake-off (result inverts the feature-proximity premise)

Emb-only LightGBM per candidate set. **Verdict rests on DB-independent metrics only** — within-set leave-one-project-out CV and the pseudo-Ireland transfer (hold out the closest project, predict it from its real ANEW label). Deep Biomass is shown for context but **NOT used to judge validity** (it has known issues). Ireland itself has no field ground-truth.

|               |   n_projects |   n_plots |   cv_lopo_rmse |   cv_lopo_r2 |   ireland_median_di |   ireland_pred_median |   ireland_vs_db_median_bias |   ireland_vs_db_mae |   ireland_vs_db_spearman |   err_vs_hdom_spearman |   pseudo_ireland_rmse |
|:--------------|-------------:|----------:|---------------:|-------------:|--------------------:|----------------------:|----------------------------:|--------------------:|-------------------------:|-----------------------:|----------------------:|
| core          |            4 |       787 |         126.33 |         0.05 |                0.90 |                184.28 |                      159.25 |              159.25 |                     0.38 |                   0.70 |                103.97 |
| extended      |            9 |      2072 |         106.33 |         0.05 |                0.84 |                132.04 |                      109.67 |              109.67 |                     0.32 |                   0.75 |                 93.58 |
| all_minus_err |           51 |     12636 |          73.69 |         0.32 |                0.94 |                135.57 |                      104.94 |              104.94 |                     0.28 |                   0.68 |                 92.65 |


**Winner on the real-label metrics: `all_minus_err`** — best pseudo-Ireland transfer RMSE (93) and best within-set CV (R² 0.32).


**The feature-closest `core` is the WORST, not the best.** It scores worst on both real-label metrics (CV R² 0.05, pseudo-transfer RMSE 104 vs 93). Feature-space proximity selected the oceanic-conifer projects (Kootznoowoo/RainierGateway — among the highest-biomass US forests); a narrow model trained on them generalises worst, even to the Ireland-like held-out project. **Embedding proximity does not imply transferable biomass labels.**


**`all_minus_err` wins** on the DB-independent metrics — more data and a broader label range give a flatter, better-calibrated, lower-variance map. If forced to train on US data only, use all of it, not a feature-matched subset.


**Ireland is extrapolation regardless of set.** Every set leaves Ireland 0% inside its AOA (median DI 0.90–0.94), and the sets **disagree strongly on the Ireland level** (median predictions 136–184 tCO2/acre; cross-model MAD up to 50) — the signature of unstable out-of-domain prediction, independent of any reference. The over-prediction also tracks stand height (err vs Hdom ρ≈0.7), pointing to a label-domain mismatch (US mature forest vs young Irish plantation) the emb-only model cannot resolve without structural (canopy-height) features.


**Bottom line:** project selection alone cannot make a trustworthy Ireland model. Ireland needs local field calibration and structural features; until then treat it as extrapolation under DI/AOA guardrails. The Part 1/2 closeness ranking still correctly identifies the *feature* analogues — it just does not translate to biomass level, and a narrow feature-matched set is actively worse than using all data.


Figure: `figures/bakeoff_ireland_levels.png`. Deep Biomass context median ≈ 25 tCO2/acre (unreliable — not a validity basis).
