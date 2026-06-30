# ANEW GT applicability: self-referential DI & AOA

**Date:** 2026-06-29 · **Experiment:** `agb_anew_gt_applicability_20260626`

## Summary

We applied the trust toolkit's Dissimilarity Index (DI) and Area of Applicability (AOA)
to the ANEW ground-truth **itself** — scoring every plot of the 51 eligible projects (52
minus the Quinte label outlier; 12,636 plots) against the rest of the GT cloud in
emb-only codec space. This produces a project dissimilarity ranking, a self-referential
AOA classification, and a regional-dependence view. The headline: **94% of all GT plots
fall inside the self-referential AOA**, and the GT space is overwhelmingly one dense
temperate-broadleaf interior with a small, genuinely isolated Alaska/Pacific-Northwest
conifer-and-tundra frontier.

See `analysis/methodology.md` for the full method and the principle behind the
leave-one-project-out reference frame.

## Cross-fold design

Two fold-aware DI passes, both the same `di.fit()` call with a different grouping array:

- **Project fold (LOPO)** — each plot scored against *other projects*. AOA threshold
  `Q75 + 1.5·IQR = 0.558`.
- **Regional fold (leave-bloc-out)** — each plot scored against *other spatial blocs*
  (KMeans on project centroids in EPSG:5070, **K = 4**: Lake-States, Northeast,
  Alaska+PNW, Appalachia/SE). The lift over LOPO is each project's regional dependence.

**Robustness:** the LOPO ranking agrees with an unweighted CAST ranking (Spearman ρ = 1.00)
and a Mahalanobis ranking (ρ = 0.85) — not a weighting or single-metric artefact.

## Project dissimilarity ranking (interior → frontier)

Median LOPO DI per project; AOA threshold = 0.558. Lowest = most redundant / interior.

| Project | Biome | Bloc | n | median DI | % inside AOA | regional dep. |
|---|---|---|---|---|---|---|
| RuskCounty | Broadleaf | 0 | 258 | 0.198 | 99.2 | 0.123 |
| AppalachianHollows | Broadleaf | 3 | 252 | 0.208 | 100.0 | 0.309 |
| McCoy | Broadleaf | 3 | 219 | 0.213 | 98.6 | 0.192 |
| GauleyRiver | Broadleaf | 3 | 199 | 0.213 | 100.0 | 0.292 |
| IronCounty | Broadleaf | 0 | 307 | 0.214 | 100.0 | 0.117 |
| … 35 more broadleaf projects, DI 0.215–0.32, ~100% inside … | | | | | | |
| LouisianaLowlands | Broadleaf | 3 | 274 | 0.351 | 100.0 | **0.622** |
| Apalachicola | Grassland | 3 | 436 | 0.360 | 97.5 | **0.573** |
| EagleMountain | Broadleaf | 1 | 144 | 0.374 | 99.3 | 0.159 |
| FundyBay | Broadleaf | 1 | 201 | 0.393 | 98.5 | 0.140 |
| Hartwood | Broadleaf | 1 | 149 | 0.397 | 98.0 | 0.054 |
| **Soterra** | Grassland | 3 | 286 | 0.445 | 85.0 | 0.477 |
| **HighCascades** | Conifer | 2 | 343 | 0.523 | 65.3 | 0.429 |
| **LongviewRanch** | Conifer | 2 | 216 | 0.567 | 46.8 | 0.402 |
| **RainierGateway** | Conifer | 2 | 204 | 0.689 | 2.0 | 0.076 |
| **Doyon** | Tundra | 2 | 102 | 0.700 | 0.0 | −0.001 |
| **Kootznoowoo** | Conifer | 2 | 166 | 0.775 | 0.0 | 0.146 |

Full table: `analysis/project_di_ranking.parquet` (all 51 rows).

## Findings

1. **The GT is one dense broadleaf interior.** 44 of 51 projects are Temperate Broadleaf &
   Mixed Forests with median DI 0.20–0.40 and ~100% of plots inside the AOA. They are
   highly mutually redundant — dropping any one barely changes coverage. This is *why* the
   Ireland bake-off found breadth (all-minus-err) beat the feature-closest core: the
   interior is dense enough that more projects keep adding label diversity at near-zero
   coverage cost.

2. **A real frontier exists, and it is conifer/tundra in Alaska + the PNW.** Kootznoowoo,
   Doyon, RainierGateway, LongviewRanch, HighCascades sit at DI 0.52–0.78 with 0–65% of
   plots inside the AOA. These five are the only projects the GT cannot vouch for from its
   own interior — they are the model's genuine extrapolation edge.

3. **Regional dependence separates two kinds of "far".** The leave-bloc-out lift reveals
   which frontier projects are *propped up by regional neighbours* vs *isolated regardless*:
   - **High dependence** (collapse when their region is removed): LouisianaLowlands (+0.62),
     Apalachicola (+0.57), Soterra (+0.48), HighCascades (+0.43), LongviewRanch (+0.40).
     These look in-domain under LOPO only because a sibling project sits nearby.
   - **Self-standing frontier** (≈ unchanged): Kootznoowoo (+0.15), Doyon (−0.00),
     RainierGateway (+0.08) — outliers no matter how the folds are drawn.

   This is the practical payload: adding a *single* project near a high-dependence frontier
   (e.g. another Gulf-coast grassland) would pull a whole cluster inside the AOA, whereas the
   self-standing AK conifer/tundra needs its own local ground truth.

4. **Biome tracks the ranking but does not equal it.** All five frontier projects are
   non-broadleaf (conifer/tundra/grassland), but Apalachicola and Soterra (grassland) are
   far less dissimilar than the AK conifer — coverage is a continuum in embedding space, not
   a biome label.

## Figures (`figures/`)

- `aoa_national_map.png` — plots in lon/lat, coloured by LOPO DI, inside/outside marker.
- `di_ranking_bar.png` — per-project median DI, coloured by bloc, AOA threshold line.
- `di_box_by_project.png` — per-project DI distributions ordered by median.
- `pca_gt_space.png` — weighted emb PCA of the GT space by spatial bloc.
- `lopo_vs_bloc_scatter.png` — LOPO vs bloc DI; distance above y=x is regional dependence.

## Limitations

- **Emb-only.** No structural features (CHM/topo/disturbance) — only available for the 23
  modelled projects. A 73-feature pass needs GEE extraction for the 28 unused projects.
- **Plot-level maps only.** Wall-to-wall raster AOA over project boundaries is a documented
  TODO (`scripts/anew_gt_applicability/extract_aoa_rasters.py`); the DI maths
  (`DISpace.di_fast`) is ready, it needs boundaries + a GEE embedding extraction.
- **Codec affine verified only for Temperate Broadleaf** (fit on Bayfield); conifer/tundra
  codec values are less certain, compounding their frontier status.

## Next steps (TODO)

- Raster AOA maps over user-supplied project boundaries (the deferred companion).
- 73-feature full-space pass once covariates are extracted for the 28 unused projects, to
  test whether structural features split the broadleaf interior that emb-only collapses.
- Feed the regional-dependence finding into acquisition planning: prioritise one project per
  high-dependence frontier cluster over more interior broadleaf plots.
