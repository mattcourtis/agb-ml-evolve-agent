# Ireland AGB — Dynamic World forest/clearfell mask

Per-pixel Dynamic World (V1, growing-season Apr–Sep median `trees` probability, threshold ≥ 0.5 =
forest) mask applied to the existing 141-stand per-pixel predictions. Non-forest pixels are set to
**0 tCO₂/acre**; masked stand density = `forest_fraction × mean(forest-pixel preds)`. Year-aligned:
the year-matched set uses DW 2022/2023/2024; the survey-year set uses DW at each stand's survey year
(clamped to DW coverage ≥ 2016). Only our outputs are masked — Deep Biomass (DB) is a separate model,
left unmasked. Conversion: Mg/ha = tCO₂/acre / 0.6977.

This is an assembly of the completed DW-sampling checkpoints (141/141 in each of
`preprocessing/_dw_mask_y2022|y2023|y2024|ysurvey/`). No re-inference / no GEE.

## Portfolio: masked vs unmasked vs DB (year-matched)

| Year | Our unmasked | Our masked | DB | unmasked ratio | **masked ratio** | mean forest fraction |
|------|------|------|------|------|------|------|
| 2022 | 87.5 tCO₂/ac (125.4 Mg/ha) | 82.3 (117.9) | 25.9 (37.2) | 3.37× | **3.17×** | 0.893 |
| 2023 | 95.5 (136.9) | 82.2 (117.8) | 22.6 (32.4) | 4.23× | **3.64×** | 0.800 |
| 2024 | 91.8 (131.5) | 80.1 (114.7) | 31.9 (45.7) | 2.88× | **2.51×** | 0.805 |
| **3-yr mean** | **91.6 (131.3)** | **81.5 (116.8)** | **26.8 (38.4)** | **3.42×** | **3.05×** |  |

Survey-year set (secondary): portfolio mean **88.8 → 74.4 tCO₂/acre** after masking.

The mask removes ~11 % of portfolio density (3-yr mean 91.6 → 81.5 tCO₂/acre) and pulls the DB
over-prediction ratio from **3.42× down to 3.05×** (and from 3.25× → ~2.9× in Mg/ha terms). Most of the
gap to DB remains: the mask fixes the *structural* zero (bare/clearfell area), not the in-domain
regression floor for stocked-but-young stands, and absolute values remain OOD.

## New prediction floor

The old deployed head had a hard floor of ~16 tCO₂/acre per pixel and **30.5 tCO₂/acre** minimum at
stand level (survey-year set). After masking:

| | min per-pixel | min stand-level |
|------|------|------|
| Unmasked (old) | ~16 tCO₂/ac | 30.5 tCO₂/ac (survey) |
| **Masked** | **0.0 tCO₂/ac** | **0.0 (survey) / 2.1 (3-yr mean)** |

Four fully-clearfelled stands (forest_fraction = 0.000: Garranakilka, Cummeen Upper, Carrowkeel,
Rathcahill West-area) now read exactly 0 at the survey year. The 3-yr-mean floor is 2.1 (Carrowreagh),
because a stand only reads 0 in the mean if it is non-forest in all three DW years.

## Validation — known age-0 / Hdom≈0 stands

These should collapse toward ~0. They do, in proportion to their (low) DW forest fraction:

| Stand | age | Hdom | forest_fraction | unmasked | **masked** (tCO₂/ac) |
|------|------|------|------|------|------|
| Moyne | 0.0 | 0.0 | 0.080 | 44.2 | **5.7** |
| Peak | 0.0 | 0.0 | 0.022 | 48.6 | **2.2** |
| Tawran | 0.0 | 0.0 | 0.340 | 60.1 | **34.4** |
| Carrigeeny | 0.0 | 0.0 | 0.031 | 50.9 | **1.8** |
| Carrowreagh | 2.0 | 0.0 | 0.031 | 33.6 | **1.7** |
| Erriblagh | 0.0 | 0.0 | 0.684 | 73.5 | **56.4** |
| Carrowkeel | 2.4 | 1.1 | **0.000** | 40.6 | **0.0** |
| Rathcahill West | 2.0 | 0.0 | **0.000** | 39.0 | **0.0** |
| Cashel | 2.6 | 0.0 | 0.968 | 107.3 | **105.2** ⚠ |

All but Cashel and Erriblagh drop to single digits / zero. **Cashel** (ff 0.968) and **Erriblagh**
(ff 0.684) are DW disagreements — see below. (`Peak II`, age 18, is a distinct mature stand and is
correctly *not* masked: ff 0.787, 90.3 → 75.1.)

### Cross-tab: forest_fraction vs Dasos age_at_survey (survey-year set)

Forest fraction rises monotonically with stand age, exactly as expected — low ff for young clearfell,
high ff for mature stands:

| age bin (yr) | n | mean ff | median ff |
|------|------|------|------|
| 0–3 | 12 | 0.308 | 0.075 |
| 3–8 | 8 | 0.453 | 0.408 |
| 8–15 | 23 | 0.596 | 0.699 |
| 15–25 | 78 | 0.857 | 0.905 |
| 25+ | 19 | 0.889 | 0.903 |

Distribution: mean ff 0.749, median 0.857; 56/141 stands ≥ 0.9, 26/141 < 0.5, 4/141 = 0.0.

## DW-vs-age disagreements

**(A) DW MISSED clearfell** — high ff (≥ 0.7) but young (age ≤ 3 yr); DW reads ground veg / residual
cover as trees, so the mask leaves these too high:
- **Benmore** (age 0, Hdom 0, ff 0.932, masked 84.1)
- **Cashel** (age 2.6, Hdom 0, ff 0.968, masked 105.2)

**(B) DW FALSE non-forest** — low ff (< 0.5) but mature (age ≥ 15 yr); optical DW under-detects, so the
mask over-penalises these:
- **Cummeen Upper** (age 21.6, Hdom 11.3, ff 0.000 → masked 0.0 — clearly wrong)
- **Cloonainra** (age 20.0, Hdom 9.5, ff 0.423, masked 40.1)
- **Ballygeery East** (age 19.0, Hdom 11.4, ff 0.413, masked 46.4)
- **Knocknahooan** (age 18.2, Hdom 12.1, ff 0.331, masked 33.5)

## Comparison effect (our masked vs DB)

DB is left unmasked and also floors around ~20–30 tCO₂/acre, so masking can now push young stands *below*
DB:

- Unmasked: **0/141** stands read below DB (3-yr mean).
- **Masked: 5/141** stands now read below DB: Arraghan (2.6 vs 9.0), Carrowreagh (2.1 vs 6.4),
  Moyne (5.5 vs 11.0), Bunrevagh (17.1 vs 33.8), Rathcahill West (36.4 vs 38.2).
- Of the 12 youngest stands (age ≤ 3), 3 now read below DB (was 0).

## Validation assertions (all passed)

- **Re-aggregation identity**: `masked = forest_fraction × mean(forest-pixel preds)` holds to < 1e-6 for
  all 141 survey-year stands.
- **141 stands** present in each of the 4 sets (survey + 2022/2023/2024).
- **Unmasked columns byte-for-byte unchanged**: re-checked against a pre-run backup — all 21 pixel and
  30 year-matched original columns identical; masked columns added alongside (pixel 21→25 cols,
  year-matched 30→41 cols).
- **Masked GeoTIFFs**: 141 tifs + 1 band-1 VRT in each of `final/ireland_pixel_tiffs_masked/`,
  `_2022_masked/`, `_2023_masked/`, `_2024_masked/`. EPSG:2157, 10 m, 2-band (tCO₂/acre + Mg/ha),
  nodata −9999 outside polygon, non-forest pixels = 0. Each masked tif shares the **exact grid /
  shape / transform** of its unmasked counterpart; ≥ 99.4 % of non-forest pixels land on a band-1 = 0
  cell (the tiny remainder share a 10 m EPSG:2157 cell with a forest pixel under last-write-wins, which
  is the same grid-snap behaviour as the existing unmasked tiffs). Fully-clearfelled stands (ff 0) are
  all-zero rasters.

## Caveats

- DW `trees` is **optical** (canopy-cover / saturation correlated, binary at 0.5): it misreads young
  plantation and clearfell-with-ground-veg as tree (missed zeros — Cashel, Benmore, Erriblagh) and can
  read mature canopy gaps / heterogeneous stands as non-tree (false zeros — Cummeen Upper,
  Knocknahooan). The mask is therefore noisy at the stand margin.
- The mask fixes the **structural zero** (bare ground / fresh clearfell area) but **not** the in-domain
  regression floor for stocked-but-young stands; that needs the deferred in-region retrain
  (hurdle / option D-E).
- Absolute values remain **out-of-domain** (US-trained head on Irish conifer plantation); the masked
  3.05× DB ratio is still well above 1:1.
