# AGB Trust Layer — Data Audit + DI/AOA (agb_trust_aoa_20260626)

Applies the `model_trust_agent_guide.md` workflow to the deployed AGB model so predictions
into the wider regional prediction space carry an applicability verdict and an expected-error
number. Scope this pass (user-set): audit + DI/AOA module, **no new GEE**, importance-weighted
CAST AOA, production retraining out of scope. Plan: `plans/compiled-scribbling-umbrella.md`.

All code is in `scripts/trust/`. Data outputs (gitignored) live under
`/home/mattc/data-space/carbonmap-embeddings/agb_trust_aoa_20260626/`.

## Part A — Audit (`scripts/trust/run_data_audit.py` → `audit/data_audit.md`)

- **Encodings identified** for all artefacts: the pool (`iter1_pool_embeddings.parquet`, all 52
  ANEW projects + Ireland) is raw GEE float; the deployed training set and Ireland features are
  training-codec. Earlier exploratory `fig7`/`fig8` mixed these — now corrected.
- **Affine verified within the modelled biome.** Applying the Bayfield-fit affine to each of the
  23 modelled projects' raw embeddings reproduces their codec values at per-plot corr ≥ 0.95
  (23/23). Slope-band drift (~0.83–0.89) is concentrated in WV but direction is faithful.
- **Unverified biomes (flagged):** all 23 modelled projects are Temperate Broadleaf. The affine is
  therefore unverified for the unused biomes (Temperate Conifer, Boreal/Taiga, Tundra, Temperate
  Grasslands) — no codec anchor exists there. It is the AEF asset's global dequantisation so it
  should hold, but DI on those projects rests on an unverified mapping.
- **Other flags:** 0 survey-year mismatches; 3 duplicate pool keys (dropped); pool CO2 ranges to
  **1262 tCO2/acre**, 2.4× the deployed model's training max of 521 (extreme-biomass PNW/old-growth);
  the deployed model carries leakage-safe `dstx_*` (not the legacy `dist_years_since`).

**Gate decision:** proceed — encodings consistent, affine faithful within the modelled biome.

## Part B — Canonical codec store (`scripts/trust/build_canonical_codec.py`)

All 52 ANEW projects mapped into codec space via the affine (pure pandas), with lon/lat joined
from the GT gpkg, CO2, eco/biome, and a `modelled` flag. 12,834 plots, 0 missing lon/lat, codec
|max| 175.9. Provenance written to `preprocessing/{feature_schema.json,data_version.txt}` and
`final/DATA_STORE.md`. Embeddings only (no co-features) — the no-GEE boundary.

## Part C — DI / AOA module

- **DI (`di.py`):** importance-weighted CAST DI (weights from the deployed LightGBM gain
  importance), nearest-neighbour weighted distance normalised by the mean pairwise training
  distance; AOA threshold = CAST Q75 + 1.5·IQR of fold-aware (leave-project-out) training DI.
  Mahalanobis retained as a cross-check. Two spaces fitted: full-feature (73-dim, operational) and
  emb-only (64-dim, for the unused-project validation and overlays). A BLAS-batched `di_fast` path
  supports raster scale.
- **AOA (`aoa.py`):** applicability API + per-project AOA report. Modelled-23 all 100% inside;
  PNW/Alaska/southern + Ireland 0% inside.
- **Validation gate — PASS (`validate_di_error.py`).** DI rank-orders true error: cross-biome
  Spearman ρ = 0.53 (p = 2e-4) on the 29 labelled unused projects (scored by the emb-only model);
  all-52 ρ = 0.63 (p = 2e-8). The in-domain reference reproduces exactly — per-region DI orders
  wv (0.40) > ne (0.24) > mw (0.20), the inverse of the known LOPO R² (wv 0.16 worst). Figure:
  `figures/fig_di_vs_error.png`.
- **Spatial CV + uncertainty (`spatial_cv.py`, `uncertainty.py`).** LOPO RMSE 57.0, leave-bloc-out
  60.9 (wv worst per region), consistent with prior results. The error=f(DI) surface is calibrated
  in emb-only space over the full DI range by combining 23-project LOPO OOF (low DI) with the 29
  labelled unused projects (high DI, the expansion regime): expected RMSE rises 57 → ~110 tCO2/acre
  across DI 0.2 → 1.2, monotone. Calibration limited to DI ≈ 1.33. Figure: `figures/fig_error_vs_di.png`.
- **Guardrails (`guardrails.py`).** `apply()` returns per-point DI / AOA mask / expected RMSE;
  `trust_header()` summarises % inside AOA + DI distribution. Demonstrated on clean codec data:
  Bayfield 100% inside (exp.RMSE 60), NortheastKingdom 96%, HighCascades & Ireland 0% (exp.RMSE 109).
- **Overlays (`overlay.py`).** Reusable `overlay_prediction_area()` + codec-space `fig7`/`fig8`
  (model-comparable replacements for the earlier raw-space figures).

## Key new finding — Bayfield inference embedding stack is band-corrupted

The existing `predictions/bayfield_emb_30m.npy` (used to produce the deployed Bayfield AGB map)
does **not** match the training codec band distribution: 7 bands (1, 5, 12, 14, 40, 49, 60) are
near-all-zero while the corresponding training bands are non-zero (e.g. band 1 mean −24). Per-pixel
DI on this stack is therefore invalid, so the wall-to-wall guardrail demo was deferred. Recorded in
`trust/bayfield_stack_flag.json`. This also calls the deployed Bayfield map's embedding inputs into
question and should be investigated (needs GEE/re-extraction — out of current scope).

## Limitations / next steps

- **Full-feature uncertainty at high DI** needs labelled OOD support with co-features → requires
  GEE extraction of topo/CHM/dstx for the 29 unused projects (deferred). The shipped error=f(DI) is
  emb-only (slightly conservative; the full model does better in-domain).
- **Unverified-biome affine** — targeted re-extraction recommended before training/trusting DI on
  conifer/boreal/tundra/grassland projects.
- **Rebuild the Bayfield embedding stack** in verified codec space before per-pixel guardrails ship.
- **Retraining** on the in-AOA unused projects remains a separate, now evidence-supported decision.
