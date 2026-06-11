# Preprocessing pipeline — Ireland AGB zero-shot transfer

Full spec: `../../preprocessing/preprocessing_spec.md`. This folder holds the **fitted, reusable
pipeline artefacts**; the scripts live in the repo root `scripts/` (referenced below — not copied,
so they stay versioned in one place).

## Artefacts here
| file | content |
|---|---|
| `aef_affine.parquet` | **production** per-band affine `emb_j = a_j·A{j} + c_j`, fit on ALL 409 valid Bayfield plots — this is the transform APPLIED to Ireland. |
| `aef_affine_gate_train287.parquet` | gate-validation affine, fit on 287 train / validated on 122 held-out — the generalisation **evidence**. |
| `encoding_gate.json` | held-out gate verdict (PASS; corr 0.986, slope median 1.006). |
| `production_refit_summary.json` | full-409 affine ranges + smoke-pred stats + held-out misfit. |
| `feature_schema.json` | per-column dtype, provenance, `affine_applied` flag (True×64 emb, False×3 dstx). |
| `data_version.txt` | input sha256[:16] + bytes + GEE asset ids + extraction date + git commit + gate verdict. |

## Reusable scripts (repo root, not copied)
- `scripts/ireland_crosswalk.py` — Location↔SiteName crosswalk, dissolve to 141 polygons, DB reference.
- `scripts/fit_aef_affine.py` — ENCODING GATE: per-band affine fit + held-out validation.
- `scripts/extract_ireland_aef.py` — Ireland AEF + Hansen dstx extraction + 67-feature assembly (gate-guarded).
- `scripts/refit_aef_affine_production.py` — post-gate refit on all 409 + re-apply to Ireland.

## Pipeline order
1. **Crosswalk + dissolve** (`ireland_crosswalk.py`): 141/141 Locations resolve (direct + `_`→`/`);
   1,053 sub-compartments → 141 dissolved MultiPolygons (EPSG:2157 dissolve, back to 4326);
   area-weighted covariates; survey_year = area-weighted mode clamped to [2017, 2025] (17 pre-2017
   fallbacks lifted to 2017). DB reference (Mg/ha → tCO₂/acre ×0.6977).
2. **ENCODING GATE** (`fit_aef_affine.py`): per-band OLS GEE A00..A63 → training `emb_*` codec,
   fit on 287 Bayfield plots, validated on 122 held-out → **PASS** (HARD precondition). NOT a scalar.
3. **Production refit** (`refit_aef_affine_production.py`): after gate PASS, refit affine on all 409
   plots; this is the applied transform.
4. **Ireland AEF extraction** (`extract_ireland_aef.py`): `reduceRegions(mean, scale=10)` of the
   AEF mosaic at each Location's survey_year over the dissolved polygon → production affine applied.
5. **Disturbance co-features**: Hansen lossyear, survey-relative timing (leakage-safe, no post-survey).
6. **Assembly**: `ireland_features.parquet` (141 × 67, exact `inference_features_embdstx.json` order,
   0 NaNs).

## Re-use for a new region
Repeat steps 2–3 (re-fit + re-pass the encoding gate) on that region's in-sample plots BEFORE
applying the head. Do not reuse the Irish/Bayfield affine as-is unless the new region shares the
same AEF codec relationship (it generally will, but the gate must confirm).
