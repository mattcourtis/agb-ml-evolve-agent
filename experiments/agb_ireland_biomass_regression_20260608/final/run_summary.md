# Run Summary — Ireland AGB zero-shot transfer (traceability)

- experiment_id: `agb_ireland_biomass_regression_20260608`
- mode: **inference-only zero-shot transfer + model-vs-model** (NO training, NO ground truth)
- decision: **RETRAIN_WARRANTED**
- generated: 2026-06-08 (Model-Saving + Final-Report actor), from ACCEPTED artefacts only.
- data outputs: not in git — see [`DATA_STORE.md`](DATA_STORE.md) for the data-space location.

## Traceability

| item | value |
|---|---|
| head | `final/model/inference_model_embdstx.txt` (LightGBM, 73 trees) |
| feature list / order | `final/model/inference_features_embdstx.json` — 67 features: `emb_00..emb_63` (affine-mapped AEF) + `dstx_pre_ysd`, `dstx_pre_loss_5yr`, `dstx_loss_frac_buf` |
| target | CO₂ standing stock, tCO₂/acre (training range [0, 520.95]); DB ref Mg/ha ×0.6977 |
| n locations | 141 (0 NaN feature vectors) |
| **encoding-gate result** | **PASS** — held-out (122 plots) mean corr 0.986, post-affine slope median 1.006, 98% bands in [0.8,1.2], intercept median 0.085·band-σ |
| prediction summary | min 26.7 / mean 91.6 / median 100.3 / max 138.4 tCO₂/acre; pred/DB 3.35× |
| H1 / H2 / H3 | SUPPORTED / NOT_SUPPORTED (comparison artefact) / MOSTLY_SUPPORTED |
| OOD | SEVERE_DOMAIN_SHIFT (Mahalanobis min 27.8 = 1.9× r99 14.79; AUC 0.999998) |
| seed | 42 (gate split + eval; LightGBM predict deterministic) |

### Inputs + versions (`preprocessing_pipeline/data_version.txt`)
| input | sha256[:16] |
|---|---|
| DB CSV | `8f34cb1eae381395` |
| Dasos gpkg | `b453cec8d320cc10` |
| training parquet (codec ref) | `3ff73b956c3043d2` |
| head model | `681d939258695f76` |
| features json | `9f80f0dfe17fdb3c` |
| `ireland_features.parquet` (141×67) content sha256 | `aee82d7b17fb357dbbc466c88c0c4e6317742b02951ad28c7068692600e519ad` |

- AEF asset: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`; Hansen: `UMD/hansen/global_forest_change_2025_v1_13`
- git HEAD: `b6d219ac5090543f58480d9df30e6a16acb35003` (branch `main`); extraction 2026-06-08.

## RETRAIN_WARRANTED decision

Per `experiment_design.md §7`: gate PASS and H1 hold, but credible transfer requires all of
H1+H2+H3 with non-catastrophic OOD. H2 fails (signed gap flat-to-declining across DB quintiles — a
DB-self-quintiling **comparison artefact**, not head saturation; error_analysis §2) and OOD is
catastrophic (AUC ≈ 1.0; 100% beyond the 99th-pct radius). Design rule (b) → escalate the conditional
analog-subset (maritime + high-biomass conifer) retrain. NOT halt; NOT fully credible.

## Justification for absent training artefacts (inference-only)

This experiment trained NO model — the pre-trained `embdstx` head was applied zero-shot. The
following training-stage artefacts therefore **do not apply and were NOT fabricated**:

- `checkpoints/best.ckpt` — **N/A (inference-only).** The applied artefact IS the pre-trained head
  text file `final/model/inference_model_embdstx.txt`; there is no checkpoint to select.
- `metrics_history.csv` — **N/A (inference-only).** No training loop ran, so there is no per-epoch
  metric history. Evaluation outputs are in `final/evaluation_matrix.yaml`.
- `training_config.yaml` — **N/A (inference-only).** No training config exists. The inference config
  is captured by `configs/experiment_config.yaml`, `configs/split_strategy.yaml`
  (`none_zero_shot_transfer`, fractions 0/0/0), and `final/preprocessing_pipeline/`.

> Reference-gate note: the skill's `references/` directory is absent (flagged by all upstream actors);
> the Final-QA gate was self-applied per `configs/experiment_design.md §7` conventions.

## Final QA gate self-check

| path | status |
|---|---|
| `final/model/inference_model_embdstx.txt` | PASS (non-empty, 205266 B) |
| `final/model/inference_features_embdstx.json` | PASS (1202 B) |
| `final/model/loader_notes.md` | PASS (2590 B) |
| `final/preprocessing_pipeline/aef_affine.parquet` | PASS (4377 B) |
| `final/preprocessing_pipeline/aef_affine_gate_train287.parquet` | PASS (4377 B) |
| `final/preprocessing_pipeline/feature_schema.json` | PASS (23326 B) |
| `final/preprocessing_pipeline/data_version.txt` | PASS (1561 B) |
| `final/preprocessing_pipeline/encoding_gate.json` | PASS (741 B) |
| `final/preprocessing_pipeline/README.md` | PASS (2979 B) |
| `final/evaluation_matrix.yaml` | PASS (7246 B) |
| `final/model_card.md` | PASS (7603 B) |
| `final/data_card.md` | PASS (4753 B) |
| `final/experiment_report.md` | PASS (8506 B) |
| `final/environment.lock` | PASS (56 packages, `uv pip freeze`) |
| `final/git_snapshot.txt` | PASS (518 B) |
| `checkpoints/best.ckpt` | **N/A — inference-only** (no training; not fabricated) |
| `metrics_history.csv` | **N/A — inference-only** (no training loop) |
| `training_config.yaml` | **N/A — inference-only** (no training config) |

All applicable gate paths resolve and are non-empty. The three training-only paths are explicitly
N/A for this inference-only experiment, justified above.
