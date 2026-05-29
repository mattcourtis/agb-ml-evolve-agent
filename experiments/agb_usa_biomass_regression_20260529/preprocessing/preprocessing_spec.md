# Preprocessing Spec — agb_usa biomass regression (iteration 0)

## Decision: reuse existing features (no re-extraction)

Iteration 0 reuses the already-extracted joint_v2 feature table rather than re-running the
extractor. Rationale:

- The batched extractor (`src.agb.extract_features_batched`) takes a single `--aoi`, but
  joint_v2 spans three regional AOIs (wv / midwest / northeast). Re-extraction would require a
  multi-AOI concat that is not a single documented entry point — out of scope for a wiring run.
- The trainer wiring (the load-bearing path for this iteration) is fully exercised by
  retraining on the reused table. Retrain reproduced the baseline **bit-identically**
  (R²=0.4182, RMSE=56.58, MAE=41.49), confirming the reused features are faithful.
- Re-extraction is deferred to iteration 1, when GEDI features are added.

## Input

- `/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet`
  (tf-deep-landcover SHA `e8c70584`).

## Transforms

- **Model features:** the 64 embedding columns `emb_00 … emb_63` only. No scaling/encoding —
  LightGBM consumes raw embeddings; there is no fitted scaler or encoder to persist.
- **Row filter:** rows with `failure == True` are dropped by the trainer (4,646 → 4,636).
- **Target:** `target` column (= `CO2`, tCO₂/acre standing stock), used as-is (no log/transform —
  log-target was a ruled-out lever).
- **Normalisation policy:** n/a (tree model on raw embeddings). No leakage surface from scaling.

## Ecoregion enrichment (for evaluation only)

`ECO_NAME` is joined onto the OOF predictions from the ANEW gpkg on `(project_name, Plot_ID)`
at evaluation time. This is a metrics-side enrichment, not a model feature.

## Feature schema

See `preprocessing/feature_schema.json`.

## Data version

See `preprocessing/data_version.txt`.

## Reproducibility footer

- input_artefact_sha256: features.parquet reused from sibling (see data_version.txt)
- libraries: pandas (read); LightGBM (trainer, sibling repo)
- seed: 42 (experiment), trainer hyperparams hardcoded in sibling
- command_or_entrypoint: reuse — no extraction command run this iteration
- timestamp_utc: 2026-05-29T09:36:00Z
