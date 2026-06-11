# Inference run (no training performed)

Zero-shot transfer: the pre-trained head was applied to Ireland; NO model was trained or fine-tuned
in this pass. This record stands in for the "training" stage under the fast-track collapse.

- head: `models/inference_model_embdstx.txt`
- features: `models/inference_features_embdstx.json` — 67 features (64 AEF embeddings + dstx_pre_ysd +
  dstx_pre_loss_5yr + dstx_loss_frac_buf), exact order matched.
- n_estimators (trees): 73
- target: CO2 standing stock, tCO2/acre; training range [0, 520.95]
- encoding gate: PASS (held-out corr 0.986, post-affine per-band slope median 1.006);
  ref `preprocessing/encoding_gate.json` and `preprocessing/preprocessing_spec.md`.
- input features: `preprocessing/ireland_features.parquet` (141 Locations x 67, training int8-codec scale)
- training_performed: false (inference only)

## Prediction summary (141 Locations)
- min 26.73 | mean 91.62 | median 100.34 | max 138.39 tCO2/acre
- 0% above training max (520.95); 79.4% above the 80 tCO2/acre optical-ceiling reference.
- Output: `evaluation/ireland_predictions.parquet` (Location, pred_tco2, db_2020_24_tco2, db_2024_tco2,
  delta_2020_24, delta_2024, covariates).

## Reproduce
- `uv run python experiments/agb_ireland_biomass_regression_20260608/evaluation/run_bias_characterisation.py`
- seed 42; deterministic LightGBM predict.
