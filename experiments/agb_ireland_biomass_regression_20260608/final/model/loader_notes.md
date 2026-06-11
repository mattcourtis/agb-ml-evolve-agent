# Loader notes — embdstx inference head

## What this is
The pre-trained `embdstx` LightGBM regression head, applied **zero-shot** (no training/fine-tuning)
to the 141 Irish Dasos Locations. It was trained on 4636 US ANEW plots (Bayfield in-sample).

- `inference_model_embdstx.txt` — LightGBM text model (73 trees).
- `inference_features_embdstx.json` — the 67-feature order + target metadata.

## Target
**CO₂ standing stock, tCO₂/acre.** Training range `[0, 520.95]`. This is AGB-only (above-ground).
Deep Biomass (DB) comparison values are converted Mg/ha → tCO₂/acre by **×0.6977**
(`0.47 · 3.667 · 0.4047`), both sides AGB-only.

## Feature order (67 — EXACT, must match `inference_features_embdstx.json`)
1. `emb_00 … emb_63` — 64 AlphaEarth (AEF) embeddings, **affine-transformed into the training int8
   codec space** (see `../preprocessing_pipeline/`). Order is strict.
2. `dstx_pre_ysd` — survey-relative years-since-disturbance (sentinel 100 if undisturbed).
3. `dstx_pre_loss_5yr` — 1 if pre-survey Hansen loss within 5 yr.
4. `dstx_loss_frac_buf` — disturbed-area fraction over the polygon.

The `dstx_*` features are NOT affine-transformed (`affine_applied=false`); the 64 `emb_*` are.

## How to load and predict
```python
import lightgbm as lgb, json, pandas as pd
m = lgb.Booster(model_file="inference_model_embdstx.txt")
feats = json.load(open("inference_features_embdstx.json"))["features"]   # 67, ordered
X = pd.read_parquet(".../preprocessing/ireland_features.parquet")[feats]  # columns in feats order
pred_tco2_acre = m.predict(X)   # deterministic; seed irrelevant at predict time
```

## ENCODING-GATE requirement for any NEW region (HARD precondition)
Before predicting on AEF features from a new region you MUST re-pass the per-band encoding gate:
fit the per-band affine (GEE AlphaEarth A00..A63 → training `emb_*` codec) on in-sample plots,
validate on a held-out split, and require **mean per-plot corr > 0.8 AND post-affine per-band slope
≈ 1 with bounded intercept**. The Irish run passed (held-out corr 0.986, slope median 1.006;
`../preprocessing_pipeline/encoding_gate.json`). The gate validates *encoding fidelity*, NOT
transfer accuracy. Without it the LightGBM absolute-threshold splits are mis-scaled and predictions
are meaningless.

## OOD caveat
For Ireland, 100% of Locations lie beyond the 99th-pct training Mahalanobis radius (domain-classifier
AUC ≈ 1.0). Treat absolute tCO₂/acre levels as deep extrapolation; trust **rankings/structure**, not
calibrated absolute values. See `../experiment_report.md`.
