# Preprocessing pipeline

Iteration 0 reuses the prior joint_v2 feature table; there is **no fitted scaler or encoder**
to ship (LightGBM consumes raw 64-dim embeddings).

## To reproduce features for inference

Use the sibling extractor (do not reimplement):

```
cd /home/mattc/code/tf-deep-landcover && \
uv run python -m src.agb.extract_features_batched \
  --plots <ANEW gpkg> --aoi <AOI geojson> --years 2022 2023 --out <parquet>
```

Output columns: `plot_id, project_name, year, lon, lat, target, failure, emb_00..emb_63`.
Drop rows where `failure` is non-null. Model features = `emb_00..emb_63`.

## Files

- `feature_schema.json` — full column contract.
- `data_version.txt` — input SHA256, source URIs, and the tf-deep-landcover SHA used.
