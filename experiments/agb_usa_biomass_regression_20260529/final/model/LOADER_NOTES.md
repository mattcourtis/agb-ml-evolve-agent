# Model loader notes

`model.txt` is a LightGBM booster in text format (also mirrored as `checkpoints/best.ckpt`).

```python
import lightgbm as lgb
import pandas as pd

booster = lgb.Booster(model_file="model.txt")

# features: the 64 AlphaEarth Foundation (AEF) embedding columns emb_00..emb_63, in that order, raw (no scaling).
feats = pd.read_parquet("<features.parquet>")
X = feats[[f"emb_{i:02d}" for i in range(64)]]
pred = booster.predict(X)   # tCO2/acre standing stock
```

- Target: `CO2`, standing-stock tCO₂/acre.
- This is the final all-data fit. The reported metrics (`metrics.json`) are the
  leave-one-project-out OOF metrics, not this final model's training fit.
- No PCA / scaler is required (raw embeddings; `pca_n_components: null`).
- Produced by `tf-deep-landcover/src/agb/train_agb_lgbm` @ `e8c70584...`.
