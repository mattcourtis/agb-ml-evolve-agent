# Data store — agb_anew_emb_weighted_20260630

Code, docs, and figures live in git. Model artifacts, CV outputs, and the trust bundle live in
the data-space (outside the repo, not tracked):

`/home/mattc/data-space/carbonmap-embeddings/agb_anew_emb_weighted_20260630/`

| Path | Description |
|---|---|
| `models/anew_emb51_model.txt` | Candidate LightGBM (emb-only, 51 projects, scheme S0, n_estimators 172). Booster text format. |
| `models/anew_emb51_features.json` | Feature order (emb_00..63), n_estimators, target stats, weighting scheme, lgb params. |
| `cv/n_trees.json` | n_estimators + grouped-holdout projects. |
| `cv/learner_check.parquet` | LightGBM vs XGBoost vs ridge, per-tier LOPO RMSE. |
| `cv/comparison_matrix.parquet` | S0–S4 × per-tier RMSE + n_eff + bloc-2 transfer. |
| `cv/per_project.parquet` | Per-project LOPO/bloc RMSE, regdep class, out-of-AOA-fold flag. |
| `cv/transfer_ladder.parquet` | LOPO / leave-bloc-out / leave-biome-out RMSE (S0) with out-of-AOA flags. |
| `cv/oof_S0.parquet` | S0 LOPO out-of-fold predictions (CO2, oof_S0, di_lopo, tier). |
| `cv/decision.json` | Chosen scheme + 4-gate audit. |
| `trust/di_space_anew51.npz` | Refit CAST DI space on the 51-project cloud (mu/sd/w/dbar/threshold). |
| `trust/error_curve.npz` | Isotonic DI→expected-RMSE knots + calibration limit. |
| `trust/thresholds.json` | AOA threshold, expected RMSE by DI / by tier. |
| `models/anew_emb51_log1p_model.txt` | Low-end de-biased candidate (log1p target). Predict = `clip(expm1(booster.predict(X)), 0)`. |
| `models/anew_emb51_log1p_features.json` | Schema for the log1p candidate (target_transform=log1p). |
| `low_end/variant_metrics.parquet` | Low-end bake-off: bias<100, rmse<100, spearman, AUC<50, zero-recall, high-end trade per variant. |
| `low_end/band_bias.parquet` | Conditional bias/RMSE by true-CO2 band, per variant. |
| `low_end/oof_by_variant.parquet` | LOPO OOF predictions per variant (backs the figures). |
| `low_end/decision.json` | Chosen variant (log1p) + decision rule + candidates. |

## Regeneration (order matters)

Deterministic from the canonical codec store + the GT-applicability DI labels; no GEE.

```
uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/anew_emb_model/cv_ladder.py
uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/anew_emb_model/decide_and_train.py
uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/anew_emb_model/trust_fit.py
uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/anew_emb_model/make_figs.py
uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/anew_emb_model/low_end.py
uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/anew_emb_model/finalise_low_end.py
```

Inputs: `agb_trust_aoa_20260626/preprocessing/anew_canonical_codec.parquet` (canonical GT) and
`agb_anew_gt_applicability_20260626/analysis/plot_level_di.parquet` (DI/bloc/AOA labels,
row-aligned). `scripts/anew_emb_model/data.py` is the shared loader (drops Quinte, attaches DI
labels, builds weight schemes).

## Promotion path

This is a **candidate**. To deploy, copy `anew_emb51_model.txt` + `anew_emb51_features.json` into
the repo `models/` directory (where the deployed `inference_model*.txt` live) and ship the
`trust/` bundle alongside. Not done automatically.
