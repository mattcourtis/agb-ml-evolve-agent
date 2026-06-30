# Low/zero-biomass de-biasing — methodology & results

## Problem

The shipped emb-only model (raw-L2, "S0") is mean-seeking: it over-predicts true CO₂ < 100
tCO₂/acre by +35 to +49 and under-predicts the high end, with the unbiased crossover at the data
mean (~130) — textbook regression-to-the-mean. 42% of plots sit below 100. Within true<100 the
predictions still rank-order truth (Spearman 0.574), so the embeddings carry low-end signal that
L2 mean-seeking compresses and shifts up. High-end saturation is an accepted feature ceiling;
the target here is the low-end **bias** and **separability**.

## Why this is a legitimate re-examination

`agb_usa_biomass_regression_20260529` previously rejected log-target ("trade-off not a fix"),
post-hoc isotonic calibration ("biases moved <3"), and Huber loss, concluding a feature deficit.
This work differs because (1) the **objective is low-end-priority** and explicitly accepts the
high-end trade, so a "low−/high+" transform is the *desired* outcome, not a wash; and (2) the
**Tweedie objective and two-stage hurdle were never built**. The governing statistical point:
post-hoc calibration maps predictions to the conditional mean `E[true|pred]`, which is the very
mean-reversion causing the bias — so it cannot de-bias. Only **re-aiming the training objective**
moves the by-true bias.

## Method

A bake-off under the **same LOPO CV** (whole-project folds, refit per fold, predictions clipped
at 0; reuses `cv_ladder.cv_predict`/`rmse`/`mae`, `data.load_eligible`). Variants, all emb-only
over the 51 projects:
- **S0_raw** — current raw-L2 baseline.
- **T_log1p / T_sqrt** — train on `log1p`/`sqrt` of the target, invert on predict.
- **tweedie_{1.1,1.3,1.5}** — LightGBM Tweedie objective (zero-inflated, right-skew), variance-power sweep.
- **hurdle_t{25,50}** — per fold: `P(true≥τ)` classifier × high-regressor + `(1−P)·low_mean`, all fit on train only.
- **Cal_isotonic** — S0 + post-hoc `IsotonicRegression(pred→true)`, the documented-weak reference.

Scored on: conditional bias by true-band (primary), low-band (<100) RMSE/MAE, within-<100
Spearman, AUC for separating true<50, zero-band recall, and the **quantified** high-end (>150) trade.

## Results

| variant | bias<100 | rmse<100 | spearman<100 | AUC<50 | zero-recall | bias>150 | rmse>150 | rmse_all |
|---|---|---|---|---|---|---|---|---|
| **S0_raw** | +43.0 | 58.5 | 0.574 | 0.874 | 0.54 | −65.6 | 102.7 | 73.7 |
| **T_log1p** | **+23.8** | **44.4** | **0.599** | **0.881** | **0.74** | −87.7 | 119.0 | 77.3 |
| T_sqrt | +32.8 | 50.4 | 0.595 | 0.881 | 0.66 | −76.3 | 110.4 | 74.5 |
| tweedie_1.5 | +36.9 | 54.0 | 0.594 | 0.881 | 0.66 | −70.9 | 106.7 | 73.9 |
| hurdle_t50 | +42.7 | 58.0 | 0.581 | 0.878 | 0.23 | −65.1 | 101.6 | 72.9 |
| Cal_isotonic | +43.1 | 58.7 | 0.575 | 0.875 | 0.48 | −64.3 | 101.6 | 73.3 |

**Chosen: T_log1p** (rule: smallest |bias<100| among variants that keep discrimination ≥ S0).

### Findings
1. **log1p halves the low-end bias and cuts low-end RMSE** (+43→+24; 58.5→44.4) while *improving*
   low-band discrimination (Spearman 0.574→0.599, AUC 0.874→0.881) and zero-detection
   (recall 0.54→0.74). The accepted, quantified cost: high-end bias>150 −65.6→−87.7,
   overall RMSE 73.7→77.3.
2. **The two-stage hurdle does not help** (bias +42.7; zero-recall *worse* at 0.23 because it adds
   `low_mean` back). Its classifier AUC (~0.878) equals the regressor's implicit separability —
   so low biomass *is* moderately separable (AUC ~0.88), but the regressor already exploits it;
   the structure adds nothing. The limit is feature signal, not model architecture.
3. **Calibration cannot de-bias.** Post-hoc isotonic on S0 leaves bias<100 at +43.1 (inert), and
   a fold-honest isotonic *on top of log1p re-inflates* it to +44.1 — because calibration targets
   `E[true|pred]`, it re-imposes the mean-reversion log1p removed. This empirically confirms the
   framing point and rules calibration out as a de-biasing tool.
4. **Tweedie** gives a milder version of the transform effect (bias +37, smaller high-end cost) —
   a reasonable middle option if the high-end trade must be limited.

## Recommendation & limits

Ship the **log1p** target transform for the low-end-priority use case (candidate
`models/anew_emb51_log1p_model.txt`; predict = `clip(expm1(booster.predict(X)), 0)`; same DI/AOA
trust layer). The residual +24 low-end bias is the floor reachable by objective changes alone on
emb-only features — closing it further needs vertical-structure features (the deferred GEE
CHM/GEDI extraction), since separating biomass *level* (not just rank) within similar-spectra
stands is feature-bound. Optionally expose the low-biomass score as a deployment flag.

Figures: `low_end_band_bias.png`, `low_end_pred_vs_true.png`, `low_end_roc_lt50.png`.
