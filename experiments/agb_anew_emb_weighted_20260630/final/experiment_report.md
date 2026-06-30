# ANEW emb-only AGB model — global, frontier-aware, trust-gated

**Date:** 2026-06-30 · **Experiment:** `agb_anew_emb_weighted_20260630`

## Summary

We trained a new AGB regression model over the **whole ANEW ground truth** (51 projects /
12,636 plots, Quinte dropped) in emb-only codec space, designed around the DI/AOA learnings
(`agb_anew_gt_applicability_20260626`). Three results, all clean and mutually reinforcing:

1. **The ceiling is feature-driven, not learner-driven.** LightGBM, untuned XGBoost, and even a
   ridge floor land within ~0.5 tCO2/acre of each other on interior LOPO RMSE (71.4 / 71.9 /
   71.6). A linear readout matching boosted trees means the 64-dim foundation-model embedding is
   so pre-digested that model choice is nearly irrelevant — accuracy is capped by the *signal*.
2. **Frontier-aware weighting does not help on emb-only data.** Every weighting scheme (S1–S4)
   made regional-frontier RMSE *worse* (58.2 → 59.7–62.8) with no range-compression gain, so the
   4-gate ship rule selected **S0 (unweighted)**. There is no extra frontier signal in the
   embeddings for weighting to exploit.
3. **The trust layer cleanly orders error by DI.** The self-referential AOA threshold (0.558) and
   the DI→expected-RMSE curve give every prediction an applicability flag and an expected error
   that rises monotonically from the interior to the frontier.

See `analysis/methodology.md` for the four-roles-of-groups framing and full method.

## How regional/groups were used

| Role | Decision | Why |
|---|---|---|
| **CV** | whole-project atomic unit; LOPO → leave-bloc-out → leave-biome-out ladder | plots within a project are autocorrelated; each rung = a different deployment question |
| **Architecture** | single global model | interior too redundant to split, frontier too small to specialise |
| **Weighting** | per-bloc capped sub-linear, *measured* | stop the broadleaf majority compressing the frontier — but it must earn its place |
| **Interpretation** | tag by regional-dependence | distinguishes "no nearby analogue" from "intrinsically isolated" |

## Learner baseline check (LOPO, S0, unweighted)

| Learner | interior RMSE | all RMSE |
|---|---|---|
| LightGBM | 71.4 | 73.7 |
| XGBoost (untuned) | 71.9 | 74.1 |
| Ridge (floor) | 71.6 | 73.9 |

→ ceiling is feature-driven. LightGBM carried forward. (A tuned model-family bake-off or tabular
deep-learning is not worth it — see methodology "modelling options considered".)

## Weighting comparison (LOPO per-tier RMSE, unweighted scoring)

| Scheme | n_eff | interior | regional-frontier | self-standing-frontier | bloc-2 transfer |
|---|---|---|---|---|---|
| **S0** unweighted | 1.00 | 71.4 | **58.2** | 122.2 | 90.0 |
| S1 per-bloc inv-density | 0.71 | 71.5 | 59.7 | 119.7 | 91.6 |
| S2 capped-DI α0.5 | 0.98 | 71.6 | 59.8 | 122.5 | 93.4 |
| S3 capped-DI α1.0 | 0.90 | 71.8 | 61.2 | 119.8 | 89.9 |
| S4 blend | 0.55 | 71.7 | 62.8 | 120.1 | 89.4 |

**Gate audit:** every scheme has *negative* frontier gain (gate 1 fails) — and S1/S4 also breach
the n_eff floor. **Ship S0.** Self-standing-frontier barely moves under any scheme, which is the
expected "needs local GT" signal, not a weighting failure.

## Transfer ladder (S0)

| Rung | group | RMSE | n | flag |
|---|---|---|---|---|
| LOPO | all | 73.7 | 12,636 | — |
| leave-bloc-out | 0 (Lake States) | 61.0 | 4,635 | in-AOA |
| | 1 (Northeast) | 65.8 | 2,638 | in-AOA |
| | **2 (AK/PNW frontier)** | **90.0** | 1,031 | **out-of-AOA** |
| | 3 (Appalachia/SE) | 93.8 | 4,332 | in-AOA (high-biomass scale) |
| leave-biome-out | HighLatitude (Doyon) | 61.8 | 102 | out-of-AOA (floor) |
| | Broadleaf | 80.2 | 10,840 | backwards fold |
| | Conifer | 94.9 | 992 | out-of-AOA (floor) |
| | Grassland | 113.4 | 702 | out-of-AOA (floor) |

Bloc-3 (Appalachia/SE) reads higher than the frontier bloc-2 *despite being in-AOA* — that is
biomass-scale (high-CO2 broadleaf + grassland), not extrapolation; the out-of-AOA flag correctly
separates the two failure modes.

## Per-project frontier errors (near vs far transfer)

| Project | biome | regdep class | LOPO | bloc | 
|---|---|---|---|---|
| LongviewRanch | Conifer | regional-frontier | 64.7 | 73.8 |
| Doyon | HighLatitude | self-standing | 61.8 | 64.0 |
| Kootznoowoo | Conifer | self-standing | 92.5 | 97.2 |
| RainierGateway | Conifer | self-standing | 160.5 | 148.8 |

RainierGateway (median 244 tCO2/acre, highest-biomass project) is the dominant error — exactly
where emb-only range compression bites hardest.

## Trust layer

AOA threshold = **0.558** (matches the applicability run). DI→expected-RMSE curve, fit on the
shipped model's LOPO OOF residuals, is monotone:

| Tier | median DI | expected RMSE |
|---|---|---|
| interior | 0.27 | 68.9 |
| regional-frontier | 0.62 | 97.1 |
| self-standing-frontier | 0.72 | 99.5 |

Calibrated to DI p99 ≈ 0.79; beyond that it is a lower bound.

## Figures (`figures/`)
- `scheme_per_tier_rmse.png` — no weighting scheme improves any tier over S0.
- `pred_vs_true.png` — LOPO OOF predictions vs truth; clear range compression at the high end.
- `per_project_lopo_vs_bloc.png` — near vs far transfer per project, by regional-dependence class.
- `error_vs_di.png` — the DI→expected-error trust curve.

## Deliverables
- Candidate model `models/anew_emb51_model.txt` + `anew_emb51_features.json` (scheme S0,
  n_estimators 172) and trust bundle `trust/{di_space_anew51,error_curve}.npz`,
  `trust/thresholds.json` — all in the data-space (see `DATA_STORE.md`). **Not** auto-promoted
  over the deployed model.

## Low-end de-biasing (follow-up)

The S0 model over-predicts true CO₂ < 100 tCO₂/acre by +35 to +49 (regression-to-the-mean) — 42%
of plots. A bake-off of objective re-aiming variants under the same LOPO CV (full method and table
in `analysis/low_end_debiasing.md`) found:

- **log1p target transform wins:** low-end bias **+43.0 → +23.8**, RMSE<100 **58.5 → 44.4**,
  zero-detection recall **0.54 → 0.74**, with low-band discrimination slightly *up*
  (Spearman 0.574→0.599). Accepted, quantified cost: high-end bias>150 −65.6→−87.7, RMSE_all 73.7→77.3.
- **Two-stage hurdle does not help** — its classifier AUC (~0.88) equals the regressor's implicit
  separability, so low biomass *is* moderately separable but already exploited; structure adds nothing.
- **Calibration cannot de-bias** — post-hoc isotonic is inert (+43.1) and *re-inflates* bias when
  stacked on log1p (it targets the conditional mean, re-imposing the mean-reversion).

Candidate `models/anew_emb51_log1p_model.txt` (predict = `clip(expm1(pred), 0)`). The residual +24
low-end bias is the floor for objective changes on emb-only features; closing it needs
vertical-structure features. Figures: `low_end_band_bias.png`, `low_end_pred_vs_true.png`,
`low_end_roc_lt50.png`.

## Conclusion & next step

The model is honest and trust-gated but emb-only-limited (R²≈0.4, range compression). Its value is
**coverage (all 51 projects, all biomes) + calibrated applicability**, not peak accuracy — and
neither learner choice nor sample weighting moves that ceiling, because **the data, not the model,
is the binding constraint**. The single highest-leverage next step is the deferred **GEE
extraction of CHM/topo/disturbance for the 28 unused projects → full-feature model over all 51**,
giving the structural (vertical) signal needed to decompress the broadleaf interior and the
high-biomass conifer frontier that embeddings alone cannot separate.
