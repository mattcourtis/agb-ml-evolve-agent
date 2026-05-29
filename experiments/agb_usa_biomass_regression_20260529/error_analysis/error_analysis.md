# Error Analysis — agb_usa biomass regression (iteration 0)

## Headline

The reproduction is exact (R²=0.4182, bit-identical to baseline), so iteration 0 has **no
wiring or tuning fault to diagnose**. The error structure is the *known, diagnosed feature
ceiling*. This analysis confirms that signature on the reproduced run and identifies the
weakest upstream stage as **feature insufficiency**, motivating the iteration-1 GEDI lever.

## 1. Dynamic-range compression (the dominant error mode)

Per-quintile bias (mean signed residual, pred − true) over the 4,636 OOF predictions:

| quintile | true mean | pred mean | bias |
|---|---:|---:|---:|
| Q1 (low) | 18.2 | 53.7 | **+35.6** (over-predict) |
| Q2 | 63.2 | 95.0 | +31.8 |
| Q3 | 101.1 | 118.4 | +17.3 |
| Q4 | 139.1 | 129.0 | −10.1 |
| Q5 (high) | 220.4 | 148.3 | **−72.1** (under-predict) |

Monotone over→under transition: the model pulls every prediction toward the centre. True
biomass spans 18→220 across quintile means (a 202 range); predicted spans only 54→148 (94).
**predicted_range_discrimination = 0.468** — the model recovers under half the true spread.
This is the textbook "collapses the dynamic range" failure, and it exceeds the comfort
threshold (|per_quintile_bias| ≤ 30) at both tails by design.

## 2. Where it bites hardest — ecoregion / region

| region | ecoregion | R² | discrimination | n |
|---|---|---:|---:|---:|
| wv | Appalachian mixed mesophytic | **0.157** | **0.19** | 598 |
| mw | Western Great Lakes | 0.415 | 0.47 | 2,619 |
| ne | New England-Acadian | 0.476 | 0.49 | 1,407 |

Appalachia is the worst by a wide margin (R²=0.16, discrimination 0.19 — matching the
research anchor's 0.21). Closed-canopy Appalachian hardwood is exactly the case where optical
embeddings saturate: a 50 tCO₂/acre stand and a 250 tCO₂/acre stand look identical from above.
The Upper Midwest forest-savanna transition fold (n=12) shows R²=−0.08 but is too small to be
material.

## 3. What this is NOT (ruled-out levers — not re-run)

Per `research/deep_research.md` and the improvement-loop addendum, the following were already
falsified on this pool and are **not** re-litigated here:

- more plots (joint pool) — WV Q1 bias unchanged
- footprint / area-weighted sampling — biases unchanged within ±2
- Huber loss — R² drops 0.07
- log-target — trades Q1 bias for worse Q5 bias
- isotonic calibration — model doesn't rank well enough; R² drops

The near-zero aggregate bias (+0.50) and well-behaved calibration (max decile residual 4.34)
confirm the model is *unbiased on average and well-ranked at the centre* — calibration cannot
fix a range the model never spread out in the first place.

## 4. Weakest upstream stage → iteration-1 trigger

- **Weakest stage:** `preprocessing / feature_source` — the AlphaEarth Foundation (AEF) embeddings do not
  carry vertical-structure information. This is upstream of the model; no model/training change
  can lift the ceiling (proven by the ruled-out levers).
- **Rerun boundary for iteration 1:** research → preprocess → train → evaluate, with a new
  feature source.
- **Lever:** add **GEDI canopy height**. Expected to help Appalachia most (worst discrimination).
- **Open decision (deferred to iteration-1 Research Actor):** GEDI access route — GEE-asset
  mirror of the sugar pipeline vs. LP-DAAC `earthaccess`. Iteration 0 does not pre-commit.

## Reproducibility footer

- inputs: `evaluation/biomass_metrics.json`, `evaluation/evaluation_matrix.yaml`,
  `checkpoints/metrics.json`, `evaluation/figures/residuals_by_quintile.png`
- libraries: analysis of computed metrics (no new computation)
- seed: 42
- command_or_entrypoint: synthesis of evaluation artefacts
- timestamp_utc: 2026-05-29T09:38:00Z
