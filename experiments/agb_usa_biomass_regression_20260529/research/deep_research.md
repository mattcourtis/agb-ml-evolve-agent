# Deep Research — agb_usa biomass regression (iteration 0 benchmark anchor)

## Scope

Iteration 0 is a **wiring validation**: reproduce the existing `joint_v2` baseline
(R²=0.42) inside this orchestrator. This research artefact fixes the benchmark range,
the default/stretch thresholds, and — critically — the **levers already ruled out**, so
no downstream Actor wastes the iteration re-litigating settled questions.

Primary source: `tf-deep-landcover/docs/runs/agb_usa.md` (investigation summary 2026-05-12)
and `docs/runs/agb-modelling-context.md`. These are internal experiment records for the
exact pool and model class being reproduced here, so they are the authoritative anchor.

## Benchmark range (CONUS forest AGB, plot-level, project-LOPO CV)

| Anchor | R² | RMSE (tCO₂/acre) | MAE | Bias | n | source |
|---|---:|---:|---:|---:|---:|---|
| Joint v1 (17 projects) | 0.39 | 61 | 45 | +0.6 | 3,229 | agb_usa.md headline table |
| **Joint v2 (23 projects) — reproduction target** | **0.42** | **57** | **41** | **+0.5** | **4,636** | agb_usa.md headline table |
| WV Appalachia (region) | 0.17 | 80 | 62 | +1.0 | 598 | agb_usa.md |
| Upper Midwest (region) | 0.42 | 55 | 41 | +1.2 | 2,631 | agb_usa.md |
| New England (region) | 0.44 | 46 | 36 | +1.0 | 1,407 | agb_usa.md |

## Thresholds (locked, mirror `configs/experiment_config.yaml`)

- **Realistic:** R² ≥ 0.40, RMSE ≤ 60, |bias| ≤ 5.
- **Stretch:** R² ≥ 0.55, RMSE ≤ 45, predicted_range_discrimination ≥ 0.6.
- **Iteration-0 acceptance:** reproduce R² = 0.42 within ±0.03 (i.e. R² ∈ [0.39, 0.45]).

## The diagnosed ceiling (why the target is 0.42, not higher)

The embeddings-only model has a **hard ceiling driven by a feature deficit, not a tuning
problem**. It compresses the dynamic range: it under-predicts high-biomass plots and
over-predicts low-biomass plots. A closed-canopy hardwood stand at 50 tCO₂/acre looks
identical from above to one at 250 tCO₂/acre, and 64-dim optical AEF embeddings cannot
separate them. Numerically, predicted-range discrimination is **21 % on WV, 46 % on the
Midwest** (a perfect model is 100 %).

## Levers already ruled out — DO NOT re-litigate in iteration 0

Per `references/improvement_loop.md` Critic addendum, the following were each run with an
explicit hypothesis and falsified; re-running them is rejected:

| # | Lever | Result | Verdict |
|---:|---|---|---|
| 1 | More plots (joint pool) | WV Q1 bias unchanged (+76→+75) | Not a data-coverage problem |
| 2 | Area-weighted footprint sampling | biases unchanged within ±2 | Not a sampling-scale problem |
| 3 | Huber robust loss | R² drops 0.07 | No outliers — systematic gaps |
| 4 | Log-target | Q1 bias −20 but Q5 bias +25 | Trade-off, not a fix |
| 5 | Isotonic post-hoc calibration | biases move <3, R² drops | Model doesn't rank well enough to calibrate |

Also no longer worth time (per source): sub-pixel reprojection, loss-function swaps
without new features, bias-correction layers without new features, and adding more plots
from the same modality.

## Next-lever pointer (iteration 1, route deferred)

The clear next step is **GEDI canopy height** as an added feature (spaceborne LiDAR
measures the vertical structure optical sensors cannot see). PALSAR-2 SAR was tested and
added only ~0.02 R² (within noise) — dropped. The GEDI **access route** (GEE-asset mirror
of the sugar pipeline vs. LP-DAAC `earthaccess`) is explicitly deferred to the iteration-1
Research Actor; iteration 0 must not pre-commit a route.

## Reproducibility footer

- input sources: `tf-deep-landcover/docs/runs/agb_usa.md`, `docs/runs/agb-modelling-context.md` @ tf-deep-landcover SHA `e8c70584`
- libraries: n/a (literature/record synthesis)
- seed: n/a
- command_or_entrypoint: manual synthesis of internal run records
- timestamp_utc: 2026-05-29T09:36:00Z
