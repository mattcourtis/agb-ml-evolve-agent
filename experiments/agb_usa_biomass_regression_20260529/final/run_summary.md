# Run Summary — agb_usa biomass regression, iteration 0

## Outcome

**Wiring validation PASSED.** The orchestrator reproduced the joint_v2 baseline
**bit-identically**: R²=0.4182 (target 0.42, within ±0.03), RMSE=56.58, MAE=41.49, bias=+0.50,
n=4,636, 23 project-LOPO folds. Decision: **advance to iteration 1** (GEDI feature test).

## What ran

| stage | how | result |
|---|---|---|
| research | Orchestrator (record synthesis) | benchmark + ruled-out levers fixed |
| data_profile | Orchestrator | ANEW gpkg + joint_v2 subset profiled |
| experiment_design | Orchestrator | acceptance metrics + split design |
| preprocessing | Orchestrator | reuse features.parquet; schema + data_version |
| split_design | Orchestrator | project_name LOPO, no random split |
| baselines | Orchestrator (`models/compute_baselines.py`) | mean R²=−0.01, ridge-PC20 R²=0.37 |
| model_selection | Orchestrator | lightgbm_emb64 fixed (reproduction) |
| **training** | **cross-repo `src.agb.train_agb_lgbm`** | **R²=0.4182, bit-identical to baseline** |
| **evaluation** | **`evaluation/compute_biomass_metrics.py`** | full biomass matrix, no nulls |
| error_analysis | Orchestrator | range compression confirmed; weakest stage = features |
| model_saving | Orchestrator | this bundle |
| final_report | Orchestrator | `final/experiment_report.md` |

Load-bearing artefacts (training, evaluation, error_analysis, model_saving, final_report) were
independently reviewed (Critic pass) before assembly.

## Model traceability

- config: `configs/training_config.yaml`
- checkpoint: `checkpoints/best.ckpt` (= `checkpoints/model.txt`); final bundle:
  `final/model/model.txt`
- metrics: `checkpoints/metrics.json` (= `final/model/metrics.json`)
- per-fold history: `checkpoints/metrics_history.csv`

## OOF-reuse justification

The breakdown metrics (per_quintile_bias, predicted_range_discrimination, per_ecoregion_r2)
were computed from the baseline `oof.parquet`. This is exact, not approximate: the iteration-0
retrain aggregate is **bit-identical** to the baseline `metrics.json`
(R²=0.4182242204386467), so the baseline OOF represents this run exactly. The sibling trainer
does not persist OOF and OOF cannot be regenerated from the final all-data `model.txt`; adding
an `--oof-out` flag is a deferred sibling-repo enhancement (needs approval).

## Git snapshot

```
crop-ml-agent-evolve: ea0551b8b7255dc1b1f3e645896638cce1d8e07b
tf-deep-landcover:    e8c70584fb1a8705308004fbed123392c8f51654
```

The `tf-deep-landcover` SHA matches `preprocessing/data_version.txt`. The
`crop-ml-agent-evolve` SHA is the HEAD at finalisation; the iteration-0 artefacts are present
in the working tree **uncommitted** — this run was not committed/pushed (the repo's `origin` is
a third-party fork; per project policy, no push). Re-run `git rev-parse HEAD` after any commit
to refresh.

## Environment

`final/environment.lock` — `uv export` of the tf-deep-landcover env (3,048 pinned deps,
Python 3.13) that ran training and evaluation.

## Unmet gaps (carried to iteration 1)

- `predicted_range_discrimination` 0.468 < 0.6 (stretch) — dynamic-range collapse.
- `per_quintile_bias` |max| 72.1 > 30 — Q1 over- / Q5 under-prediction.
- Appalachian ecoregion R²=0.157 — worst ecoregion.
- Lever: GEDI canopy height; access route deferred to iteration-1 Research Actor.
