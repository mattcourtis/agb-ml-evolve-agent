# Training

## Purpose
Make training reproducible, budget-aware, and diagnosable.

## Required rules
- fixed seeds
- config-driven runs
- deterministic data snapshot where possible
- checkpointing
- early stopping
- retry-safe logging
- hardware and runtime record
- library version capture
- command capture

## Required artifacts
- reports/training_run.md
- checkpoints/best.ckpt
- configs/training_config.yaml
- checkpoints/metrics_history.csv

## Training gates
- no training until split accepted
- no candidate training until baseline accepted
- no overwrite of prior accepted checkpoint

## Critic addendum
Reject if training config, seed, dataset version, or checkpoint lineage is missing.
