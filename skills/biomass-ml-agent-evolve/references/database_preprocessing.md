# Database Inspection and Preprocessing

## Purpose
Turn raw data into leakage-safe, reproducible model inputs.

## Mandatory checks
- schema inspection
- label audit
- missingness audit
- outlier detection
- geometry validity
- CRS validation
- spatial joins
- temporal alignment
- cloud masking
- resampling policy
- feature normalization
- duplicate removal
- leakage checks
- split design
- spatial holdout
- temporal holdout
- data versioning

## Required artifacts
- data_profile/database_profile.md
- preprocessing/preprocessing_spec.md
- configs/split_strategy.yaml
- preprocessing/feature_schema.json
- preprocessing/data_version.txt
- preprocessing/split_audit.csv (per-partition unit IDs; required for any spatial or temporal holdout)

`preprocessing/data_version.txt` must contain: SHA256 of the input manifest, full source URI list (one per line), snapshot UTC timestamp, and label-source revision tag (or `none` with justification). The Critic rejects the stage if any of those four fields is missing.

## Split requirements
- never random-only when geography or time matters
- justify region holdout (the partition key — typically `project_name` for plot data)
- justify temporal-holdout cutoff (no future-year satellite features predicting past-year biomass)
- document label availability lag

## Leakage checklist
- contemporary satellite features must be aligned to the label year; no later-year features may be used to predict an earlier-year target
- partition-level aggregate covariates (e.g., per-project mean stand age, per-project mean biomass) leaking into features for the same partition
- contemporaneous revisions in benchmark / cross-validation products (e.g., GEDI L4B, ESA CCI Biomass — never use these as labels; they are references, see `references/source_registry.md`)
- spatially adjacent plots in the same project (the partition key); `split_audit.csv` must show zero project intersection between train and test
- spatial adjacency contamination (plots within neighbour-pixel range across a partition boundary)

## Split Design Actor addendum

The Split Design Actor must produce `configs/split_strategy.yaml` containing at minimum:

- `strategy:` one of `spatial_holdout | temporal_holdout | spatiotemporal_holdout | random` (random allowed only for purely tabular, geography-irrelevant tasks)
- `holdout_proof:` partition key name (e.g., `project_name`, `tile_id`, `ecoregion_code`)
- `random_split_used:` boolean
- `temporal_cutoff_date:` ISO date (required when the target is time-aware)
- `audit_artefact:` path to `preprocessing/split_audit.csv` listing per-partition unit IDs and proving zero intersection between train and test partition keys

The Actor must explicitly justify why the chosen holdouts are appropriate for the task's spatial and temporal generalisation claim.

## Reproducibility footer (required)
Every artefact produced under this reference must end with:

```
Reproducibility:
- input_artefact_sha256: { path: sha256, ... }
- libraries: { name: version, ... }
- seed: <int>
- command_or_entrypoint: <string>
- timestamp_utc: <ISO8601>
```

## Critic addendum
Reject if:
- split logic is missing or cloud/CRS alignment is not explicitly verified;
- `configs/split_strategy.yaml` lacks any field listed in the Split Design Actor addendum;
- `preprocessing/split_audit.csv` does not exist or shows any partition-key intersection between train and test;
- `random_split_used: true` while the task requires spatial or temporal generalisation;
- the reproducibility footer is missing from any produced artefact.
