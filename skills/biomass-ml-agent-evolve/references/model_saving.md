# Model Saving and Final Packaging

## Purpose
Produce a deployable, auditable, reproducible final package.

## Must save
- final model
- best checkpoint
- preprocessing pipeline
- feature schema
- training config
- evaluation results
- error analysis
- model card
- data card
- environment/dependency record
- git commit or code snapshot if available

## Final structure
final/
├── model/
├── preprocessing_pipeline/
├── evaluation_matrix.yaml
├── model_card.md
├── data_card.md
├── experiment_report.md
├── run_summary.md
├── environment.lock          # pip freeze / conda export / uv lock
└── git_snapshot.txt          # commit SHA or code-snapshot pointer; "not_available" only if explicitly justified in run_summary.md

## Required primary artifact

`final/run_summary.md` is the Model Saving Actor's single primary artifact. The Actor returns `PATH:` and `SUMMARY:` for this file alone, in line with the SKILL.md actor contract.

## Required secondary outputs

The Model Saving Actor must also produce, and the Model Saving Critic must verify alongside the primary, every path below. Missing or empty secondary outputs cause the Critic to REJECT.

- `final/model/` — non-empty; serialised weights and loader notes
- `final/preprocessing_pipeline/` — non-empty; transform code and fitted scalers/encoders
- `final/evaluation_matrix.yaml` — copy of the ACCEPTED `evaluation/evaluation_matrix.yaml`
- `final/model_card.md` — promoted from the accepted `reports/model_card.md` draft
- `final/data_card.md` — promoted from the accepted `reports/data_card.md` draft
- `final/experiment_report.md` — produced by the Final Report Actor, but the Model Saving Critic verifies its presence
- `final/environment.lock` (or `final/environment.txt`) — pip freeze / conda export / uv lock
- `final/git_snapshot.txt` — commit SHA or code-snapshot pointer; literal `not_available` only with justification in `run_summary.md`

The Model Saving Critic must record per-path status (PRESENT, EMPTY, MISSING) before issuing ACCEPT.

## Data card schema

Generate `final/data_card.md` from the Hugging Face dataset-card pattern. Seed from `assets/data_card.template.md`. Required sections:

- header (license, language, tags, pretty_name, size_categories)
- Dataset summary (one paragraph)
- Sources — table with `{source_name, url_or_doi, access_date, license}`
- Collection window — start date, end date, label-source revision tag
- Geography — covered regions and exclusions
- Splits — counts and partition keys for train/val/test (must match `configs/split_strategy.yaml`)
- Label generation — how labels were derived; provenance and known biases
- Known limitations and biases
- Intended use and out-of-scope use
- Citation
- Versioning — content SHA256 of the input manifest (must match `preprocessing/data_version.txt`)

## Draft card lifecycle

`scripts/bootstrap_experiment.sh` seeds `reports/model_card.md` and `reports/data_card.md` as editable drafts. The Model Saving Actor must promote the accepted draft content into `final/model_card.md` and `final/data_card.md`, then verify both final files against the Final QA gate. Draft files under `reports/` are not final deliverables.

## Optional mirrors
- Hugging Face model repo
- Hugging Face dataset repo
- S3 artifact mirror
- Earth Engine assets for geospatial outputs

## Critic addendum
Reject if:
- the final model cannot be traced back to an accepted checkpoint, config, and evaluation artifact;
- any file in the Final structure above is missing or empty (with the documented exception for `git_snapshot.txt`);
- `final/data_card.md` lacks any required schema section;
- the data-card `Versioning.content_sha256` does not match `preprocessing/data_version.txt`.
