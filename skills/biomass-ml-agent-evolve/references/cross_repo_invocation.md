# Cross-Repo Invocation

## Purpose
Codify how this skill — which lives in `crop-ml-agent-evolve/` — invokes model code, fetchers, and existing training pipelines that live in the sibling repository `tf-deep-landcover/`. Actors must follow this contract to keep experiment metadata (here) and model code (there) decoupled but reproducibly linked.

## Repo split

| Lives in | Path | Authoritative for |
|---|---|---|
| this repo | `crop-ml-agent-evolve/experiments/{experiment_id}/` | experiment metadata: `IMPLEMENTATION_PLAN.md`, plans, configs, evaluation matrices, reports, model and data cards, final bundle |
| this repo | `crop-ml-agent-evolve/skills/biomass-ml-agent-evolve/` | the orchestrator/actor/critic contract, references, asset templates, scripts |
| sibling | `tf-deep-landcover/src/agb/` | feature extractors, LightGBM trainer, project-LOPO CV harness |
| sibling | `tf-deep-landcover/src/fetch/` | data fetchers (Source Coop AEF, JAXA PALSAR-2, Sentinel-1, etc.) |
| sibling | `tf-deep-landcover/experiments/agb/{pool}/` | per-pool parquet features, checkpoints, fold metrics, OOF predictions, figure outputs |
| sibling | `tf-deep-landcover/data/raw/locations/` | AOI GeoJSONs used to scope feature extraction (e.g., `agb_usa_v1_pilot_wv.geojson`) |

The skill **never** writes to `tf-deep-landcover/`. The skill **invokes** scripts there; those scripts read inputs and write outputs to locations declared in their CLI args.

## Existing entry points actors must reuse, not reimplement

Actors that need to extract features or train models must call these scripts via `uv run python -m …`. Reimplementing this logic inside an Actor artefact is rejected by the Critic.

### Feature extraction — primary
```
cd /home/mattc/code/tf-deep-landcover && \
uv run python -m src.agb.extract_features_batched \
  --plots <ANEW gpkg> \
  --aoi   <AOI geojson under data/raw/locations/> \
  --years <int...> \
  --out   <parquet path> \
  [--limit <int>]
```
- Output schema: `plot_id, project_name, year, lon, lat, target, failure, emb_00..emb_63`.
- 50× faster than the point-wise variant; tile is opened once per AOI window.
- Auth: anonymous Source Coop read. No GCP required for embedding fetch.

### Feature extraction — per-plot fallback
```
uv run python -m src.agb.extract_features_at_points \
  --plots <ANEW gpkg> \
  --pilot-bbox <AOI geojson> \
  --years <int...> \
  --out   <parquet path> \
  [--target-column CO2|Annual_CO2] \
  [--window-size <int>] \
  [--mode mean|bilinear] \
  [--palsar-dir <path>] \
  [--skip-palsar] [--limit <int>] [--seed <int>]
```
- Used for footprint-weighted / per-mode experiments, or when PALSAR features are required.

### Training — LightGBM with project-LOPO CV
```
uv run python -m src.agb.train_agb_lgbm \
  --features <parquet> \
  --out-dir  <checkpoint + metrics dir> \
  --fig-dir  <figure output dir> \
  [--pca-n-components <int>]
```
- Outputs: `{out_dir}/metrics.json` (aggregate + per-fold R²/RMSE/MAE/bias/n), `{out_dir}/model.txt` (LightGBM text format), optional `{out_dir}/pca.pkl`, `{fig_dir}/shap_*.png`, `{fig_dir}/residuals_by_quintile.png`.
- CV partition key: `project_name` — leave-one-project-out.
- Hyperparameters are hardcoded (num_leaves=31, learning_rate=0.05, n_estimators=2000, early stop=50). Changing them requires editing `tf-deep-landcover/src/agb/train_agb_lgbm.py`, which counts as a sibling-repo code change (see *Sibling-repo code edits*, below).

### Source fetchers
- `src/fetch/embeddings.py` — AlphaEarth Foundation (AEF) tile fetcher (vsicurl, anonymous Source Coop). Used implicitly by both extractors.
- `src/fetch/palsar2.py` — JAXA PALSAR-2 annual mosaic via Google Earth Engine. Requires `GCP_PROJECT` + GEE service-account credentials.
- `src/fetch/gee_utils.py` — GEE auth wrapper. Reads `GCP_PROJECT` / `GS_USER_PROJECT`.

## Path-resolution rule

Configs in the experiment dir (`configs/experiment_config.yaml`, `configs/training_config.yaml`) must record **absolute paths** for sibling-repo inputs and outputs. Actors construct commands by quoting these absolute paths verbatim. The skill never assumes the sibling repo is at any particular relative location.

Examples (illustrative; real values populate at config time):
```yaml
data:
  database_uri: "/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg"
  data_dir:    "/home/mattc/code/tf-deep-landcover"
  raw_sources:
    - "/home/mattc/code/tf-deep-landcover/data/raw/locations/agb_usa_v1_pilot_joint_v2.geojson"
```

## Dual git-SHA capture

The Model Saving Actor must write `final/git_snapshot.txt` containing **both** repo SHAs, one per line:

```
crop-ml-agent-evolve: <SHA from this repo at finalisation time>
tf-deep-landcover:    <SHA from the sibling repo at the time the extractor / trainer was invoked>
```

If either SHA is the literal string `not_available`, `run_summary.md` must justify it (e.g., `tf-deep-landcover` checked out detached, no remote). A single-line `git_snapshot.txt` naming only one repo is rejected by the Model Saving Critic.

## Reproducibility-footer extension

`preprocessing/data_version.txt` must record:
- the four base fields from `references/database_preprocessing.md` (input manifest SHA256, source URI list, snapshot UTC timestamp, label-source revision tag)
- the `tf-deep-landcover` commit SHA active when the extractor was invoked (line: `tf_deep_landcover_sha: <sha>`)

The same SHA must appear in `final/git_snapshot.txt`. The Critic cross-checks these two values; a mismatch is rejected.

## Sibling-repo code edits

Actors should treat `tf-deep-landcover/` as **read-and-invoke, not write**. If an experiment requires a new feature extractor (e.g., GEDI canopy height) or a new model variant, the Implementation Actor must:

1. Open a feature-branch on `tf-deep-landcover/` (do not push to its main without explicit user approval).
2. Add the new script under `tf-deep-landcover/src/agb/` (e.g., `extract_gedi_rh_at_points.py`).
3. Smoke the new script on a ≤10-row slice and verify the output schema matches the declaration in `preprocessing/feature_schema.json`.
4. Record the branch name and commit SHA in `preprocessing/data_version.txt` and `final/git_snapshot.txt`.
5. Critic cross-checks that the declared schema and the smoke-output schema agree before any full-pool run is approved.

## Critic addendum
Reject any Actor artefact that:
- reimplements logic already present in `tf-deep-landcover/src/agb/` or `tf-deep-landcover/src/fetch/` without an explicit RFC and user approval;
- omits absolute paths to sibling-repo inputs / outputs;
- declares `final/git_snapshot.txt` that names only one repo (must name both);
- shows a `tf-deep-landcover` SHA in `final/git_snapshot.txt` that does not match `preprocessing/data_version.txt`;
- proposes a sibling-repo code edit without the smoke-on-10-rows + schema-declaration steps above.
