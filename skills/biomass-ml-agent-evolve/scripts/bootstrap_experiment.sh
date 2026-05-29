#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bootstrap_experiment.sh <subject> <task> [geography] [output_root] [--force]

Args:
  subject       Free-form subject name (e.g., agb_usa, canopy_height_wv); will be lowercased and non-alnum slugified.
  task          One of: biomass_regression | canopy_height_regression | biomass_segmentation | change_detection
  geography     Optional, free-form. Default: unspecified.
  output_root   Optional. Default: ./experiments
  --force       Archive an existing same-day experiment dir and recreate it cleanly.

Examples:
  bootstrap_experiment.sh agb_usa biomass_regression "CONUS forest"
  bootstrap_experiment.sh canopy_height_wv canopy_height_regression "WV Appalachia" ./experiments
  bootstrap_experiment.sh biomass_change_neast change_detection "NE Maine bloc" ./experiments --force
EOF
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

# Parse args, allowing --force in any tail position.
force=0
positional=()
for arg in "$@"; do
  case "$arg" in
    --force) force=1 ;;
    -h|--help) usage; exit 0 ;;
    *) positional+=("$arg") ;;
  esac
done

subject_raw="${positional[0]:-}"
task="${positional[1]:-}"
geography="${positional[2]:-unspecified}"
root="${positional[3]:-./experiments}"

# Validate task vocabulary.
case "$task" in
  biomass_regression|canopy_height_regression|biomass_segmentation|change_detection) ;;
  *)
    echo "Error: task must be one of biomass_regression|canopy_height_regression|biomass_segmentation|change_detection (got '$task')" >&2
    exit 2
    ;;
esac

# Slugify subject: lowercase, non-alnum -> underscore, collapse, trim.
subject="$(printf '%s' "$subject_raw" \
  | tr '[:upper:]' '[:lower:]' \
  | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
if [[ -z "$subject" ]]; then
  echo "Error: subject slugified to empty string (input was '$subject_raw')" >&2
  exit 2
fi

date_tag="$(date -u +%Y%m%d)"
created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
experiment_id="${subject}_${task}_${date_tag}"
experiment_dir="${root}/${experiment_id}"

# Idempotency guard: never merge a fresh bootstrap into an existing directory.
if [[ -e "${experiment_dir}" ]]; then
  if [[ $force -eq 1 ]]; then
    archive_tag="$(date -u +%Y%m%dT%H%M%SZ)"
    archive_dir="${experiment_dir}.archive_${archive_tag}"
    if [[ -e "${archive_dir}" ]]; then
      archive_dir="${archive_dir}_$$"
    fi
    mv "${experiment_dir}" "${archive_dir}"
    echo "Archived existing experiment at ${archive_dir}" >&2
  else
    echo "Refusing to overwrite existing experiment at ${experiment_dir}." >&2
    echo "Re-run with --force to archive the existing directory and create a clean replacement." >&2
    exit 3
  fi
fi

mkdir -p "${experiment_dir}"/{research,data_profile,preprocessing,configs,models,checkpoints,evaluation,error_analysis,reports,final}
mkdir -p "${experiment_dir}"/final/{model,preprocessing_pipeline}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
skill_root="$(dirname "${script_dir}")"

# Escape a value for safe injection into a sed replacement using "|" as the delimiter.
# Escapes: backslash, the "|" delimiter, and the "&" backreference.
sed_escape() {
  printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'
}

E_experiment_id="$(sed_escape "${experiment_id}")"
E_created_at="$(sed_escape "${created_at}")"
E_subject="$(sed_escape "${subject}")"
E_task="$(sed_escape "${task}")"
E_geography="$(sed_escape "${geography}")"
E_output_dir="$(sed_escape "${experiment_dir}")"

render_template() {
  local src="$1"
  local dst="$2"
  sed \
    -e "s|{experiment_id}|${E_experiment_id}|g" \
    -e "s|{created_at}|${E_created_at}|g" \
    -e "s|{subject}|${E_subject}|g" \
    -e "s|{task}|${E_task}|g" \
    -e "s|{task_type}|${E_task}|g" \
    -e "s|{geography}|${E_geography}|g" \
    -e "s|{output_dir}|${E_output_dir}|g" \
    -e "s|{target_variable}|TBD|g" \
    -e "s|{spatial_resolution}|TBD|g" \
    -e "s|{temporal_horizon}|TBD|g" \
    -e "s|{spatial_unit}|TBD|g" \
    -e "s|{inference_unit}|TBD|g" \
    -e "s|{forecast_horizon}|TBD|g" \
    -e "s|{performance_threshold}|TBD|g" \
    -e "s|{evaluation_metrics}|TBD|g" \
    -e "s|{compute_budget}|TBD|g" \
    -e "s|{runtime_budget}|TBD|g" \
    -e "s|{yes_no}|TBD|g" \
    -e "s|{task_restatement}|TBD|g" \
    "$src" > "$dst"
}

render_template "${skill_root}/assets/IMPLEMENTATION_PLAN.template.md" "${experiment_dir}/IMPLEMENTATION_PLAN.md"
render_template "${skill_root}/assets/experiment_config.template.yaml" "${experiment_dir}/configs/experiment_config.yaml"
render_template "${skill_root}/assets/evaluation_matrix.template.yaml" "${experiment_dir}/evaluation/evaluation_matrix.yaml"
render_template "${skill_root}/assets/model_card.template.md"          "${experiment_dir}/reports/model_card.md"
render_template "${skill_root}/assets/data_card.template.md"           "${experiment_dir}/reports/data_card.md"

# Prefer the committed env example so users see one canonical copy.
if [[ -f "${skill_root}/assets/integrations.env.example" ]]; then
  cp "${skill_root}/assets/integrations.env.example" "${experiment_dir}/configs/integrations.env.example"
else
  cat > "${experiment_dir}/configs/integrations.env.example" <<'EOF'
# Earth Engine
GCP_PROJECT=
GOOGLE_APPLICATION_CREDENTIALS=
EE_SERVICE_ACCOUNT=

# Hugging Face
HF_TOKEN=
HF_HOME=

# AWS
AWS_PROFILE=
AWS_REGION=us-west-2
S3_BUCKET=
EOF
fi

{
  echo "# Bootstrap summary"
  echo
  echo "- experiment_id: ${experiment_id}"
  echo "- created_at: ${created_at}"
  echo "- subject: ${subject} (raw: ${subject_raw})"
  echo "- task: ${task}"
  echo "- geography: ${geography}"
  echo "- experiment_dir: ${experiment_dir}"
  echo
  echo "## Environment hints"
  echo "- Earth Engine project: ${GCP_PROJECT:-unset}"
  echo "- Hugging Face token present: $([[ -n "${HF_TOKEN:-}" ]] && echo yes || echo no)"
  echo "- AWS profile: ${AWS_PROFILE:-unset}"
} > "${experiment_dir}/reports/bootstrap_summary.md"

# Post-render check: any leftover {placeholder} tokens in rendered files would silently pass downstream.
leftovers="$(grep -rEln '\{[a-z_]+\}' "${experiment_dir}" || true)"
if [[ -n "${leftovers}" ]]; then
  echo "Error: unresolved {placeholder} tokens remain after render in:" >&2
  echo "${leftovers}" >&2
  exit 4
fi

echo "Created ${experiment_dir}"
echo "Next step: Orchestrator restates first-turn checklist, edits config placeholders, then starts Research Actor."
