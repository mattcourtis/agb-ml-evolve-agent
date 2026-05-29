#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
skill_root="$(dirname "${script_dir}")"
repo_root="$(cd "${skill_root}/../.." && pwd)"

failures=0
warnings=0

ok() {
  printf 'ok - %s\n' "$1"
}

warn() {
  printf 'warn - %s\n' "$1" >&2
  warnings=$((warnings + 1))
}

fail() {
  printf 'fail - %s\n' "$1" >&2
  failures=$((failures + 1))
}

check_required_command() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    ok "required command '${name}' found"
  else
    fail "required command '${name}' not found"
  fi
}

check_optional_command() {
  local name="$1"
  local why="$2"
  if command -v "${name}" >/dev/null 2>&1; then
    ok "optional command '${name}' found"
  else
    warn "optional command '${name}' not found (${why})"
  fi
}

check_env() {
  local name="$1"
  local why="$2"
  if [[ -n "${!name:-}" ]]; then
    ok "env ${name} is set"
  else
    warn "env ${name} is not set (${why})"
  fi
}

check_required_command bash
check_required_command sed
check_required_command grep
check_required_command date
check_optional_command python3 "useful for YAML checks and ML utilities"
check_optional_command ruby "preferred YAML parser for tests/bootstrap_smoke.sh"
check_optional_command git "needed for final/git_snapshot.txt"
check_optional_command gcloud "useful for Google Earth Engine project auth"
check_optional_command earthengine "useful for Earth Engine asset workflows"
check_optional_command hf "useful for Hugging Face Hub upload/download"
check_optional_command aws "useful for S3 artifact mirroring"

if ! command -v ruby >/dev/null 2>&1 && ! python3 -c 'import yaml' >/dev/null 2>&1; then
  warn "no YAML parser available (install ruby or 'pip install pyyaml'); bootstrap_smoke will skip YAML validation"
fi

if tmpfile="$(mktemp "${repo_root}/.doctor_write_test.XXXXXX" 2>/dev/null)"; then
  rm -f "${tmpfile}"
  ok "repository root is writable"
else
  fail "repository root is not writable"
fi

check_env GCP_PROJECT "Earth Engine runs require a Cloud project"
check_env HF_TOKEN "required only when pushing to Hugging Face Hub"
check_env AWS_PROFILE "required only when mirroring artifacts to S3"
check_env S3_BUCKET "required only when mirroring artifacts to S3"

if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  if [[ -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
    ok "GOOGLE_APPLICATION_CREDENTIALS points to an existing file"
  else
    warn "GOOGLE_APPLICATION_CREDENTIALS is set but the file does not exist"
  fi
else
  warn "GOOGLE_APPLICATION_CREDENTIALS is not set (service-account runs may need it)"
fi

# Biomass-specific data-access checks.

anew_path="${ANEW_GT_PATH:-/home/mattc/data-space/carbonmap-embeddings/training-data/anew_gt_with_eco_info.gpkg}"
if [[ -f "${anew_path}" ]]; then
  ok "ANEW ground-truth gpkg found at ${anew_path}"
else
  warn "ANEW ground-truth gpkg not found at ${anew_path} (set ANEW_GT_PATH if mounted elsewhere)"
fi

if command -v aws >/dev/null 2>&1; then
  if aws s3 ls --no-sign-request s3://us-west-2.opendata.source.coop/ >/dev/null 2>&1; then
    ok "Source Coop anonymous read reachable (us-west-2)"
  else
    warn "Source Coop anonymous read not reachable (AEF embedding tiles will not download)"
  fi
else
  warn "aws CLI absent — cannot probe Source Coop AEF bucket"
fi

if command -v earthengine >/dev/null 2>&1; then
  if earthengine ls projects/lp-daac-cloud-data/assets/GEDI >/dev/null 2>&1; then
    ok "GEDI collections reachable via earthengine"
  else
    warn "GEDI collections not reachable via earthengine (auth or project access may be missing)"
  fi
else
  warn "earthengine CLI absent — cannot probe GEDI access"
fi

printf '\nDoctor summary: %s failure(s), %s warning(s).\n' "${failures}" "${warnings}"

if [[ "${failures}" -gt 0 ]]; then
  exit 1
fi
