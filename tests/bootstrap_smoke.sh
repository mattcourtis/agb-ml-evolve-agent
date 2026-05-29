#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
date_tag="$(date -u +%Y%m%d)"

fail() {
  printf 'not ok - %s\n' "$1" >&2
  exit 1
}

ok() {
  printf 'ok - %s\n' "$1"
}

expect_status() {
  local expected="$1"
  shift
  set +e
  "$@" >/dev/null 2>&1
  local status=$?
  set -e
  if [[ "${status}" -eq "${expected}" ]]; then
    ok "expected exit ${expected}: $*"
  else
    fail "expected exit ${expected}, got ${status}: $*"
  fi
}

# smoke_skill <skill_name> <subject> <task> <bad_task> <expected_split_strategy>
smoke_skill() {
  local skill_name="$1"
  local subject="$2"
  local task="$3"
  local bad_task="$4"
  local expected_strategy="$5"

  local skill_root="${repo_root}/skills/${skill_name}"
  local bootstrap="${skill_root}/scripts/bootstrap_experiment.sh"
  local validator="${skill_root}/scripts/validate_skill.sh"
  local tmpdir
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' RETURN

  printf '\n=== smoke: %s ===\n' "${skill_name}"

  "${bootstrap}" "${subject}" "${task}" 'Iowa|US & Corn Belt' "${tmpdir}" >/dev/null
  local exp="${tmpdir}/${subject}_${task}_${date_tag}"

  [[ -f "${exp}/IMPLEMENTATION_PLAN.md" ]] || fail "${skill_name}: IMPLEMENTATION_PLAN.md missing"
  [[ -f "${exp}/configs/experiment_config.yaml" ]] || fail "${skill_name}: experiment_config.yaml missing"
  [[ -f "${exp}/reports/model_card.md" ]] || fail "${skill_name}: draft model card missing"
  [[ -f "${exp}/reports/data_card.md" ]] || fail "${skill_name}: draft data card missing"
  ok "${skill_name}: bootstrap created expected seed files"

  if grep -rEln '\{[a-z_]+\}' "${exp}" >/dev/null; then
    fail "${skill_name}: unresolved placeholders remain"
  else
    ok "${skill_name}: no unresolved placeholders remain"
  fi

  grep -q "strategy: \"${expected_strategy}\"" "${exp}/configs/experiment_config.yaml" \
    || fail "${skill_name}: canonical split strategy '${expected_strategy}' missing"
  grep -q 'budget_tier: "Small"' "${exp}/configs/experiment_config.yaml" || fail "${skill_name}: budget tier missing"
  grep -q 'max_full_iterations: 3' "${exp}/configs/experiment_config.yaml" || fail "${skill_name}: max_full_iterations not seeded to Small-tier cap"
  grep -q 'allow_extended_iterations: false' "${exp}/configs/experiment_config.yaml" || fail "${skill_name}: allow_extended_iterations flag missing"
  ok "${skill_name}: generated config has canonical split, budget tier, and iteration knobs"

  if command -v ruby >/dev/null 2>&1; then
    ruby -e 'require "yaml"; ARGV.each { |path| YAML.load_file(path) }' \
      "${exp}/configs/experiment_config.yaml" \
      "${exp}/evaluation/evaluation_matrix.yaml"
    ok "${skill_name}: generated YAML parses (ruby)"
  elif python3 -c 'import yaml' >/dev/null 2>&1; then
    python3 -c '
import sys, yaml
for p in sys.argv[1:]:
    with open(p) as fh:
        yaml.safe_load(fh)
' "${exp}/configs/experiment_config.yaml" "${exp}/evaluation/evaluation_matrix.yaml"
    ok "${skill_name}: generated YAML parses (python3+yaml)"
  else
    ok "${skill_name}: generated YAML parse skipped (install ruby or 'pip install pyyaml' to enable)"
  fi

  expect_status 2 "${bootstrap}" "${subject}" "${bad_task}" Iowa "${tmpdir}"
  expect_status 2 "${bootstrap}" '!!!' "${task}" Iowa "${tmpdir}"
  expect_status 3 "${bootstrap}" "${subject}" "${task}" Iowa "${tmpdir}"

  mkdir -p "${exp}/final/model"
  printf 'stale\n' > "${exp}/final/model/stale.bin"
  "${bootstrap}" "${subject}" "${task}" Iowa "${tmpdir}" --force >/dev/null

  if [[ -f "${exp}/final/model/stale.bin" ]]; then
    fail "${skill_name}: --force left stale final/model artifact in active experiment"
  fi

  local archive_dir
  archive_dir="$(find "${tmpdir}" -maxdepth 1 -type d -name "${subject}_${task}_${date_tag}.archive_*" | head -n 1)"
  [[ -n "${archive_dir}" ]] || fail "${skill_name}: --force did not create archive directory"
  [[ -f "${archive_dir}/final/model/stale.bin" ]] || fail "${skill_name}: archive does not contain prior stale artifact"
  ok "${skill_name}: --force archives previous experiment and recreates a clean active directory"

  "${validator}" >/dev/null
  ok "${skill_name}: skill validator passes"
}

smoke_skill biomass-ml-agent-evolve  agb_usa biomass_regression   invalid_task   spatial_holdout

printf '\nbootstrap_smoke passed.\n'
