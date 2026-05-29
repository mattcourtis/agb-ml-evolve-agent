#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
skill_root="$(dirname "${script_dir}")"
repo_root="$(cd "${skill_root}/../.." && pwd)"

failures=0
tmpdir=""

cleanup() {
  if [[ -n "${tmpdir}" && -d "${tmpdir}" ]]; then
    rm -rf "${tmpdir}"
  fi
}
trap cleanup EXIT

ok() {
  printf 'ok - %s\n' "$1"
}

not_ok() {
  printf 'not ok - %s\n' "$1" >&2
  failures=$((failures + 1))
}

require_file() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    ok "found ${path#${repo_root}/}"
  else
    not_ok "missing ${path#${repo_root}/}"
  fi
}

config_template="${skill_root}/assets/experiment_config.template.yaml"
skill_md="${skill_root}/SKILL.md"
model_selection="${skill_root}/references/model_selection.md"
model_saving="${skill_root}/references/model_saving.md"
orchestration="${skill_root}/references/orchestration.md"

require_file "${config_template}"
require_file "${skill_md}"
require_file "${model_selection}"
require_file "${model_saving}"
require_file "${orchestration}"

strategy="$(
  awk '
    /^splits:/ { in_splits = 1; next }
    /^[a-z_]+:/ { in_splits = 0 }
    in_splits && /^  strategy:/ {
      gsub(/"/, "", $2)
      print $2
      exit
    }
  ' "${config_template}"
)"

case "${strategy}" in
  spatial_holdout|temporal_holdout|spatiotemporal_holdout|random)
    ok "split strategy '${strategy}' is allowed"
    ;;
  *)
    not_ok "split strategy '${strategy:-missing}' is not allowed"
    ;;
esac

if awk '
  /^training:/ { in_training = 1; next }
  /^[a-z_]+:/ { in_training = 0 }
  in_training && /^  budget_tier:/ { found = 1 }
  END { exit found ? 0 : 1 }
' "${config_template}"; then
  ok "training.budget_tier is present"
else
  not_ok "training.budget_tier is missing"
fi

if grep -q 'training.budget_tier' "${model_selection}"; then
  ok "model selection references training.budget_tier"
else
  not_ok "model selection does not reference training.budget_tier"
fi

if awk '
  /^training:/ { in_training = 1; next }
  /^[a-z_]+:/ { in_training = 0 }
  in_training && /^  allow_extended_iterations:/ { found = 1 }
  END { exit found ? 0 : 1 }
' "${config_template}"; then
  ok "training.allow_extended_iterations is present"
else
  not_ok "training.allow_extended_iterations is missing"
fi

tier="$(
  awk '
    /^training:/ { in_training = 1; next }
    /^[a-z_]+:/ { in_training = 0 }
    in_training && /^  budget_tier:/ {
      gsub(/"/, "", $2); print $2; exit
    }
  ' "${config_template}"
)"

iters="$(
  awk '
    /^training:/ { in_training = 1; next }
    /^[a-z_]+:/ { in_training = 0 }
    in_training && /^  max_full_iterations:/ {
      print $2; exit
    }
  ' "${config_template}"
)"

case "${tier}" in
  Small)  tier_cap=3 ;;
  Medium) tier_cap=4 ;;
  Large)  tier_cap=6 ;;
  *)      tier_cap="" ;;
esac

if [[ -n "${tier_cap}" && -n "${iters}" && "${iters}" -le "${tier_cap}" ]]; then
  ok "max_full_iterations (${iters}) respects ${tier} tier cap (${tier_cap})"
else
  not_ok "max_full_iterations (${iters:-missing}) exceeds or mismatches ${tier:-unknown} tier cap (${tier_cap:-unknown})"
fi

model_saving_row="$(grep -E '^\| Model Saving Actor \|' "${orchestration}" || true)"
primary_field="$(printf '%s' "${model_saving_row}" | awk -F'\\|' '{print $3}')"
if printf '%s' "${primary_field}" | grep -qE '\+|,'; then
  not_ok "Model Saving Actor registry row lists multiple primary artifacts"
else
  ok "Model Saving Actor primary artifact is a single path"
fi

if grep -q 'Required primary artifact' "${model_saving}" \
  && grep -q 'Required secondary outputs' "${model_saving}"; then
  ok "model_saving.md separates primary and secondary outputs"
else
  not_ok "model_saving.md does not separate primary and secondary outputs"
fi

improvement_loop="${skill_root}/references/improvement_loop.md"
if grep -q 'compute_hours' "${improvement_loop}" \
  && grep -q 'wall_time_hours' "${improvement_loop}" \
  && grep -q 'gpu_class' "${improvement_loop}"; then
  ok "improvement_loop.md specifies expected_cost schema"
else
  not_ok "improvement_loop.md is missing expected_cost schema fields"
fi

experimental_design="${skill_root}/references/experimental_design.md"
if grep -q 'configs/experiment_design.md' "${experimental_design}"; then
  ok "experimental_design.md names the required artifact path"
else
  not_ok "experimental_design.md does not name the required artifact path"
fi

preprocess_line="$(grep -n 'preprocess data$' "${skill_md}" | cut -d: -f1 || true)"
baseline_line="$(grep -n 'train baselines$' "${skill_md}" | cut -d: -f1 || true)"
select_line="$(grep -n 'select model candidates$' "${skill_md}" | cut -d: -f1 || true)"
candidate_line="$(grep -n 'train candidate models$' "${skill_md}" | cut -d: -f1 || true)"

if [[ -n "${preprocess_line}" && -n "${baseline_line}" && -n "${select_line}" && -n "${candidate_line}" \
  && "${preprocess_line}" -lt "${baseline_line}" \
  && "${baseline_line}" -lt "${select_line}" \
  && "${select_line}" -lt "${candidate_line}" ]]; then
  ok "core loop order is baseline-first"
else
  not_ok "core loop order is not baseline-first"
fi

if grep -q 'Draft card lifecycle' "${model_saving}"; then
  ok "model/data card lifecycle is documented"
else
  not_ok "model/data card lifecycle is not documented"
fi

for final_path in 'final/model_card.md' 'final/data_card.md'; do
  if grep -q "${final_path}" "${orchestration}" && grep -q "${final_path}" "${model_saving}"; then
    ok "${final_path} appears in final QA contracts"
  else
    not_ok "${final_path} is missing from a final QA contract"
  fi
done

tmpdir="$(mktemp -d)"
if "${skill_root}/scripts/bootstrap_experiment.sh" agb_usa biomass_regression 'CONUS|forest & ecoregions' "${tmpdir}" >/dev/null; then
  experiment_dir="${tmpdir}/agb_usa_biomass_regression_$(date -u +%Y%m%d)"
  if grep -rEln '\{[a-z_]+\}' "${experiment_dir}" >/dev/null; then
    not_ok "bootstrap left unresolved placeholders"
  else
    ok "bootstrap renders without unresolved placeholders"
  fi
  if grep -q 'strategy: "spatial_holdout"' "${experiment_dir}/configs/experiment_config.yaml"; then
    ok "bootstrap renders canonical split strategy"
  else
    not_ok "bootstrap did not render canonical split strategy"
  fi
  if grep -q 'budget_tier: "Small"' "${experiment_dir}/configs/experiment_config.yaml"; then
    ok "bootstrap renders training budget tier"
  else
    not_ok "bootstrap did not render training budget tier"
  fi
  if grep -q 'allow_extended_iterations: false' "${experiment_dir}/configs/experiment_config.yaml"; then
    ok "bootstrap renders allow_extended_iterations flag"
  else
    not_ok "bootstrap did not render allow_extended_iterations flag"
  fi
else
  not_ok "bootstrap smoke render failed"
fi

if [[ "${failures}" -gt 0 ]]; then
  printf '\nvalidate_skill failed with %s issue(s).\n' "${failures}" >&2
  exit 1
fi

printf '\nvalidate_skill passed.\n'
