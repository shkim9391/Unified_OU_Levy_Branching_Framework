#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${REPO_ROOT}/configs/pediatric_leukemia_stage1.toml"

OVERWRITE=0
RUN_VALIDATION=1

for arg in "$@"; do
  case "$arg" in
    --overwrite)
      OVERWRITE=1
      ;;
    --skip-validation)
      RUN_VALIDATION=0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--overwrite] [--skip-validation]" >&2
      exit 2
      ;;
  esac
done

run_stage_script() {
  local script="$1"
  if [[ "$OVERWRITE" -eq 1 ]]; then
    python "$script" --config "$CONFIG" --overwrite
  else
    python "$script" --config "$CONFIG"
  fi
}

run_stage_script \
  "${SCRIPT_DIR}/02_project_longitudinal_cells_into_frozen_scaffold.py"

run_stage_script \
  "${SCRIPT_DIR}/04_compute_patient_interval_metrics.py"

if [[ "$RUN_VALIDATION" -eq 1 ]]; then
  python "${SCRIPT_DIR}/validate_stage1_regression.py" --config "$CONFIG"
fi
