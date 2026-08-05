#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${REPO_ROOT}/configs/pediatric_leukemia_stage2.toml"

OVERWRITE=0
RUN_MODEL_COMPARISON=1
RUN_VALIDATION=1

for arg in "$@"; do
  case "$arg" in
    --overwrite)
      OVERWRITE=1
      ;;
    --skip-model-comparison)
      RUN_MODEL_COMPARISON=0
      ;;
    --skip-validation)
      RUN_VALIDATION=0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--overwrite] [--skip-model-comparison] [--skip-validation]" >&2
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

echo "========================================================================"
echo "Stage 2: sample-level effective dynamic parameters"
echo "========================================================================"
run_stage_script "${SCRIPT_DIR}/11_compute_sample_dynamic_parameters.py"

if [[ "$RUN_MODEL_COMPARISON" -eq 1 ]]; then
  echo
  echo "========================================================================"
  echo "Stage 2: Gaussian/Student-t model comparison"
  echo "========================================================================"
  run_stage_script "${SCRIPT_DIR}/make_figure4_model_comparison.py"
fi

if [[ "$RUN_VALIDATION" -eq 1 ]]; then
  echo
  echo "========================================================================"
  echo "Stage 2: regression validation"
  echo "========================================================================"
  if [[ "$RUN_MODEL_COMPARISON" -eq 1 ]]; then
    python "${SCRIPT_DIR}/validate_stage2_regression.py" --config "$CONFIG"
  else
    python "${SCRIPT_DIR}/validate_stage2_regression.py" \
      --config "$CONFIG" --skip-model-comparison
  fi
fi

echo
echo "========================================================================"
echo "Stage 2 completed successfully"
echo "========================================================================"
