#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${REPO_ROOT}/configs/pediatric_leukemia_stage3.toml"

OVERWRITE=0
RUN_JUMPS=1
RUN_BRANCH_TABLES=1
RUN_THRESHOLD_SENSITIVITY=1
RUN_VALIDATION=1

for arg in "$@"; do
  case "$arg" in
    --overwrite)
      OVERWRITE=1
      ;;
    --skip-jumps)
      RUN_JUMPS=0
      ;;
    --skip-branch-tables)
      RUN_BRANCH_TABLES=0
      ;;
    --skip-threshold-sensitivity)
      RUN_THRESHOLD_SENSITIVITY=0
      ;;
    --skip-validation)
      RUN_VALIDATION=0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--overwrite] [--skip-jumps] [--skip-branch-tables] [--skip-threshold-sensitivity] [--skip-validation]" >&2
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

if [[ "$RUN_JUMPS" -eq 1 ]]; then
  echo "========================================================================"
  echo "Stage 3: operational jump-candidate ranking"
  echo "========================================================================"
  run_stage_script "${SCRIPT_DIR}/08_compute_jump_candidates.py"
fi

if [[ "$RUN_BRANCH_TABLES" -eq 1 ]]; then
  echo
  echo "========================================================================"
  echo "Stage 3: frozen-scaffold branch transition tables"
  echo "========================================================================"
  run_stage_script "${SCRIPT_DIR}/17_compute_branch_transition_tables.py"
fi

if [[ "$RUN_THRESHOLD_SENSITIVITY" -eq 1 ]]; then
  echo
  echo "========================================================================"
  echo "Stage 3: B1-B3 malignant-cell-threshold sensitivity"
  echo "========================================================================"
  run_stage_script "${SCRIPT_DIR}/transition_summary_threshold_sensitivity.py"
fi

if [[ "$RUN_VALIDATION" -eq 1 ]]; then
  echo
  echo "========================================================================"
  echo "Stage 3: regression validation"
  echo "========================================================================"
  VALIDATION_ARGS=("--config" "$CONFIG")
  if [[ "$RUN_JUMPS" -eq 0 ]]; then
    VALIDATION_ARGS+=("--skip-jumps")
  fi
  if [[ "$RUN_BRANCH_TABLES" -eq 0 ]]; then
    VALIDATION_ARGS+=("--skip-branch-tables")
  fi
  if [[ "$RUN_THRESHOLD_SENSITIVITY" -eq 0 ]]; then
    VALIDATION_ARGS+=("--skip-threshold-sensitivity")
  fi
  python "${SCRIPT_DIR}/validate_stage3_regression.py" "${VALIDATION_ARGS[@]}"
fi

echo
echo "========================================================================"
echo "Stage 3 completed successfully"
echo "========================================================================"
