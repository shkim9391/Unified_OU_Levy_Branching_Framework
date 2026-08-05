#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ "${1:-}" == "--overwrite" ]]; then
  python "$SCRIPT_DIR/run_stage5_benchmarks.py" --config "$ROOT/configs/pediatric_leukemia_stage5.toml" --overwrite
elif [[ $# -eq 0 ]]; then
  python "$SCRIPT_DIR/run_stage5_benchmarks.py" --config "$ROOT/configs/pediatric_leukemia_stage5.toml"
else
  echo "Usage: $0 [--overwrite]" >&2; exit 2
fi
python -m pytest
