#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/pediatric_leukemia_stage4.toml"
OVERWRITE=""
if [[ "${1:-}" == "--overwrite" ]]; then
  OVERWRITE="--overwrite"
fi

printf '%s\n' '========================================================================'
printf '%s\n' 'Stage 4A: archive Figure 3 statistics and selections'
printf '%s\n' '========================================================================'
python application/pediatric_leukemia/prepare_figure3_non_gaussian_results.py \
  --config "$CONFIG" $OVERWRITE

printf '\n%s\n' '========================================================================'
printf '%s\n' 'Stage 4B: plot Figure 3 from archived tables'
printf '%s\n' '========================================================================'
python application/pediatric_leukemia/plot_figure3_non_gaussian.py --config "$CONFIG"

printf '\n%s\n' '========================================================================'
printf '%s\n' 'Stage 4C: plot Figure 4 from archived Stage 2 tables'
printf '%s\n' '========================================================================'
python application/pediatric_leukemia/plot_figure4_model_comparison.py --config "$CONFIG"

printf '\n%s\n' '========================================================================'
printf '%s\n' 'Stage 4D: validate computation/graphics separation'
printf '%s\n' '========================================================================'
python application/pediatric_leukemia/validate_stage4_separation.py --config "$CONFIG"
