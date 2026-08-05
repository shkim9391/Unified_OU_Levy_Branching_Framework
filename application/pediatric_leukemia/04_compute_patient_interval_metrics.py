from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oulb.dynamics import (  # noqa: E402
    CentroidColumnSpec,
    IntervalDefinition,
    compute_intervals,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "pediatric_leukemia_stage1.toml"
TIME_ORDER = {"DX": 0, "EOI_REM": 1, "REL": 2}
INTERVAL_DEFINITIONS = (
    IntervalDefinition(name="DX_to_REL", start="DX", end="REL"),
    IntervalDefinition(name="DX_to_EOI_REM", start="DX", end="EOI_REM"),
    IntervalDefinition(
        name="EOI_REM_to_REL",
        start="EOI_REM",
        end="REL",
        prior="DX",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Stage 1 TOML configuration (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing Stage 1 output files.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def require_input(path: Path) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def prepare_output(path: Path, *, overwrite: bool) -> Path:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}\nRe-run with --overwrite to replace it."
        )
    return path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["intervals"]

    input_all = require_input(Path(paths["centroids_all_csv"]))
    input_main = require_input(Path(paths["centroids_main_csv"]))
    output_all = prepare_output(
        Path(paths["output_intervals_all_csv"]), overwrite=args.overwrite
    )
    output_main = prepare_output(
        Path(paths["output_intervals_main_csv"]), overwrite=args.overwrite
    )
    centroids_all = pd.read_csv(input_all)
    centroids_main = pd.read_csv(input_main)

    common_arguments = {
        "time_order": TIME_ORDER,
        "interval_definitions": INTERVAL_DEFINITIONS,
        "columns": CentroidColumnSpec(),
        "hd_prefix": "hd_",
        # Unlike the original dictionary overwrite, duplicate centroids now
        # fail explicitly. This does not alter validated unique inputs.
        "duplicate_policy": "error",
        "allow_unordered_timepoints": False,
        "tail_reference_interval": "DX_to_REL",
        "tail_quantile": 0.90,
        "tail_threshold_column": "tail_threshold_dx_rel_q90",
        "tail_flag_column": "tail_flag_dx_rel_q90",
    }
    intervals_all = compute_intervals(centroids_all, **common_arguments)
    intervals_main = compute_intervals(centroids_main, **common_arguments)

    intervals_all.to_csv(output_all, index=False)
    intervals_main.to_csv(output_main, index=False)

    print(f"[DONE] Saved all intervals:  {output_all}")
    print(f"[DONE] Saved main intervals: {output_main}")

    print("\n[SUMMARY: all interval counts]")
    print(intervals_all["interval_class"].value_counts(dropna=False).sort_index())

    print("\n[SUMMARY: main interval counts]")
    print(intervals_main["interval_class"].value_counts(dropna=False).sort_index())

    print("\n[SUMMARY: main DX_to_REL displacement]")
    dx_rel = pd.to_numeric(
        intervals_main.loc[
            intervals_main["interval_class"] == "DX_to_REL", "displacement_hd"
        ],
        errors="coerce",
    )
    if dx_rel.notna().any():
        print(dx_rel.describe())

    aml21_mask = intervals_main["patient_id"].astype(str) == "AML21"
    if aml21_mask.any():
        print("\n[INFO] AML21 interval records:")
        print(
            intervals_main.loc[
                aml21_mask,
                [
                    "patient_id",
                    "interval_class",
                    "sample_start",
                    "sample_end",
                    "displacement_2d",
                    "displacement_hd",
                    "branch_start",
                    "branch_end",
                    "branch_switch",
                    "directional_discontinuity",
                ],
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
