from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import tomllib

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pediatric_leukemia_stage3.toml"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--skip-jumps", action="store_true")
    parser.add_argument("--skip-branch-tables", action="store_true")
    parser.add_argument("--skip-threshold-sensitivity", action="store_true")
    parser.add_argument("--skip-summary-text", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def read_table(path: Path, *, separator: str | None = None) -> pd.DataFrame:
    if separator is not None:
        return pd.read_csv(path, sep=separator)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<NA>")


def is_fully_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    nonmissing = series.notna()
    if not nonmissing.any():
        return False
    numeric = pd.to_numeric(series, errors="coerce")
    return bool(numeric[nonmissing].notna().all())


def compare_aligned_frames(
    new: pd.DataFrame,
    old: pd.DataFrame,
    *,
    name: str,
    atol: float,
    rtol: float,
) -> CheckResult:
    max_abs_difference = 0.0
    for column in new.columns:
        if is_fully_numeric(new[column]) and is_fully_numeric(old[column]):
            left = pd.to_numeric(new[column], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(old[column], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True):
                difference = np.abs(left - right)
                local_max = (
                    float(np.nanmax(difference))
                    if np.isfinite(difference).any()
                    else np.nan
                )
                return CheckResult(
                    name,
                    False,
                    f"Numeric mismatch in {column}; max_abs_difference={local_max}",
                )
            difference = np.abs(left - right)
            if np.isfinite(difference).any():
                max_abs_difference = max(
                    max_abs_difference, float(np.nanmax(difference))
                )
        else:
            left = normalize_text(new[column])
            right = normalize_text(old[column])
            mismatch = left != right
            if mismatch.any():
                location = mismatch[mismatch].index[0]
                return CheckResult(
                    name,
                    False,
                    f"Text mismatch in {column} at {location}: "
                    f"new={left.loc[location]!r}, legacy={right.loc[location]!r}",
                )
    return CheckResult(
        name,
        True,
        f"{len(new)} rows; max numeric absolute difference={max_abs_difference:.3g}",
    )


def compare_tables_by_key(
    new_path: Path,
    old_path: Path,
    *,
    keys: Sequence[str],
    name: str,
    atol: float,
    rtol: float,
    separator: str | None = None,
) -> CheckResult:
    for path in (new_path, old_path):
        if not path.exists():
            return CheckResult(name, False, f"Missing file: {path}")
    new = read_table(new_path, separator=separator)
    old = read_table(old_path, separator=separator)
    missing_keys = [key for key in keys if key not in new.columns or key not in old.columns]
    if missing_keys:
        return CheckResult(name, False, f"Missing comparison keys: {missing_keys}")
    if set(new.columns) != set(old.columns):
        return CheckResult(
            name,
            False,
            "Column mismatch: "
            f"new_only={sorted(set(new.columns) - set(old.columns))}, "
            f"legacy_only={sorted(set(old.columns) - set(new.columns))}",
        )
    if new.duplicated(list(keys)).any() or old.duplicated(list(keys)).any():
        return CheckResult(name, False, "Comparison keys are not unique.")
    new_aligned = new.set_index(list(keys)).sort_index()
    old_aligned = old.set_index(list(keys)).sort_index()
    if not new_aligned.index.equals(old_aligned.index):
        return CheckResult(name, False, "Row keys differ between new and legacy tables.")
    return compare_aligned_frames(
        new_aligned,
        old_aligned,
        name=name,
        atol=atol,
        rtol=rtol,
    )


def compare_text_files(new_path: Path, old_path: Path, *, name: str) -> CheckResult:
    for path in (new_path, old_path):
        if not path.exists():
            return CheckResult(name, False, f"Missing file: {path}")
    new_text = new_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    old_text = old_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if new_text != old_text:
        return CheckResult(name, False, "Text content differs.")
    return CheckResult(name, True, f"{len(new_text.splitlines())} lines agree")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    results: list[CheckResult] = []

    if not args.skip_jumps:
        results.append(
            compare_tables_by_key(
                Path(config["jump_candidates"]["output_csv"]).expanduser(),
                Path(config["legacy_jump_candidates"]["output_csv"]).expanduser(),
                keys=("patient_id", "sample_start", "sample_end"),
                name="Jump-candidate table",
                atol=args.atol,
                rtol=args.rtol,
            )
        )

    if not args.skip_branch_tables:
        new = config["branch_tables"]
        old = config["legacy_branch_tables"]
        results.extend(
            [
                compare_tables_by_key(
                    Path(new["output_transition_csv"]).expanduser(),
                    Path(old["output_transition_csv"]).expanduser(),
                    keys=("patient_id", "sample_start", "sample_end"),
                    name="Branch transition table",
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                compare_tables_by_key(
                    Path(new["output_ecology_csv"]).expanduser(),
                    Path(old["output_ecology_csv"]).expanduser(),
                    keys=("branch_id_dominant", "ecotype_label"),
                    name="Branch ecology summary",
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                compare_tables_by_key(
                    Path(new["output_program_csv"]).expanduser(),
                    Path(old["output_program_csv"]).expanduser(),
                    keys=("branch_id_dominant",),
                    name="Branch scaffold-program summary",
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                compare_tables_by_key(
                    Path(new["output_risk_csv"]).expanduser(),
                    Path(old["output_risk_csv"]).expanduser(),
                    keys=("branch_id_start",),
                    name="Branch escape-risk summary",
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                compare_tables_by_key(
                    Path(new["output_stats_tsv"]).expanduser(),
                    Path(old["output_stats_tsv"]).expanduser(),
                    keys=("section", "item"),
                    name="Figure 5 statistics summary",
                    atol=args.atol,
                    rtol=args.rtol,
                    separator="\t",
                ),
            ]
        )

    if not args.skip_threshold_sensitivity:
        new = config["threshold_sensitivity"]
        old = config["legacy_threshold_sensitivity"]
        results.extend(
            [
                compare_tables_by_key(
                    Path(new["output_counts_csv"]).expanduser(),
                    Path(old["output_counts_csv"]).expanduser(),
                    keys=("threshold", "timepoint"),
                    name="Threshold branch-count table",
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                compare_tables_by_key(
                    Path(new["output_transition_csv"]).expanduser(),
                    Path(old["output_transition_csv"]).expanduser(),
                    keys=("threshold", "dx_branch", "rel_branch"),
                    name="Threshold transition table",
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                compare_tables_by_key(
                    Path(new["output_patient_csv"]).expanduser(),
                    Path(old["output_patient_csv"]).expanduser(),
                    keys=("threshold", "sample"),
                    name="Threshold patient branch table",
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                compare_tables_by_key(
                    Path(new["output_switch_csv"]).expanduser(),
                    Path(old["output_switch_csv"]).expanduser(),
                    keys=("threshold",),
                    name="Threshold switch-rate table",
                    atol=args.atol,
                    rtol=args.rtol,
                ),
            ]
        )
        if not args.skip_summary_text:
            results.append(
                compare_text_files(
                    Path(new["output_summary_txt"]).expanduser(),
                    Path(old["output_summary_txt"]).expanduser(),
                    name="Threshold sensitivity text summary",
                )
            )

    print("Stage 3 regression validation")
    print("=" * 72)
    for result in results:
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.detail}")
    failures = [result for result in results if not result.passed]
    if failures:
        raise SystemExit(1)
    print("\nAll requested Stage 3 regression checks passed.")


if __name__ == "__main__":
    main()
