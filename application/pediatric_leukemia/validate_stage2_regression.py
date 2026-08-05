from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pediatric_leukemia_stage2.toml"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument(
        "--skip-model-comparison",
        action="store_true",
        help="Validate only sample dynamic parameters and the DX attractor.",
    )
    parser.add_argument(
        "--skip-summary-text",
        action="store_true",
        help="Do not compare the plain-text fit summary.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<NA>")


def _read_table(path: Path, *, separator: str | None = None) -> pd.DataFrame:
    if separator is not None:
        return pd.read_csv(path, sep=separator)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def compare_tables_by_key(
    new_path: Path,
    legacy_path: Path,
    *,
    keys: Sequence[str],
    name: str,
    atol: float,
    rtol: float,
    separator: str | None = None,
    ignore_columns: Sequence[str] = (),
) -> CheckResult:
    for path in (new_path, legacy_path):
        if not path.exists():
            return CheckResult(name, False, f"Missing file: {path}")

    new = _read_table(new_path, separator=separator)
    old = _read_table(legacy_path, separator=separator)
    ignored = {str(column) for column in ignore_columns}
    new = new.drop(columns=[column for column in ignored if column in new.columns])
    old = old.drop(columns=[column for column in ignored if column in old.columns])
    missing_keys = [key for key in keys if key not in new.columns or key not in old.columns]
    if missing_keys:
        return CheckResult(name, False, f"Missing comparison keys: {missing_keys}")
    if new.duplicated(list(keys)).any() or old.duplicated(list(keys)).any():
        return CheckResult(name, False, "Comparison keys are not unique.")

    if set(new.columns) != set(old.columns):
        return CheckResult(
            name,
            False,
            "Column mismatch: "
            f"new_only={sorted(set(new.columns) - set(old.columns))}, "
            f"legacy_only={sorted(set(old.columns) - set(new.columns))}",
        )

    new_aligned = new.set_index(list(keys)).sort_index()
    old_aligned = old.set_index(list(keys)).sort_index()
    if not new_aligned.index.equals(old_aligned.index):
        return CheckResult(name, False, "Row keys differ between new and legacy tables.")
    return _compare_aligned_frames(
        new_aligned,
        old_aligned,
        name=name,
        atol=atol,
        rtol=rtol,
    )


def compare_tables_by_order(
    new_path: Path,
    legacy_path: Path,
    *,
    name: str,
    atol: float,
    rtol: float,
) -> CheckResult:
    for path in (new_path, legacy_path):
        if not path.exists():
            return CheckResult(name, False, f"Missing file: {path}")
    new = pd.read_csv(new_path)
    old = pd.read_csv(legacy_path)
    if list(new.columns) != list(old.columns):
        return CheckResult(
            name,
            False,
            f"Ordered column mismatch: new={list(new.columns)}, legacy={list(old.columns)}",
        )
    if len(new) != len(old):
        return CheckResult(
            name, False, f"Row-count mismatch: new={len(new)}, legacy={len(old)}"
        )
    return _compare_aligned_frames(
        new.reset_index(drop=True),
        old.reset_index(drop=True),
        name=name,
        atol=atol,
        rtol=rtol,
    )


def _compare_aligned_frames(
    new: pd.DataFrame,
    old: pd.DataFrame,
    *,
    name: str,
    atol: float,
    rtol: float,
) -> CheckResult:
    max_abs_difference = 0.0
    for column in new.columns:
        new_numeric = pd.to_numeric(new[column], errors="coerce")
        old_numeric = pd.to_numeric(old[column], errors="coerce")
        numeric_mask_match = new_numeric.notna().equals(old_numeric.notna())
        treat_as_numeric = (
            numeric_mask_match
            and (new_numeric.notna().any() or old_numeric.notna().any())
            and not (
                pd.api.types.is_bool_dtype(new[column])
                or pd.api.types.is_bool_dtype(old[column])
            )
        )
        if treat_as_numeric:
            left = new_numeric.to_numpy(dtype=float)
            right = old_numeric.to_numpy(dtype=float)
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
            left_text = _normalize_text(new[column])
            right_text = _normalize_text(old[column])
            mismatch = left_text != right_text
            if mismatch.any():
                location = mismatch[mismatch].index[0]
                return CheckResult(
                    name,
                    False,
                    f"Text mismatch in {column} at {location}: "
                    f"new={left_text.loc[location]!r}, legacy={right_text.loc[location]!r}",
                )
    return CheckResult(
        name,
        True,
        f"{len(new)} rows; max numeric absolute difference={max_abs_difference:.3g}",
    )


def compare_text_files(new_path: Path, legacy_path: Path, *, name: str) -> CheckResult:
    for path in (new_path, legacy_path):
        if not path.exists():
            return CheckResult(name, False, f"Missing file: {path}")
    new_text = new_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    old_text = legacy_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if new_text != old_text:
        return CheckResult(name, False, "Text content differs.")
    return CheckResult(name, True, f"{len(new_text.splitlines())} lines agree")


def model_output_paths(prefix: Path) -> dict[str, Path]:
    return {
        "comparison": prefix.parent / f"{prefix.name}_model_comparison.csv",
        "parameters": prefix.parent / f"{prefix.name}_parameter_summary.csv",
        "tail": prefix.parent / f"{prefix.name}_tail_fit_grid.csv",
        "casewise": prefix.parent / f"{prefix.name}_casewise_loglik_gain.csv",
        "summary": prefix.parent / f"{prefix.name}_fit_summary.txt",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dynamic = config["dynamic_parameters"]
    legacy_dynamic = config["legacy_dynamic_parameters"]

    results = [
        compare_tables_by_key(
            Path(dynamic["output_csv"]).expanduser(),
            Path(legacy_dynamic["output_csv"]).expanduser(),
            keys=("sample_id",),
            name="Sample dynamic parameter table",
            atol=args.atol,
            rtol=args.rtol,
        ),
        compare_tables_by_key(
            Path(dynamic["output_attractor_tsv"]).expanduser(),
            Path(legacy_dynamic["output_attractor_tsv"]).expanduser(),
            keys=("feature",),
            name="DX attractor table",
            atol=args.atol,
            rtol=args.rtol,
            separator="\t",
        ),
    ]

    if not args.skip_model_comparison:
        new_prefix = Path(config["model_comparison"]["output_prefix"]).expanduser()
        old_prefix = Path(config["legacy_model_comparison"]["output_prefix"]).expanduser()
        new_paths = model_output_paths(new_prefix)
        old_paths = model_output_paths(old_prefix)
        results.extend(
            [
                compare_tables_by_key(
                    new_paths["comparison"],
                    old_paths["comparison"],
                    keys=("model_id",),
                    name="Model comparison table",
                    atol=args.atol,
                    rtol=args.rtol,
                    ignore_columns=("message",),
                ),
                compare_tables_by_key(
                    new_paths["parameters"],
                    old_paths["parameters"],
                    keys=("model_id", "parameter"),
                    name="Model parameter table",
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                compare_tables_by_order(
                    new_paths["tail"],
                    old_paths["tail"],
                    name="Tail-fit grid",
                    atol=args.atol,
                    rtol=args.rtol,
                ),
                compare_tables_by_key(
                    new_paths["casewise"],
                    old_paths["casewise"],
                    keys=("sample_std",),
                    name="Case-wise log-likelihood table",
                    atol=args.atol,
                    rtol=args.rtol,
                ),
            ]
        )
        if not args.skip_summary_text:
            results.append(
                compare_text_files(
                    new_paths["summary"],
                    old_paths["summary"],
                    name="Model fit summary text",
                )
            )

    print("Stage 2 regression validation")
    print("=" * 72)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    failures = [result for result in results if not result.passed]
    if failures:
        raise SystemExit(1)
    print("\nAll requested Stage 2 regression checks passed.")


if __name__ == "__main__":
    main()
