from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pediatric_leukemia_stage1.toml"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument(
        "--skip-h5ad",
        action="store_true",
        help="Compare only CSV outputs.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<NA>")


def compare_tables(
    new_path: Path,
    legacy_path: Path,
    *,
    keys: Sequence[str],
    name: str,
    atol: float,
    rtol: float,
) -> CheckResult:
    for path in (new_path, legacy_path):
        if not path.exists():
            return CheckResult(name, False, f"Missing file: {path}")

    new = pd.read_csv(new_path)
    old = pd.read_csv(legacy_path)
    missing_keys = [key for key in keys if key not in new.columns or key not in old.columns]
    if missing_keys:
        return CheckResult(name, False, f"Missing comparison keys: {missing_keys}")
    if new.duplicated(list(keys)).any() or old.duplicated(list(keys)).any():
        return CheckResult(name, False, "Comparison keys are not unique.")

    new_columns = set(new.columns)
    old_columns = set(old.columns)
    if new_columns != old_columns:
        return CheckResult(
            name,
            False,
            "Column mismatch: "
            f"new_only={sorted(new_columns - old_columns)}, "
            f"legacy_only={sorted(old_columns - new_columns)}",
        )

    new_aligned = new.set_index(list(keys)).sort_index()
    old_aligned = old.set_index(list(keys)).sort_index()
    if not new_aligned.index.equals(old_aligned.index):
        new_only = new_aligned.index.difference(old_aligned.index).tolist()[:10]
        old_only = old_aligned.index.difference(new_aligned.index).tolist()[:10]
        return CheckResult(
            name,
            False,
            f"Row-key mismatch: new_only={new_only}, legacy_only={old_only}",
        )

    max_abs_difference = 0.0
    for column in new_aligned.columns:
        new_numeric = pd.to_numeric(new_aligned[column], errors="coerce")
        old_numeric = pd.to_numeric(old_aligned[column], errors="coerce")
        numeric_mask_match = new_numeric.notna().equals(old_numeric.notna())
        treat_as_numeric = (
            numeric_mask_match
            and (new_numeric.notna().any() or old_numeric.notna().any())
            and not (
                pd.api.types.is_bool_dtype(new_aligned[column])
                or pd.api.types.is_bool_dtype(old_aligned[column])
            )
        )
        if treat_as_numeric:
            left = new_numeric.to_numpy(dtype=float)
            right = old_numeric.to_numpy(dtype=float)
            if not np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True):
                difference = np.abs(left - right)
                local_max = float(np.nanmax(difference)) if np.isfinite(difference).any() else np.nan
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
            left_text = _normalize_text(new_aligned[column])
            right_text = _normalize_text(old_aligned[column])
            mismatch = left_text != right_text
            if mismatch.any():
                first_key = mismatch[mismatch].index[0]
                return CheckResult(
                    name,
                    False,
                    f"Text mismatch in {column} at key {first_key}: "
                    f"new={left_text.loc[first_key]!r}, legacy={right_text.loc[first_key]!r}",
                )

    return CheckResult(
        name,
        True,
        f"{len(new_aligned)} rows; max numeric absolute difference={max_abs_difference:.3g}",
    )


def compare_h5ad(
    new_path: Path,
    legacy_path: Path,
    *,
    atol: float,
    rtol: float,
) -> CheckResult:
    if not new_path.exists() or not legacy_path.exists():
        missing = [str(path) for path in (new_path, legacy_path) if not path.exists()]
        return CheckResult("Projected H5AD", False, f"Missing file(s): {missing}")
    try:
        import anndata as ad
    except ImportError:
        return CheckResult(
            "Projected H5AD",
            False,
            "anndata is not installed; use --skip-h5ad or install application dependencies.",
        )

    new = ad.read_h5ad(new_path)
    old = ad.read_h5ad(legacy_path)
    if new.n_obs != old.n_obs or new.n_vars != old.n_vars:
        return CheckResult(
            "Projected H5AD",
            False,
            f"Shape mismatch: new={new.shape}, legacy={old.shape}",
        )
    if not new.obs_names.equals(old.obs_names):
        return CheckResult("Projected H5AD", False, "Observation names differ.")
    if not new.var_names.equals(old.var_names):
        return CheckResult("Projected H5AD", False, "Variable names differ.")

    obs_columns = [
        "PC1",
        "PC2",
        "ecotype_label",
        "state_HSC",
        "state_Prog",
        "state_GMP",
        "state_MonoDC",
        "aux_EryBaso",
        "aux_CLP",
        "branch_id",
        "branch_maxprob",
        "branch_entropy",
        "sample_has_projected_scores",
    ]
    for column in obs_columns:
        if column not in new.obs.columns or column not in old.obs.columns:
            return CheckResult(
                "Projected H5AD", False, f"Missing obs comparison column: {column}"
            )
        if pd.api.types.is_numeric_dtype(new.obs[column]) and pd.api.types.is_numeric_dtype(
            old.obs[column]
        ):
            if not np.allclose(
                pd.to_numeric(new.obs[column], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(old.obs[column], errors="coerce").to_numpy(dtype=float),
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            ):
                return CheckResult(
                    "Projected H5AD", False, f"Numeric obs mismatch: {column}"
                )
        else:
            if not _normalize_text(new.obs[column]).equals(
                _normalize_text(old.obs[column])
            ):
                return CheckResult(
                    "Projected H5AD", False, f"Text obs mismatch: {column}"
                )

    for key in ("X_fig2", "X_scaffold"):
        if key not in new.obsm or key not in old.obsm:
            return CheckResult("Projected H5AD", False, f"Missing obsm key: {key}")
        if not np.allclose(
            np.asarray(new.obsm[key], dtype=float),
            np.asarray(old.obsm[key], dtype=float),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        ):
            return CheckResult("Projected H5AD", False, f"obsm mismatch: {key}")

    return CheckResult(
        "Projected H5AD",
        True,
        f"shape={new.shape}; selected obs and obsm outputs agree",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    projection = config["projection"]
    intervals = config["intervals"]
    legacy = config["legacy"]

    results = [
        compare_tables(
            Path(projection["output_sample_scores_csv"]).expanduser(),
            Path(legacy["sample_scores_csv"]).expanduser(),
            keys=("sample_id",),
            name="Sample projection table",
            atol=args.atol,
            rtol=args.rtol,
        ),
        compare_tables(
            Path(intervals["output_intervals_all_csv"]).expanduser(),
            Path(legacy["intervals_all_csv"]).expanduser(),
            keys=("patient_id", "interval_class", "sample_start", "sample_end"),
            name="All interval table",
            atol=args.atol,
            rtol=args.rtol,
        ),
        compare_tables(
            Path(intervals["output_intervals_main_csv"]).expanduser(),
            Path(legacy["intervals_main_csv"]).expanduser(),
            keys=("patient_id", "interval_class", "sample_start", "sample_end"),
            name="Main interval table",
            atol=args.atol,
            rtol=args.rtol,
        ),
    ]
    if not args.skip_h5ad:
        results.append(
            compare_h5ad(
                Path(projection["output_projected_h5ad"]).expanduser(),
                Path(legacy["projected_h5ad"]).expanduser(),
                atol=args.atol,
                rtol=args.rtol,
            )
        )

    print("Stage 1 regression validation")
    print("=" * 72)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    failures = [result for result in results if not result.passed]
    if failures:
        raise SystemExit(1)
    print("\nAll requested Stage 1 regression checks passed.")


if __name__ == "__main__":
    main()
