from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PROJECT_ROOT = Path(
    "/Unified_OU_Levy_Branching_Framework"
)

PANEL_SPEC = [
    {"panel": "A", "scenario": "brownian", "parameter": "drift",
     "panel_title": "Brownian drift recovery", "symbol": r"$\beta$"},
    {"panel": "B", "scenario": "brownian", "parameter": "sigma",
     "panel_title": "Brownian diffusion recovery", "symbol": r"$\sigma$"},
    {"panel": "C", "scenario": "ou", "parameter": "theta",
     "panel_title": r"Standard OU $\theta$ recovery", "symbol": r"$\theta$"},
    {"panel": "D", "scenario": "ou", "parameter": "mu",
     "panel_title": r"Standard OU $\mu$ recovery", "symbol": r"$\mu$"},
    {"panel": "E", "scenario": "ou", "parameter": "sigma",
     "panel_title": r"Standard OU $\sigma$ recovery", "symbol": r"$\sigma$"},
    {"panel": "F", "scenario": "shifted_ou", "parameter": "delta",
     "panel_title": r"Treatment-shift $\Delta$ recovery", "symbol": r"$\Delta$"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find {label}. Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    scenario_col = resolve_column(raw, ["scenario", "model"], "scenario")
    parameter_col = resolve_column(raw, ["parameter", "parameter_name"], "parameter")
    replicate_col = resolve_column(raw, ["replicate", "replicate_id"], "replicate")
    truth_col = resolve_column(raw, ["truth", "true_value"], "truth")
    estimate_col = resolve_column(raw, ["estimate", "estimated_value"], "estimate")

    out = pd.DataFrame({
        "scenario": raw[scenario_col].astype(str).str.strip(),
        "parameter": raw[parameter_col].astype(str).str.strip(),
        "replicate": pd.to_numeric(raw[replicate_col], errors="raise").astype(int),
        "truth": pd.to_numeric(raw[truth_col], errors="coerce"),
        "estimate": pd.to_numeric(raw[estimate_col], errors="coerce"),
    }).dropna(subset=["truth", "estimate"])

    out["error"] = out["estimate"] - out["truth"]
    out["absolute_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    return out


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser()
    stage5 = root / "application/pediatric_leukemia/outputs/stage5"
    data_dir = root / "figures/data"
    data_dir.mkdir(parents=True, exist_ok=True)

    recovery_path = stage5 / "parameter_recovery.csv"
    legacy_summary_path = stage5 / "recovery_summary.csv"

    points_path = data_dir / "Figure3_parameter_recovery_points.csv"
    summary_path = data_dir / "Figure3_parameter_recovery_summary.csv"
    metadata_path = data_dir / "Figure3_parameter_recovery_metadata.json"

    if not args.overwrite and any(
        path.exists() for path in [points_path, summary_path, metadata_path]
    ):
        raise FileExistsError(
            "Figure 3 archived outputs already exist; use --overwrite."
        )

    for path in [recovery_path, legacy_summary_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    raw = pd.read_csv(recovery_path)
    legacy_summary = pd.read_csv(legacy_summary_path)

    points = normalize(raw).merge(
        pd.DataFrame(PANEL_SPEC),
        on=["scenario", "parameter"],
        how="inner",
        validate="many_to_one",
    )

    expected = {(x["scenario"], x["parameter"]) for x in PANEL_SPEC}
    observed = set(zip(points["scenario"], points["parameter"]))
    if expected - observed:
        raise ValueError(
            f"Missing required scenario/parameter pairs: {sorted(expected - observed)}"
        )

    points = points[
        [
            "panel", "scenario", "parameter", "panel_title", "symbol",
            "replicate", "truth", "estimate", "error",
            "absolute_error", "squared_error",
        ]
    ].sort_values(["panel", "replicate"]).reset_index(drop=True)

    summary = (
        points.groupby(
            ["panel", "scenario", "parameter", "panel_title", "symbol"],
            dropna=False,
        )
        .agg(
            n=("estimate", "size"),
            truth=("truth", "mean"),
            estimate_mean=("estimate", "mean"),
            estimate_median=("estimate", "median"),
            estimate_sd=("estimate", "std"),
            bias=("error", "mean"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            estimate_min=("estimate", "min"),
            estimate_max=("estimate", "max"),
        )
        .reset_index()
    )

    summary["boundary_hit_fraction"] = np.nan
    summary["boundary_annotation"] = (
        "Boundary flags not archived in baseline recovery tables"
    )

    points.to_csv(points_path, index=False)
    summary.to_csv(summary_path, index=False)

    metadata = {
        "figure": "Figure 3",
        "title": "Recovery of continuous dynamic parameters",
        "source_files": [str(recovery_path), str(legacy_summary_path)],
        "source_rows": {
            "parameter_recovery": int(len(raw)),
            "recovery_summary": int(len(legacy_summary)),
        },
        "archived_rows": {
            "points": int(len(points)),
            "summary": int(len(summary)),
        },
        "panel_specification": PANEL_SPEC,
        "boundary_note": (
            "Baseline Stage 5 recovery tables do not archive optimizer "
            "boundary-hit flags. Boundary-hit rates should be shown in a "
            "stress-test figure unless those flags are archived upstream."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"[SAVED] {points_path}")
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {metadata_path}")
    print()
    print(summary[
        ["panel", "scenario", "parameter", "n", "truth", "bias", "rmse"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
