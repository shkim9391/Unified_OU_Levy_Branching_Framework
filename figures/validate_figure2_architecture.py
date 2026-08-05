from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "/Unified_OU_Levy_Branching_Framework"
)

SCENARIOS = [
    "brownian",
    "ou",
    "shifted_ou",
    "ou_jump",
    "ou_branching",
    "full_oulb",
]

FORBIDDEN_PLOTTING_CALLS = {
    "simulate_process",
    "observe_latent",
    "fit_ou_mle",
    "fit_brownian_mle",
    "fit_shifted_ou_mle",
    "minimize",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    return parser.parse_args()


def called_functions(path: Path) -> set[str]:
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )
    calls: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)

    return calls


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()

    datadir = root / "figures" / "data"
    trajectory_path = datadir / "Figure2_representative_trajectories.csv"
    event_path = datadir / "Figure2_event_ledger.csv"
    metadata_path = datadir / "Figure2_simulation_metadata.json"
    plotting_path = root / "figures" / "plot_figure2_simulation_trajectories.py"

    for path in [
        trajectory_path,
        event_path,
        metadata_path,
        plotting_path,
    ]:
        if not path.exists():
            raise SystemExit(f"[FAIL] Missing file: {path}")

    trajectories = pd.read_csv(trajectory_path)
    events = pd.read_csv(event_path)
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    observed_scenarios = set(
        trajectories["scenario"].astype(str)
    )

    if observed_scenarios != set(SCENARIOS):
        raise SystemExit(
            "[FAIL] Scenario mismatch: "
            f"{sorted(observed_scenarios)}"
        )

    counts = trajectories.groupby("scenario").size()

    if counts.nunique() != 1:
        raise SystemExit(
            "[FAIL] Scenarios do not share a common latent grid."
        )

    numeric_columns = [
        "time",
        "latent_state",
        "branch_state",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(
            trajectories[column],
            errors="coerce",
        )
        if not np.isfinite(values).all():
            raise SystemExit(
                f"[FAIL] Nonfinite values in {column}."
            )

    if trajectories["is_observed"].astype(bool).sum() == 0:
        raise SystemExit(
            "[FAIL] No noisy observations were archived."
        )

    if not (
        trajectories["scenario"].isin(
            ["shifted_ou", "full_oulb"]
        )
        & trajectories["treatment_time"].notna()
    ).any():
        raise SystemExit(
            "[FAIL] Treatment timing was not archived."
        )

    if events.empty:
        raise SystemExit(
            "[FAIL] Event ledger is empty."
        )

    plotting_calls = called_functions(plotting_path)
    forbidden_found = sorted(
        plotting_calls & FORBIDDEN_PLOTTING_CALLS
    )

    if forbidden_found:
        raise SystemExit(
            "[FAIL] Plotting layer contains computation calls: "
            + ", ".join(forbidden_found)
        )

    if int(metadata["trajectory_rows"]) != len(trajectories):
        raise SystemExit(
            "[FAIL] Metadata trajectory count does not agree."
        )

    print("Figure 2 archive validation")
    print("=" * 72)
    print(
        f"[PASS] Six scenarios archived: "
        f"{', '.join(SCENARIOS)}"
    )
    print(
        f"[PASS] Common latent grid: "
        f"{int(counts.iloc[0])} rows per scenario"
    )
    print(
        f"[PASS] Noisy observations archived: "
        f"{int(trajectories['is_observed'].astype(bool).sum())}"
    )
    print(
        f"[PASS] Event ledger rows: {len(events)}"
    )
    print(
        "[PASS] Treatment timing and branch-specific attractors archived"
    )
    print(
        "[PASS] Plotting script contains no simulation or fitting calls"
    )
    print()
    print(
        "Figure 2 computation/graphics separation validated."
    )


if __name__ == "__main__":
    main()
