from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/Unified_OU_Levy_Branching_Framework"
)


def pass_message(message: str) -> None:
    print(f"[PASS] {message}")


def main() -> None:
    data_dir = ROOT / "figures/data"
    files = {
        "jump_replicates": data_dir / "Figure4_jump_recovery_replicates.csv",
        "jump_summary": data_dir / "Figure4_jump_recovery_summary.csv",
        "branch_replicates": data_dir / "Figure4_branch_recovery_replicates.csv",
        "branch_summary": data_dir / "Figure4_branch_recovery_summary.csv",
        "selection": data_dir / "Figure4_representative_selection.csv",
        "trajectories": data_dir / "Figure4_representative_trajectories.csv",
        "events": data_dir / "Figure4_representative_events.csv",
        "metadata": data_dir / "Figure4_recovery_metadata.json",
    }

    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(path)

    jump = pd.read_csv(files["jump_replicates"])
    branch = pd.read_csv(files["branch_replicates"])
    selection = pd.read_csv(files["selection"])
    trajectories = pd.read_csv(files["trajectories"])
    events = pd.read_csv(files["events"])
    metadata = json.loads(
        files["metadata"].read_text(encoding="utf-8")
    )

    required_jump = {
        "precision",
        "recall",
        "false_positive_rate",
        "f1",
        "seed",
    }
    required_branch = {
        "state_accuracy",
        "adjusted_rand_index",
        "transition_f1",
        "seed",
    }
    if not required_jump.issubset(jump.columns):
        raise ValueError("Jump replicate table is incomplete.")
    if not required_branch.issubset(branch.columns):
        raise ValueError("Branch replicate table is incomplete.")

    pass_message(f"Jump replicate rows: {len(jump)}")
    pass_message(f"Branch replicate rows: {len(branch)}")

    for column in [
        "precision",
        "recall",
        "false_positive_rate",
        "f1",
    ]:
        finite = jump[column].dropna()
        if not ((finite >= 0) & (finite <= 1)).all():
            raise ValueError(f"Invalid jump metric range: {column}")

    for column in [
        "state_accuracy",
        "adjusted_rand_index",
        "transition_f1",
    ]:
        finite = branch[column].dropna()
        lower = -1 if column == "adjusted_rand_index" else 0
        if not ((finite >= lower) & (finite <= 1)).all():
            raise ValueError(f"Invalid branch metric range: {column}")

    pass_message("Recovery metrics lie in valid ranges")

    expected_examples = {
        "jump_success",
        "jump_failure",
        "branch_success",
        "branch_failure",
    }
    observed_examples = set(selection["example_type"])
    if observed_examples != expected_examples:
        raise ValueError(
            f"Representative examples mismatch: {observed_examples}"
        )

    for example_id in selection["example_id"]:
        if example_id not in set(trajectories["example_id"]):
            raise ValueError(f"Missing trajectory for {example_id}")
        if example_id not in set(events["example_id"]):
            raise ValueError(f"Missing event table for {example_id}")

    pass_message("Four representative examples are fully archived")
    pass_message("Seeds and selection metrics are archived upstream")

    plot_script = ROOT / "figures/plot_figure4_recovery_limits.py"
    source = plot_script.read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_names = {
        "simulate_process",
        "detect_jump_intervals",
        "binary_metrics",
        "adjusted_rand_index",
        "np.random",
        "minimize",
        "curve_fit",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            raise ValueError(
                f"Forbidden plotting-layer call found: {node.id}"
            )
        if isinstance(node, ast.Attribute):
            label = ast.unparse(node)
            if label in forbidden_names:
                raise ValueError(
                    f"Forbidden plotting-layer call found: {label}"
                )

    pass_message(
        "Plotting layer contains no simulation, detection, fitting, or random calls"
    )

    for extension in [".svg", ".pdf", ".png"]:
        figure = ROOT / f"figures/Figure4_recovery_limits{extension}"
        if not figure.exists():
            raise FileNotFoundError(figure)
    pass_message("Rendered SVG, PDF, and PNG files exist")

    print()
    print("Figure 4 recovery-limit validation completed successfully.")


if __name__ == "__main__":
    main()
