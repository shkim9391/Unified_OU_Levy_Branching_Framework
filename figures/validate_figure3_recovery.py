from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(
    "/Unified_OU_Levy_Branching_Framework"
)


def main() -> None:
    data_dir = ROOT / "figures/data"
    points_path = data_dir / "Figure3_parameter_recovery_points.csv"
    summary_path = data_dir / "Figure3_parameter_recovery_summary.csv"
    plot_path = ROOT / "figures/plot_figure3_parameter_recovery.py"

    for path in [points_path, summary_path, plot_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    points = pd.read_csv(points_path)
    summary = pd.read_csv(summary_path)

    expected_panels = {"A", "B", "C", "D", "E", "F"}
    if set(points["panel"]) != expected_panels:
        raise AssertionError(
            f"Unexpected point-table panels: {sorted(set(points['panel']))}"
        )
    if set(summary["panel"]) != expected_panels:
        raise AssertionError(
            f"Unexpected summary-table panels: {sorted(set(summary['panel']))}"
        )

    if not np.isfinite(points["truth"]).all():
        raise AssertionError("Nonfinite truth values found.")
    if not np.isfinite(points["estimate"]).all():
        raise AssertionError("Nonfinite estimate values found.")

    tree = ast.parse(plot_path.read_text(encoding="utf-8"))
    forbidden = {
        "simulate_process",
        "fit_brownian_mle",
        "fit_ou_mle",
        "fit_shifted_ou_mle",
        "minimize",
        "curve_fit",
        "bootstrap",
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }
    found = forbidden & called
    if found:
        raise AssertionError(
            f"Forbidden computation calls in plotting script: {sorted(found)}"
        )

    print("Figure 3 recovery validation")
    print("=" * 72)
    print("[PASS] Six recovery panels archived")
    print(f"[PASS] Recovery point rows: {len(points)}")
    print(f"[PASS] Recovery summary rows: {len(summary)}")
    print("[PASS] Truth and estimate values are finite")
    print("[PASS] Plotting script contains no simulation, fitting, or optimization")
    print()
    print("Figure 3 computation/graphics separation validated.")


if __name__ == "__main__":
    main()
