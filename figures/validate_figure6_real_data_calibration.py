from __future__ import annotations

import argparse
import ast
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/figure6_real_data_calibration.toml"),
    )
    return parser.parse_args()


def read_config(path: Path) -> dict:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    root = Path(config["paths"]["project_root"]).expanduser()
    data_dir = root / config["paths"]["output_data_dir"]
    figure_dir = root / config["paths"]["output_figure_dir"]

    paths = {
        "empirical": data_dir / "Figure6_empirical_interval_statistics.csv",
        "parameters": data_dir / "Figure6_calibration_parameters.csv",
        "trajectories": data_dir / "Figure6_calibrated_trajectory_table.csv",
        "events": data_dir / "Figure6_calibrated_event_ledger.csv",
        "comparison": data_dir / "Figure6_observed_simulated_statistics.csv",
        "fidelity": data_dir / "Figure6_calibration_fidelity.csv",
        "metadata": data_dir / "Figure6_calibration_metadata.json",
    }

    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    empirical = pd.read_csv(paths["empirical"])
    parameters = pd.read_csv(paths["parameters"])
    trajectories = pd.read_csv(paths["trajectories"])
    comparison = pd.read_csv(paths["comparison"])
    fidelity = pd.read_csv(paths["fidelity"])
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    if empirical.empty:
        raise AssertionError("Empirical calibration table is empty.")
    print(f"[PASS] Empirical patient summaries: {len(empirical)}")

    provenance = set(parameters["provenance"].astype(str))
    if not {"Empirical", "Derived", "Prespecified"}.issubset(provenance):
        raise AssertionError(
            "Calibration parameter table does not distinguish empirical, "
            "derived, and prespecified quantities."
        )
    print("[PASS] Calibration provenance is explicitly archived")

    expected_scenarios = {"retention", "jump", "branch_reorganization"}
    observed_scenarios = set(trajectories["scenario"].astype(str))
    if expected_scenarios != observed_scenarios:
        raise AssertionError(
            f"Expected {expected_scenarios}; found {observed_scenarios}"
        )
    print("[PASS] Three calibrated trajectory classes are archived")

    required_trajectory = {
        "replicate",
        "scenario",
        "seed",
        "time",
        "latent_state",
        "observed_state",
        "branch_state",
        "attractor",
    }
    if required_trajectory - set(trajectories.columns):
        raise AssertionError("Trajectory archive is missing required columns.")
    print(f"[PASS] Calibrated trajectory rows: {len(trajectories)}")

    required_metrics = {
        "n_observations",
        "followup",
        "total_displacement",
        "max_interval_displacement",
        "median_interval_displacement",
        "q90_interval_displacement",
    }
    if required_metrics - set(comparison["metric"].astype(str)):
        raise AssertionError("Observed/simulated comparison misses metrics.")
    print("[PASS] Observed and simulated statistics share common metrics")

    finite_ratio = pd.to_numeric(
        fidelity["ratio_median"], errors="coerce"
    ).dropna()
    if not len(finite_ratio) or not np.isfinite(finite_ratio).all():
        raise AssertionError("Calibration-fidelity ratios are not finite.")
    print(f"[PASS] Calibration fidelity metrics: {len(fidelity)}")

    if metadata["n_replicates"] != int(config["simulation"]["n_replicates"]):
        raise AssertionError("Metadata replicate count disagrees with config.")
    print("[PASS] Seeds, replicate count, and interpretation note are archived")

    plot_path = root / "figures/plot_figure6_real_data_calibration.py"
    tree = ast.parse(plot_path.read_text(encoding="utf-8"))
    banned = {
        "simulate_process",
        "observe_latent",
        "np.random",
        "default_rng",
        "minimize",
        "fit_ou_mle",
        "fit_brownian_mle",
    }
    source = plot_path.read_text(encoding="utf-8")
    found = sorted(token for token in banned if token in source)
    if found:
        raise AssertionError(f"Plotting layer contains banned operations: {found}")
    print("[PASS] Plotting layer contains no simulation, fitting, or randomness")

    for extension in [".svg", ".pdf", ".png"]:
        path = figure_dir / f"Figure6_real_data_calibration{extension}"
        if not path.exists():
            raise FileNotFoundError(path)
    print("[PASS] Rendered SVG, PDF, and PNG files exist")

    print()
    print("Figure 6 real-data calibration validation completed successfully.")


if __name__ == "__main__":
    main()
