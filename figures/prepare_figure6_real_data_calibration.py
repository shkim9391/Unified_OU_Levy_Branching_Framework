from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oulb.observation import ObservationSpec, observe_latent, regular_schedule
from oulb.real_data_calibration import (
    empirical_interval_count,
    empirical_schedule,
    first_existing_column,
    ratio_fidelity,
    summarize_empirical_intervals,
    transition_statistics,
)
from oulb.simulation import BranchSpec, JumpSpec, simulate_process


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/figure6_real_data_calibration.toml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def infer_branch_switches(
    jump_table: pd.DataFrame,
    interval_table: pd.DataFrame,
) -> np.ndarray | None:
    for table in [jump_table, interval_table]:
        column = first_existing_column(
            table,
            [
                "branch_switch",
                "dx_to_rel_switch",
                "switch",
                "branch_changed",
            ],
            required=False,
        )
        if column is not None:
            values = pd.to_numeric(table[column], errors="coerce").dropna()
            if len(values):
                return (values.to_numpy(float) != 0).astype(float)
    return None


def make_dense_schedule(observed_times: np.ndarray, points_per_interval: int) -> np.ndarray:
    dense_parts = []
    for index in range(len(observed_times) - 1):
        part = np.linspace(
            observed_times[index],
            observed_times[index + 1],
            points_per_interval + 1,
        )
        if index:
            part = part[1:]
        dense_parts.append(part)
    return np.concatenate(dense_parts)


def map_observations(
    dense_times: np.ndarray,
    observed_times: np.ndarray,
    observed_states: np.ndarray,
) -> np.ndarray:
    mapped = np.full(len(dense_times), np.nan, dtype=float)
    for time, value in zip(observed_times, observed_states):
        index = int(np.argmin(np.abs(dense_times - time)))
        mapped[index] = float(value)
    return mapped


def event_rows(
    replicate: int,
    scenario: str,
    result,
    times: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for event in result.event_ledger:
        interval = int(event.get("interval", -1))
        event_time = (
            float(times[min(interval + 1, len(times) - 1)])
            if interval >= 0
            else np.nan
        )
        row = {
            "replicate": replicate,
            "scenario": scenario,
            "event": str(event.get("event", "unknown")),
            "interval": interval,
            "event_time": event_time,
        }
        for key, value in event.items():
            if key not in row:
                row[key] = value.item() if isinstance(value, np.generic) else value
        rows.append(row)

    if result.branches is not None:
        branches = np.asarray(result.branches, dtype=int)
        changes = np.flatnonzero(np.diff(branches) != 0) + 1
        for index in changes:
            rows.append(
                {
                    "replicate": replicate,
                    "scenario": scenario,
                    "event": "branch_transition",
                    "interval": int(index - 1),
                    "event_time": float(times[index]),
                    "branch_from": int(branches[index - 1]),
                    "branch_to": int(branches[index]),
                }
            )

    return rows


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    root = Path(config["paths"]["project_root"]).expanduser()

    interval_path = resolve_path(root, config["paths"]["interval_table"])
    jump_path = resolve_path(root, config["paths"]["jump_table"])
    branch_path = resolve_path(root, config["paths"]["branch_transition_table"])
    stage5_json_path = resolve_path(root, config["paths"]["stage5_calibration_json"])

    for path in [interval_path, jump_path, branch_path, stage5_json_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = resolve_path(root, config["paths"]["output_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "empirical": output_dir / "Figure6_empirical_interval_statistics.csv",
        "parameters": output_dir / "Figure6_calibration_parameters.csv",
        "trajectories": output_dir / "Figure6_calibrated_trajectory_table.csv",
        "events": output_dir / "Figure6_calibrated_event_ledger.csv",
        "comparison": output_dir / "Figure6_observed_simulated_statistics.csv",
        "fidelity": output_dir / "Figure6_calibration_fidelity.csv",
        "metadata": output_dir / "Figure6_calibration_metadata.json",
    }

    if not args.overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Figure 6 outputs exist; use --overwrite.")

    interval = pd.read_csv(interval_path)
    jumps = pd.read_csv(jump_path)
    branch = pd.read_csv(branch_path)
    stage5_calibration = json.loads(stage5_json_path.read_text(encoding="utf-8"))

    patient_column = first_existing_column(
        interval,
        ["patient_id", "patient", "sample_id", "case_id"],
    )
    dt_column = first_existing_column(
        interval,
        [
            "dt",
            "delta_t",
            "interval_duration",
            "time_interval",
            "dt_years",
            "dt_months",
            "dt_days",
        ],
        required=False,
    )

    if dt_column is None:
        interval["_figure6_dt"] = 1.0
        dt_column = "_figure6_dt"
        time_calibration_mode = (
            "normalized_stage_transition"
        )
        time_calibration_unit = (
            "normalized transition units"
        )

        print(
            "[INFO] No sample-specific numerical timing "
            "column was found. Using one normalized time "
            "unit per observed stage transition."
        )
    else:
        time_calibration_mode = "empirical_numeric"
        time_calibration_unit = "source-table units"

    displacement_column = first_existing_column(
        interval,
        [
            "displacement_hd",
            "total_displacement",
            "disp_total_6d",
            "displacement",
            "distance",
        ],
    )

    switch_values = infer_branch_switches(jumps, interval)
    calibration = summarize_empirical_intervals(
        interval,
        patient_column=patient_column,
        dt_column=dt_column,
        displacement_column=displacement_column,
        branch_switch=switch_values,
    )

    dt_values = pd.to_numeric(interval[dt_column], errors="coerce")
    displacement_values = pd.to_numeric(interval[displacement_column], errors="coerce")
    patient_interval_counts = (
        interval.groupby(patient_column, dropna=False).size().to_numpy(dtype=int)
    )

    empirical_rows = []
    for patient_id, group in interval.groupby(patient_column, dropna=False):
        group_dt = pd.to_numeric(group[dt_column], errors="coerce").dropna().to_numpy(float)
        group_disp = pd.to_numeric(
            group[displacement_column], errors="coerce"
        ).dropna().to_numpy(float)
        empirical_rows.append(
            {
                "patient_id": str(patient_id),
                "n_intervals": int(len(group)),
                "n_observations": int(len(group) + 1),
                "followup": float(np.sum(group_dt[group_dt > 0]))
                if len(group_dt)
                else np.nan,
                "median_interval_duration": float(np.median(group_dt[group_dt > 0]))
                if np.any(group_dt > 0)
                else np.nan,
                "total_displacement": float(np.sum(np.abs(group_disp)))
                if len(group_disp)
                else np.nan,
                "max_interval_displacement": float(np.max(np.abs(group_disp)))
                if len(group_disp)
                else np.nan,
                "median_interval_displacement": float(np.median(np.abs(group_disp)))
                if len(group_disp)
                else np.nan,
                "q90_interval_displacement": float(np.quantile(np.abs(group_disp), 0.90))
                if len(group_disp)
                else np.nan,
            }
        )

    empirical_df = pd.DataFrame(empirical_rows)
    empirical_df.to_csv(outputs["empirical"], index=False)

    sigma = max(
        float(stage5_calibration.get("displacement_sd", calibration.displacement_sd)),
        float(config["simulation"]["minimum_sigma"]),
    )
    jump_scale = max(
        calibration.displacement_q90 - calibration.median_displacement,
        calibration.displacement_sd,
        float(config["simulation"]["minimum_sigma"]),
    ) * float(config["simulation"]["jump_scale_multiplier"])

    empirical_switch = calibration.branch_switch_fraction
    if not np.isfinite(empirical_switch):
        empirical_switch = 0.25

    mean_dt = max(calibration.median_dt, 1e-6)
    switch_rate = (
        -np.log(max(1.0 - min(empirical_switch, 0.95), 0.05)) / mean_dt
    ) * float(config["simulation"]["branch_switch_multiplier"])

    jump_rate = (
        max(1.0 / max(calibration.n_intervals, 1), 0.02)
        * float(config["simulation"]["jump_rate_multiplier"])
    )

    separation = (
        max(calibration.displacement_q90, sigma)
        * float(config["simulation"]["branch_separation_multiplier"])
    )

    parameter_rows = [
        {
            "quantity": "n_empirical_intervals",
            "value": calibration.n_intervals,
            "provenance": "Empirical",
            "description": "Intervals contributing to calibration",
        },
        {
            "quantity": "median_interval_duration",
            "value": calibration.median_dt,
            "provenance": "Empirical",
            "description": "Median positive observation interval",
        },
        {
            "quantity": "displacement_sd",
            "value": sigma,
            "provenance": "Derived",
            "description": "Characteristic continuous fluctuation scale",
        },
        {
            "quantity": "displacement_q90",
            "value": calibration.displacement_q90,
            "provenance": "Empirical",
            "description": "Upper-tail displacement scale",
        },
        {
            "quantity": "jump_scale",
            "value": jump_scale,
            "provenance": "Derived",
            "description": "Compound-Poisson jump magnitude scale",
        },
        {
            "quantity": "jump_rate",
            "value": jump_rate,
            "provenance": "Prespecified from empirical scale",
            "description": "Rare-event intensity used in calibration simulations",
        },
        {
            "quantity": "branch_switch_fraction",
            "value": empirical_switch,
            "provenance": "Empirical",
            "description": "Observed interval-level branch-switch prevalence",
        },
        {
            "quantity": "branch_switch_rate",
            "value": switch_rate,
            "provenance": "Derived",
            "description": "Continuous-time switching-rate approximation",
        },
        {
            "quantity": "branch_separation",
            "value": separation,
            "provenance": "Derived",
            "description": "Attractor separation on the calibrated state scale",
        },
        {
            "quantity": "theta",
            "value": float(config["simulation"]["theta"]),
            "provenance": "Prespecified",
            "description": "Restoring-rate setting for calibrated simulation",
        },
    ]
    parameter_df = pd.DataFrame(parameter_rows)
    parameter_df.to_csv(outputs["parameters"], index=False)

    seed = int(config["simulation"]["seed"])
    rng = np.random.default_rng(seed)
    n_replicates = int(config["simulation"]["n_replicates"])
    points_per_interval = int(config["simulation"]["latent_points_per_interval"])
    noise_sd = (
        float(config["simulation"]["observation_noise_fraction"]) * sigma
    )

    rate_matrix = np.array(
        [[-switch_rate, switch_rate], [switch_rate, -switch_rate]],
        dtype=float,
    )
    branch_spec = BranchSpec(
        rate_matrix,
        np.array(
            [
                float(config["simulation"]["theta"]),
                float(config["simulation"]["theta"]),
            ]
        ),
        np.array([[-0.5 * separation], [0.5 * separation]]),
        np.array([sigma, sigma]),
    )

    trajectory_rows = []
    event_ledger = []
    simulated_stat_rows = []

    scenarios = ["retention", "jump", "branch_reorganization"]

    for replicate in range(n_replicates):
        n_intervals = empirical_interval_count(
            patient_interval_counts, rng=rng, minimum=1
        )
        observed_times = empirical_schedule(
            dt_values.to_numpy(float),
            rng=rng,
            n_intervals=n_intervals,
        )
        dense_times = make_dense_schedule(observed_times, points_per_interval)
        scenario = scenarios[replicate % len(scenarios)]
        replicate_seed = int(rng.integers(0, 2**31 - 1))

        if scenario == "retention":
            result = simulate_process(
                dense_times,
                [0.0],
                model="ou",
                theta=float(config["simulation"]["theta"]),
                mu=float(config["simulation"]["mu"]),
                sigma=sigma,
                seed=replicate_seed,
            )
        elif scenario == "jump":
            result = simulate_process(
                dense_times,
                [0.0],
                model="ou_jump",
                theta=float(config["simulation"]["theta"]),
                mu=float(config["simulation"]["mu"]),
                sigma=sigma,
                jump=JumpSpec(rate=jump_rate, scale=jump_scale),
                seed=replicate_seed,
            )
        else:
            result = simulate_process(
                dense_times,
                [-0.5 * separation],
                model="ou_branching",
                branch=branch_spec,
                seed=replicate_seed,
            )

        obs_indices = np.searchsorted(dense_times, observed_times)
        latent_at_observed = result.states[obs_indices]
        observed_times_returned, observed_values = observe_latent(
            observed_times,
            latent_at_observed,
            ObservationSpec(
                noise_sd=noise_sd,
                missing_probability=0.0,
                retain_endpoints=True,
            ),
            rng,
        )
        mapped_observations = map_observations(
            dense_times,
            observed_times_returned,
            observed_values[:, 0],
        )

        branches = (
            np.asarray(result.branches, dtype=int)
            if result.branches is not None
            else np.full(len(dense_times), -1, dtype=int)
        )
        attractor = np.full(len(dense_times), float(config["simulation"]["mu"]))
        if result.branches is not None:
            attractor = np.where(
                branches == 0,
                -0.5 * separation,
                0.5 * separation,
            )

        for index, time in enumerate(dense_times):
            trajectory_rows.append(
                {
                    "replicate": replicate,
                    "scenario": scenario,
                    "seed": replicate_seed,
                    "time": float(time),
                    "latent_state": float(result.states[index, 0]),
                    "observed_state": mapped_observations[index],
                    "branch_state": int(branches[index]),
                    "attractor": float(attractor[index]),
                }
            )

        event_ledger.extend(event_rows(replicate, scenario, result, dense_times))

        stats = transition_statistics(
            observed_times,
            latent_at_observed[:, 0],
            branches=branches[obs_indices] if result.branches is not None else None,
            events=result.event_ledger,
        )
        stats.update(
            {
                "replicate": replicate,
                "scenario": scenario,
                "seed": replicate_seed,
            }
        )
        simulated_stat_rows.append(stats)

    trajectory_df = pd.DataFrame(trajectory_rows)
    events_df = pd.DataFrame(event_ledger)
    simulated_df = pd.DataFrame(simulated_stat_rows)

    trajectory_df.to_csv(outputs["trajectories"], index=False)
    events_df.to_csv(outputs["events"], index=False)

    observed_summary = {
        "n_observations": float(empirical_df["n_observations"].median()),
        "followup": float(empirical_df["followup"].median()),
        "total_displacement": float(empirical_df["total_displacement"].median()),
        "max_interval_displacement": float(
            empirical_df["max_interval_displacement"].median()
        ),
        "median_interval_displacement": float(
            empirical_df["median_interval_displacement"].median()
        ),
        "q90_interval_displacement": float(
            empirical_df["q90_interval_displacement"].median()
        ),
        "branch_switch_fraction": float(empirical_switch),
    }

    comparison_rows = []
    for metric, observed_value in observed_summary.items():
        values = pd.to_numeric(simulated_df[metric], errors="coerce").dropna()
        for value in values:
            comparison_rows.append(
                {
                    "metric": metric,
                    "source": "Simulated",
                    "value": float(value),
                }
            )
        comparison_rows.append(
            {
                "metric": metric,
                "source": "Observed",
                "value": observed_value,
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(outputs["comparison"], index=False)

    fidelity_metrics = [
        "n_observations",
        "followup",
        "total_displacement",
        "max_interval_displacement",
        "median_interval_displacement",
        "q90_interval_displacement",
        "branch_switch_fraction",
    ]
    fidelity_df = ratio_fidelity(
        observed_summary,
        simulated_df,
        fidelity_metrics,
    )
    fidelity_df.to_csv(outputs["fidelity"], index=False)

    metadata = {
        "figure": "Figure 6",
        "title": "Real-data calibration",
        "source_files": {
            "interval_table": str(interval_path),
            "jump_table": str(jump_path),
            "branch_transition_table": str(branch_path),
            "stage5_calibration_json": str(stage5_json_path),
        },
        "resolved_columns": {
            "patient": patient_column,
            "dt": dt_column,
            "displacement": displacement_column,
        },
        "simulation_seed": seed,
        "n_replicates": n_replicates,
        "scenario_order": scenarios,
        "archived_rows": {
            "empirical": int(len(empirical_df)),
            "parameters": int(len(parameter_df)),
            "trajectories": int(len(trajectory_df)),
            "events": int(len(events_df)),
            "comparison": int(len(comparison_df)),
            "fidelity": int(len(fidelity_df)),
        },
        "interpretation_note": (
            "Empirical data calibrate the observation architecture and "
            "characteristic dynamical scales. Parameters not identifiable "
            "from sparse intervals are explicitly marked as prespecified."
        ),
    }
    outputs["metadata"].write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    for path in outputs.values():
        print(f"[SAVED] {path}")


if __name__ == "__main__":
    main()
