from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oulb.observation import ObservationSpec, observe_latent, regular_schedule
from oulb.simulation import BranchSpec, JumpSpec, simulate_process


DEFAULT_OUTDIR = Path(
    "/Unified_OU_Levy_Branching_Framework/figures/data"
)

SCENARIO_ORDER = [
    "brownian",
    "ou",
    "shifted_ou",
    "ou_jump",
    "ou_branching",
    "full_oulb",
]

SCENARIO_LABELS = {
    "brownian": "Brownian drift",
    "ou": "Standard OU",
    "shifted_ou": "Treatment-shifted OU",
    "ou_jump": "OU with compound-Poisson jumps",
    "ou_branching": "OU with branch switching",
    "full_oulb": "Full OU-Lévy-Branching",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare archived representative trajectories for Figure 2."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pediatric_leukemia_stage5.toml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def branch_attractor(
    branches: np.ndarray | None,
    branch_spec: BranchSpec | None,
    times: np.ndarray,
    treatment_time: float | None = None,
    treatment_delta: float = 0.0,
) -> np.ndarray:
    if branches is None or branch_spec is None:
        return np.full(len(times), np.nan, dtype=float)

    mu = np.asarray(branch_spec.mu, dtype=float)
    if mu.ndim == 1:
        values = mu[branches]
    else:
        values = mu[branches, 0]

    values = values.astype(float)

    if treatment_time is not None:
        values = values + treatment_delta * (times >= treatment_time)

    return values


def treatment_indicator(
    times: np.ndarray,
    treatment_time: float | None,
) -> np.ndarray:
    if treatment_time is None:
        return np.zeros(len(times), dtype=int)
    return (times >= treatment_time).astype(int)


def map_observations_to_latent_grid(
    latent_times: np.ndarray,
    observed_times: np.ndarray,
    observed_values: np.ndarray,
) -> np.ndarray:
    mapped = np.full(len(latent_times), np.nan, dtype=float)

    for time, value in zip(observed_times, observed_values):
        index = int(np.argmin(np.abs(latent_times - time)))
        if not np.isclose(latent_times[index], time, atol=1e-10, rtol=0):
            raise ValueError(
                f"Observed time {time} does not match the latent grid."
            )
        mapped[index] = float(value)

    return mapped


def event_rows(
    scenario: str,
    result,
    times: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for event in result.event_ledger:
        interval = int(event.get("interval", -1))

        if 0 <= interval < len(times) - 1:
            event_time = float(times[interval + 1])
        else:
            event_time = np.nan

        record = {
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "event": str(event.get("event", "unknown")),
            "interval": interval,
            "event_time": event_time,
        }

        for key, value in event.items():
            if key in record:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            record[key] = value

        rows.append(record)

    return rows


def scenario_row_table(
    *,
    scenario: str,
    times: np.ndarray,
    result,
    observed_times: np.ndarray,
    observed_values: np.ndarray,
    treatment_time: float | None,
    attractor: np.ndarray,
) -> pd.DataFrame:
    states = np.asarray(result.states, dtype=float)

    if states.ndim != 2 or states.shape[1] < 1:
        raise ValueError(
            f"Scenario {scenario} returned an invalid state matrix."
        )

    branches = (
        np.asarray(result.branches, dtype=int)
        if result.branches is not None
        else np.full(len(times), -1, dtype=int)
    )

    observations = map_observations_to_latent_grid(
        times,
        observed_times,
        observed_values[:, 0],
    )

    return pd.DataFrame(
        {
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "scenario_order": SCENARIO_ORDER.index(scenario) + 1,
            "time": times,
            "latent_state": states[:, 0],
            "observed_state": observations,
            "is_observed": np.isfinite(observations),
            "treatment_indicator": treatment_indicator(
                times,
                treatment_time,
            ),
            "treatment_time": treatment_time,
            "branch_state": branches,
            "branch_attractor": attractor,
        }
    )



def has_jump_events(result, minimum: int = 1) -> bool:
    """Return True when the event ledger contains enough jump events."""
    count = sum(
        str(event.get("event", "")).lower() == "jump"
        for event in result.event_ledger
    )
    return count >= minimum


def has_branch_transition(result, minimum: int = 1) -> bool:
    """Return True when the latent branch sequence changes enough times."""
    if result.branches is None:
        return False

    branches = np.asarray(result.branches, dtype=int)

    if branches.size < 2:
        return False

    transition_count = int(
        np.sum(np.diff(branches) != 0)
    )

    return transition_count >= minimum


def branch_transition_rows(
    scenario: str,
    result,
    times: np.ndarray,
) -> list[dict[str, Any]]:
    if result.branches is None:
        return []

    branches = np.asarray(result.branches, dtype=int)
    change_indices = np.flatnonzero(
        np.diff(branches) != 0
    ) + 1

    rows = []

    for index in change_indices:
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "event": "branch_switch",
                "interval": int(index - 1),
                "event_time": float(times[index]),
                "branch_from": int(branches[index - 1]),
                "branch_to": int(branches[index]),
            }
        )

    return rows


def find_representative_seed(
    simulator,
    *,
    first_seed: int,
    require_jumps: int = 0,
    require_branch_transitions: int = 0,
    maximum_attempts: int = 10000,
):
    """
    Deterministically find the first seed producing the requested mechanisms.

    The chosen seed should subsequently be archived in the metadata.
    """
    for offset in range(maximum_attempts):
        seed = first_seed + offset
        result = simulator(seed)

        jump_ok = (
            require_jumps == 0
            or has_jump_events(
                result,
                minimum=require_jumps,
            )
        )

        branch_ok = (
            require_branch_transitions == 0
            or has_branch_transition(
                result,
                minimum=require_branch_transitions,
            )
        )

        if jump_ok and branch_ok:
            return result, seed

    raise RuntimeError(
        "No representative simulation was found after "
        f"{maximum_attempts} deterministic seed attempts."
    )


def main() -> None:
    args = parse_args()
    config = read_config(args.config)

    outdir = args.output_dir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    trajectory_path = outdir / "Figure2_representative_trajectories.csv"
    event_path = outdir / "Figure2_event_ledger.csv"
    metadata_path = outdir / "Figure2_simulation_metadata.json"

    targets = [trajectory_path, event_path, metadata_path]

    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError(
            "Figure 2 archived outputs already exist; use --overwrite."
        )

    benchmark = config.get("benchmark", {})
    base_seed = int(benchmark.get("seed", 20260803))
    followup = float(benchmark.get("followup", 8.0))

    # A dense latent grid supports smooth trajectories, whereas observations
    # are deliberately sparse and noisy.
    n_latent = 241
    times = regular_schedule(0.0, followup, n_latent)

    observation_spec = ObservationSpec(
        noise_sd=0.10,
        missing_probability=0.55,
        retain_endpoints=True,
    )

    treatment_time = followup * 0.48

    # Shared branch structure for branch-specific panels.
    rate_matrix = np.array(
        [
            [-0.28, 0.28],
            [0.16, -0.16],
        ],
        dtype=float,
    )
    branch_spec = BranchSpec(
        rate_matrix,
        np.array([0.90, 0.65], dtype=float),
        np.array([[-0.25], [0.85]], dtype=float),
        np.array([0.18, 0.24], dtype=float),
    )

    simulations: dict[str, dict[str, Any]] = {}

    simulations["brownian"] = {
        "result": simulate_process(
            times,
            [0.0],
            model="brownian",
            drift=0.10,
            sigma=0.20,
            seed=base_seed + 1,
        ),
        "seed": base_seed + 1,
        "treatment_time": None,
        "attractor": np.full(
            len(times),
            np.nan,
            dtype=float,
        ),
    }

    simulations["ou"] = {
        "result": simulate_process(
            times,
            [1.15],
            model="ou",
            theta=0.85,
            mu=0.20,
            sigma=0.18,
            seed=base_seed + 2,
        ),
        "seed": base_seed + 2,
        "treatment_time": None,
        "attractor": np.full(
            len(times),
            0.20,
            dtype=float,
        ),
    }

    treatment_shift = (
        lambda time: 0.70
        if time >= treatment_time
        else 0.0
    )

    simulations["shifted_ou"] = {
        "result": simulate_process(
            times,
            [0.0],
            model="shifted_ou",
            theta=0.80,
            mu=0.05,
            sigma=0.17,
            treatment_shift=treatment_shift,
            seed=base_seed + 3,
        ),
        "seed": base_seed + 3,
        "treatment_time": treatment_time,
        "attractor": (
            0.05
            + 0.70
            * (
                times >= treatment_time
            ).astype(float)
        ),
    }

    ou_jump_result, ou_jump_seed = find_representative_seed(
        lambda seed: simulate_process(
            times,
            [0.10],
            model="ou_jump",
            theta=0.90,
            mu=0.05,
            sigma=0.14,
            jump=JumpSpec(
                rate=0.55,
                scale=0.65,
            ),
            seed=seed,
        ),
        first_seed=base_seed + 4,
        require_jumps=2,
    )

    simulations["ou_jump"] = {
        "result": ou_jump_result,
        "seed": ou_jump_seed,
        "treatment_time": None,
        "attractor": np.full(
            len(times),
            0.05,
            dtype=float,
        ),
    }

    branching_result, branching_seed = find_representative_seed(
        lambda seed: simulate_process(
            times,
            [-0.20],
            model="ou_branching",
            branch=branch_spec,
            seed=seed,
        ),
        first_seed=base_seed + 5,
        require_branch_transitions=2,
    )

    simulations["ou_branching"] = {
        "result": branching_result,
        "seed": branching_seed,
        "treatment_time": None,
        "attractor": branch_attractor(
            branching_result.branches,
            branch_spec,
            times,
        ),
    }

    full_result, full_seed = find_representative_seed(
        lambda seed: simulate_process(
            times,
            [-0.20],
            model="full",
            branch=branch_spec,
            jump=JumpSpec(
                rate=0.45,
                scale=0.60,
            ),
            treatment_shift=treatment_shift,
            seed=seed,
        ),
        first_seed=base_seed + 6,
        require_jumps=2,
        require_branch_transitions=1,
    )

    simulations["full_oulb"] = {
        "result": full_result,
        "seed": full_seed,
        "treatment_time": treatment_time,
        "attractor": branch_attractor(
            full_result.branches,
            branch_spec,
            times,
            treatment_time=treatment_time,
            treatment_delta=0.70,
        ),
    }

    rng = np.random.default_rng(base_seed + 100)

    trajectory_tables: list[pd.DataFrame] = []
    ledger_rows: list[dict[str, Any]] = []

    for scenario in SCENARIO_ORDER:
        entry = simulations[scenario]
        result = entry["result"]

        observed_times, observed_values = observe_latent(
            times,
            result.states,
            observation_spec,
            rng,
        )

        table = scenario_row_table(
            scenario=scenario,
            times=times,
            result=result,
            observed_times=observed_times,
            observed_values=observed_values,
            treatment_time=entry["treatment_time"],
            attractor=np.asarray(
                entry["attractor"],
                dtype=float,
            ),
        )

        trajectory_tables.append(table)

        existing_event_names = {
            str(event.get("event", "")).lower()
            for event in result.event_ledger
        }

        ledger_rows.extend(
            event_rows(
                scenario,
                result,
                times,
            )
        )

        if not (
            "branch_switch" in existing_event_names
            or "branch_transition" in existing_event_names
        ):
            ledger_rows.extend(
                branch_transition_rows(
                    scenario,
                    result,
                    times,
                )
            )

    # Construct the archived trajectory table only after all six
    # scenarios have been processed.
    trajectory_df = pd.DataFrame.from_records(
        record
        for table in trajectory_tables
        for record in table.to_dict(
            orient="records"
        )
    )

    trajectory_df["time"] = pd.to_numeric(
        trajectory_df["time"],
        errors="raise",
    ).astype(float)

    for column in [
        "latent_state",
        "observed_state",
        "treatment_time",
        "branch_attractor",
    ]:
        trajectory_df[column] = pd.to_numeric(
            trajectory_df[column],
            errors="coerce",
        ).astype(float)

    trajectory_df["branch_state"] = pd.to_numeric(
        trajectory_df["branch_state"],
        errors="raise",
    ).astype(int)

    event_df = pd.DataFrame.from_records(
        ledger_rows
    )

    if event_df.empty:
        event_df = pd.DataFrame(
            columns=[
                "scenario",
                "scenario_label",
                "event",
                "interval",
                "event_time",
            ]
        )
    else:
        event_df["scenario_order"] = (
            event_df["scenario"]
            .map(
                {
                    scenario: index + 1
                    for index, scenario
                    in enumerate(SCENARIO_ORDER)
                }
            )
        )

        event_df = event_df.sort_values(
            [
                "scenario_order",
                "event_time",
                "event",
            ],
            na_position="last",
        ).reset_index(drop=True)

    trajectory_df.to_csv(
        trajectory_path,
        index=False,
    )
    event_df.to_csv(
        event_path,
        index=False,
    )

    metadata = {
        "figure": "Figure 2",
        "purpose": (
            "Representative archived trajectories for the nested "
            "OU-Lévy-Branching mechanisms."
        ),
        "source_modules": [
            "src/oulb/simulation.py",
            "src/oulb/observation.py",
        ],
        "source_config": str(args.config),
        "base_seed": base_seed,
        "scenario_seeds": {
            scenario: int(
                simulations[scenario]["seed"]
            )
            for scenario in SCENARIO_ORDER
        },
        "followup": followup,
        "n_latent": n_latent,
        "observation_spec": {
            "noise_sd": 0.10,
            "missing_probability": 0.55,
            "retain_endpoints": True,
        },
        "treatment_time": treatment_time,
        "scenario_order": SCENARIO_ORDER,
        "scenario_labels": SCENARIO_LABELS,
        "trajectory_rows": int(len(trajectory_df)),
        "event_rows": int(len(event_df)),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[SAVED] {trajectory_path}")
    print(f"[SAVED] {event_path}")
    print(f"[SAVED] {metadata_path}")
    print()
    print("Scenario event counts:")
    if event_df.empty:
        print("No events recorded.")
    else:
        print(
            event_df.groupby(
                ["scenario", "event"],
                dropna=False,
            )
            .size()
            .to_string()
        )


if __name__ == "__main__":
    main()
