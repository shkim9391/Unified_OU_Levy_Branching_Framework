from __future__ import annotations

import argparse
import json
import tomllib
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from oulb.recovery_limits import (
    BranchRecoveryCondition,
    JumpRecoveryCondition,
    run_branch_recovery_replicate,
    run_jump_recovery_replicate,
    select_representative_examples,
    summarize_branch_recovery,
    summarize_jump_recovery,
)


DEFAULT_ROOT = Path(
    "/Unified_OU_Levy_Branching_Framework"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/figure4_recovery_limits.toml"),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Use the final replicate count rather than the development count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def _read_config(path: Path) -> dict:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def _seed_stream(base_seed: int):
    rng = np.random.default_rng(base_seed)
    while True:
        yield int(rng.integers(0, 2**31 - 1))


def _condition_match(row: pd.Series, record: pd.Series) -> bool:
    shared = [
        column
        for column in row.index
        if column in record.index
        and column not in {"replicate", "seed"}
    ]
    for column in shared:
        left = row[column]
        right = record[column]
        if isinstance(left, (float, np.floating)):
            if not np.isclose(float(left), float(right)):
                return False
        elif left != right:
            return False
    return True


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser()
    config = _read_config(args.config)

    output_dir = root / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "jump_replicates": output_dir / "Figure4_jump_recovery_replicates.csv",
        "jump_summary": output_dir / "Figure4_jump_recovery_summary.csv",
        "branch_replicates": output_dir / "Figure4_branch_recovery_replicates.csv",
        "branch_summary": output_dir / "Figure4_branch_recovery_summary.csv",
        "selection": output_dir / "Figure4_representative_selection.csv",
        "trajectories": output_dir / "Figure4_representative_trajectories.csv",
        "events": output_dir / "Figure4_representative_events.csv",
        "metadata": output_dir / "Figure4_recovery_metadata.json",
    }

    if not args.overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(
            "Figure 4 archive files already exist; use --overwrite."
        )

    benchmark = config["benchmark"]
    followup = float(benchmark["followup"])
    seed_iter = _seed_stream(int(benchmark["seed"]))

    jump_cfg = config["jump_recovery"]
    branch_cfg = config["branch_recovery"]
    replicate_key = (
        "replicates_final"
        if args.final
        else "replicates_development"
    )
    jump_replicates_n = int(jump_cfg[replicate_key])
    branch_replicates_n = int(branch_cfg[replicate_key])

    jump_rows: list[dict] = []
    jump_payload: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    jump_grid = product(
        jump_cfg["jump_rates"],
        jump_cfg["jump_scales"],
        jump_cfg["diffusion_scales"],
        jump_cfg["observation_noise"],
        jump_cfg["n_observations"],
    )

    for condition_index, values in enumerate(jump_grid):
        condition = JumpRecoveryCondition(
            jump_rate=float(values[0]),
            jump_scale=float(values[1]),
            diffusion_sigma=float(values[2]),
            observation_noise=float(values[3]),
            n_observations=int(values[4]),
        )
        for replicate in range(jump_replicates_n):
            seed = next(seed_iter)
            row, trajectory, events = run_jump_recovery_replicate(
                condition,
                replicate=replicate,
                seed=seed,
                followup=followup,
                jump_z_threshold=float(
                    jump_cfg["jump_z_threshold"]
                ),
            )
            row["condition_index"] = condition_index
            jump_rows.append(row)
            jump_payload[seed] = (trajectory, events)

    branch_rows: list[dict] = []
    branch_payload: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    branch_grid = product(
        branch_cfg["branch_separations"],
        branch_cfg["switching_rates"],
        branch_cfg["diffusion_scales"],
        branch_cfg["observation_noise"],
        branch_cfg["n_observations"],
    )

    for condition_index, values in enumerate(branch_grid):
        condition = BranchRecoveryCondition(
            branch_separation=float(values[0]),
            switching_rate=float(values[1]),
            diffusion_sigma=float(values[2]),
            observation_noise=float(values[3]),
            n_observations=int(values[4]),
        )
        for replicate in range(branch_replicates_n):
            seed = next(seed_iter)
            row, trajectory, events = run_branch_recovery_replicate(
                condition,
                replicate=replicate,
                seed=seed,
                followup=followup,
            )
            row["condition_index"] = condition_index
            branch_rows.append(row)
            branch_payload[seed] = (trajectory, events)

    jump_df = pd.DataFrame(jump_rows)
    branch_df = pd.DataFrame(branch_rows)
    jump_summary = summarize_jump_recovery(jump_df)
    branch_summary = summarize_branch_recovery(branch_df)

    selection = select_representative_examples(
        jump_df,
        branch_df,
        minimum_true_jumps=int(
            jump_cfg["minimum_true_jumps_for_example"]
        ),
        minimum_true_transitions=int(
            branch_cfg["minimum_true_transitions_for_example"]
        ),
    )

    representative_trajectories: list[pd.DataFrame] = []
    representative_events: list[pd.DataFrame] = []

    for _, selected in selection.iterrows():
        seed = int(selected["seed"])
        mechanism = selected["mechanism"]
        payload = (
            jump_payload[seed]
            if mechanism == "jump"
            else branch_payload[seed]
        )
        trajectory, events = payload

        trajectory = trajectory.copy()
        events = events.copy()

        for table in [trajectory, events]:
            table.insert(0, "example_id", selected["example_id"])
            table.insert(1, "example_type", selected["example_type"])
            table.insert(2, "mechanism", mechanism)
            table.insert(3, "seed", seed)
            table.insert(4, "selection_metric", selected["selection_metric"])
            table.insert(5, "selection_value", selected["selection_value"])

        representative_trajectories.append(trajectory)
        representative_events.append(events)

    trajectory_df = pd.concat(
        representative_trajectories,
        ignore_index=True,
    )
    event_df = pd.concat(
        representative_events,
        ignore_index=True,
    )

    jump_df.to_csv(paths["jump_replicates"], index=False)
    jump_summary.to_csv(paths["jump_summary"], index=False)
    branch_df.to_csv(paths["branch_replicates"], index=False)
    branch_summary.to_csv(paths["branch_summary"], index=False)
    selection.to_csv(paths["selection"], index=False)
    trajectory_df.to_csv(paths["trajectories"], index=False)
    event_df.to_csv(paths["events"], index=False)

    metadata = {
        "figure": "Figure 4",
        "title": (
            "Recovery limits of discontinuous and branching dynamics"
        ),
        "config": str(args.config),
        "mode": "final" if args.final else "development",
        "replicate_key": replicate_key,
        "jump_replicate_rows": int(len(jump_df)),
        "jump_summary_rows": int(len(jump_summary)),
        "branch_replicate_rows": int(len(branch_df)),
        "branch_summary_rows": int(len(branch_summary)),
        "representative_examples": selection.to_dict(
            orient="records"
        ),
        "display_slice": config["display"],
        "selection_is_upstream": True,
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    for path in paths.values():
        print(f"[SAVED] {path}")


if __name__ == "__main__":
    main()
