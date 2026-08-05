from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from oulb.benchmarking import (
    adjusted_rand_index,
    binary_metrics,
    detect_jump_intervals,
    standardized_ou_innovations,
)
from oulb.observation import ObservationSpec, observe_latent, regular_schedule
from oulb.simulation import BranchSpec, JumpSpec, simulate_process


@dataclass(frozen=True)
class JumpRecoveryCondition:
    jump_rate: float
    jump_scale: float
    diffusion_sigma: float
    observation_noise: float
    n_observations: int


@dataclass(frozen=True)
class BranchRecoveryCondition:
    branch_separation: float
    switching_rate: float
    diffusion_sigma: float
    observation_noise: float
    n_observations: int


def _safe_metric(metric: dict[str, float], key: str) -> float:
    value = metric.get(key, np.nan)
    return float(value) if value is not None else np.nan


def false_positive_rate(truth: np.ndarray, pred: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    fp = int(np.sum((~truth) & pred))
    tn = int(np.sum((~truth) & (~pred)))
    return fp / (fp + tn) if (fp + tn) > 0 else np.nan


def f1_score(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return np.nan
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def nearest_attractor_states(
    observed_values: np.ndarray,
    attractors: np.ndarray,
) -> np.ndarray:
    values = np.asarray(observed_values, dtype=float).reshape(-1, 1)
    centers = np.asarray(attractors, dtype=float).reshape(1, -1)
    distance = np.abs(values - centers)
    return np.argmin(distance, axis=1).astype(int)


def interval_event_truth(
    event_ledger: list[dict],
    n_intervals: int,
    event_name: str,
) -> np.ndarray:
    truth = np.zeros(n_intervals, dtype=bool)
    target = event_name.lower()
    for event in event_ledger:
        if str(event.get("event", "")).lower() != target:
            continue
        interval = int(event.get("interval", -1))
        if 0 <= interval < n_intervals:
            truth[interval] = True
    return truth


def transition_truth(branches: np.ndarray) -> np.ndarray:
    branches = np.asarray(branches, dtype=int)
    return np.diff(branches) != 0


def run_jump_recovery_replicate(
    condition: JumpRecoveryCondition,
    *,
    replicate: int,
    seed: int,
    followup: float,
    jump_z_threshold: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    times = regular_schedule(
        0.0,
        float(followup),
        int(condition.n_observations),
    )

    latent = simulate_process(
        times,
        [0.0],
        model="ou_jump",
        theta=0.8,
        mu=0.0,
        sigma=float(condition.diffusion_sigma),
        jump=JumpSpec(
            rate=float(condition.jump_rate),
            scale=float(condition.jump_scale),
        ),
        seed=int(seed),
    )

    rng = np.random.default_rng(seed + 1000003)
    obs_times, obs_values = observe_latent(
        times,
        latent.states,
        ObservationSpec(
            noise_sd=float(condition.observation_noise),
            missing_probability=0.0,
            retain_endpoints=True,
        ),
        rng,
    )

    y = np.asarray(obs_values[:, 0], dtype=float)
    z = standardized_ou_innovations(
        obs_times,
        y,
        mu=0.0,
        theta=0.8,
        sigma=max(float(condition.diffusion_sigma), 1e-8),
    )
    predicted = detect_jump_intervals(
        z,
        float(jump_z_threshold),
    )

    truth = interval_event_truth(
        latent.event_ledger,
        n_intervals=len(times) - 1,
        event_name="jump",
    )

    # Observation times equal the simulation grid in this benchmark.
    metrics = binary_metrics(truth, predicted)
    precision = _safe_metric(metrics, "precision")
    recall = _safe_metric(metrics, "recall")

    row = {
        "replicate": int(replicate),
        "seed": int(seed),
        "jump_rate": float(condition.jump_rate),
        "jump_scale": float(condition.jump_scale),
        "diffusion_sigma": float(condition.diffusion_sigma),
        "observation_noise": float(condition.observation_noise),
        "n_observations": int(condition.n_observations),
        "n_observed": int(len(obs_times)),
        "n_true_jumps": int(np.sum(truth)),
        "n_detected_jumps": int(np.sum(predicted)),
        "tp": int(np.sum(truth & predicted)),
        "fp": int(np.sum((~truth) & predicted)),
        "tn": int(np.sum((~truth) & (~predicted))),
        "fn": int(np.sum(truth & (~predicted))),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate(truth, predicted),
        "f1": f1_score(precision, recall),
    }

    trajectory = pd.DataFrame(
        {
            "time": times,
            "latent_state": latent.states[:, 0],
            "observed_state": y,
        }
    )

    events = pd.DataFrame(
        {
            "interval": np.arange(len(times) - 1, dtype=int),
            "event_time": times[1:],
            "true_jump": truth,
            "detected_jump": predicted,
            "standardized_innovation": z,
        }
    )

    return row, trajectory, events


def run_branch_recovery_replicate(
    condition: BranchRecoveryCondition,
    *,
    replicate: int,
    seed: int,
    followup: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    times = regular_schedule(
        0.0,
        float(followup),
        int(condition.n_observations),
    )

    separation = float(condition.branch_separation)
    rate = float(condition.switching_rate)
    attractors = np.array(
        [-0.5 * separation, 0.5 * separation],
        dtype=float,
    )

    branch_spec = BranchSpec(
        np.array(
            [
                [-rate, rate],
                [rate, -rate],
            ],
            dtype=float,
        ),
        np.array([0.8, 0.8], dtype=float),
        attractors[:, None],
        np.array(
            [
                condition.diffusion_sigma,
                condition.diffusion_sigma,
            ],
            dtype=float,
        ),
    )

    latent = simulate_process(
        times,
        [attractors[0]],
        model="ou_branching",
        branch=branch_spec,
        initial_branch=0,
        seed=int(seed),
    )

    rng = np.random.default_rng(seed + 2000003)
    obs_times, obs_values = observe_latent(
        times,
        latent.states,
        ObservationSpec(
            noise_sd=float(condition.observation_noise),
            missing_probability=0.0,
            retain_endpoints=True,
        ),
        rng,
    )

    observed = np.asarray(obs_values[:, 0], dtype=float)
    predicted_branch = nearest_attractor_states(
        observed,
        attractors,
    )
    true_branch = np.asarray(latent.branches, dtype=int)

    true_transition = transition_truth(true_branch)
    predicted_transition = transition_truth(predicted_branch)
    transition_metrics = binary_metrics(
        true_transition,
        predicted_transition,
    )

    row = {
        "replicate": int(replicate),
        "seed": int(seed),
        "branch_separation": separation,
        "switching_rate": rate,
        "diffusion_sigma": float(condition.diffusion_sigma),
        "observation_noise": float(condition.observation_noise),
        "n_observations": int(condition.n_observations),
        "n_observed": int(len(obs_times)),
        "n_true_transitions": int(np.sum(true_transition)),
        "state_accuracy": float(
            np.mean(true_branch == predicted_branch)
        ),
        "adjusted_rand_index": float(
            adjusted_rand_index(
                true_branch,
                predicted_branch,
            )
        ),
        "transition_precision": _safe_metric(
            transition_metrics,
            "precision",
        ),
        "transition_recall": _safe_metric(
            transition_metrics,
            "recall",
        ),
    }
    row["transition_f1"] = f1_score(
        row["transition_precision"],
        row["transition_recall"],
    )

    attractor_series = attractors[true_branch]
    trajectory = pd.DataFrame(
        {
            "time": times,
            "latent_state": latent.states[:, 0],
            "observed_state": observed,
            "true_branch": true_branch,
            "predicted_branch": predicted_branch,
            "true_attractor": attractor_series,
            "attractor_0": attractors[0],
            "attractor_1": attractors[1],
        }
    )

    transitions = pd.DataFrame(
        {
            "interval": np.arange(len(times) - 1, dtype=int),
            "event_time": times[1:],
            "true_transition": true_transition,
            "predicted_transition": predicted_transition,
        }
    )

    return row, trajectory, transitions


def summarize_jump_recovery(replicates: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "jump_rate",
        "jump_scale",
        "diffusion_sigma",
        "observation_noise",
        "n_observations",
    ]
    return (
        replicates.groupby(group_columns, dropna=False)
        .agg(
            n_replicates=("replicate", "size"),
            n_precision_evaluable=("precision", "count"),
            n_recall_evaluable=("recall", "count"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            precision_median=("precision", "median"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            false_positive_rate_mean=("false_positive_rate", "mean"),
            f1_mean=("f1", "mean"),
            mean_true_jumps=("n_true_jumps", "mean"),
        )
        .reset_index()
    )


def summarize_branch_recovery(replicates: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "branch_separation",
        "switching_rate",
        "diffusion_sigma",
        "observation_noise",
        "n_observations",
    ]
    return (
        replicates.groupby(group_columns, dropna=False)
        .agg(
            n_replicates=("replicate", "size"),
            mean_true_transitions=("n_true_transitions", "mean"),
            state_accuracy_mean=("state_accuracy", "mean"),
            state_accuracy_sd=("state_accuracy", "std"),
            ari_mean=("adjusted_rand_index", "mean"),
            ari_sd=("adjusted_rand_index", "std"),
            transition_f1_mean=("transition_f1", "mean"),
        )
        .reset_index()
    )


def select_representative_examples(
    jump_replicates: pd.DataFrame,
    branch_replicates: pd.DataFrame,
    *,
    minimum_true_jumps: int = 1,
    minimum_true_transitions: int = 1,
) -> pd.DataFrame:
    rows: list[dict] = []

    jump_eligible = jump_replicates[
        jump_replicates["n_true_jumps"] >= minimum_true_jumps
    ].copy()
    if jump_eligible.empty:
        raise ValueError("No jump replicate satisfies the example eligibility rule.")

    jump_success = jump_eligible.sort_values(
        ["f1", "recall", "precision", "seed"],
        ascending=[False, False, False, True],
        na_position="last",
    ).iloc[0]
    jump_failure = jump_eligible.sort_values(
        ["f1", "recall", "precision", "seed"],
        ascending=[True, True, True, True],
        na_position="last",
    ).iloc[0]

    for example_id, example_type, record in [
        ("J_success", "jump_success", jump_success),
        ("J_failure", "jump_failure", jump_failure),
    ]:
        rows.append(
            {
                "example_id": example_id,
                "example_type": example_type,
                "mechanism": "jump",
                "selection_metric": "f1",
                "selection_value": float(record["f1"])
                if np.isfinite(record["f1"]) else np.nan,
                "eligibility_rule": (
                    f"n_true_jumps >= {minimum_true_jumps}"
                ),
                "replicate": int(record["replicate"]),
                "seed": int(record["seed"]),
            }
        )

    branch_eligible = branch_replicates[
        branch_replicates["n_true_transitions"]
        >= minimum_true_transitions
    ].copy()
    if branch_eligible.empty:
        raise ValueError(
            "No branch replicate satisfies the example eligibility rule."
        )

    branch_success = branch_eligible.sort_values(
        ["adjusted_rand_index", "state_accuracy", "seed"],
        ascending=[False, False, True],
    ).iloc[0]
    branch_failure = branch_eligible.sort_values(
        ["adjusted_rand_index", "state_accuracy", "seed"],
        ascending=[True, True, True],
    ).iloc[0]

    for example_id, example_type, record in [
        ("B_success", "branch_success", branch_success),
        ("B_failure", "branch_failure", branch_failure),
    ]:
        rows.append(
            {
                "example_id": example_id,
                "example_type": example_type,
                "mechanism": "branch",
                "selection_metric": "adjusted_rand_index",
                "selection_value": float(
                    record["adjusted_rand_index"]
                ),
                "eligibility_rule": (
                    "n_true_transitions >= "
                    f"{minimum_true_transitions}"
                ),
                "replicate": int(record["replicate"]),
                "seed": int(record["seed"]),
            }
        )

    return pd.DataFrame(rows)
