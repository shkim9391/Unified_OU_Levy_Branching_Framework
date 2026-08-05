import numpy as np
import pandas as pd

from oulb.recovery_limits import (
    BranchRecoveryCondition,
    JumpRecoveryCondition,
    f1_score,
    nearest_attractor_states,
    run_branch_recovery_replicate,
    run_jump_recovery_replicate,
    summarize_branch_recovery,
    summarize_jump_recovery,
)


def test_f1_score_handles_defined_metrics():
    assert np.isclose(f1_score(0.5, 0.5), 0.5)
    assert f1_score(0.0, 0.0) == 0.0


def test_nearest_attractor_states():
    observed = np.array([-1.0, -0.2, 0.4, 1.2])
    predicted = nearest_attractor_states(
        observed,
        np.array([-0.5, 0.5]),
    )
    assert predicted.tolist() == [0, 0, 1, 1]


def test_jump_recovery_replicate_is_reproducible():
    condition = JumpRecoveryCondition(
        jump_rate=0.5,
        jump_scale=0.8,
        diffusion_sigma=0.2,
        observation_noise=0.1,
        n_observations=20,
    )
    first, _, _ = run_jump_recovery_replicate(
        condition,
        replicate=0,
        seed=123,
        followup=5.0,
        jump_z_threshold=3.0,
    )
    second, _, _ = run_jump_recovery_replicate(
        condition,
        replicate=0,
        seed=123,
        followup=5.0,
        jump_z_threshold=3.0,
    )
    assert first == second


def test_branch_recovery_replicate_ranges():
    condition = BranchRecoveryCondition(
        branch_separation=1.0,
        switching_rate=0.2,
        diffusion_sigma=0.2,
        observation_noise=0.1,
        n_observations=20,
    )
    result, trajectory, events = run_branch_recovery_replicate(
        condition,
        replicate=0,
        seed=321,
        followup=5.0,
    )
    assert 0.0 <= result["state_accuracy"] <= 1.0
    assert -1.0 <= result["adjusted_rand_index"] <= 1.0
    assert len(trajectory) == 20
    assert len(events) == 19


def test_summary_shapes():
    jump = pd.DataFrame(
        [
            {
                "jump_rate": 0.2,
                "jump_scale": 0.5,
                "diffusion_sigma": 0.2,
                "observation_noise": 0.1,
                "n_observations": 20,
                "replicate": 0,
                "precision": 1.0,
                "recall": 0.5,
                "false_positive_rate": 0.0,
                "f1": 2 / 3,
                "n_true_jumps": 2,
            }
        ]
    )
    branch = pd.DataFrame(
        [
            {
                "branch_separation": 1.0,
                "switching_rate": 0.2,
                "diffusion_sigma": 0.2,
                "observation_noise": 0.1,
                "n_observations": 20,
                "replicate": 0,
                "n_true_transitions": 1,
                "state_accuracy": 0.9,
                "adjusted_rand_index": 0.8,
                "transition_f1": 0.7,
            }
        ]
    )
    assert len(summarize_jump_recovery(jump)) == 1
    assert len(summarize_branch_recovery(branch)) == 1
