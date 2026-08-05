import numpy as np
import pandas as pd

from oulb.real_data_calibration import (
    empirical_schedule,
    first_existing_column,
    ratio_fidelity,
    summarize_empirical_intervals,
    transition_statistics,
)


def test_first_existing_column_supports_canonical_names():
    table = pd.DataFrame({"Patient_ID": ["A"], "delta_t": [1.0]})
    assert first_existing_column(table, ["patient_id"]) == "Patient_ID"


def test_empirical_schedule_is_increasing():
    rng = np.random.default_rng(1)
    schedule = empirical_schedule(
        np.array([0.2, 0.5, 1.0]),
        rng=rng,
        n_intervals=5,
    )
    assert len(schedule) == 6
    assert np.all(np.diff(schedule) > 0)


def test_interval_summary_is_finite():
    table = pd.DataFrame(
        {
            "patient_id": ["A", "A", "B"],
            "dt": [0.5, 1.0, 0.25],
            "displacement": [0.2, 0.4, 0.1],
        }
    )
    summary = summarize_empirical_intervals(
        table,
        patient_column="patient_id",
        dt_column="dt",
        displacement_column="displacement",
        branch_switch=np.array([0, 1, 0]),
    )
    assert summary.n_patients == 2
    assert np.isfinite(summary.displacement_q90)
    assert np.isclose(summary.branch_switch_fraction, 1 / 3)


def test_transition_statistics_returns_expected_metrics():
    stats = transition_statistics(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.5, 0.25]),
        branches=np.array([0, 0, 1]),
        events=[{"event": "jump"}],
    )
    assert stats["n_observations"] == 3
    assert np.isclose(stats["max_interval_displacement"], 0.5)
    assert np.isclose(stats["branch_switch_fraction"], 0.5)


def test_ratio_fidelity_has_reference_ratio():
    simulated = pd.DataFrame({"total_displacement": [1.0, 2.0, 3.0]})
    result = ratio_fidelity(
        {"total_displacement": 2.0},
        simulated,
        ["total_displacement"],
    )
    assert len(result) == 1
    assert np.isclose(result.iloc[0]["ratio_median"], 1.0)
