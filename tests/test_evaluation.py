from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oulb.evaluation import (
    aicc_from_loglik,
    build_fit_summary_text,
    prepare_transition_metric_data,
    run_default_model_comparison,
    standardize_stability,
)


def make_transition_table() -> pd.DataFrame:
    stable = [0.18, 0.21, 0.24, 0.26, 0.29, 0.31, 0.34, 0.36]
    switching = [0.22, 0.27, 0.32, 0.37, 0.44, 0.58, 0.95, 1.45]
    rows = []
    for index, value in enumerate(stable):
        rows.append(
            {
                "sample": f"S{index}",
                "metric": value,
                "dx_branch": "A",
                "rel_branch": "A",
                "switch": 0,
            }
        )
    for index, value in enumerate(switching):
        rows.append(
            {
                "sample": f"W{index}",
                "metric": value,
                "dx_branch": "A",
                "rel_branch": "B",
                "switch": 1,
            }
        )
    return pd.DataFrame(rows)


def test_standardize_stability_common_encodings() -> None:
    assert standardize_stability(0) == "Stable"
    assert standardize_stability("same") == "Stable"
    assert standardize_stability(1) == "Switching"
    assert standardize_stability("changed") == "Switching"
    assert standardize_stability(None) is None


def test_prepare_transition_metric_data_and_group_counts() -> None:
    prepared = prepare_transition_metric_data(
        make_transition_table(),
        metric_column="metric",
        sample_column="sample",
        start_branch_column="dx_branch",
        end_branch_column="rel_branch",
        stability_column="switch",
    )
    assert prepared.shape[0] == 16
    assert prepared["group_code"].value_counts().to_dict() == {0: 8, 1: 8}
    assert set(prepared["stability_std"]) == {"Stable", "Switching"}


def test_default_model_comparison_builds_all_outputs() -> None:
    prepared = prepare_transition_metric_data(
        make_transition_table(),
        metric_column="metric",
        sample_column="sample",
        start_branch_column="dx_branch",
        end_branch_column="rel_branch",
        stability_column="switch",
    )
    artifacts = run_default_model_comparison(prepared)
    assert set(artifacts.results) == {"M0", "M1", "M2", "M3"}
    assert artifacts.comparison_table.shape[0] == 4
    assert np.isclose(artifacts.comparison_table["aicc_weight"].sum(), 1.0)
    assert artifacts.tail_fit_grid.shape[0] == 300
    assert artifacts.casewise_loglik_gain.shape[0] == 16
    assert set(artifacts.parameter_table["model_id"]) == {"M0", "M1", "M2", "M3"}
    summary = build_fit_summary_text(artifacts)
    assert "Model ranking by AICc" in summary
    assert "Best Gaussian model" in summary


def test_aicc_is_infinite_when_sample_size_is_too_small() -> None:
    assert np.isinf(aicc_from_loglik(-1.0, k=5, n=6))


def test_preparation_rejects_insufficient_groups() -> None:
    table = make_transition_table().iloc[:9].copy()
    with pytest.raises(ValueError, match="Need at least"):
        prepare_transition_metric_data(
            table,
            metric_column="metric",
            sample_column="sample",
            start_branch_column="dx_branch",
            end_branch_column="rel_branch",
            stability_column="switch",
            minimum_per_group=2,
        )
