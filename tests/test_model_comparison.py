from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oulb.evaluation import (
    best_model_by_family,
    build_casewise_loglik_gain,
    build_switching_tail_grid,
    compute_information_criteria,
    fit_displacement_model_ladder,
    prepare_displacement_data,
    results_to_tables,
    standardize_stability,
)


def make_displacement_table() -> pd.DataFrame:
    stable = [0.20, 0.24, 0.28, 0.31, 0.34, 0.38]
    switching = [0.46, 0.50, 0.54, 0.58, 0.62, 1.45]
    rows: list[dict[str, object]] = []
    for index, value in enumerate(stable):
        rows.append(
            {
                "sample": f"S{index}",
                "disp_total_6d": value,
                "DX_branch_ge50": "A",
                "REL_branch_ge50": "A",
                "dx_to_rel_switch": 0,
            }
        )
    for index, value in enumerate(switching):
        rows.append(
            {
                "sample": f"W{index}",
                "disp_total_6d": value,
                "DX_branch_ge50": "A",
                "REL_branch_ge50": "B",
                "dx_to_rel_switch": 1,
            }
        )
    return pd.DataFrame(rows)


def prepare() -> pd.DataFrame:
    return prepare_displacement_data(
        make_displacement_table(),
        metric_column="disp_total_6d",
        sample_column="sample",
        dx_branch_column="DX_branch_ge50",
        rel_branch_column="REL_branch_ge50",
        stability_column="dx_to_rel_switch",
    )


def test_stability_standardization() -> None:
    assert standardize_stability(0) == "Stable"
    assert standardize_stability("no_switch") == "Stable"
    assert standardize_stability(1) == "Switching"
    assert standardize_stability("changed") == "Switching"
    assert standardize_stability(None) is None


def test_prepare_and_fit_default_ladder() -> None:
    prepared = prepare()
    assert prepared["stability_std"].value_counts().to_dict() == {
        "Stable": 6,
        "Switching": 6,
    }

    results = fit_displacement_model_ladder(
        prepared["disp_std"].to_numpy(dtype=float),
        prepared["group_code"].to_numpy(dtype=int),
    )
    assert list(results) == ["M0", "M1", "M2", "M3"]
    assert all(result.n == 12 for result in results.values())
    assert all(np.isfinite(result.loglik) for result in results.values())
    assert all(np.all(np.isfinite(result.logpdf)) for result in results.values())
    assert results["M1"].params["sigma_stable"] > 0
    assert results["M3"].params["nu"] > 2

    comparison, parameters = results_to_tables(results)
    assert comparison.shape[0] == 4
    assert np.isclose(comparison["aicc_weight"].sum(), 1.0)
    assert comparison["delta_aicc"].min() == 0
    assert set(parameters["model_id"]) == {"M0", "M1", "M2", "M3"}
    assert best_model_by_family(comparison, "gaussian") in {"M0", "M1"}
    assert best_model_by_family(comparison, "student_t") in {"M2", "M3"}


def test_tail_and_casewise_tables_have_validated_schema() -> None:
    prepared = prepare()
    results = fit_displacement_model_ladder(
        prepared["disp_std"].to_numpy(dtype=float),
        prepared["group_code"].to_numpy(dtype=int),
    )
    comparison, _ = results_to_tables(results)
    best_gaussian = results[best_model_by_family(comparison, "gaussian")]
    best_student = results[best_model_by_family(comparison, "student_t")]

    tail = build_switching_tail_grid(
        prepared, best_gaussian, best_student, grid_size=300
    )
    assert tail.shape == (300, 6)
    assert tail["x"].is_monotonic_increasing
    assert np.all((tail["observed_switching_survival"] >= 0) & (tail["observed_switching_survival"] <= 1))

    casewise = build_casewise_loglik_gain(prepared, results)
    assert casewise.shape[0] == len(prepared)
    assert casewise["delta_loglik"].is_monotonic_decreasing
    assert casewise["sample_std"].is_unique


def test_information_criteria_and_group_size_guard() -> None:
    aic, aicc, bic = compute_information_criteria(-10.0, k=2, n=20)
    assert np.isclose(aic, 24.0)
    assert aicc > aic
    assert bic > aic

    too_small = make_displacement_table().iloc[:7].copy()
    with pytest.raises(ValueError, match="Need at least 2 Stable"):
        prepare_displacement_data(
            too_small,
            metric_column="disp_total_6d",
            sample_column="sample",
            dx_branch_column="DX_branch_ge50",
            rel_branch_column="REL_branch_ge50",
            stability_column="dx_to_rel_switch",
        )
