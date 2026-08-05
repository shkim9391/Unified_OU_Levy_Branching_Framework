from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oulb.dynamics import (
    IntervalDefinition,
    compute_intervals,
    directional_discontinuity,
)


TIME_ORDER = {"DX": 0, "EOI_REM": 1, "REL": 2}
INTERVALS = (
    IntervalDefinition(name="DX_to_REL", start="DX", end="REL"),
    IntervalDefinition(name="DX_to_EOI_REM", start="DX", end="EOI_REM"),
    IntervalDefinition(
        name="EOI_REM_to_REL", start="EOI_REM", end="REL", prior="DX"
    ),
)


def make_centroids() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "AML21",
                "sample_id": "AML21_DX",
                "clinical_timepoint_coarse": "DX",
                "x2d": 0.0,
                "y2d": 0.0,
                "n_cells": 100,
                "hd_1": 0.0,
                "hd_2": 0.0,
                "branch_id_dominant": "HSC-like basin",
                "ecotype_dominant": "E1",
                "reg_program_score_median": 0.1,
                "disease_subgroup": "AML",
            },
            {
                "patient_id": "AML21",
                "sample_id": "AML21_EOI",
                "clinical_timepoint_coarse": "EOI_REM",
                "x2d": 1.0,
                "y2d": 0.0,
                "n_cells": 80,
                "hd_1": 1.0,
                "hd_2": 0.0,
                "branch_id_dominant": "Progenitor-like basin",
                "ecotype_dominant": "E2",
                "reg_program_score_median": 0.2,
                "disease_subgroup": "AML",
            },
            {
                "patient_id": "AML21",
                "sample_id": "AML21_REL",
                "clinical_timepoint_coarse": "REL",
                "x2d": 1.0,
                "y2d": 1.0,
                "n_cells": 90,
                "hd_1": 1.0,
                "hd_2": 1.0,
                "branch_id_dominant": "Progenitor-like basin",
                "ecotype_dominant": "E2",
                "reg_program_score_median": 0.3,
                "disease_subgroup": "AML",
            },
            {
                "patient_id": "AML22",
                "sample_id": "AML22_DX",
                "clinical_timepoint_coarse": "DX",
                "x2d": 0.0,
                "y2d": 0.0,
                "n_cells": 70,
                "hd_1": 0.0,
                "hd_2": 0.0,
                "branch_id_dominant": "GMP-like basin",
                "ecotype_dominant": "E3",
                "reg_program_score_median": 0.0,
                "disease_subgroup": "AML",
            },
            {
                "patient_id": "AML22",
                "sample_id": "AML22_REL",
                "clinical_timepoint_coarse": "REL",
                "x2d": 2.0,
                "y2d": 0.0,
                "n_cells": 65,
                "hd_1": 2.0,
                "hd_2": 0.0,
                "branch_id_dominant": "GMP-like basin",
                "ecotype_dominant": "E3",
                "reg_program_score_median": 0.1,
                "disease_subgroup": "AML",
            },
        ]
    )


def test_directional_discontinuity_for_orthogonal_vectors() -> None:
    assert np.isclose(
        directional_discontinuity(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        1.0,
    )


def test_compute_intervals_matches_validated_structure() -> None:
    out = compute_intervals(
        make_centroids(),
        time_order=TIME_ORDER,
        interval_definitions=INTERVALS,
        tail_reference_interval="DX_to_REL",
        tail_threshold_column="tail_threshold_dx_rel_q90",
        tail_flag_column="tail_flag_dx_rel_q90",
    )
    assert out.shape[0] == 4

    aml21_direct = out.loc[
        (out["patient_id"] == "AML21")
        & (out["interval_class"] == "DX_to_REL")
    ].iloc[0]
    assert np.isclose(aml21_direct["displacement_2d"], np.sqrt(2.0))
    assert np.isclose(aml21_direct["displacement_hd"], np.sqrt(2.0))
    assert aml21_direct["branch_switch"] == 1

    aml21_second = out.loc[
        (out["patient_id"] == "AML21")
        & (out["interval_class"] == "EOI_REM_to_REL")
    ].iloc[0]
    assert np.isclose(aml21_second["directional_discontinuity"], 1.0)
    assert aml21_second["branch_switch"] == 0

    direct = out.loc[out["interval_class"] == "DX_to_REL"].copy()
    flagged = direct.loc[direct["tail_flag_dx_rel_q90"], "patient_id"].tolist()
    assert flagged == ["AML22"]


def test_duplicate_patient_timepoint_is_rejected() -> None:
    centroids = make_centroids()
    centroids = pd.concat([centroids, centroids.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="multiple rows for a patient-timepoint"):
        compute_intervals(
            centroids,
            time_order=TIME_ORDER,
            interval_definitions=INTERVALS,
            duplicate_policy="error",
            tail_reference_interval="DX_to_REL",
            tail_threshold_column="tail_threshold_dx_rel_q90",
            tail_flag_column="tail_flag_dx_rel_q90",
        )
