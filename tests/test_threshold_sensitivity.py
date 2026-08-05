import numpy as np
import pandas as pd

from oulb.branching import (
    ThresholdBranchSpec,
    add_threshold_branch_assignments,
    build_threshold_branch_counts,
    build_threshold_patient_transition_tables,
    run_threshold_sensitivity,
)


def projection_table() -> pd.DataFrame:
    rows = []
    patterns = {
        "P1": [("DX", 0, 120, (0.1, 1.0, 2.0)), ("REL", 2, 110, (0.2, 1.2, 2.1))],
        "P2": [("DX", 0, 80, (0.2, 0.1, 2.0)), ("REL", 2, 75, (1.1, 0.2, 2.2))],
        "P3": [
            ("DX", 0, 55, (2.0, 0.2, 0.3)),
            ("REM", 1, 45, (2.0, 0.4, 0.2)),
            ("REL", 2, 65, (2.0, 1.0, 0.1)),
        ],
        "P4": [("DX", 0, 25, (0.2, 1.2, 2.0)), ("REL", 2, 30, (1.5, 0.2, 2.0))],
        "P5": [("DX", 0, 15, (0.2, 1.0, 2.0)), ("REL", 2, 18, (0.2, 1.0, 2.0))],
    }
    for patient, values in patterns.items():
        for timepoint, time_index, cells, distances in values:
            rows.append(
                {
                    "sample": patient,
                    "Patient_ID": f"ID_{patient}",
                    "sample_id": f"{patient}_{timepoint}",
                    "timepoint": timepoint,
                    "time_index": time_index,
                    "malignant_cells": cells,
                    "projection_eligible": True,
                    "dist_B1": distances[0],
                    "dist_B2": distances[1],
                    "dist_B3": distances[2],
                    "ilr_stem_vs_committed": 0.1,
                    "ilr_prog_vs_mature": 0.2,
                    "ilr_gmp_vs_monodc": 0.3,
                    "T_NK_given_known_z": 0.4,
                    "Myeloid_APC_given_known_z": 0.5,
                    "B_Plasma_given_known_z": 0.6,
                    "Biopsy_Origin": "BM",
                    "Subgroup": "AML",
                    "Treatment_Outcome": "Response",
                }
            )
    return pd.DataFrame(rows)


def test_threshold_assignments_use_minimum_distance_and_cell_cutoff():
    projected = add_threshold_branch_assignments(
        projection_table(), thresholds=[20, 50, 100]
    )
    p2_dx = projected.loc[projected.sample_id == "P2_DX"].iloc[0]
    assert p2_dx["branch_ge20"] == "B2"
    assert p2_dx["branch_ge50"] == "B2"
    assert pd.isna(p2_dx["branch_ge100"])
    p5_dx = projected.loc[projected.sample_id == "P5_DX"].iloc[0]
    assert not bool(p5_dx["eligible_ge20"])


def test_missing_state_value_makes_sample_ineligible():
    table = projection_table()
    table.loc[table.sample_id == "P1_DX", "T_NK_given_known_z"] = np.nan
    projected = add_threshold_branch_assignments(table, thresholds=[20])
    row = projected.loc[projected.sample_id == "P1_DX"].iloc[0]
    assert not bool(row["eligible_ge20"])
    assert pd.isna(row["branch_ge20"])


def test_threshold_counts_include_ineligible_na_column():
    projected = add_threshold_branch_assignments(
        projection_table(), thresholds=[20]
    )
    counts = build_threshold_branch_counts(projected, thresholds=[20])
    dx = counts.loc[counts.timepoint == "DX"].iloc[0]
    assert dx["n_total_samples"] == 5
    assert dx["n_eligible"] == 4
    assert dx["NA"] == 1
    assert dx[["B1", "B2", "B3"]].sum() == 4


def test_patient_transition_and_switch_tables_have_expected_counts():
    projected = add_threshold_branch_assignments(
        projection_table(), thresholds=[20, 50, 100]
    )
    patient, transition, switch = build_threshold_patient_transition_tables(
        projected, thresholds=[20, 50, 100]
    )
    assert len(patient) == 15
    assert len(transition) == 27
    row20 = switch.loc[switch.threshold == 20].iloc[0]
    assert row20["n_dx_rel_pairs"] == 4
    assert row20["n_switch"] == 2
    assert row20["switch_rate"] == 0.5


def test_complete_threshold_artifacts_include_stable_text_report():
    artifacts = run_threshold_sensitivity(
        projection_table(), thresholds=[20, 50, 100]
    )
    assert artifacts.summary_text.startswith(
        "GSE235063 transition summary and threshold sensitivity"
    )
    assert "Threshold >= 50 malignant cells" in artifacts.summary_text
    assert "DX -> REL transition matrix:" in artifacts.summary_text
    assert len(artifacts.switch_rates) == 3
