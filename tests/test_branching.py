import numpy as np
import pandas as pd
import pytest

from oulb.branching import (
    build_branch_summary_artifacts,
    build_branch_transition_table,
    zscore_series,
)
from oulb.jumps import compute_jump_candidates

BRANCH_ORDER = ["A", "B", "C", "D"]
SCAFFOLD = ["s1", "s2", "s3", "s4", "aux1", "aux2"]


def interval_table(include_tail=True):
    table = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3", "P4"],
            "interval_class": ["DX_to_REL"] * 4,
            "t_start": ["DX"] * 4,
            "t_end": ["REL"] * 4,
            "sample_start": [f"P{i}_DX" for i in range(1, 5)],
            "sample_end": [f"P{i}_REL" for i in range(1, 5)],
            "n_cells_start": [100] * 4,
            "n_cells_end": [100] * 4,
            "displacement_2d": [0.1, 0.2, 0.3, 0.4],
            "displacement_hd": [0.2, 0.4, 0.8, 1.6],
            "direction_x": [0.1] * 4,
            "direction_y": [0.0] * 4,
            "branch_start": ["A", "A", "B", "C"],
            "branch_end": ["A", "B", "B", "A"],
            "branch_switch": [0, 1, 0, 1],
            "ecotype_start": ["E1"] * 4,
            "ecotype_end": ["E1"] * 4,
            "reg_score_start": [0.0] * 4,
            "reg_score_end": [0.0] * 4,
            "disease_subgroup": ["AML"] * 4,
            "directional_discontinuity": [np.nan] * 4,
        }
    )
    if include_tail:
        table["tail_threshold_dx_rel_q90"] = 1.36
        table["tail_flag_dx_rel_q90"] = [False, False, False, True]
    return table


def sample_table():
    rows = []
    branches = {"P1": ("A", "A"), "P2": ("A", "B"), "P3": ("B", "B"), "P4": ("C", "A")}
    for index, patient in enumerate(branches, start=1):
        for phase, branch in zip(("DX", "REL"), branches[patient]):
            row = {
                "sample_id": f"{patient}_{phase}",
                "patient_id": patient,
                "clinical_timepoint_coarse": phase,
                "n_cells": 100,
                "branch_id_dominant": branch,
                "ecotype_label": f"E{index % 2 + 1}",
                "theta_eff": 0.2 + index / 10,
                "sigma_eff": 0.8 - index / 20,
                "mu_shift_from_dx": index / 10,
                "is_main_analysis_sample": True,
                "PC1": float(index),
                "PC2": float(-index),
            }
            for j, column in enumerate(SCAFFOLD):
                row[column] = (index + j) / 20
            rows.append(row)
    return pd.DataFrame(rows)


def test_branch_transition_table_joins_start_and_end_annotations():
    intervals = interval_table()
    jumps = compute_jump_candidates(intervals)
    output = build_branch_transition_table(
        intervals, jumps, sample_table(), scaffold_columns=SCAFFOLD
    )
    assert len(output) == 4
    row = output.loc[output.patient_id == "P2"].iloc[0]
    assert row["transition_label"] == "A -> B"
    assert row["transition_class"] == "Branch-switching"
    assert row["ecotype_label_start"] == "E1"
    assert row["ecotype_label_end"] == "E1"


def test_branch_summary_bundle_builds_all_five_tables():
    intervals = interval_table()
    jumps = compute_jump_candidates(intervals)
    artifacts = build_branch_summary_artifacts(
        intervals,
        jumps,
        sample_table(),
        scaffold_columns=SCAFFOLD,
        branch_order=BRANCH_ORDER,
    )
    assert len(artifacts.transition_table) == 4
    assert set(artifacts.program_summary.branch_id_dominant) == {"A", "B", "C"}
    assert artifacts.escape_risk_summary["n_intervals"].sum() == 4
    assert set(artifacts.stats_summary.section) == {
        "transition_count",
        "dominant_ecotype_by_branch",
        "branch_escape_risk",
    }


def test_transition_fallback_tail_flag_is_inclusive():
    intervals = interval_table(include_tail=False)
    jumps = compute_jump_candidates(intervals)
    output = build_branch_transition_table(
        intervals, jumps, sample_table(), scaffold_columns=SCAFFOLD
    )
    threshold = output["tail_threshold_dx_rel_q90"].iloc[0]
    assert np.isfinite(threshold)
    assert output["tail_flag_dx_rel_q90"].sum() == 1


def test_duplicate_conflicting_jump_rows_fail_one_to_one_merge():
    intervals = interval_table()
    jumps = compute_jump_candidates(intervals)
    duplicate = jumps.iloc[[0]].copy()
    duplicate["jump_score"] += 1.0
    jumps = pd.concat([jumps, duplicate], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        build_branch_transition_table(
            intervals, jumps, sample_table(), scaffold_columns=SCAFFOLD
        )


def test_population_zscore_handles_zero_variance():
    result = zscore_series(pd.Series([3.0, 3.0, 3.0]))
    assert np.allclose(result, 0.0)
