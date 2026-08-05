import numpy as np
import pandas as pd
import pytest

from oulb.jumps import JumpCandidateSpec, compute_jump_candidates


def interval_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3", "P4"],
            "interval_class": ["DX_to_REL", "DX_to_REL", "DX_to_REL", "DX_to_EOI_REM"],
            "sample_start": ["P1_DX", "P2_DX", "P3_DX", "P4_DX"],
            "sample_end": ["P1_REL", "P2_REL", "P3_REL", "P4_REM"],
            "displacement_hd": [1.0, 2.0, 3.0, 100.0],
            "displacement_2d": [2.0, 4.0, 8.0, 100.0],
            "branch_start": ["A", "A", "B", "A"],
            "branch_end": ["A", "B", "B", "A"],
            "branch_switch": [0, 1, 0, 0],
        }
    )


def test_compute_jump_candidates_preserves_validated_score_and_order():
    output = compute_jump_candidates(interval_table())
    assert output["patient_id"].tolist() == ["P3", "P2", "P1"]
    assert output["jump_rank"].tolist() == [1, 2, 3]
    assert output.loc[output["patient_id"] == "P2", "jump_class"].item() == "Branch-switching"
    p2 = output.loc[output["patient_id"] == "P2"].iloc[0]
    assert p2["jump_score"] == pytest.approx(p2["z_displacement_hd"] + 0.5)


def test_jump_candidate_branch_weight_is_configurable():
    base = compute_jump_candidates(interval_table())
    weighted = compute_jump_candidates(
        interval_table(), spec=JumpCandidateSpec(branch_weight=5.0)
    )
    assert weighted.iloc[0]["patient_id"] == "P2"
    assert base.loc[base.patient_id == "P2", "jump_score"].item() != weighted.loc[
        weighted.patient_id == "P2", "jump_score"
    ].item()


def test_zero_mad_produces_zero_displacement_z_scores():
    table = interval_table().iloc[:3].copy()
    table["displacement_hd"] = 2.0
    output = compute_jump_candidates(table)
    assert np.allclose(output["z_displacement_hd"], 0.0)
    assert output.loc[output.patient_id == "P2", "jump_score"].item() == 0.5


def test_missing_requested_interval_raises_clear_error():
    with pytest.raises(ValueError, match="No 'REL_to_NEXT'"):
        compute_jump_candidates(
            interval_table(), spec=JumpCandidateSpec(interval_value="REL_to_NEXT")
        )
