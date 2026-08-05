import numpy as np
import pandas as pd

from oulb.non_gaussian import (
    build_non_gaussian_result_tables,
    prepare_non_gaussian_dataframe,
)


def fixture_table():
    return pd.DataFrame({
        "sample": [f"P{i}" for i in range(8)],
        "DX_branch_ge50": ["B1", "B1", "B2", "B2", "B1", "B2", "B3", "B1"],
        "REL_branch_ge50": ["B1", "B1", "B2", "B2", "B2", "B3", "B1", "B3"],
        "disp_total_6d": [0.1, 0.2, 0.25, 0.3, 0.7, 0.8, 1.0, 1.2],
        "disp_malignant_3d": [0.1, 0.15, 0.2, 0.25, 0.5, 0.6, 0.7, 0.9],
        "disp_tme_3d": [0.05, 0.1, 0.1, 0.2, 0.4, 0.5, 0.8, 0.7],
        "dx_to_rel_switch": [0, 0, 0, 0, 1, 1, 1, 1],
    })


def test_non_gaussian_tables_are_materialized():
    prepared, resolved = prepare_non_gaussian_dataframe(fixture_table())
    result = build_non_gaussian_result_tables(prepared, n_boot=100, seed=2, top_jump_candidates=3)
    assert resolved["total_disp"] == "disp_total_6d"
    assert len(result["ranked"]) == 8
    assert len(result["qq"]) == 8
    assert len(result["effects"]) == 3
    assert len(result["jumps"]) == 3
    assert np.isfinite(result["metadata"]["baseline_scale"])


def test_rank_and_qq_tables_are_precomputed():
    prepared, _ = prepare_non_gaussian_dataframe(fixture_table())
    result = build_non_gaussian_result_tables(prepared, n_boot=20, seed=1)
    assert result["ranked"]["rank"].tolist() == list(range(1, 9))
    assert "theoretical_q" in result["qq"].columns
    assert "jump_score_std" in result["jumps"].columns
