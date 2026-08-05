import json
from pathlib import Path

import pandas as pd

from oulb.model_comparison_plotting import render_model_comparison_figure
from oulb.non_gaussian import build_non_gaussian_result_tables, prepare_non_gaussian_dataframe
from oulb.non_gaussian_plotting import render_non_gaussian_figure


def test_non_gaussian_plot_reads_tables(tmp_path: Path):
    raw = pd.DataFrame({
        "sample": ["A", "B", "C", "D"],
        "DX_branch_ge50": ["B1", "B1", "B1", "B2"],
        "REL_branch_ge50": ["B1", "B1", "B2", "B1"],
        "disp_total_6d": [0.1, 0.2, 0.8, 1.0],
        "disp_malignant_3d": [0.1, 0.2, 0.6, 0.8],
        "disp_tme_3d": [0.1, 0.15, 0.5, 0.7],
        "dx_to_rel_switch": [0, 0, 1, 1],
    })
    prepared, _ = prepare_non_gaussian_dataframe(raw)
    tables = build_non_gaussian_result_tables(prepared, n_boot=20, seed=1, top_jump_candidates=4)
    png, pdf = render_non_gaussian_figure(
        tables["ranked"], tables["qq"], tables["effects"], tables["jumps"],
        tables["metadata"], tmp_path / "figure", dpi=72
    )
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0


def test_model_comparison_plot_reads_archived_tables(tmp_path: Path):
    comp = pd.DataFrame({
        "model_id": ["M3", "M1", "M2", "M0"],
        "model_label": ["Student-t branch-aware", "Gaussian branch-aware", "Student-t pooled", "Gaussian pooled"],
        "aicc": [10.0, 12.0, 13.0, 14.0],
        "delta_aicc": [0.0, 2.0, 3.0, 4.0],
    })
    tail = pd.DataFrame({
        "x": [0.1, 0.2, 0.3],
        "observed_switching_survival": [1.0, 0.6, 0.2],
        "switching_gaussian_survival": [0.9, 0.5, 0.1],
        "switching_student_t_survival": [0.95, 0.62, 0.22],
    })
    case = pd.DataFrame({
        "sample_std": ["A", "B", "C", "D"],
        "stability_std": ["Stable", "Switching", "Stable", "Switching"],
        "delta_loglik": [0.1, 0.5, -0.1, 0.3],
    })
    png, pdf = render_model_comparison_figure(comp, tail, case, tmp_path / "model", dpi=72)
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0
