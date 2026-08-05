from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oulb.dynamics import (
    add_effective_dynamic_parameters,
    collapse_replicated_sample_scores,
    compute_reference_attractor,
    robust_zscore,
)


SCAFFOLD = (
    "state_HSC",
    "state_Prog",
    "state_GMP",
    "state_MonoDC",
    "aux_EryBaso",
    "aux_CLP",
)


def _cell_rows(
    sample_id: str,
    patient_id: str,
    phase: str,
    branch: str,
    max_probability: float,
    entropy: float,
    scaffold: tuple[float, ...],
    n_cells: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(n_cells):
        record: dict[str, object] = {
            "sample_id": sample_id,
            "patient_id": patient_id,
            "clinical_timepoint_coarse": phase,
            "branch_id": branch,
            "branch_maxprob": max_probability,
            "branch_entropy": entropy,
            "ecotype_label": "E1" if patient_id == "P1" else "E2",
            "PC1": float(index) * 0.0,
            "PC2": float(index) * 0.0,
        }
        record.update(dict(zip(SCAFFOLD, scaffold)))
        rows.append(record)
    return rows


def make_observations() -> pd.DataFrame:
    rows = []
    rows += _cell_rows(
        "P1_DX",
        "P1",
        "DX",
        "HSC-like basin",
        0.8,
        0.4,
        (0.8, 0.1, 0.1, 0.0, 0.02, 0.01),
        2,
    )
    rows += _cell_rows(
        "P2_DX",
        "P2",
        "DX",
        "Progenitor-like basin",
        0.6,
        0.8,
        (0.4, 0.4, 0.1, 0.1, 0.04, 0.03),
        3,
    )
    rows += _cell_rows(
        "P1_REL",
        "P1",
        "REL",
        "GMP-like basin",
        0.5,
        1.0,
        (0.2, 0.2, 0.5, 0.1, 0.02, 0.01),
        4,
    )
    return pd.DataFrame(rows)


def test_collapse_and_reference_attractor() -> None:
    sample_table = collapse_replicated_sample_scores(
        make_observations(), scaffold_columns=SCAFFOLD
    )
    assert sample_table.shape[0] == 3
    p1_rel = sample_table.loc[sample_table["sample_id"] == "P1_REL"].iloc[0]
    assert p1_rel["n_cells"] == 4
    assert p1_rel["branch_id_dominant"] == "GMP-like basin"
    assert np.isclose(p1_rel["branch_id_dominant_frac"], 1.0)

    attractor = compute_reference_attractor(
        sample_table,
        scaffold_columns=SCAFFOLD,
        reference_phase="DX",
    )
    expected = np.array([0.6, 0.25, 0.1, 0.05, 0.03, 0.02])
    assert np.allclose(attractor[list(SCAFFOLD)].to_numpy(dtype=float), expected)


def test_effective_parameters_and_flags() -> None:
    sample_table = collapse_replicated_sample_scores(
        make_observations(), scaffold_columns=SCAFFOLD
    )
    attractor = compute_reference_attractor(
        sample_table,
        scaffold_columns=SCAFFOLD,
        reference_phase="DX",
    )
    output = add_effective_dynamic_parameters(
        sample_table,
        attractor,
        scaffold_columns=SCAFFOLD,
        max_branch_entropy=float(np.log(4.0)),
        main_patient_ids={"P1", "P2"},
        qc_flagged_patient_ids={"P2"},
        minimum_main_cells=3,
        phase_order={"DX": 0, "REL": 2},
    )

    p1_rel = output.loc[output["sample_id"] == "P1_REL"].iloc[0]
    assert np.isclose(p1_rel["theta_eff"], 0.5)
    assert np.isclose(p1_rel["sigma_eff"], 1.0 / np.log(4.0))
    expected_shift = np.linalg.norm(
        np.array([0.2, 0.2, 0.5, 0.1, 0.02, 0.01])
        - np.array([0.6, 0.25, 0.1, 0.05, 0.03, 0.02])
    )
    assert np.isclose(p1_rel["mu_shift_from_dx"], expected_shift)
    assert bool(p1_rel["is_main_analysis_sample"])
    assert p1_rel["phase_order"] == 2

    p2_dx = output.loc[output["sample_id"] == "P2_DX"].iloc[0]
    assert bool(p2_dx["is_qc_flagged_patient"])
    assert not bool(p2_dx["is_main_analysis_sample"])


def test_robust_zscore_constant_series_returns_zeros() -> None:
    output = robust_zscore(pd.Series([3.0, 3.0, 3.0]))
    assert np.array_equal(output.to_numpy(), np.zeros(3))


def test_missing_scaffold_column_is_rejected() -> None:
    observations = make_observations().drop(columns=["aux_CLP"])
    with pytest.raises(ValueError, match="missing required columns"):
        collapse_replicated_sample_scores(observations, scaffold_columns=SCAFFOLD)
