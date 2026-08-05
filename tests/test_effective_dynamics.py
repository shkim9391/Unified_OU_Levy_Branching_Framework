from __future__ import annotations

import numpy as np
import pandas as pd

from oulb.dynamics import (
    add_effective_dynamic_parameters,
    collapse_replicated_sample_scores,
    compute_reference_attractor,
    compute_sample_effective_dynamics,
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


def make_observations() -> pd.DataFrame:
    sample_rows = [
        {
            "sample_id": "P1_DX",
            "patient_id": "P1",
            "clinical_timepoint_coarse": "DX",
            "branch_id": "HSC-like basin",
            "branch_maxprob": 0.7,
            "branch_entropy": np.log(2.0),
            "ecotype_label": "E1",
            "PC1": 1.0,
            "PC2": 2.0,
            "state_HSC": 0.7,
            "state_Prog": 0.1,
            "state_GMP": 0.1,
            "state_MonoDC": 0.1,
            "aux_EryBaso": 0.2,
            "aux_CLP": 0.0,
        },
        {
            "sample_id": "P1_REL",
            "patient_id": "P1",
            "clinical_timepoint_coarse": "REL",
            "branch_id": "Progenitor-like basin",
            "branch_maxprob": 0.5,
            "branch_entropy": np.log(4.0),
            "ecotype_label": "E2",
            "PC1": 3.0,
            "PC2": 4.0,
            "state_HSC": 0.2,
            "state_Prog": 0.5,
            "state_GMP": 0.2,
            "state_MonoDC": 0.1,
            "aux_EryBaso": 0.0,
            "aux_CLP": 0.1,
        },
        {
            "sample_id": "P2_DX",
            "patient_id": "P2",
            "clinical_timepoint_coarse": "DX",
            "branch_id": "GMP-like basin",
            "branch_maxprob": 0.6,
            "branch_entropy": np.log(2.0),
            "ecotype_label": "E3",
            "PC1": -1.0,
            "PC2": 0.0,
            "state_HSC": 0.1,
            "state_Prog": 0.2,
            "state_GMP": 0.6,
            "state_MonoDC": 0.1,
            "aux_EryBaso": 0.1,
            "aux_CLP": 0.0,
        },
    ]
    rows: list[dict[str, object]] = []
    for sample in sample_rows:
        repetitions = 60 if sample["sample_id"] != "P2_DX" else 40
        rows.extend([sample.copy() for _ in range(repetitions)])
    return pd.DataFrame(rows)


def test_collapse_replicated_scores_preserves_sample_values() -> None:
    collapsed = collapse_replicated_sample_scores(
        make_observations(), scaffold_columns=SCAFFOLD
    )
    assert collapsed.shape[0] == 3
    p1_dx = collapsed.set_index("sample_id").loc["P1_DX"]
    assert p1_dx["n_cells"] == 60
    assert np.isclose(p1_dx["branch_maxprob"], 0.7)
    assert np.isclose(p1_dx["PC1"], 1.0)
    assert p1_dx["branch_id_dominant_frac"] == 1.0


def test_reference_attractor_and_effective_parameters() -> None:
    collapsed = collapse_replicated_sample_scores(
        make_observations(), scaffold_columns=SCAFFOLD
    )
    attractor = compute_reference_attractor(
        collapsed,
        scaffold_columns=SCAFFOLD,
        reference_timepoint="DX",
    )
    assert np.isclose(attractor["state_HSC"], 0.4)
    assert np.isclose(attractor["state_GMP"], 0.35)

    output = add_effective_dynamic_parameters(
        collapsed,
        attractor,
        scaffold_columns=SCAFFOLD,
        main_patients={"P1", "P2"},
        flagged_patients={"P2"},
        min_main_cells=50,
        n_main_states=4,
    )
    indexed = output.set_index("sample_id")
    assert np.isclose(indexed.loc["P1_DX", "theta_eff"], 0.7)
    assert np.isclose(indexed.loc["P1_REL", "sigma_eff"], 1.0)
    assert bool(indexed.loc["P1_DX", "is_main_analysis_sample"])
    assert not bool(indexed.loc["P2_DX", "is_main_analysis_sample"])
    assert indexed.loc["P1_DX", "phase_order"] == 0
    assert indexed.loc["P1_REL", "phase_order"] == 2


def test_high_level_effective_dynamics_matches_composed_steps() -> None:
    output, attractor = compute_sample_effective_dynamics(
        make_observations(),
        scaffold_columns=SCAFFOLD,
        main_patients={"P1", "P2"},
        exploratory_patients={"P2"},
        minimum_main_cells=50,
    )
    assert len(output) == 3
    assert list(attractor.index) == list(SCAFFOLD)
    assert set(output["sample_id"]) == {"P1_DX", "P1_REL", "P2_DX"}


def test_robust_zscore_returns_zero_when_mad_is_zero() -> None:
    result = robust_zscore(pd.Series([1.0, 1.0, 1.0]))
    assert np.array_equal(result.to_numpy(), np.zeros(3))
