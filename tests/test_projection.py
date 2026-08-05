from __future__ import annotations

import numpy as np
import pandas as pd

from oulb.projection import (
    CoarseStateDefinition,
    assign_nearest_reference,
    attach_sample_scores_to_anndata,
    compute_coarse_state_scores,
    compute_group_fractions,
    merge_sample_score_tables,
    project_fraction_row,
)
from oulb.scaffold import FrozenPCARecipe, FrozenScaffold


class FakeAnnData:
    def __init__(self, obs: pd.DataFrame):
        self.obs = obs.copy()
        self.obsm: dict[str, np.ndarray] = {}
        self.uns: dict[str, object] = {}


def make_scaffold() -> FrozenScaffold:
    recipe = FrozenPCARecipe(
        feature_order=("A", "B"),
        scaler_mean=np.array([0.0, 0.0]),
        scaler_scale=np.array([1.0, 1.0]),
        components=np.eye(2),
    )
    reference = pd.DataFrame(
        {
            "sample_id": ["ref_A", "ref_B"],
            "PC1": [1.0, 0.0],
            "PC2": [0.0, 1.0],
            "ecotype_label": ["E_A", "E_B"],
        }
    )
    return FrozenScaffold(
        normal_group_map={"fine_A": "A", "fine_B": "B"},
        pca_recipe=recipe,
        reference_scores=reference,
    )


def make_state_definition() -> CoarseStateDefinition:
    return CoarseStateDefinition(
        main_states={
            "state_HSC": ("HSC",),
            "state_Prog": ("Progenitor",),
            "state_GMP": ("GMP",),
            "state_MonoDC": ("Monocytes", "cDC"),
        },
        auxiliary_states={
            "aux_EryBaso": ("Early.Basophil", "Early.Erythrocyte"),
            "aux_CLP": ("CLP",),
        },
        branch_labels={
            "state_HSC": "HSC-like basin",
            "state_Prog": "Progenitor-like basin",
            "state_GMP": "GMP-like basin",
            "state_MonoDC": "Mono/DC-like basin",
        },
    )


def test_fraction_projection_and_nearest_reference() -> None:
    metadata = pd.DataFrame(
        {
            "Malignant": ["Normal", "Normal", "Normal", "Malignant"],
            "Classified_Celltype": ["fine_A", "fine_A", "fine_B", "fine_B"],
        }
    )
    scaffold = make_scaffold()
    fractions = compute_group_fractions(
        metadata,
        fine_to_group=scaffold.normal_group_map,
        feature_order=scaffold.pca_recipe.feature_order,
    )
    np.testing.assert_allclose(
        fractions[["normal_frac__A", "normal_frac__B"]].to_numpy(),
        np.array([2.0 / 3.0, 1.0 / 3.0]),
    )

    projected = project_fraction_row(fractions, scaffold)
    np.testing.assert_allclose(projected, np.array([2.0 / 3.0, 1.0 / 3.0]))
    label, reference_id, distance = assign_nearest_reference(projected, scaffold)
    assert label == "E_A"
    assert reference_id == "ref_A"
    assert distance > 0.0


def test_coarse_state_scores_preserve_validated_normalization() -> None:
    observations = pd.DataFrame(
        {
            "sample_id": ["s1"] * 6,
            "Classified_Celltype": [
                "HSC",
                "HSC",
                "Progenitor",
                "Monocytes",
                "Early.Basophil",
                "CLP",
            ],
        }
    )
    scores = compute_coarse_state_scores(
        observations,
        make_state_definition(),
    )
    row = scores.iloc[0]
    assert row["n_malignant_cells"] == 6
    np.testing.assert_allclose(
        row[["state_HSC", "state_Prog", "state_GMP", "state_MonoDC"]]
        .to_numpy(dtype=float),
        np.array([0.50, 0.25, 0.00, 0.25]),
    )
    np.testing.assert_allclose(
        row[["aux_EryBaso", "aux_CLP"]].to_numpy(dtype=float),
        np.array([1.0 / 6.0, 1.0 / 6.0]),
    )
    assert row["branch_id"] == "HSC-like basin"
    assert np.isclose(row["branch_maxprob"], 0.5)


def test_merge_and_attach_preserve_observation_count() -> None:
    context = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "PC1": [0.2],
            "PC2": [-0.1],
            "ecotype_label": ["E1"],
        }
    )
    state = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "n_malignant_cells": [2],
            "state_HSC": [0.4],
            "state_Prog": [0.3],
            "state_GMP": [0.2],
            "state_MonoDC": [0.1],
            "aux_EryBaso": [0.0],
            "aux_CLP": [0.0],
            "branch_id": ["HSC-like basin"],
            "branch_maxprob": [0.4],
            "branch_entropy": [1.2798542258336676],
        }
    )
    merged = merge_sample_score_tables(
        context, state, expected_sample_ids={"s1"}
    )
    adata = FakeAnnData(
        pd.DataFrame(
            {
                "sample_id": ["s1", "s1"],
                "Classified_Celltype": ["HSC", "GMP"],
            },
            index=["cell1", "cell2"],
        )
    )
    attach_sample_scores_to_anndata(
        adata,
        merged,
        sample_column="sample_id",
        required_score_columns=(
            "PC1",
            "PC2",
            "ecotype_label",
            "state_HSC",
            "state_Prog",
            "state_GMP",
            "state_MonoDC",
            "aux_EryBaso",
            "aux_CLP",
            "branch_id",
            "branch_maxprob",
            "branch_entropy",
        ),
        coordinate_columns=("PC1", "PC2"),
        scaffold_columns=(
            "state_HSC",
            "state_Prog",
            "state_GMP",
            "state_MonoDC",
            "aux_EryBaso",
            "aux_CLP",
        ),
    )
    assert adata.obs.shape[0] == 2
    assert adata.obs["sample_has_projected_scores"].all()
    assert adata.obsm["X_fig2"].shape == (2, 2)
    assert adata.obsm["X_scaffold"].shape == (2, 6)
