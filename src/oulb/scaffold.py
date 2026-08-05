from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _as_string_tuple(values: Any, *, field_name: str) -> tuple[str, ...]:
    """Convert list-like values stored in ``AnnData.uns`` to a string tuple."""
    arr = np.asarray(values, dtype=object).reshape(-1)
    result = tuple(str(value) for value in arr.tolist())
    if not result:
        raise ValueError(f"{field_name} must contain at least one value.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} contains duplicate values: {result}")
    return result


@dataclass(frozen=True)
class FrozenPCARecipe:
    """Immutable scaling and PCA transformation learned from a reference set."""

    feature_order: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    components: np.ndarray
    component_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        feature_order = tuple(str(value) for value in self.feature_order)
        if not feature_order:
            raise ValueError("feature_order must contain at least one feature.")
        if len(set(feature_order)) != len(feature_order):
            raise ValueError("feature_order contains duplicate features.")

        mean = np.asarray(self.scaler_mean, dtype=float).reshape(-1).copy()
        scale = np.asarray(self.scaler_scale, dtype=float).reshape(-1).copy()
        components = np.asarray(self.components, dtype=float).copy()

        if components.ndim != 2:
            raise ValueError(
                f"PCA components must be two-dimensional; received shape {components.shape}."
            )
        n_features = len(feature_order)
        if mean.shape != (n_features,):
            raise ValueError(
                f"scaler_mean has shape {mean.shape}; expected {(n_features,)}."
            )
        if scale.shape != (n_features,):
            raise ValueError(
                f"scaler_scale has shape {scale.shape}; expected {(n_features,)}."
            )
        if components.shape[1] != n_features:
            raise ValueError(
                "PCA component width does not match feature_order: "
                f"{components.shape[1]} versus {n_features}."
            )
        if components.shape[0] < 1:
            raise ValueError("At least one PCA component is required.")
        if not np.isfinite(mean).all():
            raise ValueError("scaler_mean contains non-finite values.")
        if not np.isfinite(scale).all():
            raise ValueError("scaler_scale contains non-finite values.")
        if not np.isfinite(components).all():
            raise ValueError("PCA components contain non-finite values.")

        names = self.component_names
        if names is None:
            names = tuple(f"PC{i + 1}" for i in range(components.shape[0]))
        else:
            names = tuple(str(value) for value in names)
            if len(names) != components.shape[0]:
                raise ValueError(
                    "component_names length must equal the number of PCA components."
                )
            if len(set(names)) != len(names):
                raise ValueError("component_names contains duplicate names.")

        # A zero scale is treated as one during projection, matching the
        # validated implementation while avoiding division by zero.
        safe_scale = scale.copy()
        safe_scale[safe_scale == 0.0] = 1.0

        object.__setattr__(self, "feature_order", feature_order)
        object.__setattr__(self, "scaler_mean", mean)
        object.__setattr__(self, "scaler_scale", safe_scale)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "component_names", names)

    @property
    def n_features(self) -> int:
        return len(self.feature_order)

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        """Project one or more feature vectors using the frozen recipe.

        Parameters
        ----------
        values:
            Array with shape ``(n_features,)`` or ``(n_rows, n_features)``.
        """
        matrix = np.asarray(values, dtype=float)
        was_vector = matrix.ndim == 1
        if was_vector:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features:
            raise ValueError(
                "Projection input must have shape (n_rows, n_features) with "
                f"n_features={self.n_features}; received {matrix.shape}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Projection input contains non-finite values.")

        standardized = (matrix - self.scaler_mean[None, :]) / self.scaler_scale[None, :]
        projected = standardized @ self.components.T
        return projected[0] if was_vector else projected


@dataclass(frozen=True)
class FrozenScaffold:
    """Reference mapping, frozen PCA recipe, and labeled reference coordinates."""

    normal_group_map: Mapping[str, str]
    pca_recipe: FrozenPCARecipe
    reference_scores: pd.DataFrame
    coordinate_columns: tuple[str, ...] = ("PC1", "PC2")
    label_column: str = "ecotype_label"
    reference_id_column: str = "sample_id"
    reference_score_source: Path | None = None

    def __post_init__(self) -> None:
        normal_map = {str(key): str(value) for key, value in dict(self.normal_group_map).items()}
        coordinates = tuple(str(value) for value in self.coordinate_columns)
        if not coordinates:
            raise ValueError("coordinate_columns must contain at least one coordinate.")
        if len(coordinates) > self.pca_recipe.n_components:
            raise ValueError(
                "The number of scaffold coordinates exceeds the number of PCA components."
            )

        score_table = self.reference_scores.copy()
        required = [self.reference_id_column, *coordinates, self.label_column]
        missing = [column for column in required if column not in score_table.columns]
        if missing:
            raise ValueError(f"Reference score table is missing columns: {missing}")

        score_table = score_table[required].copy()
        score_table[self.reference_id_column] = score_table[self.reference_id_column].astype(str)
        score_table[self.label_column] = score_table[self.label_column].astype(str)
        if score_table[self.reference_id_column].duplicated().any():
            duplicated = score_table.loc[
                score_table[self.reference_id_column].duplicated(keep=False),
                self.reference_id_column,
            ].tolist()
            raise ValueError(f"Reference IDs are not unique: {duplicated}")

        for column in coordinates:
            score_table[column] = pd.to_numeric(score_table[column], errors="raise")
        coordinate_values = score_table[list(coordinates)].to_numpy(dtype=float)
        if not np.isfinite(coordinate_values).all():
            raise ValueError("Reference coordinate table contains non-finite values.")
        if score_table.empty:
            raise ValueError("Reference score table is empty.")

        source = None if self.reference_score_source is None else Path(self.reference_score_source)
        object.__setattr__(self, "normal_group_map", normal_map)
        object.__setattr__(self, "reference_scores", score_table)
        object.__setattr__(self, "coordinate_columns", coordinates)
        object.__setattr__(self, "label_column", str(self.label_column))
        object.__setattr__(self, "reference_id_column", str(self.reference_id_column))
        object.__setattr__(self, "reference_score_source", source)


def load_reference_score_table(
    path: str | Path,
    *,
    coordinate_columns: Sequence[str] = ("PC1", "PC2"),
    label_column: str = "ecotype_label",
    reference_id_column: str = "sample_id",
) -> pd.DataFrame:
    """Load the reference labels and coordinates used for nearest-neighbor transfer."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)

    table = pd.read_csv(source)
    required = [reference_id_column, *coordinate_columns, label_column]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Reference source CSV missing required columns: {missing}")
    return table[required].copy()


def load_frozen_scaffold_from_anndata(
    reference: Any,
    *,
    reference_scores_path: str | Path | None = None,
    normal_grouping_key: str = "normal_broad_grouping",
    pca_key: str = "ecotype_pca",
    transfer_key: str = "transfer_frozen",
    frozen_score_source_key: str = "frozen_score_source",
    coordinate_columns: Sequence[str] = ("PC1", "PC2"),
    label_column: str = "ecotype_label",
    reference_id_column: str = "sample_id",
) -> FrozenScaffold:
    """Reconstruct a validated frozen scaffold from an AnnData-like object.

    The object only needs an ``uns`` mapping; importing :mod:`anndata` is not
    required by the reusable core.
    """
    if not hasattr(reference, "uns"):
        raise TypeError("reference must expose an AnnData-like 'uns' mapping.")

    uns = reference.uns
    required_uns = [normal_grouping_key, pca_key, transfer_key]
    missing_uns = [key for key in required_uns if key not in uns]
    if missing_uns:
        raise ValueError(f"Reference object missing required uns entries: {missing_uns}")

    normal_map = dict(uns[normal_grouping_key])
    pca_payload = uns[pca_key]
    for key in ("feature_order", "scaler_mean", "scaler_scale", "pca_components"):
        if key not in pca_payload:
            raise ValueError(f"Reference PCA recipe missing key: {key}")

    feature_order = _as_string_tuple(
        pca_payload["feature_order"], field_name="feature_order"
    )
    components = np.asarray(pca_payload["pca_components"], dtype=float)
    if components.ndim != 2:
        raise ValueError(
            "Reference PCA components must be a two-dimensional array; "
            f"received shape {components.shape}."
        )
    component_names = tuple(f"PC{i + 1}" for i in range(components.shape[0]))
    recipe = FrozenPCARecipe(
        feature_order=feature_order,
        scaler_mean=np.asarray(pca_payload["scaler_mean"], dtype=float),
        scaler_scale=np.asarray(pca_payload["scaler_scale"], dtype=float),
        components=components,
        component_names=component_names,
    )

    if reference_scores_path is None:
        transfer_payload = uns[transfer_key]
        if frozen_score_source_key not in transfer_payload:
            raise ValueError(
                f"Reference transfer recipe missing key: {frozen_score_source_key}"
            )
        reference_scores_path = Path(str(transfer_payload[frozen_score_source_key]))
    else:
        reference_scores_path = Path(reference_scores_path).expanduser()

    reference_scores = load_reference_score_table(
        reference_scores_path,
        coordinate_columns=coordinate_columns,
        label_column=label_column,
        reference_id_column=reference_id_column,
    )

    return FrozenScaffold(
        normal_group_map=normal_map,
        pca_recipe=recipe,
        reference_scores=reference_scores,
        coordinate_columns=tuple(coordinate_columns),
        label_column=label_column,
        reference_id_column=reference_id_column,
        reference_score_source=Path(reference_scores_path),
    )
