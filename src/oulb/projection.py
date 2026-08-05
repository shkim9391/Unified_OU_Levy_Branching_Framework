from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .scaffold import FrozenScaffold


@dataclass(frozen=True)
class CoarseStateDefinition:
    """Map fine cell labels into normalized main states and auxiliary states."""

    main_states: Mapping[str, Sequence[str]]
    auxiliary_states: Mapping[str, Sequence[str]]
    branch_labels: Mapping[str, str]
    unknown_branch_label: str = "Unknown"

    def __post_init__(self) -> None:
        main_states = {
            str(state): tuple(str(label) for label in labels)
            for state, labels in dict(self.main_states).items()
        }
        auxiliary_states = {
            str(state): tuple(str(label) for label in labels)
            for state, labels in dict(self.auxiliary_states).items()
        }
        branch_labels = {
            str(state): str(label) for state, label in dict(self.branch_labels).items()
        }
        if not main_states:
            raise ValueError("At least one main state is required.")
        missing_branch_labels = [state for state in main_states if state not in branch_labels]
        if missing_branch_labels:
            raise ValueError(
                f"Missing branch labels for main states: {missing_branch_labels}"
            )

        main_fine_labels = [label for labels in main_states.values() for label in labels]
        if len(main_fine_labels) != len(set(main_fine_labels)):
            raise ValueError(
                "Fine labels cannot belong to more than one normalized main state."
            )

        object.__setattr__(self, "main_states", main_states)
        object.__setattr__(self, "auxiliary_states", auxiliary_states)
        object.__setattr__(self, "branch_labels", branch_labels)
        object.__setattr__(self, "unknown_branch_label", str(self.unknown_branch_label))

    @property
    def main_state_columns(self) -> tuple[str, ...]:
        return tuple(self.main_states.keys())

    @property
    def auxiliary_state_columns(self) -> tuple[str, ...]:
        return tuple(self.auxiliary_states.keys())

    @property
    def scaffold_columns(self) -> tuple[str, ...]:
        return self.main_state_columns + self.auxiliary_state_columns


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def compute_group_fractions(
    metadata: pd.DataFrame,
    *,
    fine_to_group: Mapping[str, str],
    feature_order: Sequence[str],
    malignancy_column: str = "Malignant",
    cell_type_column: str = "Classified_Celltype",
    excluded_malignancy_values: Sequence[str] = ("Malignant",),
    output_prefix: str = "normal_frac__",
) -> pd.Series:
    """Compute selected broad-group fractions from non-excluded cells.

    The denominator is the sum of counts assigned to ``feature_order``. Cells
    mapped to ``Unknown`` or to groups outside that order are excluded from the
    denominator. This exactly preserves the behavior of the validated script.
    """
    require_columns(
        metadata,
        [malignancy_column, cell_type_column],
        context="Cell metadata",
    )
    ordered_features = tuple(str(value) for value in feature_order)
    if not ordered_features:
        raise ValueError("feature_order must not be empty.")

    table = metadata[[malignancy_column, cell_type_column]].copy()
    table[malignancy_column] = table[malignancy_column].astype(str)
    table[cell_type_column] = table[cell_type_column].astype(str)

    excluded = {str(value) for value in excluded_malignancy_values}
    table = table.loc[~table[malignancy_column].isin(excluded)].copy()
    mapping = {str(key): str(value) for key, value in dict(fine_to_group).items()}
    table["__broad_group"] = table[cell_type_column].map(mapping).fillna("Unknown")

    selected_counts = (
        table["__broad_group"]
        .value_counts()
        .reindex(ordered_features, fill_value=0)
        .astype(float)
    )
    denominator = float(selected_counts.sum())
    if denominator == 0.0:
        fractions = pd.Series(0.0, index=ordered_features, dtype=float)
    else:
        fractions = selected_counts / denominator

    fractions.index = [f"{output_prefix}{feature}" for feature in ordered_features]
    return fractions.astype(float)


def project_fraction_row(
    fractions: pd.Series,
    scaffold: FrozenScaffold,
    *,
    input_prefix: str = "normal_frac__",
) -> np.ndarray:
    """Project one sample's ordered broad-group fractions into PCA space."""
    expected_columns = [
        f"{input_prefix}{feature}" for feature in scaffold.pca_recipe.feature_order
    ]
    missing = [column for column in expected_columns if column not in fractions.index]
    if missing:
        raise ValueError(f"Fraction row missing projection features: {missing}")
    values = fractions.loc[expected_columns].to_numpy(dtype=float)
    return scaffold.pca_recipe.transform_array(values)


def assign_nearest_reference(
    coordinates: Sequence[float],
    scaffold: FrozenScaffold,
) -> tuple[str, str, float]:
    """Assign the label of the nearest reference sample in frozen coordinate space."""
    query = np.asarray(coordinates, dtype=float).reshape(-1)
    n_coordinates = len(scaffold.coordinate_columns)
    if query.shape != (n_coordinates,):
        raise ValueError(
            f"Expected {n_coordinates} query coordinates; received shape {query.shape}."
        )
    if not np.isfinite(query).all():
        raise ValueError("Query coordinates contain non-finite values.")

    reference_matrix = scaffold.reference_scores[
        list(scaffold.coordinate_columns)
    ].to_numpy(dtype=float)
    distances = np.sqrt(np.sum((reference_matrix - query[None, :]) ** 2, axis=1))
    index = int(np.argmin(distances))
    row = scaffold.reference_scores.iloc[index]
    return (
        str(row[scaffold.label_column]),
        str(row[scaffold.reference_id_column]),
        float(distances[index]),
    )


def project_samples_from_manifest(
    manifest: pd.DataFrame,
    requested_sample_ids: Iterable[str],
    scaffold: FrozenScaffold,
    *,
    metadata_root: str | Path,
    metadata_reader: Callable[[Path], pd.DataFrame] | None = None,
    metadata_read_kwargs: Mapping[str, Any] | None = None,
    sample_column: str = "sample_id",
    metadata_file_column: str = "metadata_file",
    manifest_passthrough: Mapping[str, str] | None = None,
    malignancy_column: str = "Malignant",
    cell_type_column: str = "Classified_Celltype",
    excluded_malignancy_values: Sequence[str] = ("Malignant",),
    fraction_prefix: str = "normal_frac__",
) -> pd.DataFrame:
    """Project selected manifest samples using their cell-metadata files."""
    require_columns(
        manifest,
        [sample_column, metadata_file_column],
        context="Manifest",
    )
    passthrough = dict(manifest_passthrough or {})
    require_columns(manifest, passthrough.keys(), context="Manifest")

    requested = {str(value) for value in requested_sample_ids}
    if not requested:
        raise ValueError("requested_sample_ids is empty.")

    table = manifest.copy()
    table["__sample_key"] = table[sample_column].astype(str)
    table = table.loc[table["__sample_key"].isin(requested)].copy()

    duplicated = table["__sample_key"].duplicated(keep=False)
    if duplicated.any():
        values = table.loc[duplicated, "__sample_key"].tolist()
        raise ValueError(f"Manifest contains duplicate selected sample IDs: {values}")

    found = set(table["__sample_key"])
    missing_samples = sorted(requested - found)
    if missing_samples:
        raise ValueError(
            "Selected query samples are absent from the manifest: "
            f"{missing_samples}"
        )

    root = Path(metadata_root).expanduser()
    read_kwargs = dict(metadata_read_kwargs or {})
    if metadata_reader is None:
        metadata_reader = lambda path: pd.read_csv(path, **read_kwargs)

    rows: list[dict[str, Any]] = []
    coordinate_columns = scaffold.coordinate_columns
    for _, manifest_row in table.iterrows():
        metadata_value = manifest_row[metadata_file_column]
        if pd.isna(metadata_value):
            raise ValueError(
                f"Sample {manifest_row['__sample_key']} has no metadata file."
            )
        metadata_path = Path(str(metadata_value)).expanduser()
        if not metadata_path.is_absolute():
            metadata_path = root / metadata_path
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        metadata = metadata_reader(metadata_path)
        fractions = compute_group_fractions(
            metadata,
            fine_to_group=scaffold.normal_group_map,
            feature_order=scaffold.pca_recipe.feature_order,
            malignancy_column=malignancy_column,
            cell_type_column=cell_type_column,
            excluded_malignancy_values=excluded_malignancy_values,
            output_prefix=fraction_prefix,
        )
        projected_all = project_fraction_row(
            fractions,
            scaffold,
            input_prefix=fraction_prefix,
        )
        projected = projected_all[: len(coordinate_columns)]
        label, nearest_id, distance = assign_nearest_reference(projected, scaffold)

        record: dict[str, Any] = {sample_column: str(manifest_row["__sample_key"])}
        for source_column, output_column in passthrough.items():
            record[str(output_column)] = str(manifest_row[source_column])
        for index, coordinate_column in enumerate(coordinate_columns):
            record[coordinate_column] = float(projected[index])
        record[scaffold.label_column] = label
        record["ecotype_nearest_reference_sample"] = nearest_id
        record["ecotype_nearest_reference_distance"] = distance
        record.update({str(key): float(value) for key, value in fractions.items()})
        rows.append(record)

    return pd.DataFrame(rows)


def _left_associative_sum(values: Iterable[float]) -> float:
    """Sum floats in explicit left-to-right order.

    Python 3.12+ may use compensated summation in :func:`sum`. The validated
    implementation used explicit additions, so this helper preserves its
    floating-point operation order for regression equivalence.
    """
    total = 0.0
    for value in values:
        total = total + float(value)
    return float(total)


def compute_coarse_state_scores(
    observations: pd.DataFrame,
    state_definition: CoarseStateDefinition,
    *,
    sample_column: str = "sample_id",
    cell_type_column: str = "Classified_Celltype",
    count_column: str = "n_malignant_cells",
    branch_id_column: str = "branch_id",
    branch_max_probability_column: str = "branch_maxprob",
    branch_entropy_column: str = "branch_entropy",
) -> pd.DataFrame:
    """Aggregate fine cell labels into normalized main and auxiliary states."""
    require_columns(
        observations,
        [sample_column, cell_type_column],
        context="Observation table",
    )
    obs = observations[[sample_column, cell_type_column]].copy()
    obs[sample_column] = obs[sample_column].astype(str)
    obs[cell_type_column] = obs[cell_type_column].astype(str)

    rows: list[dict[str, Any]] = []
    for sample_id, sample_obs in obs.groupby(
        sample_column, sort=False, observed=True
    ):
        counts = sample_obs[cell_type_column].value_counts()
        total = float(counts.sum())
        fractions = counts.astype(float) / total if total > 0.0 else counts.astype(float) * np.nan

        main_raw = {
            state: _left_associative_sum(
                float(fractions.get(label, 0.0)) for label in labels
            )
            for state, labels in state_definition.main_states.items()
        }
        denominator = _left_associative_sum(main_raw.values())
        if denominator <= 0.0 or not np.isfinite(denominator):
            normalized_main = {state: np.nan for state in main_raw}
        else:
            normalized_main = {
                state: float(value / denominator) for state, value in main_raw.items()
            }

        auxiliary = {
            state: _left_associative_sum(
                float(fractions.get(label, 0.0)) for label in labels
            )
            for state, labels in state_definition.auxiliary_states.items()
        }

        state_vector = np.asarray(
            [normalized_main[state] for state in state_definition.main_state_columns],
            dtype=float,
        )
        if np.isfinite(state_vector).all():
            branch_max_probability = float(np.nanmax(state_vector))
            branch_entropy = float(
                -(state_vector * np.log(np.clip(state_vector, 1e-12, None))).sum()
            )
            branch_state = state_definition.main_state_columns[
                int(np.nanargmax(state_vector))
            ]
            branch_id = state_definition.branch_labels[branch_state]
        else:
            branch_max_probability = np.nan
            branch_entropy = np.nan
            branch_id = state_definition.unknown_branch_label

        record: dict[str, Any] = {
            sample_column: str(sample_id),
            count_column: int(sample_obs.shape[0]),
        }
        record.update(normalized_main)
        record.update(auxiliary)
        record[branch_id_column] = branch_id
        record[branch_max_probability_column] = branch_max_probability
        record[branch_entropy_column] = branch_entropy
        rows.append(record)

    return pd.DataFrame(rows)


def merge_sample_score_tables(
    context_scores: pd.DataFrame,
    state_scores: pd.DataFrame,
    *,
    sample_column: str = "sample_id",
    expected_sample_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One-to-one merge of context and malignant-state sample summaries."""
    require_columns(context_scores, [sample_column], context="Context score table")
    require_columns(state_scores, [sample_column], context="State score table")

    left = context_scores.copy()
    right = state_scores.copy()
    left[sample_column] = left[sample_column].astype(str)
    right[sample_column] = right[sample_column].astype(str)
    if left[sample_column].duplicated().any():
        raise ValueError("Context score table contains duplicate sample IDs.")
    if right[sample_column].duplicated().any():
        raise ValueError("State score table contains duplicate sample IDs.")

    merged = left.merge(
        right,
        on=sample_column,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if expected_sample_ids is not None:
        expected = {str(value) for value in expected_sample_ids}
        observed = set(merged[sample_column].astype(str))
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise ValueError(
                "Merged sample score coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
    return merged


def attach_sample_scores_to_anndata(
    adata: Any,
    sample_scores: pd.DataFrame,
    *,
    sample_column: str,
    required_score_columns: Sequence[str],
    coordinate_columns: Sequence[str],
    scaffold_columns: Sequence[str],
    coordinate_obsm_key: str = "X_fig2",
    scaffold_obsm_key: str = "X_scaffold",
    projected_flag_column: str = "sample_has_projected_scores",
    note_key: str = "figure3_projection_note",
    note: str | None = None,
    scaffold_feature_order_key: str = "figure3_scaffold_feature_order",
    overwrite_existing: bool = False,
) -> Any:
    """Attach sample-level scores to every matching observation in an AnnData object."""
    for attribute in ("obs", "obsm", "uns"):
        if not hasattr(adata, attribute):
            raise TypeError(f"adata must expose an AnnData-like '{attribute}' attribute.")
    require_columns(adata.obs, [sample_column], context="AnnData.obs")
    require_columns(sample_scores, [sample_column], context="Sample score table")

    scores = sample_scores.copy()
    scores[sample_column] = scores[sample_column].astype(str)
    if scores[sample_column].duplicated().any():
        duplicates = scores.loc[
            scores[sample_column].duplicated(keep=False), sample_column
        ].tolist()
        raise ValueError(f"Sample score table contains duplicate sample IDs: {duplicates}")

    required = [str(column) for column in required_score_columns]
    require_columns(scores, required, context="Sample score table")

    score_columns = [column for column in scores.columns if column != sample_column]
    overlap = [column for column in score_columns if column in adata.obs.columns]
    if overlap and not overwrite_existing:
        raise ValueError(
            "AnnData.obs already contains score columns. Pass overwrite_existing=True "
            f"to replace them: {overlap}"
        )

    observation_keys = adata.obs[sample_column].astype(str)
    lookup = scores.set_index(sample_column)
    missing_samples = sorted(set(observation_keys) - set(lookup.index))
    if missing_samples:
        raise ValueError(
            f"No projected sample scores are available for sample IDs: {missing_samples}"
        )

    aligned = lookup.reindex(observation_keys.to_numpy())
    aligned.index = adata.obs.index
    obs_out = adata.obs.copy()
    for column in score_columns:
        obs_out[column] = aligned[column].to_numpy()

    for column in required:
        missing_mask = obs_out[column].isna()
        if missing_mask.any():
            bad = (
                obs_out.loc[missing_mask, sample_column]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Projected column {column} contains missing values for sample IDs: {bad}"
            )

    obs_out[projected_flag_column] = True
    adata.obs = obs_out
    adata.obsm[coordinate_obsm_key] = obs_out[list(coordinate_columns)].to_numpy(
        dtype=float
    )
    adata.obsm[scaffold_obsm_key] = obs_out[list(scaffold_columns)].to_numpy(
        dtype=float
    )
    if note is not None:
        adata.uns[note_key] = str(note)
    adata.uns[scaffold_feature_order_key] = [str(value) for value in scaffold_columns]
    return adata
