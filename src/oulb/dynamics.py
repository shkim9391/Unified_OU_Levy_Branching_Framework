from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CentroidColumnSpec:
    """Column names used by a sample-centroid table."""

    patient: str = "patient_id"
    sample: str = "sample_id"
    timepoint: str = "clinical_timepoint_coarse"
    x: str = "x2d"
    y: str = "y2d"
    n_cells: str = "n_cells"
    branch: str = "branch_id_dominant"
    ecotype: str = "ecotype_dominant"
    regulatory_score: str = "reg_program_score_median"
    disease_subgroup: str = "disease_subgroup"


@dataclass(frozen=True)
class IntervalDefinition:
    """A requested start-to-end interval and optional preceding timepoint."""

    name: str
    start: str
    end: str
    prior: str | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        start = str(self.start).strip()
        end = str(self.end).strip()
        prior = None if self.prior is None else str(self.prior).strip()
        if not name or not start or not end:
            raise ValueError("Interval name, start, and end must be non-empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "prior", prior)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between equally shaped numeric vectors."""
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError(f"Vector shape mismatch: {left.shape} versus {right.shape}")
    return float(np.sqrt(np.sum((left - right) ** 2)))


def directional_discontinuity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return ``1 - cosine_similarity`` for two successive movement vectors."""
    first = np.asarray(v1, dtype=float)
    second = np.asarray(v2, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"Vector shape mismatch: {first.shape} versus {second.shape}")
    n1 = np.linalg.norm(first)
    n2 = np.linalg.norm(second)
    if n1 == 0.0 or n2 == 0.0:
        return np.nan
    cosine = float(np.dot(first, second) / (n1 * n2))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return 1.0 - cosine


def _optional_value(row: pd.Series, column: str, default: Any) -> Any:
    return row[column] if column in row.index else default


def build_interval_record(
    row_start: pd.Series,
    row_end: pd.Series,
    hd_columns: Sequence[str],
    *,
    columns: CentroidColumnSpec,
    interval_name: str,
    prior_row: pd.Series | None = None,
) -> dict[str, Any]:
    """Build one standardized interval record."""
    start_xy = np.asarray([row_start[columns.x], row_start[columns.y]], dtype=float)
    end_xy = np.asarray([row_end[columns.x], row_end[columns.y]], dtype=float)
    start_hd = row_start[list(hd_columns)].to_numpy(dtype=float)
    end_hd = row_end[list(hd_columns)].to_numpy(dtype=float)

    branch_start = _optional_value(row_start, columns.branch, "Unknown")
    branch_end = _optional_value(row_end, columns.branch, "Unknown")

    record: dict[str, Any] = {
        "patient_id": row_start[columns.patient],
        "interval_class": str(interval_name),
        "t_start": row_start[columns.timepoint],
        "t_end": row_end[columns.timepoint],
        "sample_start": row_start[columns.sample],
        "sample_end": row_end[columns.sample],
        "n_cells_start": int(row_start[columns.n_cells]),
        "n_cells_end": int(row_end[columns.n_cells]),
        "displacement_2d": euclidean_distance(start_xy, end_xy),
        "displacement_hd": euclidean_distance(start_hd, end_hd),
        "direction_x": float(end_xy[0] - start_xy[0]),
        "direction_y": float(end_xy[1] - start_xy[1]),
        "branch_start": branch_start,
        "branch_end": branch_end,
        # String comparison intentionally preserves the validated behavior,
        # including how an explicit "Unknown" value is handled.
        "branch_switch": int(str(branch_start) != str(branch_end)),
        "ecotype_start": _optional_value(row_start, columns.ecotype, "Unknown"),
        "ecotype_end": _optional_value(row_end, columns.ecotype, "Unknown"),
        "reg_score_start": _optional_value(
            row_start, columns.regulatory_score, np.nan
        ),
        "reg_score_end": _optional_value(row_end, columns.regulatory_score, np.nan),
        "disease_subgroup": _optional_value(
            row_start, columns.disease_subgroup, "Unknown"
        ),
    }

    if prior_row is None:
        record["directional_discontinuity"] = np.nan
    else:
        prior_xy = np.asarray(
            [prior_row[columns.x], prior_row[columns.y]], dtype=float
        )
        v1 = start_xy - prior_xy
        v2 = end_xy - start_xy
        record["directional_discontinuity"] = directional_discontinuity(v1, v2)
    return record


def add_reference_tail_flag(
    intervals: pd.DataFrame,
    *,
    reference_interval: str,
    value_column: str = "displacement_hd",
    quantile: float = 0.90,
    threshold_column: str = "tail_threshold",
    flag_column: str = "tail_flag",
    inclusive: bool = False,
) -> pd.DataFrame:
    """Add a table-level upper-tail threshold and event flag.

    The default strict ``>`` comparison and NumPy linear quantile reproduce the
    validated Figure 3 implementation.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")
    required = ["interval_class", value_column]
    missing = [column for column in required if column not in intervals.columns]
    if missing:
        raise ValueError(f"Interval table missing columns required for tail flag: {missing}")

    out = intervals.copy()
    reference_values = pd.to_numeric(
        out.loc[out["interval_class"] == reference_interval, value_column],
        errors="coerce",
    ).dropna()
    all_values = pd.to_numeric(out[value_column], errors="coerce")

    if reference_values.empty:
        out[threshold_column] = np.nan
        out[flag_column] = False
        return out

    threshold = float(np.nanquantile(reference_values.to_numpy(dtype=float), quantile))
    out[threshold_column] = threshold
    if inclusive:
        out[flag_column] = all_values >= threshold
    else:
        out[flag_column] = all_values > threshold
    return out


def compute_intervals(
    centroids: pd.DataFrame,
    *,
    time_order: Mapping[str, int],
    interval_definitions: Sequence[IntervalDefinition],
    columns: CentroidColumnSpec = CentroidColumnSpec(),
    hd_prefix: str = "hd_",
    duplicate_policy: str = "error",
    allow_unordered_timepoints: bool = False,
    tail_reference_interval: str | None = None,
    tail_quantile: float = 0.90,
    tail_threshold_column: str = "tail_threshold",
    tail_flag_column: str = "tail_flag",
) -> pd.DataFrame:
    """Construct requested patient intervals from a centroid table."""
    required = [
        columns.patient,
        columns.sample,
        columns.timepoint,
        columns.x,
        columns.y,
        columns.n_cells,
    ]
    missing = [column for column in required if column not in centroids.columns]
    if missing:
        raise ValueError(f"Missing required centroid columns: {missing}")

    hd_columns = [column for column in centroids.columns if column.startswith(hd_prefix)]
    if not hd_columns:
        raise ValueError(
            f"No high-dimensional centroid columns found with prefix {hd_prefix!r}."
        )

    if duplicate_policy not in {"error", "first", "last"}:
        raise ValueError("duplicate_policy must be 'error', 'first', or 'last'.")

    table = centroids.copy()
    table["__time_order"] = table[columns.timepoint].map(dict(time_order))
    if not allow_unordered_timepoints and table["__time_order"].isna().any():
        unknown = sorted(
            table.loc[table["__time_order"].isna(), columns.timepoint]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"No ordering was supplied for timepoints: {unknown}")

    duplicate_mask = table.duplicated(
        subset=[columns.patient, columns.timepoint], keep=False
    )
    if duplicate_mask.any():
        duplicate_rows = table.loc[
            duplicate_mask,
            [columns.patient, columns.timepoint, columns.sample],
        ].copy()
        if duplicate_policy == "error":
            raise ValueError(
                "Centroid table has multiple rows for a patient-timepoint pair:\n"
                + duplicate_rows.to_string(index=False)
            )

    table = table.sort_values(
        [columns.patient, "__time_order", columns.sample],
        na_position="last",
    ).reset_index(drop=True)
    if duplicate_mask.any() and duplicate_policy in {"first", "last"}:
        table = table.drop_duplicates(
            subset=[columns.patient, columns.timepoint],
            keep=duplicate_policy,
        ).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, patient_table in table.groupby(columns.patient, sort=False, observed=True):
        records = {
            str(row[columns.timepoint]): row for _, row in patient_table.iterrows()
        }
        for definition in interval_definitions:
            if definition.start not in records or definition.end not in records:
                continue
            prior_row = None
            if definition.prior is not None:
                prior_row = records.get(definition.prior)
            rows.append(
                build_interval_record(
                    records[definition.start],
                    records[definition.end],
                    hd_columns,
                    columns=columns,
                    interval_name=definition.name,
                    prior_row=prior_row,
                )
            )

    if not rows:
        raise ValueError("No interval records were generated.")

    intervals = pd.DataFrame(rows)
    if tail_reference_interval is not None:
        intervals = add_reference_tail_flag(
            intervals,
            reference_interval=tail_reference_interval,
            value_column="displacement_hd",
            quantile=tail_quantile,
            threshold_column=tail_threshold_column,
            flag_column=tail_flag_column,
            inclusive=False,
        )
    intervals = intervals.sort_values(
        ["patient_id", "interval_class", "sample_start", "sample_end"]
    ).reset_index(drop=True)
    return intervals

# ---------------------------------------------------------------------------
# Sample-level effective dynamic summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleObservationColumnSpec:
    """Source-column names for repeated observations carrying sample scores."""

    sample: str = "sample_id"
    patient: str = "patient_id"
    timepoint: str = "clinical_timepoint_coarse"
    branch: str = "branch_id"
    branch_max_probability: str = "branch_maxprob"
    branch_entropy: str = "branch_entropy"
    ecotype: str = "ecotype_label"


def require_columns(
    table: pd.DataFrame,
    required: Sequence[str],
    *,
    context: str = "table",
) -> None:
    """Raise a clear error when required columns are absent."""
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def dominant_value(series: pd.Series, *, default: str = "Unknown") -> str:
    """Most frequent non-missing string, preserving legacy tie behavior."""
    values = series.dropna().astype(str)
    if values.empty:
        return default
    return str(values.value_counts().idxmax())


def dominant_fraction(series: pd.Series) -> float:
    """Fraction occupied by the dominant non-missing string value."""
    values = series.dropna().astype(str)
    if values.empty:
        return np.nan
    proportions = values.value_counts(normalize=True)
    return float(proportions.iloc[0])


def robust_zscore(series: pd.Series) -> pd.Series:
    """Median/MAD robust z-score used by the validated Figure 4 workflow."""
    numeric = pd.to_numeric(series, errors="coerce")
    median = np.nanmedian(numeric)
    mad = np.nanmedian(np.abs(numeric - median))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(numeric)), index=series.index, dtype=float)
    return 0.6745 * (numeric - median) / mad


def collapse_observations_to_sample_table(
    observations: pd.DataFrame,
    *,
    scaffold_columns: Sequence[str],
    columns: SampleObservationColumnSpec = SampleObservationColumnSpec(),
    coordinate_columns: Sequence[str] = ("PC1", "PC2"),
) -> pd.DataFrame:
    """Collapse repeated cell-level observations to one row per sample.

    Numeric sample-replicated values are summarized by the median; categorical
    annotations are summarized by their dominant value. These choices reproduce
    the validated pediatric-leukemia implementation.
    """
    scaffold_columns = tuple(str(column) for column in scaffold_columns)
    if not scaffold_columns:
        raise ValueError("At least one scaffold column is required.")
    required = [
        columns.sample,
        columns.patient,
        columns.timepoint,
        columns.branch,
        columns.branch_max_probability,
        columns.branch_entropy,
        columns.ecotype,
        *scaffold_columns,
    ]
    require_columns(observations, required, context="Observation table")

    rows: list[dict[str, Any]] = []
    for sample_id, sample_table in observations.groupby(columns.sample, sort=False, observed=False):
        record: dict[str, Any] = {
            "sample_id": str(sample_id),
            "patient_id": dominant_value(sample_table[columns.patient]),
            "clinical_timepoint_coarse": dominant_value(
                sample_table[columns.timepoint]
            ),
            "n_cells": int(sample_table.shape[0]),
            "branch_id_dominant": dominant_value(sample_table[columns.branch]),
            "branch_id_dominant_frac": dominant_fraction(
                sample_table[columns.branch]
            ),
            "ecotype_label": dominant_value(sample_table[columns.ecotype]),
            "branch_maxprob": float(
                np.nanmedian(
                    pd.to_numeric(
                        sample_table[columns.branch_max_probability], errors="coerce"
                    )
                )
            ),
            "branch_entropy": float(
                np.nanmedian(
                    pd.to_numeric(sample_table[columns.branch_entropy], errors="coerce")
                )
            ),
        }
        for column in scaffold_columns:
            record[column] = float(
                np.nanmedian(pd.to_numeric(sample_table[column], errors="coerce"))
            )
        for column in coordinate_columns:
            output_name = str(column)
            if column in sample_table.columns:
                record[output_name] = float(
                    np.nanmedian(
                        pd.to_numeric(sample_table[column], errors="coerce")
                    )
                )
            else:
                record[output_name] = np.nan
        rows.append(record)

    if not rows:
        raise ValueError("No sample-level records could be constructed.")
    output = pd.DataFrame(rows)
    if output["sample_id"].duplicated().any():
        duplicates = output.loc[
            output["sample_id"].duplicated(), "sample_id"
        ].tolist()
        raise ValueError(f"Duplicate sample records were generated: {duplicates}")
    return output


def compute_reference_attractor(
    sample_table: pd.DataFrame,
    *,
    scaffold_columns: Sequence[str],
    timepoint_column: str = "clinical_timepoint_coarse",
    reference_timepoint: str = "DX",
    phase_column: str | None = None,
    reference_phase: str | None = None,
) -> pd.Series:
    """Coordinate-wise median attractor for a designated reference phase.

    ``phase_column`` and ``reference_phase`` are accepted as descriptive aliases
    for ``timepoint_column`` and ``reference_timepoint``.
    """
    if phase_column is not None:
        timepoint_column = str(phase_column)
    if reference_phase is not None:
        reference_timepoint = str(reference_phase)
    scaffold_columns = tuple(str(column) for column in scaffold_columns)
    require_columns(
        sample_table,
        [timepoint_column, *scaffold_columns],
        context="Sample table",
    )
    reference = sample_table[
        sample_table[timepoint_column].astype(str) == str(reference_timepoint)
    ].copy()
    if reference.empty:
        raise ValueError(
            f"No {reference_timepoint!r} samples found; cannot compute reference attractor."
        )
    attractor = reference[list(scaffold_columns)].median(axis=0)
    attractor.name = f"{str(reference_timepoint).lower()}_attractor"
    return attractor


def add_effective_dynamic_parameters(
    sample_table: pd.DataFrame,
    reference_attractor: pd.Series,
    *,
    scaffold_columns: Sequence[str],
    main_patients: Collection[str] | None = None,
    flagged_patients: Collection[str] = (),
    min_main_cells: int = 50,
    n_main_states: int = 4,
    phase_order: Mapping[str, int] | None = None,
    # Descriptive compatibility aliases used by application/tests.
    max_branch_entropy: float | None = None,
    main_patient_ids: Collection[str] | None = None,
    qc_flagged_patient_ids: Collection[str] | None = None,
    minimum_main_cells: int | None = None,
) -> pd.DataFrame:
    """Add validated sample-level effective dynamic summaries.

    These quantities are descriptive proxies rather than fitted parameters of a
    continuous-time OU SDE:

    - ``theta_eff`` is dominant branch probability;
    - ``sigma_eff`` is branch entropy divided by its configured maximum;
    - ``mu_shift_from_dx`` is scaffold distance from the reference attractor.
    """
    scaffold_columns = tuple(str(column) for column in scaffold_columns)
    if main_patient_ids is not None:
        if main_patients is not None and {
            str(value) for value in main_patients
        } != {str(value) for value in main_patient_ids}:
            raise ValueError("Conflicting main-patient collections were supplied.")
        main_patients = main_patient_ids
    if main_patients is None:
        main_patients = ()
    if qc_flagged_patient_ids is not None:
        if flagged_patients and {
            str(value) for value in flagged_patients
        } != {str(value) for value in qc_flagged_patient_ids}:
            raise ValueError("Conflicting flagged-patient collections were supplied.")
        flagged_patients = qc_flagged_patient_ids
    if minimum_main_cells is not None:
        min_main_cells = int(minimum_main_cells)

    if min_main_cells < 0:
        raise ValueError("min_main_cells must be non-negative.")
    if max_branch_entropy is None:
        if n_main_states <= 1:
            raise ValueError("n_main_states must be greater than one.")
        entropy_denominator = float(np.log(float(n_main_states)))
    else:
        entropy_denominator = float(max_branch_entropy)
        if not np.isfinite(entropy_denominator) or entropy_denominator <= 0:
            raise ValueError("max_branch_entropy must be finite and positive.")

    required = [
        "patient_id",
        "n_cells",
        "clinical_timepoint_coarse",
        "branch_maxprob",
        "branch_entropy",
        *scaffold_columns,
    ]
    require_columns(sample_table, required, context="Sample table")
    missing_attractor = [
        column for column in scaffold_columns if column not in reference_attractor.index
    ]
    if missing_attractor:
        raise ValueError(
            f"Reference attractor missing scaffold features: {missing_attractor}"
        )

    output = sample_table.copy()
    output["theta_eff"] = pd.to_numeric(
        output["branch_maxprob"], errors="coerce"
    )
    output["sigma_eff"] = (
        pd.to_numeric(output["branch_entropy"], errors="coerce")
        / entropy_denominator
    )

    attractor_vector = reference_attractor[list(scaffold_columns)].to_numpy(
        dtype=float
    )
    shifts: list[float] = []
    for _, row in output.iterrows():
        sample_vector = row[list(scaffold_columns)].to_numpy(dtype=float)
        shifts.append(euclidean_distance(sample_vector, attractor_vector))
    output["mu_shift_from_dx"] = shifts

    main_set = {str(patient) for patient in main_patients}
    flagged_set = {str(patient) for patient in flagged_patients}
    patient_strings = output["patient_id"].astype(str)
    output["is_main_analysis_patient"] = patient_strings.isin(main_set)
    output["is_qc_flagged_patient"] = patient_strings.isin(flagged_set)
    output["is_main_analysis_sample"] = (
        output["is_main_analysis_patient"]
        & (~output["is_qc_flagged_patient"])
        & (pd.to_numeric(output["n_cells"], errors="coerce") >= min_main_cells)
    )

    output["theta_eff_z"] = robust_zscore(output["theta_eff"])
    output["sigma_eff_z"] = robust_zscore(output["sigma_eff"])
    output["mu_shift_from_dx_z"] = robust_zscore(output["mu_shift_from_dx"])
    output["theta_sigma_ratio"] = output["theta_eff"] / (
        output["sigma_eff"] + 1e-6
    )

    ordering = dict(phase_order or {"DX": 0, "EOI_REM": 1, "REL": 2})
    output["phase_order"] = (
        output["clinical_timepoint_coarse"]
        .map(ordering)
        .fillna(999)
        .astype(int)
    )
    return output


def compute_sample_effective_dynamics(
    observations: pd.DataFrame,
    *,
    scaffold_columns: Sequence[str],
    main_patients: Collection[str],
    exploratory_patients: Collection[str] = (),
    reference_timepoint: str = "DX",
    minimum_main_cells: int = 50,
    main_state_count: int = 4,
    phase_order: Mapping[str, int] | None = None,
    columns: SampleObservationColumnSpec = SampleObservationColumnSpec(),
    coordinate_columns: Sequence[str] = ("PC1", "PC2"),
) -> tuple[pd.DataFrame, pd.Series]:
    """Collapse observations, estimate the attractor, and add all proxies."""
    sample_table = collapse_observations_to_sample_table(
        observations,
        scaffold_columns=scaffold_columns,
        columns=columns,
        coordinate_columns=coordinate_columns,
    )
    attractor = compute_reference_attractor(
        sample_table,
        scaffold_columns=scaffold_columns,
        reference_timepoint=reference_timepoint,
    )
    output = add_effective_dynamic_parameters(
        sample_table,
        attractor,
        scaffold_columns=scaffold_columns,
        main_patients=main_patients,
        flagged_patients=exploratory_patients,
        min_main_cells=minimum_main_cells,
        n_main_states=main_state_count,
        phase_order=phase_order,
    )
    return output, attractor


# Public aliases used by different stages of the refactor.
SampleDynamicColumnSpec = SampleObservationColumnSpec
SampleColumnSpec = SampleObservationColumnSpec
collapse_replicated_sample_scores = collapse_observations_to_sample_table
