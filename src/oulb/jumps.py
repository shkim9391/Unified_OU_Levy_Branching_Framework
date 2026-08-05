from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .dynamics import require_columns, robust_zscore


@dataclass(frozen=True)
class JumpCandidateSpec:
    """Column and scoring definition for an operational jump ranking."""

    interval_column: str = "interval_class"
    interval_value: str = "DX_to_REL"
    displacement_hd_column: str = "displacement_hd"
    displacement_2d_column: str = "displacement_2d"
    branch_switch_column: str = "branch_switch"
    branch_weight: float = 0.50
    switching_label: str = "Branch-switching"
    continuous_label: str = "Branch-continuous"

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.branch_weight)):
            raise ValueError("branch_weight must be finite.")
        for name in (
            "interval_column",
            "interval_value",
            "displacement_hd_column",
            "displacement_2d_column",
            "branch_switch_column",
            "switching_label",
            "continuous_label",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")


def compute_jump_candidates(
    intervals: pd.DataFrame,
    *,
    spec: JumpCandidateSpec = JumpCandidateSpec(),
    required_identity_columns: Sequence[str] = (
        "patient_id",
        "sample_start",
        "sample_end",
        "branch_start",
        "branch_end",
    ),
) -> pd.DataFrame:
    """Rank interval-level operational jump candidates.

    The validated score is

    ``robust_z(displacement_hd) + branch_weight * branch_switch``.

    The robust z-score is based on the median and MAD.  Both high-dimensional
    and two-dimensional displacement z-scores are retained in the output, while
    only the high-dimensional value contributes to the validated jump score.
    """

    required = [
        spec.interval_column,
        spec.displacement_hd_column,
        spec.displacement_2d_column,
        spec.branch_switch_column,
        *[str(column) for column in required_identity_columns],
    ]
    require_columns(intervals, required, context="Interval table")

    output = intervals[
        intervals[spec.interval_column].astype(str) == str(spec.interval_value)
    ].copy()
    if output.empty:
        raise ValueError(
            f"No {spec.interval_value!r} intervals found in column "
            f"{spec.interval_column!r}."
        )

    output[spec.displacement_hd_column] = pd.to_numeric(
        output[spec.displacement_hd_column], errors="coerce"
    )
    output[spec.displacement_2d_column] = pd.to_numeric(
        output[spec.displacement_2d_column], errors="coerce"
    )
    output[spec.branch_switch_column] = (
        pd.to_numeric(output[spec.branch_switch_column], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    output["z_displacement_hd"] = robust_zscore(
        output[spec.displacement_hd_column]
    )
    output["z_displacement_2d"] = robust_zscore(
        output[spec.displacement_2d_column]
    )
    output["jump_score"] = output["z_displacement_hd"].fillna(0.0) + float(
        spec.branch_weight
    ) * output[spec.branch_switch_column].astype(float)
    output["jump_class"] = np.where(
        output[spec.branch_switch_column] == 1,
        spec.switching_label,
        spec.continuous_label,
    )

    output = output.sort_values(
        ["jump_score", spec.displacement_hd_column],
        ascending=[False, False],
    ).reset_index(drop=True)
    output["jump_rank"] = np.arange(1, output.shape[0] + 1)

    front = [
        "jump_rank",
        "patient_id",
        spec.interval_column,
        "sample_start",
        "sample_end",
        spec.displacement_hd_column,
        spec.displacement_2d_column,
        "branch_start",
        "branch_end",
        spec.branch_switch_column,
        "jump_class",
        "z_displacement_hd",
        "jump_score",
    ]
    remaining = [column for column in output.columns if column not in front]
    return output[front + remaining]
