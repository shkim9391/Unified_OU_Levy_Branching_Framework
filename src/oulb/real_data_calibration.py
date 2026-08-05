from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationSummary:
    n_intervals: int
    n_patients: int
    median_dt: float
    q25_dt: float
    q75_dt: float
    median_displacement: float
    displacement_sd: float
    displacement_q90: float
    branch_switch_fraction: float


def first_existing_column(
    table: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    """Return the first exact or canonicalized candidate present in a table."""
    canonical = {
        "".join(ch for ch in str(column).lower() if ch.isalnum()): column
        for column in table.columns
    }

    for candidate in candidates:
        if candidate in table.columns:
            return candidate

    for candidate in candidates:
        key = "".join(ch for ch in str(candidate).lower() if ch.isalnum())
        if key in canonical:
            return canonical[key]

    if required:
        raise ValueError(
            f"Could not resolve any of {list(candidates)} from "
            f"columns {list(table.columns)}"
        )
    return None


def robust_positive(values: Iterable[float]) -> np.ndarray:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    return array[np.isfinite(array) & (array > 0)]


def summarize_empirical_intervals(
    interval_table: pd.DataFrame,
    *,
    patient_column: str,
    dt_column: str,
    displacement_column: str,
    branch_switch: np.ndarray | None = None,
) -> CalibrationSummary:
    dt = robust_positive(interval_table[dt_column])
    displacement = pd.to_numeric(
        interval_table[displacement_column], errors="coerce"
    ).to_numpy(float)
    displacement = displacement[np.isfinite(displacement) & (displacement >= 0)]

    if dt.size == 0:
        raise ValueError("No positive finite interval durations were found.")
    if displacement.size == 0:
        raise ValueError("No finite nonnegative displacements were found.")

    patients = interval_table[patient_column].astype(str)
    switch_fraction = (
        float(np.mean(np.asarray(branch_switch, dtype=float)))
        if branch_switch is not None and len(branch_switch)
        else np.nan
    )

    return CalibrationSummary(
        n_intervals=int(min(len(dt), len(displacement))),
        n_patients=int(patients.nunique()),
        median_dt=float(np.median(dt)),
        q25_dt=float(np.quantile(dt, 0.25)),
        q75_dt=float(np.quantile(dt, 0.75)),
        median_displacement=float(np.median(displacement)),
        displacement_sd=float(np.std(displacement, ddof=1))
        if displacement.size > 1
        else 0.0,
        displacement_q90=float(np.quantile(displacement, 0.90)),
        branch_switch_fraction=switch_fraction,
    )


def empirical_schedule(
    interval_durations: np.ndarray,
    *,
    rng: np.random.Generator,
    n_intervals: int,
) -> np.ndarray:
    """Sample an empirical-style irregular observation schedule."""
    pool = robust_positive(interval_durations)
    if pool.size == 0:
        raise ValueError("Interval-duration pool is empty.")

    sampled = rng.choice(pool, size=int(n_intervals), replace=True)
    return np.concatenate([[0.0], np.cumsum(sampled)])


def empirical_interval_count(
    counts: np.ndarray,
    *,
    rng: np.random.Generator,
    minimum: int = 1,
) -> int:
    valid = np.asarray(counts, dtype=int)
    valid = valid[valid >= minimum]
    if valid.size == 0:
        return minimum
    return int(rng.choice(valid))


def transition_statistics(
    times: np.ndarray,
    states: np.ndarray,
    *,
    branches: np.ndarray | None = None,
    events: Sequence[Mapping] | None = None,
) -> dict[str, float]:
    """Return summary statistics shared by empirical and simulated intervals."""
    t = np.asarray(times, dtype=float)
    x = np.asarray(states, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]

    displacement = np.abs(np.diff(x))
    total = float(abs(x[-1] - x[0]))
    maximum = float(np.max(displacement)) if displacement.size else 0.0
    median = float(np.median(displacement)) if displacement.size else 0.0
    q90 = float(np.quantile(displacement, 0.90)) if displacement.size else 0.0

    branch_switch_fraction = np.nan
    if branches is not None:
        branch_array = np.asarray(branches, dtype=int)
        if branch_array.size > 1:
            branch_switch_fraction = float(np.mean(np.diff(branch_array) != 0))

    jump_fraction = np.nan
    if events is not None and displacement.size:
        jump_count = sum(
            str(event.get("event", "")).lower() == "jump"
            for event in events
        )
        jump_fraction = float(jump_count / displacement.size)

    return {
        "n_observations": int(len(t)),
        "followup": float(t[-1] - t[0]),
        "total_displacement": total,
        "max_interval_displacement": maximum,
        "median_interval_displacement": median,
        "q90_interval_displacement": q90,
        "branch_switch_fraction": branch_switch_fraction,
        "jump_fraction": jump_fraction,
    }


def ratio_fidelity(
    observed: Mapping[str, float],
    simulated: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Summarize simulated-to-observed ratios with empirical quantiles."""
    rows = []

    for metric in metrics:
        observed_value = float(observed[metric])
        values = pd.to_numeric(simulated[metric], errors="coerce").dropna().to_numpy(float)

        if not np.isfinite(observed_value) or values.size == 0:
            continue

        if np.isclose(observed_value, 0.0):
            ratio = np.full(values.size, np.nan)
        else:
            ratio = values / observed_value

        finite_ratio = ratio[np.isfinite(ratio)]
        rows.append(
            {
                "metric": metric,
                "observed_value": observed_value,
                "simulated_median": float(np.median(values)),
                "simulated_q025": float(np.quantile(values, 0.025)),
                "simulated_q975": float(np.quantile(values, 0.975)),
                "ratio_median": float(np.median(finite_ratio))
                if finite_ratio.size
                else np.nan,
                "ratio_q025": float(np.quantile(finite_ratio, 0.025))
                if finite_ratio.size
                else np.nan,
                "ratio_q975": float(np.quantile(finite_ratio, 0.975))
                if finite_ratio.size
                else np.nan,
            }
        )

    return pd.DataFrame(rows)
