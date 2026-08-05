from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_switch_rate_sensitivity_figure(
    switch_rates: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (6.0, 4.0),
    title: str = "DX→REL switch-rate sensitivity",
) -> plt.Figure:
    """Plot switch rate against malignant-cell eligibility threshold."""
    required = ["threshold", "switch_rate"]
    missing = [column for column in required if column not in switch_rates.columns]
    if missing:
        raise ValueError(f"Switch-rate table missing required columns: {missing}")

    figure, axis = plt.subplots(figsize=figsize)
    ordered = switch_rates.sort_values("threshold")
    axis.plot(ordered["threshold"], ordered["switch_rate"], marker="o")
    axis.set_xlabel("Malignant-cell threshold")
    axis.set_ylabel("DX→REL switch rate")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def create_branch_count_sensitivity_figure(
    counts_by_timepoint: pd.DataFrame,
    *,
    branch_labels: Sequence[str] = ("B1", "B2", "B3"),
    diagnosis_timepoint: str = "DX",
    relapse_timepoint: str = "REL",
    figsize: tuple[float, float] = (10.0, 4.0),
    title: str = "Projected branch counts by threshold",
) -> plt.Figure:
    """Plot stacked diagnosis and relapse branch counts across thresholds."""
    required = ["threshold", "timepoint", *[str(x) for x in branch_labels]]
    missing = [
        column for column in required if column not in counts_by_timepoint.columns
    ]
    if missing:
        raise ValueError(f"Branch-count table missing required columns: {missing}")

    selected_rows: list[dict] = []
    for threshold in counts_by_timepoint["threshold"].drop_duplicates():
        for timepoint in (diagnosis_timepoint, relapse_timepoint):
            subset = counts_by_timepoint[
                (counts_by_timepoint["threshold"] == threshold)
                & (counts_by_timepoint["timepoint"] == timepoint)
            ]
            if len(subset) == 0:
                continue
            selected_rows.append(subset.iloc[0].to_dict())
    plot_table = pd.DataFrame(selected_rows)

    figure, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for axis, timepoint in zip(
        axes, (diagnosis_timepoint, relapse_timepoint), strict=True
    ):
        subset = plot_table[plot_table["timepoint"] == timepoint].sort_values(
            "threshold"
        )
        x = np.arange(len(subset))
        bottom = np.zeros(len(subset), dtype=float)
        for label in branch_labels:
            values = subset[str(label)].to_numpy(dtype=float)
            axis.bar(x, values, bottom=bottom, label=str(label))
            bottom = bottom + values
        axis.set_xticks(x)
        axis.set_xticklabels(subset["threshold"].astype(str))
        axis.set_xlabel("Threshold")
        axis.set_title(str(timepoint))
    axes[0].set_ylabel("Eligible projected samples")
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    return figure
