from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .evaluation import FitResult, ModelComparisonArtifacts


COLOR_STABLE = "#4C9ED9"
COLOR_SWITCHING = "#F28E2B"
COLOR_GAUSSIAN = "#6E6E6E"
COLOR_HEAVY = "#B22222"
COLOR_BEST = "#B22222"
COLOR_OTHER = "#9A9A9A"
COLOR_REF = "#7A7A7A"

PANEL_FONT_SIZE = 18
TITLE_FONT_SIZE = 15
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 11
ANNOT_FONT_SIZE = 10

DEFAULT_FIGSIZE = (16.0, 10.0)
DEFAULT_DPI = 600


def panel_label(ax: plt.Axes, label: str) -> None:
    """Place a manuscript panel label in axes coordinates."""
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_FONT_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_model_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
) -> None:
    """Draw one model node in the nested-model ladder."""
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.3,
        edgecolor=COLOR_REF,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + 0.03,
        xy[1] + height - 0.08,
        title,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        xy[0] + 0.03,
        xy[1] + height - 0.18,
        body,
        fontsize=11,
        ha="left",
        va="top",
    )


def plot_model_ladder(
    ax: plt.Axes,
    *,
    title: str = "Nested Gaussian and heavy-tail model ladder",
) -> None:
    """Plot the four-model pooled/group-aware, Gaussian/heavy-tail ladder."""
    ax.set_axis_off()
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)

    draw_model_box(
        ax,
        (0.03, 0.56),
        0.42,
        0.32,
        "M0  Gaussian pooled",
        r"$\Delta_i \sim \mathcal{N}(\mu,\sigma^2)$",
    )
    draw_model_box(
        ax,
        (0.53, 0.56),
        0.42,
        0.32,
        "M1  Gaussian branch-aware",
        r"$\Delta_i \sim \mathcal{N}(\mu_g,\sigma_g^2)$"
        + "\n"
        + r"$g \in \{\mathrm{Stable, Switching}\}$",
    )
    draw_model_box(
        ax,
        (0.03, 0.12),
        0.42,
        0.32,
        "M2  Student-t pooled",
        r"$\Delta_i \sim t_{\nu}(\mu,\sigma)$",
    )
    draw_model_box(
        ax,
        (0.53, 0.12),
        0.42,
        0.32,
        "M3  Student-t branch-aware",
        r"$\Delta_i \sim t_{\nu}(\mu_g,\sigma_g)$"
        + "\n"
        + r"$g \in \{\mathrm{Stable, Switching}\}$",
    )

    ax.text(
        0.49,
        0.735,
        "add branch structure",
        fontsize=11,
        ha="center",
        va="center",
        color=COLOR_REF,
    )
    ax.text(
        0.49,
        0.295,
        "add branch structure",
        fontsize=11,
        ha="center",
        va="center",
        color=COLOR_REF,
    )
    ax.text(
        0.24,
        0.49,
        "add heavy tails",
        fontsize=11,
        ha="center",
        va="center",
        color=COLOR_REF,
    )
    ax.text(
        0.75,
        0.49,
        "add heavy tails",
        fontsize=11,
        ha="center",
        va="center",
        color=COLOR_REF,
    )

    ax.annotate(
        "",
        xy=(0.53, 0.72),
        xytext=(0.45, 0.72),
        arrowprops=dict(arrowstyle="->", color=COLOR_REF, lw=1.3),
    )
    ax.annotate(
        "",
        xy=(0.53, 0.28),
        xytext=(0.45, 0.28),
        arrowprops=dict(arrowstyle="->", color=COLOR_REF, lw=1.3),
    )
    ax.annotate(
        "",
        xy=(0.24, 0.44),
        xytext=(0.24, 0.56),
        arrowprops=dict(arrowstyle="->", color=COLOR_REF, lw=1.3),
    )
    ax.annotate(
        "",
        xy=(0.75, 0.44),
        xytext=(0.75, 0.56),
        arrowprops=dict(arrowstyle="->", color=COLOR_REF, lw=1.3),
    )


def plot_model_comparison(
    ax: plt.Axes,
    comparison: pd.DataFrame,
    *,
    title: str = "Branch-aware heavy-tail model provides the best predictive fit",
) -> None:
    """Plot delta-AICc bars from a model-comparison table."""
    plot_table = comparison.sort_values("delta_aicc", ascending=True).reset_index(
        drop=True
    )
    y = np.arange(len(plot_table))[::-1]
    colors = [
        COLOR_BEST if delta == 0 else COLOR_OTHER
        for delta in plot_table["delta_aicc"]
    ]

    ax.barh(y, plot_table["delta_aicc"], color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_table["model_label"], fontsize=TICK_FONT_SIZE)
    ax.set_xlabel(r"$\Delta$AICc  (lower is better)", fontsize=LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)

    for yi, (_, row) in zip(y, plot_table.iterrows()):
        ax.text(
            row["delta_aicc"] + 0.08,
            yi,
            f"AICc={row['aicc']:.2f}",
            fontsize=10,
            ha="left",
            va="center",
        )

    ax.axvline(0, color=COLOR_REF, linestyle="--", linewidth=1.2)
    ax.set_xlim(0, float(plot_table["delta_aicc"].max()) + 1.3)


def plot_tail_fit(
    ax: plt.Axes,
    tail_grid: pd.DataFrame,
    *,
    best_gaussian: FitResult,
    best_student_t: FitResult,
    n_switching: int,
    metric_label: str = "DX→REL total displacement (6D)",
    title: str = "Switching-group tail fit highlights the heavy-tail comparison",
) -> None:
    """Plot empirical and fitted Switching-group survival curves."""
    if n_switching <= 0:
        raise ValueError("n_switching must be positive.")
    floor = 0.5 / n_switching
    x_grid = tail_grid["x"].to_numpy(dtype=float)

    ax.plot(
        x_grid,
        np.maximum(
            tail_grid["observed_switching_survival"].to_numpy(dtype=float), floor
        ),
        color="black",
        linewidth=2.2,
        label="Observed Switching survival",
    )
    ax.plot(
        x_grid,
        np.maximum(
            tail_grid["switching_gaussian_survival"].to_numpy(dtype=float), floor
        ),
        color=COLOR_GAUSSIAN,
        linewidth=2.0,
        linestyle="--",
        label=f"Switching Gaussian fit: {best_gaussian.model_label}",
    )
    ax.plot(
        x_grid,
        np.maximum(
            tail_grid["switching_student_t_survival"].to_numpy(dtype=float), floor
        ),
        color=COLOR_HEAVY,
        linewidth=2.0,
        label=f"Switching heavy-tail fit: {best_student_t.model_label}",
    )

    ax.set_yscale("log")
    ax.set_xlabel(metric_label, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(
        r"Switching survival  $P(\Delta \geq x)$", fontsize=LABEL_FONT_SIZE
    )
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.legend(frameon=False, fontsize=9, loc="upper right")


def plot_casewise_gain(
    ax: plt.Axes,
    casewise: pd.DataFrame,
    *,
    top_annotate: int,
    title: str = "Model improvement is concentrated in extreme displacement cases",
) -> None:
    """Plot case-wise M3-minus-M1 log-likelihood gain."""
    ranked = casewise.sort_values("delta_loglik", ascending=False).reset_index(
        drop=True
    )
    y = np.arange(len(ranked))[::-1]
    colors = ranked["stability_std"].map(
        {"Stable": COLOR_STABLE, "Switching": COLOR_SWITCHING}
    ).tolist()

    for yi, delta, color in zip(y, ranked["delta_loglik"], colors):
        ax.hlines(yi, 0.0, delta, color=COLOR_REF, linewidth=1.4, zorder=1)
        ax.scatter(
            delta,
            yi,
            s=70,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

    ax.axvline(0, color=COLOR_REF, linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(ranked["sample_std"], fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.set_xlabel(
        r"Case-wise log-likelihood gain  "
        r"$\log p(\Delta_i \mid M3)-\log p(\Delta_i \mid M1)$",
        fontsize=LABEL_FONT_SIZE,
    )
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)

    top = ranked.head(min(top_annotate, len(ranked)))
    for _, row in top.iterrows():
        position = int(
            ranked.index[ranked["sample_std"] == row["sample_std"]][0]
        )
        yi = len(ranked) - 1 - position
        ax.annotate(
            row["sample_std"],
            xy=(row["delta_loglik"], yi),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=ANNOT_FONT_SIZE,
            ha="left",
            va="bottom",
        )

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_STABLE,
            markeredgecolor="white",
            markersize=8,
            label="Branch-continuous",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_SWITCHING,
            markeredgecolor="white",
            markersize=8,
            label="Branch-switching",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        frameon=False,
        fontsize=10,
        loc="lower right",
    )


def create_model_comparison_figure(
    prepared_data: pd.DataFrame,
    artifacts: ModelComparisonArtifacts,
    *,
    top_annotate: int = 5,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    panel_labels: Sequence[str] = ("E", "F", "G", "H"),
    metric_label: str = "DX→REL total displacement (6D)",
    comparison_title: str = (
        "Branch-aware heavy-tail model provides the best predictive fit"
    ),
    tail_title: str = (
        "Switching-group tail fit highlights the heavy-tail comparison"
    ),
    case_title: str = (
        "Model improvement is concentrated in extreme displacement cases"
    ),
    x_label: str | None = None,
) -> plt.Figure:
    """Create the four-panel model-comparison figure without writing files."""
    if len(panel_labels) != 4:
        raise ValueError("Exactly four panel labels are required.")
    n_switching = int(
        (prepared_data["stability_std"] == "Switching").sum()
    )
    best_gaussian = artifacts.results[artifacts.best_gaussian_id]
    best_student = artifacts.results[artifacts.best_student_t_id]

    figure, axes = plt.subplots(2, 2, figsize=figsize)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()
    plot_model_ladder(ax_a)
    plot_model_comparison(
        ax_b,
        artifacts.comparison_table,
        title=comparison_title,
    )
    plot_tail_fit(
        ax_c,
        artifacts.tail_fit_grid,
        best_gaussian=best_gaussian,
        best_student_t=best_student,
        n_switching=n_switching,
        metric_label=metric_label if x_label is None else str(x_label),
        title=tail_title,
    )
    plot_casewise_gain(
        ax_d,
        artifacts.casewise_loglik_gain,
        top_annotate=top_annotate,
        title=case_title,
    )

    for label, ax in zip(panel_labels, [ax_a, ax_b, ax_c, ax_d]):
        panel_label(ax, str(label))
    figure.tight_layout()
    return figure
