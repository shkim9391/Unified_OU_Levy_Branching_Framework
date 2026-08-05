from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CONFIG = Path(
    "configs/figure5_observation_robustness.toml"
)

NAVY = "#173B6C"
ORANGE = "#F28E2B"
GRAY = "#6B7280"
LIGHT_GRAY = "#E6E8EB"
DARK = "#202124"
WHITE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render Figure 5 from archived "
            "observation-robustness tables."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    return parser.parse_args()


def read_config(path: Path) -> dict:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.12,
    y: float = 1.06,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="top",
    )


def format_missingness(value: float) -> str:
    return f"{value:g}"


def heatmap_text_color(
    value: float,
    bound: float,
) -> str:
    if bound <= 0:
        return DARK

    normalized = abs(value) / bound
    return WHITE if normalized >= 0.55 else DARK


def facet_heatmaps(
    fig: plt.Figure,
    parent_spec,
    summary: pd.DataFrame,
    *,
    value_column: str,
    panel: str,
    title: str,
    colorbar_label: str = "Naïve − measurement-aware",
    colorbar_pad: float = 0.012,
) -> None:
    patterns = sorted(
        summary["interval_pattern"]
        .astype(str)
        .unique()
    )
    missing_levels = sorted(
        summary["missing_probability"].unique()
    )

    subgrid = parent_spec.subgridspec(
        nrows=len(patterns),
        ncols=len(missing_levels),
        hspace=0.38,
        wspace=0.24,
    )

    values = pd.to_numeric(
        summary[value_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        raise ValueError(
            f"No finite values found for {value_column}."
        )

    bound = max(
        abs(float(np.min(finite_values))),
        abs(float(np.max(finite_values))),
        1e-12,
    )

    axes: list[plt.Axes] = []
    image = None

    for row_index, pattern in enumerate(patterns):
        for column_index, missingness in enumerate(
            missing_levels
        ):
            ax = fig.add_subplot(
                subgrid[row_index, column_index]
            )
            axes.append(ax)

            subset = summary[
                (
                    summary["interval_pattern"]
                    .astype(str)
                    == pattern
                )
                & np.isclose(
                    summary["missing_probability"],
                    missingness,
                )
            ]

            pivot = (
                subset.pivot(
                    index="noise_sd",
                    columns="n_observations",
                    values=value_column,
                )
                .sort_index(axis=0)
                .sort_index(axis=1)
            )

            matrix = pivot.to_numpy(dtype=float)

            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
                vmin=-bound,
                vmax=bound,
            )

            ax.set_xticks(
                np.arange(len(pivot.columns))
            )
            ax.set_xticklabels(
                [
                    str(int(value))
                    for value in pivot.columns
                ],
                fontsize=7.6,
            )

            ax.set_yticks(
                np.arange(len(pivot.index))
            )
            ax.set_yticklabels(
                [
                    f"{value:g}"
                    for value in pivot.index
                ],
                fontsize=7.6,
            )

            for matrix_row in range(matrix.shape[0]):
                for matrix_column in range(
                    matrix.shape[1]
                ):
                    cell_value = matrix[
                        matrix_row,
                        matrix_column,
                    ]

                    if not np.isfinite(cell_value):
                        continue

                    ax.text(
                        matrix_column,
                        matrix_row,
                        f"{cell_value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color=heatmap_text_color(
                            cell_value,
                            bound,
                        ),
                    )

            if row_index == len(patterns) - 1:
                ax.set_xlabel(
                    "Observation count",
                    fontsize=8.4,
                )

            if column_index == 0:
                ax.set_ylabel(
                    f"{pattern}\nNoise SD",
                    fontsize=8.4,
                )

            ax.set_title(
                (
                    "Missing="
                    f"{format_missingness(missingness)}"
                ),
                fontsize=8.8,
                pad=3,
            )

            ax.tick_params(
                axis="both",
                length=2.5,
            )

    if image is None or not axes:
        raise RuntimeError(
            f"No heatmaps were rendered for {value_column}."
        )

    first_axis = axes[0]

    first_axis.text(
        -0.12,
        1.30,
        panel,
        transform=first_axis.transAxes,
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="top",
    )

    first_axis.text(
        0.0,
        1.30,
        title,
        transform=first_axis.transAxes,
        fontsize=13.2,
        fontweight="bold",
        color=DARK,
        ha="left",
        va="top",
    )

    colorbar = fig.colorbar(
        image,
        ax=axes,
        fraction=0.015,
        pad=colorbar_pad,
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=8.2,
        labelpad=6,
    )

    colorbar.ax.tick_params(
        labelsize=7.5,
    )


def paired_condition_plot(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    naive_column: str,
    aware_column: str,
    panel: str,
    title: str,
    ylabel: str,
    clip_quantile: float,
) -> None:
    ordered = (
        summary.sort_values(
            aware_column,
            ascending=True,
        )
        .reset_index(drop=True)
    )

    x_positions = np.arange(len(ordered))

    naive_values = pd.to_numeric(
        ordered[naive_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    aware_values = pd.to_numeric(
        ordered[aware_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    for index in range(len(ordered)):
        if not (
            np.isfinite(naive_values[index])
            and np.isfinite(aware_values[index])
        ):
            continue

        ax.plot(
            [x_positions[index], x_positions[index]],
            [naive_values[index], aware_values[index]],
            color=LIGHT_GRAY,
            linewidth=0.7,
            zorder=1,
        )

    ax.scatter(
        x_positions,
        naive_values,
        s=15,
        facecolor=WHITE,
        edgecolor=GRAY,
        linewidth=0.7,
        label="Naïve",
        zorder=2,
    )

    ax.scatter(
        x_positions,
        aware_values,
        s=18,
        facecolor=ORANGE,
        edgecolor=NAVY,
        linewidth=0.7,
        label="Measurement-aware",
        zorder=3,
    )

    finite_values = np.concatenate(
        [
            naive_values[np.isfinite(naive_values)],
            aware_values[np.isfinite(aware_values)],
        ]
    )

    if finite_values.size:
        upper_limit = float(
            np.quantile(
                finite_values,
                clip_quantile,
            )
        )
        upper_limit = max(
            upper_limit * 1.08,
            1e-6,
        )
        ax.set_ylim(0.0, upper_limit)

    panel_label(
        ax,
        panel,
        x=-0.11,
    )

    ax.set_title(
        title,
        fontsize=12.8,
        fontweight="bold",
        color=DARK,
        pad=8,
    )
    ax.set_xlabel(
        (
            "Stress-test conditions ordered by "
            "measurement-aware performance"
        ),
        fontsize=8.8,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=9.5,
    )
    ax.set_xticks([])

    ax.grid(
        axis="y",
        color=LIGHT_GRAY,
        linewidth=0.7,
        alpha=0.85,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        fontsize=8.8,
        loc="upper left",
    )


def overall_improvement_plot(
    ax: plt.Axes,
    overall: pd.DataFrame,
) -> None:
    specifications = [
        (
            "theta",
            "rmse",
            r"$\theta$ RMSE",
        ),
        (
            "sigma",
            "rmse",
            r"$\sigma$ RMSE",
        ),
        (
            "theta",
            "boundary_fraction",
            r"$\theta$ boundary hits",
        ),
    ]

    labels: list[str] = []
    relative_improvements: list[float] = []

    for parameter, metric, label in specifications:
        naive = overall[
            (overall["parameter"] == parameter)
            & (overall["metric"] == metric)
            & (overall["estimator"] == "naive")
        ]

        aware = overall[
            (overall["parameter"] == parameter)
            & (overall["metric"] == metric)
            & (
                overall["estimator"]
                == "measurement_aware"
            )
        ]

        if len(naive) != 1 or len(aware) != 1:
            raise ValueError(
                "Missing or duplicated overall summary row "
                f"for {parameter}, {metric}."
            )

        naive_value = float(
            naive.iloc[0]["mean"]
        )
        aware_value = float(
            aware.iloc[0]["mean"]
        )

        if np.isclose(naive_value, 0.0):
            relative_improvement = np.nan
        else:
            relative_improvement = (
                100.0
                * (
                    naive_value
                    - aware_value
                )
                / naive_value
            )

        labels.append(label)
        relative_improvements.append(
            relative_improvement
        )

    y_positions = np.arange(len(labels))
    values = np.asarray(
        relative_improvements,
        dtype=float,
    )

    colors = [
        ORANGE if value >= 0 else GRAY
        for value in values
    ]

    ax.barh(
        y_positions,
        values,
        color=colors,
        edgecolor=WHITE,
        alpha=0.90,
    )

    ax.axvline(
        0.0,
        color=NAVY,
        linestyle="--",
        linewidth=1.1,
    )

    for y_position, value in zip(
        y_positions,
        values,
    ):
        if not np.isfinite(value):
            continue

        offset = 1.0 if value >= 0 else -1.0

        ax.text(
            value + offset,
            y_position,
            f"{value:.1f}%",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9.2,
            color=DARK,
        )

    panel_label(
        ax,
        "F",
        x=-0.055,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        labels,
        fontsize=9.5,
    )

    ax.set_xlabel(
        (
            "Relative improvement from measurement-aware "
            "fitting (%)"
        ),
        fontsize=9.5,
    )
    ax.set_title(
        "Overall robustness improvement",
        fontsize=12.8,
        fontweight="bold",
        color=DARK,
        pad=8,
    )

    ax.grid(
        axis="x",
        color=LIGHT_GRAY,
        linewidth=0.7,
        alpha=0.85,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    finite_values = values[np.isfinite(values)]

    if finite_values.size:
        padding = max(
            5.0,
            0.15
            * (
                float(np.max(finite_values))
                - float(np.min(finite_values))
            ),
        )

        ax.set_xlim(
            min(0.0, float(np.min(finite_values)))
            - padding,
            max(0.0, float(np.max(finite_values)))
            + padding,
        )


def validate_input_columns(
    summary: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    summary_columns = {
        "n_observations",
        "noise_sd",
        "missing_probability",
        "interval_pattern",
        "theta_rmse_improvement",
        "sigma_rmse_improvement",
        "theta_boundary_reduction",
        "theta_rmse_naive",
        "theta_rmse_measurement_aware",
        "sigma_rmse_naive",
        "sigma_rmse_measurement_aware",
    }

    overall_columns = {
        "parameter",
        "metric",
        "estimator",
        "mean",
        "ci_lower",
        "ci_upper",
    }

    missing_summary = (
        summary_columns
        - set(summary.columns)
    )
    missing_overall = (
        overall_columns
        - set(overall.columns)
    )

    if missing_summary:
        raise ValueError(
            "Figure 5 summary table is missing columns: "
            f"{sorted(missing_summary)}"
        )

    if missing_overall:
        raise ValueError(
            "Figure 5 overall table is missing columns: "
            f"{sorted(missing_overall)}"
        )


def main() -> None:
    arguments = parse_args()
    config = read_config(arguments.config)

    project_root = Path(
        config["paths"]["project_root"]
    ).expanduser()

    data_directory = (
        project_root
        / config["paths"]["output_data_dir"]
    )
    output_directory = (
        project_root
        / config["paths"]["output_figure_dir"]
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        data_directory
        / "Figure5_observation_robustness_summary.csv"
    )
    overall_path = (
        data_directory
        / "Figure5_observation_robustness_overall.csv"
    )
    metadata_path = (
        data_directory
        / "Figure5_observation_robustness_metadata.json"
    )

    for path in [
        summary_path,
        overall_path,
        metadata_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    summary = pd.read_csv(summary_path)
    overall = pd.read_csv(overall_path)

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    validate_input_columns(
        summary,
        overall,
    )

    figure_width = float(
        config["display"]["figure_width"]
    )
    configured_height = float(
        config["display"]["figure_height"]
    )

    # Four complete rows require a slightly taller canvas than
    # the original three-row version.
    figure_height = max(
        configured_height,
        15.2,
    )

    figure = plt.figure(
        figsize=(
            figure_width,
            figure_height,
        ),
        constrained_layout=False,
    )

    grid = figure.add_gridspec(
        nrows=4,
        ncols=2,
        left=0.055,
        right=0.965,
        top=0.905,
        bottom=0.070,
        height_ratios=[
            1.20,
            1.05,
            0.80,
            0.62,
        ],
        hspace=0.46,
        wspace=0.28,
    )

    facet_heatmaps(
        figure,
        grid[0, 0],
        summary,
        value_column="theta_rmse_improvement",
        panel="A",
        title=r"Improvement in $\theta$ RMSE",
        colorbar_label="Naïve − measurement-aware",
    )

    facet_heatmaps(
        figure,
        grid[0, 1],
        summary,
        value_column="sigma_rmse_improvement",
        panel="B",
        title=r"Improvement in $\sigma$ RMSE",
        colorbar_label="Naïve − measurement-aware",
    )

    facet_heatmaps(
        figure,
        grid[1, :],
        summary,
        value_column="theta_boundary_reduction",
        panel="C",
        title=(
            r"Reduction in $\theta$ "
            "boundary-hit frequency"
        ),
        colorbar_label="Reduction",
        colorbar_pad=0.018,
    )

    axis_d = figure.add_subplot(
        grid[2, 0]
    )
    paired_condition_plot(
        axis_d,
        summary,
        naive_column="theta_rmse_naive",
        aware_column=(
            "theta_rmse_measurement_aware"
        ),
        panel="D",
        title=r"Condition-level $\theta$ RMSE",
        ylabel=r"$\theta$ RMSE",
        clip_quantile=float(
            config["display"][
                "theta_clip_quantile"
            ]
        ),
    )

    axis_e = figure.add_subplot(
        grid[2, 1]
    )
    paired_condition_plot(
        axis_e,
        summary,
        naive_column="sigma_rmse_naive",
        aware_column=(
            "sigma_rmse_measurement_aware"
        ),
        panel="E",
        title=r"Condition-level $\sigma$ RMSE",
        ylabel=r"$\sigma$ RMSE",
        clip_quantile=float(
            config["display"][
                "sigma_clip_quantile"
            ]
        ),
    )

    axis_f = figure.add_subplot(
        grid[3, :]
    )
    overall_improvement_plot(
        axis_f,
        overall,
    )

#    figure.suptitle(
#        config["labels"]["title"],
#        fontsize=17.5,
#        fontweight="bold",
#        color=DARK,
#        y=0.992,
#    )

#    figure.text(
#        0.5,
#        0.965,
#        config["labels"]["subtitle"],
#        ha="center",
#        va="top",
#        fontsize=11.3,
#        color=GRAY,
#        style="italic",
#    )

    figure.text(
        0.055,
        0.935,
        (
            "Positive values favor measurement-aware fitting; "
            "negative values favor naïve fitting."
        ),
        ha="left",
        fontsize=8.8,
        color=GRAY,
    )

    figure.text(
        0.5,
        0.026,
        (
            "All condition summaries, estimator comparisons, "
            "and display selections were archived upstream "
            "before plotting."
        ),
        ha="center",
        fontsize=9.0,
        color=GRAY,
    )

    figure.text(
        1.005,
        0.50,
        (
            "Archived stress-test fits: "
            f"{metadata['archived_rows']['replicates']}"
        ),
        rotation=270,
        va="center",
        ha="right",
        fontsize=8.5,
        color=GRAY,
    )

    output_stem = (
        output_directory
        / "Figure5_observation_model_robustness"
    )

    figure.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=int(config["display"]["dpi"]),
        bbox_inches="tight",
    )

    plt.close(figure)

    for extension in [
        ".svg",
        ".pdf",
        ".png",
    ]:
        print(
            f"[SAVED] "
            f"{output_stem.with_suffix(extension)}"
        )


if __name__ == "__main__":
    main()
