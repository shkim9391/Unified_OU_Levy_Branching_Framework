from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


NAVY = "#173B6C"
BLUE = "#4C78A8"
PURPLE = "#7A5195"
ORANGE = "#F28E2B"
GREEN = "#59A14F"
RED = "#C44E52"
GRAY = "#6B7280"
LIGHT_GRAY = "#E6E8EB"
DARK = "#202124"
WHITE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/figure6_real_data_calibration.toml"),
    )
    return parser.parse_args()


def read_config(path: Path) -> dict:
    with path.expanduser().open("rb") as handle:
        return tomllib.load(handle)


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.05) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        va="top",
    )


def calibration_pipeline(ax) -> None:
    ax.set_axis_off()

    labels = [
        "Longitudinal\nleukemia samples",
        "Observed stage\ntransitions",
        "Transition architecture\nand displacement",
        "Calibrated OULB\nscales",
        "Synthetic cancer-\nevolution trajectories",
    ]

    x_positions = np.linspace(
        0.10,
        0.90,
        len(labels),
    )

    # Move the complete pipeline upward.
    box_y = 0.48
    box_height = 0.28
    box_center_y = box_y + box_height / 2

    for index, (x, label) in enumerate(
        zip(x_positions, labels)
    ):
        ax.add_patch(
            FancyBboxPatch(
                (
                    x - 0.078,
                    box_y,
                ),
                0.156,
                box_height,
                boxstyle="round,pad=0.015",
                transform=ax.transAxes,
                facecolor=WHITE,
                edgecolor=(
                    BLUE
                    if index < 3
                    else PURPLE
                ),
                linewidth=1.2,
            )
        )

        ax.text(
            x,
            box_center_y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.4,
            fontweight=(
                "bold"
                if index in {0, 4}
                else "normal"
            ),
        )

        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(
                    x_positions[index + 1] - 0.086,
                    box_center_y,
                ),
                xytext=(
                    x + 0.086,
                    box_center_y,
                ),
                xycoords=ax.transAxes,
                arrowprops=dict(
                    arrowstyle="-|>",
                    linewidth=1.4,
                    color=GRAY,
                ),
            )

    panel_label(
        ax,
        "A",
        x=-0.03,
        y=1.12,
    )

    ax.set_title(
        "Empirical calibration pipeline",
        fontsize=13.5,
        fontweight="bold",
        pad=2,
    )

    ax.text(
        0.5,
        0.32,
        (
            "Empirical quantities calibrate the observation architecture "
            "and characteristic scales;\n"
            "non-identifiable mechanistic quantities remain explicitly "
            "prespecified."
        ),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.3,
        color=GRAY,
    )


def plot_empirical_architecture(
    ax: plt.Axes,
    empirical: pd.DataFrame,
) -> None:
    counts = (
        empirical["n_intervals"]
        .value_counts()
        .sort_index()
    )

    x = np.arange(len(counts))

    bars = ax.bar(
        x,
        counts.to_numpy(),
        width=0.62,
        color=BLUE,
        alpha=0.82,
        edgecolor=WHITE,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(int(value)) for value in counts.index]
    )

    for bar, count in zip(bars, counts.to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=NAVY,
        )

    panel_label(ax, "B")

    ax.set_title(
        "Empirical longitudinal observation architecture",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Observed transitions per patient")
    ax.set_ylabel("Patients")
    ax.set_xlim(-0.55, len(counts) - 0.45)
    ax.set_ylim(0, max(counts) * 1.15)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)


def plot_parameter_provenance(ax, parameters: pd.DataFrame) -> None:
    display_names = {
        "transition_time_unit": "Normalized transition unit",
        "displacement_sd": "Displacement SD",
        "displacement_q90": "Displacement q90",
        "jump_scale": "Jump scale",
        "branch_switch_fraction": "Branch-switch fraction",
        "branch_separation": "Branch separation",
        "theta": r"Restoring rate $\theta$",
    }
    selected = parameters[
        parameters["quantity"].isin(display_names)
    ].copy()
    selected["display"] = selected["quantity"].map(display_names)
    selected = selected.reset_index(drop=True)

    colors = {
        "Empirical": BLUE,
        "Derived": ORANGE,
        "Prespecified": PURPLE,
        "Prespecified from empirical scale": GREEN,
    }
    y = np.arange(len(selected))[::-1]
    ax.scatter(
        selected["value"],
        y,
        s=80,
        color=[colors.get(value, GRAY) for value in selected["provenance"]],
        edgecolor=WHITE,
        linewidth=0.8,
    )
    for yi, (_, row) in zip(y, selected.iterrows()):
        ax.hlines(yi, 0, row["value"], color=LIGHT_GRAY, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(selected["display"], fontsize=8.8)
    ax.set_xlabel("Calibrated value")
    ax.set_title("Calibration quantities and provenance", fontsize=13, fontweight="bold")
    panel_label(ax, "C")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.7)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=color, markeredgecolor=WHITE,
            markersize=7, label=label
        )
        for label, color in colors.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.8, loc="lower left")


def representative_replicates(
    trajectories: pd.DataFrame,
    n: int,
) -> list[tuple[str, int]]:
    order = ["retention", "jump", "branch_reorganization"]
    selected = []
    for scenario in order[:n]:
        subset = trajectories[trajectories["scenario"] == scenario]
        if subset.empty:
            continue
        replicate = int(subset["replicate"].drop_duplicates().iloc[0])
        selected.append((scenario, replicate))
    return selected


def plot_trajectory(ax, data: pd.DataFrame, events: pd.DataFrame, title: str) -> None:
    data = data.sort_values("time")
    ax.plot(data["time"], data["latent_state"], color=NAVY, linewidth=1.9)
    mask = pd.to_numeric(data["observed_state"], errors="coerce").notna()
    ax.scatter(
        data.loc[mask, "time"],
        data.loc[mask, "observed_state"],
        s=18,
        facecolor=WHITE,
        edgecolor=ORANGE,
        linewidth=0.8,
        zorder=4,
    )
    if pd.to_numeric(data["attractor"], errors="coerce").notna().any():
        ax.plot(
            data["time"],
            data["attractor"],
            color=PURPLE,
            linestyle="--",
            linewidth=1.3,
        )
    for _, event in events.iterrows():
        if event["event"] == "jump":
            ax.axvline(event["event_time"], color=RED, linestyle=":", linewidth=1.0)
        elif event["event"] == "branch_transition":
            ax.axvline(event["event_time"], color=GREEN, linestyle="--", linewidth=1.0)

    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Time", fontsize=8.8)
    ax.set_ylabel(r"State $X_t$", fontsize=8.8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def comparison_plot(
    ax: plt.Axes,
    comparison: pd.DataFrame,
) -> None:
    metrics = [
        "n_observations",
        "followup",
        "total_displacement",
        "max_interval_displacement",
        "median_interval_displacement",
        "q90_interval_displacement",
    ]

    labels = [
        "Observation\ncount",
        "Transition\nspan",
        "Total\ndisplacement",
        "Maximum\ntransition",
        "Median\ntransition",
        "q90\ntransition",
    ]

    normalized_values = []

    for metric in metrics:
        observed_row = comparison[
            (comparison["source"] == "Observed")
            & (comparison["metric"] == metric)
        ]

        if len(observed_row) != 1:
            raise ValueError(
                f"Expected one observed value for {metric}."
            )

        observed_value = float(
            observed_row.iloc[0]["value"]
        )

        simulated_values = pd.to_numeric(
            comparison.loc[
                (comparison["source"] == "Simulated")
                & (comparison["metric"] == metric),
                "value",
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        if np.isclose(observed_value, 0.0):
            normalized = np.full(
                len(simulated_values),
                np.nan,
            )
        else:
            normalized = (
                simulated_values / observed_value
            )

        normalized_values.append(
            normalized[np.isfinite(normalized)]
        )

    positions = np.arange(len(metrics))

    boxplot = ax.boxplot(
        normalized_values,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(
            color=NAVY,
            linewidth=1.4,
        ),
    )

    for patch in boxplot["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.35)
        patch.set_edgecolor(NAVY)

    ax.axhline(
        1.0,
        color=ORANGE,
        linestyle="--",
        linewidth=1.5,
    )
    
    ax.text(
        0.98,
        0.96,
        "Observed reference",
        transform=ax.get_yaxis_transform(),
        color=ORANGE,
        fontsize=8.5,
        ha="right",
        va="top",
    )
    
    ax.set_xticks(positions)
    ax.set_xticklabels(
        labels,
        fontsize=8.2,
    )
    ax.set_ylabel("Simulated / observed ratio")
    ax.set_title(
        "Observed statistics versus calibrated simulations",
        fontsize=13,
        fontweight="bold",
    )

    panel_label(ax, "E")

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
    ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper right",
    )


def fidelity_plot(
    ax,
    fidelity: pd.DataFrame,
) -> None:
    display_names = {
        "n_observations": "Observation count",
        "followup": "Transition span",
        "total_displacement": "Total displacement",
        "max_interval_displacement":
            "Maximum transition",
        "median_interval_displacement":
            "Median transition",
        "q90_interval_displacement":
            "q90 transition",
        "branch_switch_fraction":
            "Branch-switch frequency",
    }

    table = fidelity[
        fidelity["metric"].isin(display_names)
    ].copy()

    table["display"] = (
        table["metric"]
        .map(display_names)
    )

    # Preserve the archived table order while displaying
    # the first metric at the top of the panel.
    y = np.arange(len(table))[::-1]

    lower = (
        table["ratio_median"]
        - table["ratio_q025"]
    )
    upper = (
        table["ratio_q975"]
        - table["ratio_median"]
    )

    ax.errorbar(
        table["ratio_median"],
        y,
        xerr=np.vstack(
            [
                lower,
                upper,
            ]
        ),
        fmt="o",
        color=NAVY,
        ecolor=BLUE,
        capsize=3,
    )

    # Add descriptive metric labels.
    ax.set_yticks(y)
    ax.set_yticklabels(
        table["display"],
        fontsize=9,
    )

    ax.set_xscale("log")
    ax.set_xlim(
        0.1,
        10.0,
    )

    ax.axvline(
        1.0,
        color=ORANGE,
        linestyle="--",
        linewidth=1.4,
    )

    ax.set_xlabel(
        "Simulated / observed ratio (log scale)"
    )

    ax.set_title(
        "Calibration fidelity across empirical statistics",
        fontsize=13,
        fontweight="bold",
    )

    panel_label(
        ax,
        "F",
    )

    ax.spines[
        [
            "top",
            "right",
        ]
    ].set_visible(False)

    ax.grid(
        axis="x",
        color=LIGHT_GRAY,
        linewidth=0.7,
    )

    ax.tick_params(
        axis="y",
        length=0,
        pad=5,
    )


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    root = Path(config["paths"]["project_root"]).expanduser()
    data_dir = root / config["paths"]["output_data_dir"]
    output_dir = root / config["paths"]["output_figure_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    empirical = pd.read_csv(data_dir / "Figure6_empirical_interval_statistics.csv")
    parameters = pd.read_csv(data_dir / "Figure6_calibration_parameters.csv")
    trajectories = pd.read_csv(data_dir / "Figure6_calibrated_trajectory_table.csv")
    events = pd.read_csv(data_dir / "Figure6_calibrated_event_ledger.csv")
    comparison = pd.read_csv(data_dir / "Figure6_observed_simulated_statistics.csv")
    fidelity = pd.read_csv(data_dir / "Figure6_calibration_fidelity.csv")
    metadata = json.loads(
        (data_dir / "Figure6_calibration_metadata.json").read_text(encoding="utf-8")
    )

    fig = plt.figure(
        figsize=(
            float(config["display"]["figure_width"]),
            float(config["display"]["figure_height"]),
        )
    )
    grid = fig.add_gridspec(
        nrows=7,
        ncols=2,
        left=0.055,
        right=0.965,
        top=0.91,
        bottom=0.075,
        height_ratios=[
            0.48,  # row 0: Panel A
            0.05,  # row 1: small A → B/C gap
            1.00,  # row 2: Panels B and C
            0.30,  # row 3: larger B/C → D gap
            1.00,  # row 4: Panel D
            0.30,  # row 5: larger D → E/F gap
            1.02,  # row 6: Panels E and F
        ],
        hspace=0.0,
        wspace=0.28,
    )

    ax_a = fig.add_subplot(grid[0, :])
    calibration_pipeline(ax_a)

    ax_b = fig.add_subplot(grid[2, 0])
    plot_empirical_architecture(ax_b, empirical)

    ax_c = fig.add_subplot(grid[2, 1])
    plot_parameter_provenance(ax_c, parameters)

   # ----------------------------------------------------------
    # Row 4: Panel D spanning both columns
    # ----------------------------------------------------------
    trajectory_grid = grid[4, :].subgridspec(
        nrows=1,
        ncols=3,
        wspace=0.30,
    )
    
    selected = representative_replicates(
        trajectories,
        int(config["display"]["trajectory_count"]),
    )
    
    scenario_titles = {
        "retention": "Retention-dominated",
        "jump": "Jump-dominated",
        "branch_reorganization": "Branch reorganization",
    }
    
    for index, (scenario, replicate) in enumerate(selected):
        ax = fig.add_subplot(
            trajectory_grid[0, index]
        )
    
        data = trajectories[
            (trajectories["scenario"] == scenario)
            & (trajectories["replicate"] == replicate)
        ]
    
        event_table = events[
            (events["scenario"] == scenario)
            & (events["replicate"] == replicate)
        ]
    
        plot_trajectory(
            ax,
            data,
            event_table,
            scenario_titles[scenario],
        )
    
        if index == 0:
            panel_label(
                ax,
                "D",
                x=-0.16,
            )
    
    
    # ----------------------------------------------------------
    # Row 6: Panels E and F
    # ----------------------------------------------------------
    ax_e = fig.add_subplot(
        grid[6, 0]
    )
    comparison_plot(
        ax_e,
        comparison,
    )
    
    ax_f = fig.add_subplot(
        grid[6, 1]
    )
    fidelity_plot(
        ax_f,
        fidelity,
    )
    
    fig.text(
        0.5,
        0.025,
        metadata["interpretation_note"],
        ha="center",
        fontsize=8.8,
        color=GRAY,
    )

    stem = output_dir / "Figure6_real_data_calibration"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=int(config["display"]["dpi"]),
        bbox_inches="tight",
    )
    plt.close(fig)

    for extension in [".svg", ".pdf", ".png"]:
        print(f"[SAVED] {stem.with_suffix(extension)}")


if __name__ == "__main__":
    main()
