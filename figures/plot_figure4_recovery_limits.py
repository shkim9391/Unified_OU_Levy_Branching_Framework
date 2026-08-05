from __future__ import annotations

import argparse
import json
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "/Unified_OU_Levy_Branching_Framework"
)

FIGSIZE = (15.5, 14.0)
DPI = 600

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
        "--project-root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def _heatmap(
    ax,
    table: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    title: str,
    x_label: str,
    y_label: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    pivot = table.pivot(
        index=y,
        columns=x,
        values=value,
    ).sort_index(
        axis=0,
    ).sort_index(
        axis=1,
    )

    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [f"{value:g}" for value in pivot.columns],
        fontsize=9,
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(
        [f"{value:g}" for value in pivot.index],
        fontsize=9,
    )
    ax.set_xlabel(x_label, fontsize=10.5)
    ax.set_ylabel(y_label, fontsize=10.5)
    ax.set_title(
        title,
        fontsize=10,
        fontweight="bold",
        pad=10,
    )

    cmap = image.cmap
    norm = image.norm
    
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value_at_cell = pivot.iloc[row, column]
    
            if np.isfinite(value_at_cell):
    
                r, g, b, _ = cmap(norm(value_at_cell))
    
                # perceived luminance
                luminance = 0.2126*r + 0.7152*g + 0.0722*b
    
                text_color = WHITE if luminance < 0.45 else DARK
    
                ax.text(
                    column,
                    row,
                    f"{value_at_cell:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.3,
                    color=text_color,
                )

    return image


def _panel_label(
    ax,
    label: str,
    x: float = -0.12,
    y: float = 1.04,
) -> None:

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        va="top",
        ha="left",
    )


def _plot_example(ax, trajectory, events, title: str) -> None:
    trajectory = trajectory.sort_values("time")
    ax.plot(
        trajectory["time"],
        trajectory["latent_state"],
        color=NAVY,
        linewidth=2.0,
        label="Latent state",
    )
    ax.scatter(
        trajectory["time"],
        trajectory["observed_state"],
        s=20,
        facecolor=WHITE,
        edgecolor=ORANGE,
        linewidth=0.9,
        label="Observed",
        zorder=3,
    )

    if "true_attractor" in trajectory.columns:
        attractor = pd.to_numeric(
            trajectory["true_attractor"],
            errors="coerce",
        )
        if attractor.notna().any():
            ax.plot(
                trajectory["time"],
                attractor,
                color=PURPLE,
                linestyle="--",
                linewidth=1.5,
                label="True attractor",
            )

    if "true_branch" in trajectory.columns:
        true_branch = pd.to_numeric(
            trajectory["true_branch"],
            errors="coerce",
        )
        predicted_branch = pd.to_numeric(
            trajectory["predicted_branch"],
            errors="coerce",
        )
        if true_branch.notna().any():
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.step(
                trajectory["time"],
                y_min + 0.04 * y_range
                + true_branch * 0.05 * y_range,
                where="post",
                color=GREEN,
                linewidth=1.4,
                label="True branch",
            )
            ax.step(
                trajectory["time"],
                y_min + 0.14 * y_range
                + predicted_branch * 0.05 * y_range,
                where="post",
                color=RED,
                linewidth=1.2,
                linestyle=":",
                label="Recovered branch",
            )

    if "true_jump" in events.columns:
        true_events = events[
            events["true_jump"].astype(bool)
        ]
        detected_events = events[
            events["detected_jump"].astype(bool)
        ]
        for time in true_events["event_time"]:
            ax.axvline(
                time,
                color=RED,
                linestyle="-",
                linewidth=1.0,
                alpha=0.75,
            )
        for time in detected_events["event_time"]:
            ax.axvline(
                time,
                color=ORANGE,
                linestyle=":",
                linewidth=1.1,
                alpha=0.9,
            )

    ax.set_title(title, fontsize=11.0, fontweight="bold")
    ax.set_xlabel("Time", fontsize=9.5)
    ax.set_ylabel(r"State $X_t$", fontsize=9.5)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.7)


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser()
    data_dir = root / "figures/data"

    paths = {
        "jump_summary": data_dir / "Figure4_jump_recovery_summary.csv",
        "branch_summary": data_dir / "Figure4_branch_recovery_summary.csv",
        "selection": data_dir / "Figure4_representative_selection.csv",
        "trajectories": data_dir / "Figure4_representative_trajectories.csv",
        "events": data_dir / "Figure4_representative_events.csv",
        "metadata": data_dir / "Figure4_recovery_metadata.json",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    jump = pd.read_csv(paths["jump_summary"])
    branch = pd.read_csv(paths["branch_summary"])
    selection = pd.read_csv(paths["selection"])
    trajectories = pd.read_csv(paths["trajectories"])
    events = pd.read_csv(paths["events"])
    metadata = json.loads(
        paths["metadata"].read_text(encoding="utf-8")
    )
    display = metadata["display_slice"]

    jump_slice = jump[
        np.isclose(
            jump["diffusion_sigma"],
            float(display["jump_diffusion_sigma"]),
        )
        & np.isclose(
            jump["jump_rate"],
            float(display["jump_rate"]),
        )
        & (
            jump["n_observations"]
            == int(display["jump_n_observations"])
        )
    ].copy()

    branch_slice = branch[
        np.isclose(
            branch["diffusion_sigma"],
            float(display["branch_diffusion_sigma"]),
        )
        & np.isclose(
            branch["switching_rate"],
            float(display["switching_rate"]),
        )
        & (
            branch["n_observations"]
            == int(display["branch_n_observations"])
        )
    ].copy()

    if jump_slice.empty or branch_slice.empty:
        raise ValueError("Configured display slice is absent from archived summaries.")

    # A taller canvas is needed for four non-overlapping rows.
    fig = plt.figure(
        figsize=FIGSIZE,
        constrained_layout=False,
    )

    # Unified layout:
    #
    # Row 1: A spans columns 0–1 | B spans columns 2–3
    # Row 2: C spans columns 0–1 | D spans columns 2–3
    # Row 3: E spans all four columns
    # Row 4: F1 | F2 | F3 | F4
    grid = fig.add_gridspec(
        nrows=4,
        ncols=4,
        left=0.060,
        right=0.955,
        top=0.905,
        bottom=0.080,
        height_ratios=[
            1.00,   # A–B
            1.00,   # C–D
            1.05,   # E
            0.84,   # F1–F4
        ],
        hspace=0.60,
        wspace=0.42,
    )

    # ----------------------------------------------------------
    # Rows 1 and 2: four recovery heatmaps
    # ----------------------------------------------------------
    ax_a = fig.add_subplot(grid[0, 0:2])
    ax_b = fig.add_subplot(grid[0, 2:4])

    ax_c = fig.add_subplot(grid[1, 0:2])
    ax_d = fig.add_subplot(grid[1, 2:4])

    image_a = _heatmap(
        ax_a,
        jump_slice,
        x="jump_scale",
        y="observation_noise",
        value="precision_mean",
        title="Jump-detection precision",
        x_label="Jump magnitude",
        y_label="Observation noise",
    )

    image_b = _heatmap(
        ax_b,
        jump_slice,
        x="jump_scale",
        y="observation_noise",
        value="recall_mean",
        title="Jump-detection recall",
        x_label="Jump magnitude",
        y_label="Observation noise",
    )

    image_c = _heatmap(
        ax_c,
        jump_slice,
        x="jump_scale",
        y="observation_noise",
        value="false_positive_rate_mean",
        title="Jump false-positive rate",
        x_label="Jump magnitude",
        y_label="Observation noise",
    )

    image_d = _heatmap(
        ax_d,
        branch_slice,
        x="branch_separation",
        y="observation_noise",
        value="state_accuracy_mean",
        title="Branch-state accuracy",
        x_label="Branch separation",
        y_label="Observation noise",
    )

    # ----------------------------------------------------------
    # Row 3: ARI heatmap spanning the complete figure width
    # ----------------------------------------------------------
    ax_e = fig.add_subplot(grid[2, 0:4])

    image_e = _heatmap(
        ax_e,
        branch_slice,
        x="branch_separation",
        y="observation_noise",
        value="ari_mean",
        title="Adjusted Rand index",
        x_label="Branch separation",
        y_label="Observation noise",
    )

    # Put the color bar inside Panel E's allocated row rather than
    # creating another free-floating figure axis.
    cbar_ax = inset_axes(
        ax_e,
        width="1.4%",
        height="80%",
        loc="center right",
        bbox_to_anchor=(0.03, 0.0, 1.0, 1.0),
        bbox_transform=ax_e.transAxes,
        borderpad=0,
    )

    colorbar = fig.colorbar(
        image_e,
        cax=cbar_ax,
    )
    colorbar.set_label(
        "Recovery metric",
        fontsize=9.5,
    )
    colorbar.ax.tick_params(
        labelsize=8.5,
    )


    # ----------------------------------------------------------
    # Panel labels A–E
    # ----------------------------------------------------------
    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")
    _panel_label(ax_c, "C")
    _panel_label(ax_d, "D")
    _panel_label(
    ax_e,
    "E",
    x=-0.055,
    )

    # ----------------------------------------------------------
    # Row 4: four independently archived representative examples
    # ----------------------------------------------------------
    example_order = [
        ("jump_success", "Jump success", "Precision"),
        ("jump_failure", "Jump failure", "Precision"),
        ("branch_success", "Branch success", "ARI"),
        ("branch_failure", "Branch failure", "ARI"),
    ]
    
    example_axes = []
    
    for index, (example_type, label, metric_label) in enumerate(
        example_order
    ):
        axis = fig.add_subplot(grid[3, index])
        example_axes.append(axis)
    
        selected = selection[
            selection["example_type"] == example_type
        ]
    
        if len(selected) != 1:
            raise ValueError(
                f"Expected exactly one {example_type} example; "
                f"found {len(selected)}."
            )
    
        example_id = selected.iloc[0]["example_id"]
    
        trajectory = trajectories[
            trajectories["example_id"] == example_id
        ]
    
        event_table = events[
            events["example_id"] == example_id
        ]
    
        selection_value = float(
            selected.iloc[0]["selection_value"]
        )
    
        title = (
            f"{label}\n"
            f"{metric_label}={selection_value:.2f}"
        )
    
        _plot_example(
            axis,
            trajectory,
            event_table,
            title,
        )
    
        axis.text(
            -0.16,
            1.08,
            f"F{index + 1}",
            transform=axis.transAxes,
            fontsize=15,
            fontweight="bold",
            color=NAVY,
            va="top",
            ha="left",
        )


# ----------------------------------------------------------
# Row 4: four independently archived representative examples
# ----------------------------------------------------------
    example_order = [
        ("jump_success", "Jump success"),
        ("jump_failure", "Jump failure"),
        ("branch_success", "Branch success"),
        ("branch_failure", "Branch failure"),
    ]
    
    example_axes = []
    
    for index, (example_type, label) in enumerate(example_order):
        axis = fig.add_subplot(grid[3, index])
        example_axes.append(axis)
    
        selected = selection[
            selection["example_type"] == example_type
        ]
    
        if len(selected) != 1:
            raise ValueError(
                f"Expected exactly one {example_type} example; "
                f"found {len(selected)}."
            )
    
        selected_row = selected.iloc[0]
        example_id = selected_row["example_id"]
    
        trajectory = trajectories[
            trajectories["example_id"] == example_id
        ]
    
        event_table = events[
            events["example_id"] == example_id
        ]
    
        selection_value = float(
            selected_row["selection_value"]
        )
    
        if example_type.startswith("jump_"):
            metric_label = "Precision"
        elif example_type.startswith("branch_"):
            metric_label = "ARI"
        else:
            metric_label = "Metric"
    
        _plot_example(
            axis,
            trajectory,
            event_table,
            (
                f"{label}\n"
                f"{metric_label}={selection_value:.2f}"
            ),
        )
    
        axis.text(
            -0.16,
            1.08,
            f"F{index + 1}",
            transform=axis.transAxes,
            fontsize=15,
            fontweight="bold",
            color=NAVY,
            va="top",
            ha="left",
        )

#    fig.suptitle(
#        "Figure 4. Recovery limits of discontinuous and branching dynamics",
#        fontsize=20,
#        fontweight="bold",
#        y=0.975,
#    )

#    fig.text(
#        0.5,
#        0.947,
#        (
#            "Recovery improves with stronger discontinuous signals and greater "
#            "branch separation, and deteriorates with observation noise."
#        ),
#        ha="center",
#        fontsize=12.2,
#        color=GRAY,
#        style="italic",
#    )

    fig.text(
        0.5,
        0.035,
        (
            "Heatmaps show archived condition means at the prespecified display "
            "slice; representative cases were selected upstream before plotting."
        ),
        ha="center",
        fontsize=9.2,
        color=GRAY,
    )

    output = root / "figures/Figure4_recovery_limits"
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        output.with_suffix(".png"),
        dpi=args.dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[SAVED] {output.with_suffix('.svg')}")
    print(f"[SAVED] {output.with_suffix('.pdf')}")
    print(f"[SAVED] {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
