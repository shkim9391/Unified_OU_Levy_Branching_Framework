from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATADIR = Path(
    "/Unified_OU_Levy_Branching_Framework/figures/data"
)
DEFAULT_OUTDIR = Path(
    "/Unified_OU_Levy_Branching_Framework/figures"
)

FIGSIZE = (15.5, 11.0)
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

BRANCH_COLORS = {
    -1: "#D9D9D9",
    0: "#DCEAF7",
    1: "#FCE8D5",
}

SCENARIO_ORDER = [
    "brownian",
    "ou",
    "shifted_ou",
    "ou_jump",
    "ou_branching",
    "full_oulb",
]

PANEL_LABELS = {
    "brownian": "A",
    "ou": "B",
    "shifted_ou": "C",
    "ou_jump": "D",
    "ou_branching": "E",
    "full_oulb": "F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure 2 from archived simulation tables."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATADIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
    )
    return parser.parse_args()


def contiguous_segments(
    time: np.ndarray,
    states: np.ndarray,
) -> list[tuple[float, float, int]]:
    segments: list[tuple[float, float, int]] = []

    start = 0

    for index in range(1, len(states)):
        if states[index] != states[start]:
            segments.append(
                (
                    float(time[start]),
                    float(time[index]),
                    int(states[start]),
                )
            )
            start = index

    segments.append(
        (
            float(time[start]),
            float(time[-1]),
            int(states[start]),
        )
    )

    return segments


def add_branch_background(
    ax,
    time: np.ndarray,
    branches: np.ndarray,
) -> None:
    if np.all(branches < 0):
        return

    for start, end, state in contiguous_segments(time, branches):
        ax.axvspan(
            start,
            end,
            facecolor=BRANCH_COLORS.get(state, LIGHT_GRAY),
            alpha=0.52,
            edgecolor="none",
            zorder=0,
        )


def event_times_for(
    events: pd.DataFrame,
    scenario: str,
    event_name: str,
) -> np.ndarray:
    subset = events[
        (events["scenario"] == scenario)
        & (events["event"].astype(str).str.lower() == event_name.lower())
    ]

    return pd.to_numeric(
        subset["event_time"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)


def annotate_jump_events(
    ax,
    data: pd.DataFrame,
    jump_times: np.ndarray,
) -> None:
    if len(jump_times) == 0:
        return

    time = data["time"].to_numpy(dtype=float)
    latent = data["latent_state"].to_numpy(dtype=float)

    for event_time in jump_times:
        index = int(np.argmin(np.abs(time - event_time)))
        ax.axvline(
            event_time,
            color=RED,
            linestyle=":",
            linewidth=1.2,
            alpha=0.85,
            zorder=2,
        )
        ax.scatter(
            [time[index]],
            [latent[index]],
            marker="*",
            s=85,
            facecolor=RED,
            edgecolor=WHITE,
            linewidth=0.6,
            zorder=6,
        )


def annotate_branch_events(
    ax,
    data: pd.DataFrame,
    branch_times: np.ndarray,
) -> None:
    for event_time in branch_times:
        ax.axvline(
            event_time,
            color=GREEN,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            zorder=2,
        )


def plot_panel(
    ax,
    data: pd.DataFrame,
    events: pd.DataFrame,
    scenario: str,
) -> None:
    time = data["time"].to_numpy(dtype=float)
    latent = data["latent_state"].to_numpy(dtype=float)
    observed = data["observed_state"].to_numpy(dtype=float)
    branches = data["branch_state"].to_numpy(dtype=int)
    attractor = data["branch_attractor"].to_numpy(dtype=float)

    add_branch_background(
        ax,
        time,
        branches,
    )

    if np.isfinite(attractor).any():
        ax.plot(
            time,
            attractor,
            color=PURPLE,
            linewidth=1.7,
            linestyle="--",
            label="Attractor",
            zorder=2,
        )

    ax.plot(
        time,
        latent,
        color=NAVY,
        linewidth=2.2,
        label="Latent trajectory",
        zorder=4,
    )

    observed_mask = np.isfinite(observed)
    ax.scatter(
        time[observed_mask],
        observed[observed_mask],
        s=22,
        facecolor=WHITE,
        edgecolor=ORANGE,
        linewidth=1.0,
        label="Noisy observations",
        zorder=5,
    )

    treatment_time = pd.to_numeric(
        data["treatment_time"],
        errors="coerce",
    ).dropna()

    if len(treatment_time) > 0:
        treatment_time_value = float(treatment_time.iloc[0])
        ax.axvline(
            treatment_time_value,
            color=ORANGE,
            linewidth=1.5,
            linestyle="-.",
            label="Treatment",
            zorder=3,
        )

    jump_times = event_times_for(
        events,
        scenario,
        "jump",
    )
    branch_times = event_times_for(
        events,
        scenario,
        "branch_switch",
    )

    # Some implementations may label branch events differently.
    if len(branch_times) == 0:
        branch_times = event_times_for(
            events,
            scenario,
            "branch",
        )

    annotate_jump_events(
        ax,
        data,
        jump_times,
    )
    annotate_branch_events(
        ax,
        data,
        branch_times,
    )

    title = str(data["scenario_label"].iloc[0])

    ax.set_title(
        title,
        fontsize=13.5,
        fontweight="bold",
        color=DARK,
        pad=10,
    )

    ax.text(
        -0.10,
        1.04,
        PANEL_LABELS[scenario],
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        va="top",
    )

    ax.set_xlabel("Time", fontsize=10.5)
    ax.set_ylabel(r"State $X_t$", fontsize=10.5)

    ax.tick_params(
        axis="both",
        labelsize=9.5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis="y",
        color=LIGHT_GRAY,
        linewidth=0.7,
        alpha=0.75,
    )

    mechanism_parts = []

    if scenario == "brownian":
        mechanism_parts = ["drift", "diffusion"]
    elif scenario == "ou":
        mechanism_parts = ["mean reversion", "diffusion"]
    elif scenario == "shifted_ou":
        mechanism_parts = ["mean reversion", "therapy shift"]
    elif scenario == "ou_jump":
        mechanism_parts = ["mean reversion", "rare jumps"]
    elif scenario == "ou_branching":
        mechanism_parts = ["branch-specific reversion", "switching"]
    elif scenario == "full_oulb":
        mechanism_parts = [
            "therapy",
            "jumps",
            "branch switching",
            "diffusion",
        ]

    ax.text(
        0.02,
        0.04,
        " + ".join(mechanism_parts),
        transform=ax.transAxes,
        fontsize=9.0,
        color=GRAY,
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor=WHITE,
            edgecolor=LIGHT_GRAY,
            alpha=0.88,
        ),
    )


def add_global_legend(fig) -> None:
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=NAVY,
            linewidth=2.2,
            label="Latent trajectory",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=WHITE,
            markeredgecolor=ORANGE,
            markersize=6,
            label="Noisy observation",
        ),
        plt.Line2D(
            [0],
            [0],
            color=PURPLE,
            linewidth=1.7,
            linestyle="--",
            label="Attractor",
        ),
        plt.Line2D(
            [0],
            [0],
            color=ORANGE,
            linewidth=1.5,
            linestyle="-.",
            label="Treatment time",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor=RED,
            markeredgecolor=WHITE,
            markersize=10,
            label="Jump event",
        ),
        plt.Line2D(
            [0],
            [0],
            color=GREEN,
            linewidth=1.0,
            linestyle="--",
            label="Branch transition",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.018),
    )


def main() -> None:
    args = parse_args()

    datadir = args.data_dir.expanduser()
    outdir = args.output_dir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    trajectory_path = datadir / "Figure2_representative_trajectories.csv"
    event_path = datadir / "Figure2_event_ledger.csv"
    metadata_path = datadir / "Figure2_simulation_metadata.json"

    for path in [
        trajectory_path,
        event_path,
        metadata_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    trajectories = pd.read_csv(trajectory_path)
    events = pd.read_csv(event_path)
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    required_columns = [
        "scenario",
        "scenario_label",
        "scenario_order",
        "time",
        "latent_state",
        "observed_state",
        "branch_state",
        "branch_attractor",
        "treatment_time",
    ]

    missing = [
        column
        for column in required_columns
        if column not in trajectories.columns
    ]

    if missing:
        raise ValueError(
            f"Trajectory table missing required columns: {missing}"
        )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=FIGSIZE,
        sharex=True,
        constrained_layout=False,
    )

    for axis, scenario in zip(
        axes.flatten(),
        SCENARIO_ORDER,
    ):
        data = trajectories[
            trajectories["scenario"] == scenario
        ].sort_values("time")

        if data.empty:
            raise ValueError(
                f"No archived rows found for scenario: {scenario}"
            )

        plot_panel(
            axis,
            data,
            events,
            scenario,
        )

    all_values = pd.concat(
        [
            trajectories["latent_state"],
            trajectories["observed_state"],
            trajectories["branch_attractor"],
        ],
        ignore_index=True,
    )
    all_values = pd.to_numeric(
        all_values,
        errors="coerce",
    ).dropna()

    lower = float(all_values.quantile(0.01))
    upper = float(all_values.quantile(0.99))
    padding = max(0.15, 0.12 * (upper - lower))

    for axis in axes.flatten():
        axis.set_ylim(
            lower - padding,
            upper + padding,
        )

    fig.text(
        0.012,
        0.50,
        "Branch-state shading indicates the active lineage regime",
        rotation=90,
        va="center",
        fontsize=9.5,
        color=GRAY,
    )

    fig.text(
        0.99,
        0.50,
        (
            f"Archived trajectories: seed={metadata['base_seed']}; "
            f"latent grid={metadata['n_latent']} time points"
        ),
        rotation=270,
        va="center",
        ha="right",
        fontsize=8.8,
        color=GRAY,
    )

    add_global_legend(fig)

    fig.tight_layout(
        rect=[0.03, 0.065, 0.97, 0.925],
        h_pad=2.2,
        w_pad=1.7,
    )

    stem = outdir / "Figure2_simulation_architecture"

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=args.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVED] {stem.with_suffix('.svg')}")
    print(f"[SAVED] {stem.with_suffix('.pdf')}")
    print(f"[SAVED] {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
