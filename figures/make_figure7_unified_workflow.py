from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

DEFAULT_OUTDIR = Path(
    "/Unified_OU_Levy_Branching_Framework/figures"
)
FIGSIZE = (18, 12)
DPI = 600

NAVY = "#173B6C"
BLUE = "#4C78A8"
TEAL = "#3A9D9A"
GREEN = "#59A14F"
ORANGE = "#F28E2B"
RED = "#C44E52"
PURPLE = "#7A5195"
GRAY = "#6B7280"
LIGHT_GRAY = "#E6E8EB"
PALE_BLUE = "#EAF2F8"
PALE_TEAL = "#E8F5F4"
PALE_GREEN = "#EDF6EA"
PALE_ORANGE = "#FFF2E3"
PALE_RED = "#FBE9EA"
PALE_PURPLE = "#F2ECF6"
WHITE = "#FFFFFF"
DARK = "#202124"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 7: unified OULB computational workflow."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def panel_frame(ax, label: str, title: str, edgecolor: str = LIGHT_GRAY) -> None:
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor=edgecolor,
            linewidth=1.2,
            clip_on=False,
        )
    )
    ax.text(
        0.035,
        0.955,
        label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color=WHITE,
        ha="center",
        va="center",
        bbox=dict(boxstyle="circle,pad=0.22", facecolor=NAVY, edgecolor="none"),
    )
    ax.text(
        0.105,
        0.955,
        title,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="center",
    )


def arrow(ax, start, end, color: str = GRAY, linewidth: float = 1.6,
          mutation_scale: float = 14) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            zorder=5,
        )
    )


def workflow_box(ax, y: float, title: str, body: str, facecolor: str,
                 edgecolor: str, number: int) -> None:
    x, width, height = 0.06, 0.66, 0.078
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
        )
    )
    ax.text(
        x + 0.035,
        y + height / 2,
        str(number),
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=WHITE,
        ha="center",
        va="center",
        bbox=dict(boxstyle="circle,pad=0.22", facecolor=edgecolor, edgecolor="none"),
    )
    ax.text(
        x + 0.085,
        y + height * 0.66,
        title,
        transform=ax.transAxes,
        fontsize=9.7,
        fontweight="bold",
        color=DARK,
        ha="left",
        va="center",
    )
    ax.text(
        x + 0.085,
        y + height * 0.30,
        body,
        transform=ax.transAxes,
        fontsize=8.3,
        color=DARK,
        ha="left",
        va="center",
    )


def panel_a(ax) -> None:
    panel_frame(ax, "A", "End-to-end computational workflow", BLUE)
    boxes = [
        ("Raw observations", "single-cell, spatial, longitudinal, bulk", PALE_BLUE, BLUE),
        ("Observation model", "measurement error, irregular sampling, missingness", PALE_TEAL, TEAL),
        ("Latent OULB process", "OU dynamics + Lévy jumps + branch switching", PALE_GREEN, GREEN),
        ("Bayesian inference", "posterior estimation, simulation, uncertainty", PALE_ORANGE, ORANGE),
        ("Parameter recovery", r"$\theta,\ \mu,\ \sigma,\ \lambda,\ \kappa,\ Q$", PALE_ORANGE, ORANGE),
        ("Jump detection", "timing, probability, and magnitude of events", PALE_RED, RED),
        ("Branch inference", "lineage states, transition matrix, probabilities", PALE_PURPLE, PURPLE),
        ("Prediction", "future states, event risk, credible intervals", PALE_BLUE, BLUE),
        ("Biological interpretation", "therapy response, constraint, escape, diversification", PALE_RED, RED),
    ]
    y_positions = np.linspace(0.83, 0.07, len(boxes))
    for index, (title, body, facecolor, edgecolor) in enumerate(boxes):
        workflow_box(ax, float(y_positions[index]), title, body, facecolor, edgecolor, index + 1)
        if index < len(boxes) - 1:
            arrow(
                ax,
                (0.39, y_positions[index] - 0.005),
                (0.39, y_positions[index + 1] + 0.088),
                linewidth=1.3,
                mutation_scale=12,
            )


def panel_b(ax) -> None:
    panel_frame(ax, "B", "Nested model hierarchy", PURPLE)
    models = [
        ("Brownian motion", "diffusion only"),
        ("Standard OU", "mean reversion"),
        ("Treatment-shifted OU", "time-varying attractor"),
        ("OU + Lévy jumps", "rare discontinuities"),
        ("OU + branching", "lineage diversification"),
        ("Full OULB", "OU + Lévy + branching"),
    ]
    y_positions = np.linspace(0.80, 0.20, len(models))
    for index, ((name, description), y) in enumerate(zip(models, y_positions)):
        facecolor = PALE_PURPLE if index == len(models) - 1 else PALE_BLUE
        edgecolor = PURPLE if index == len(models) - 1 else BLUE
        ax.add_patch(
            FancyBboxPatch(
                (0.18, y),
                0.56,
                0.08,
                boxstyle="round,pad=0.012",
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.0,
            )
        )
        ax.text(0.46, y + 0.052, name, transform=ax.transAxes, fontsize=10.2,
                fontweight="bold", color=DARK, ha="center", va="center")
        ax.text(0.46, y + 0.022, description, transform=ax.transAxes, fontsize=8.5,
                color=GRAY, ha="center", va="center")
        if index < len(models) - 1:
            arrow(ax, (0.46, y - 0.002), (0.46, y_positions[index + 1] + 0.085),
                  linewidth=1.2, mutation_scale=10)
    ax.text(0.84, 0.78, "Simpler", transform=ax.transAxes, fontsize=9.3,
            color=GRAY, ha="center")
    arrow(ax, (0.84, 0.73), (0.84, 0.22), color=PURPLE,
          linewidth=2.0, mutation_scale=17)
    ax.text(0.84, 0.15, "More expressive", transform=ax.transAxes,
            fontsize=9.3, color=PURPLE, ha="center")


def modality_icon(ax, x: float, y: float, kind: int) -> None:
    if kind == 0:
        for index in range(10):
            angle = 2 * np.pi * index / 10
            ax.add_patch(
                Circle(
                    (x + 0.035 * np.cos(angle), y + 0.035 * np.sin(angle)),
                    0.010,
                    transform=ax.transAxes,
                    facecolor=BLUE if index % 2 else TEAL,
                    edgecolor=NAVY,
                    linewidth=0.4,
                )
            )
    elif kind == 1:
        for row in range(5):
            for column in range(5):
                ax.add_patch(
                    Rectangle(
                        (x - 0.05 + column * 0.021, y - 0.05 + row * 0.021),
                        0.018,
                        0.018,
                        transform=ax.transAxes,
                        facecolor=PURPLE if (row + column) % 2 else GREEN,
                        edgecolor=WHITE,
                        linewidth=0.3,
                    )
                )
    elif kind == 2:
        for layer in range(4):
            ax.add_patch(
                Rectangle(
                    (x - 0.05 + layer * 0.014, y - 0.04 + layer * 0.010),
                    0.085,
                    0.065,
                    transform=ax.transAxes,
                    facecolor=PALE_BLUE,
                    edgecolor=GRAY,
                    linewidth=0.6,
                )
            )
    elif kind == 3:
        nodes = [(x - 0.04, y + 0.03), (x, y), (x + 0.04, y + 0.03),
                 (x + 0.02, y - 0.045), (x - 0.03, y - 0.04)]
        for first, second in [(0, 1), (1, 2), (1, 3), (0, 4)]:
            ax.plot([nodes[first][0], nodes[second][0]],
                    [nodes[first][1], nodes[second][1]],
                    transform=ax.transAxes, color=GRAY, linewidth=1.0)
        for index, (nx, ny) in enumerate(nodes):
            ax.add_patch(
                Circle((nx, ny), 0.013, transform=ax.transAxes,
                       facecolor=PURPLE if index % 2 else TEAL,
                       edgecolor=NAVY, linewidth=0.5)
            )
    else:
        ax.add_patch(
            Rectangle((x - 0.035, y - 0.05), 0.07, 0.10,
                      transform=ax.transAxes, facecolor=PALE_ORANGE,
                      edgecolor=ORANGE, linewidth=0.8)
        )
        for row in range(3):
            ax.plot([x - 0.022, x + 0.022], [y + 0.02 - row * 0.025] * 2,
                    transform=ax.transAxes, color=GRAY, linewidth=0.8)


def panel_c(ax) -> None:
    panel_frame(ax, "C", "Broad data modalities supported", BLUE)
    labels = [
        "Longitudinal\nsingle-cell",
        "Spatial\ntranscriptomics",
        "Bulk\nomics",
        "Pathway\nactivities",
        "Future multimodal\ndatasets",
    ]
    x_positions = np.linspace(0.10, 0.90, len(labels))
    for index, (x, label) in enumerate(zip(x_positions, labels)):
        ax.text(x, 0.76, label, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color=DARK, ha="center", va="center")
        modality_icon(ax, float(x), 0.60, index)
        arrow(ax, (x, 0.48), (x, 0.39), linewidth=1.3, mutation_scale=11)
    ax.add_patch(
        FancyBboxPatch((0.08, 0.29), 0.84, 0.10, boxstyle="round,pad=0.012",
                       transform=ax.transAxes, facecolor=PALE_BLUE,
                       edgecolor=BLUE, linewidth=1.0)
    )
    ax.text(0.50, 0.34, "Observation model", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=NAVY, ha="center", va="center")
    arrow(ax, (0.50, 0.29), (0.50, 0.21), linewidth=1.4)
    ax.add_patch(
        FancyBboxPatch((0.08, 0.10), 0.84, 0.10, boxstyle="round,pad=0.012",
                       transform=ax.transAxes, facecolor=PALE_GREEN,
                       edgecolor=GREEN, linewidth=1.0)
    )
    ax.text(0.50, 0.15, "Latent OULB process", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=DARK, ha="center", va="center")


def panel_d(ax) -> None:
    panel_frame(ax, "D", "Outputs of the unified framework", TEAL)
    center = (0.50, 0.46)
    ax.add_patch(
        Circle(center, 0.14, transform=ax.transAxes,
               facecolor=PALE_BLUE, edgecolor=BLUE, linewidth=1.2)
    )
    ax.text(center[0], center[1] + 0.03, "Posterior", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", color=NAVY, ha="center")
    ax.text(center[0], center[1] - 0.02, "OULB model", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", color=NAVY, ha="center")
    outputs = [
        ("Parameter\nestimates", 0.50, 0.78, GREEN),
        ("Trajectory\nprediction", 0.78, 0.65, TEAL),
        ("Jump\ndetection", 0.80, 0.35, RED),
        ("Branch\ninference", 0.68, 0.16, PURPLE),
        ("Simulation\n& what-if", 0.38, 0.13, BLUE),
        ("Calibration\n& validation", 0.20, 0.29, ORANGE),
        ("Uncertainty\nquantification", 0.20, 0.60, GRAY),
    ]
    for label, x, y, color in outputs:
        ax.plot([center[0], x], [center[1], y], transform=ax.transAxes,
                color=LIGHT_GRAY, linewidth=1.2, zorder=1)
        ax.add_patch(Circle((x, y), 0.055, transform=ax.transAxes,
                            facecolor=WHITE, edgecolor=color,
                            linewidth=1.2, zorder=3))
        ax.text(x, y - 0.090, label, transform=ax.transAxes,
                fontsize=8.6, color=DARK, ha="center", va="center")


def software_stage_box(ax, y: float, title: str, body: str,
                       facecolor: str, edgecolor: str) -> None:
    ax.add_patch(
        FancyBboxPatch((0.07, y), 0.58, 0.105, boxstyle="round,pad=0.012",
                       transform=ax.transAxes, facecolor=facecolor,
                       edgecolor=edgecolor, linewidth=1.0)
    )
    ax.text(0.11, y + 0.073, title, transform=ax.transAxes,
            fontsize=9.3, fontweight="bold", color=edgecolor, ha="left")
    ax.text(0.11, y + 0.032, body, transform=ax.transAxes,
            fontsize=7.9, color=DARK, ha="left")


def panel_e(ax) -> None:
    panel_frame(ax, "E", "Software architecture and reproducible pipeline", ORANGE)
    stages = [
        ("Stage 1 — Projection and preprocessing",
         "map data into ecological or latent state space", PALE_BLUE, BLUE),
        ("Stage 2 — Dynamic parameter estimation",
         r"estimate $\theta,\ \mu,\ \sigma$ and observation terms", PALE_TEAL, TEAL),
        ("Stage 3 — Jump and branching analysis",
         "detect jumps and infer branch transitions", PALE_GREEN, GREEN),
        ("Stage 4 — Archived results and graphics",
         "store tables, trajectories, events, and summaries", PALE_ORANGE, ORANGE),
        ("Stage 5 — Benchmarking and calibration",
         "simulation studies, stress tests, empirical calibration", PALE_RED, RED),
    ]
    y_positions = [0.76, 0.61, 0.46, 0.31, 0.16]
    for index, (title, body, facecolor, edgecolor) in enumerate(stages):
        software_stage_box(ax, y_positions[index], title, body, facecolor, edgecolor)
        if index < len(stages) - 1:
            arrow(ax, (0.36, y_positions[index] - 0.01),
                  (0.36, y_positions[index + 1] + 0.115),
                  linewidth=1.2, mutation_scale=10)
    ax.add_patch(
        FancyBboxPatch((0.70, 0.16), 0.24, 0.70, boxstyle="round,pad=0.012",
                       transform=ax.transAxes, facecolor=WHITE, edgecolor=BLUE,
                       linestyle="--", linewidth=1.0)
    )
    ax.text(0.82, 0.80, "Reproducibility", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", color=NAVY, ha="center")
    principles = [
        "modular codebase",
        "versioned analyses",
        "archived intermediate tables",
        "deterministic seeds",
        "table-backed plotting",
        "provenance tracking",
    ]
    for index, principle in enumerate(principles):
        y = 0.70 - index * 0.09
        ax.add_patch(Circle((0.75, y), 0.012, transform=ax.transAxes,
                            facecolor=TEAL, edgecolor="none"))
        ax.text(0.78, y, principle, transform=ax.transAxes,
                fontsize=8.2, color=DARK, va="center")


def panel_f(ax) -> None:
    panel_frame(ax, "F", "Take-home message", NAVY)
    ax.add_patch(
        FancyBboxPatch((0.08, 0.12), 0.84, 0.76,
                       boxstyle="round,pad=0.025,rounding_size=0.025",
                       transform=ax.transAxes, facecolor=WHITE,
                       edgecolor=NAVY, linewidth=2.0)
    )
    ax.text(
        0.50,
        0.70,
        (
            "The OULB framework integrates\n"
            "constrained stochastic dynamics,\n"
            "discontinuous evolutionary events,\n"
            "lineage diversification, observation\n"
            "processes, Bayesian inference,\n"
            "empirical calibration, and\n"
            "reproducible benchmarking."
        ),
        transform=ax.transAxes,
        fontsize=11.2,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="center",
        linespacing=1.25,
    )
    conclusions = [
        "Unified mathematical foundation",
        "Flexible across data modalities",
        "Validated by simulation and calibration",
        "Supports interpretable biological insight",
    ]
    for index, conclusion in enumerate(conclusions):
        y = 0.38 - index * 0.08
        ax.add_patch(Circle((0.20, y), 0.020, transform=ax.transAxes,
                            facecolor=PALE_BLUE, edgecolor=NAVY, linewidth=1.0))
        ax.text(0.26, y, conclusion, transform=ax.transAxes,
                fontsize=9.1, color=DARK, va="center")


def add_footer(fig) -> None:
    ax = fig.add_axes([0.025, 0.015, 0.95, 0.035])
    ax.set_axis_off()
    footer_items = [
        ("Empirical", BLUE),
        ("Derived", ORANGE),
        ("Prespecified", PURPLE),
        ("Prespecified from empirical scale", GREEN),
    ]
    x = 0.02
    for label, color in footer_items:
        ax.add_patch(Circle((x, 0.50), 0.010, transform=ax.transAxes,
                            facecolor=color, edgecolor=NAVY, linewidth=0.4))
        ax.text(x + 0.018, 0.50, label, transform=ax.transAxes,
                fontsize=8.3, color=DARK, va="center")
        x += 0.18 if label != "Prespecified from empirical scale" else 0.27
    ax.text(
        0.98,
        0.50,
        "Figure 7 summarizes the complete OULB workflow from observations to biological interpretation.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=GRAY,
        ha="right",
        va="center",
        style="italic",
    )


def build_figure(output_dir: Path, dpi: int = DPI) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    fig.suptitle(
        "Figure 7. Unified computational workflow of the Ornstein–Uhlenbeck–Lévy–Branching framework",
        fontsize=21,
        fontweight="bold",
        color=DARK,
        y=0.982,
    )
    fig.text(
        0.5,
        0.948,
        "From raw observations to biological insights through probabilistic modeling, inference, validation, and prediction.",
        ha="center",
        fontsize=12.5,
        color=GRAY,
        style="italic",
    )
    grid = fig.add_gridspec(
        nrows=2,
        ncols=3,
        left=0.025,
        right=0.975,
        top=0.91,
        bottom=0.065,
        width_ratios=[1.02, 0.84, 1.46],
        height_ratios=[1.02, 1.10],
        hspace=0.025,
        wspace=0.020,
    )
    panel_a(fig.add_subplot(grid[:, 0]))
    panel_b(fig.add_subplot(grid[0, 1]))
    panel_c(fig.add_subplot(grid[0, 2]))
    lower_grid = grid[1, 1:].subgridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.00, 1.25, 0.95],
        wspace=0.022,
    )
    panel_d(fig.add_subplot(lower_grid[0, 0]))
    panel_e(fig.add_subplot(lower_grid[0, 1]))
    panel_f(fig.add_subplot(lower_grid[0, 2]))
    add_footer(fig)

    stem = output_dir / "Figure7_unified_OULB_workflow"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {stem.with_suffix('.svg')}")
    print(f"[SAVED] {stem.with_suffix('.pdf')}")
    print(f"[SAVED] {stem.with_suffix('.png')}")


def main() -> None:
    args = parse_args()
    build_figure(args.output_dir.expanduser(), dpi=args.dpi)


if __name__ == "__main__":
    main()
