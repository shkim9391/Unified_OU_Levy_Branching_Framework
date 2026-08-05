from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_PROJECT_ROOT = Path(
    "/Unified_OU_Levy_Branching_Framework"
)
FIGSIZE = (15.5, 11.0)
DPI = 600

NAVY = "#173B6C"
BLUE = "#4C78A8"
ORANGE = "#F28E2B"
RED = "#C44E52"
GRAY = "#6B7280"
LIGHT_GRAY = "#E6E8EB"
DARK = "#202124"
WHITE = "#FFFFFF"

PANEL_ORDER = ["A", "B", "C", "D", "E", "F"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def deterministic_jitter(
    replicate: np.ndarray,
    width: float = 0.13,
) -> np.ndarray:
    """
    Deterministic horizontal jitter around the single categorical x-position.

    The golden-angle sequence distributes replicate points without introducing
    additional randomness, so repeated plotting produces identical output.
    """
    replicate = np.asarray(replicate, dtype=float)
    golden_angle = 2.399963229728653

    return width * np.sin(replicate * golden_angle)


def plot_panel(
    ax,
    points: pd.DataFrame,
    summary: pd.Series,
) -> None:
    """
    Display the sampling distribution of estimates around one fixed truth.

    Elements
    --------
    - violin: overall estimate distribution;
    - box: median and interquartile range;
    - open circles: all replicate estimates, including outliers;
    - orange diamond: mean estimate;
    - orange error bar: mean ± 1 SD;
    - dashed horizontal line: true generative value.
    """
    truth = float(summary["truth"])
    estimate = points["estimate"].to_numpy(dtype=float)
    replicate = points["replicate"].to_numpy(dtype=int)

    estimate_mean = float(summary["estimate_mean"])
    estimate_sd = float(summary["estimate_sd"])

    # ------------------------------------------------------------
    # Vertical range: preserve all observed estimates and truth.
    # ------------------------------------------------------------
    combined = np.concatenate(
        [
            estimate,
            np.array([truth], dtype=float),
        ]
    )

    lower = float(np.min(combined))
    upper = float(np.max(combined))
    span = upper - lower

    if not np.isfinite(span) or span <= 0:
        span = max(abs(truth), 1.0) * 0.20

    padding = max(
        0.10 * span,
        0.04 * max(abs(truth), 1.0),
    )

    axis_min = lower - padding
    axis_max = upper + padding

    # ------------------------------------------------------------
    # Distribution layer
    # ------------------------------------------------------------
    violin = ax.violinplot(
        dataset=[estimate],
        positions=[0.0],
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method="scott",
    )

    for body in violin["bodies"]:
        body.set_facecolor(BLUE)
        body.set_edgecolor(BLUE)
        body.set_linewidth(1.0)
        body.set_alpha(0.16)
        body.set_zorder(1)

    # Narrow boxplot. Fliers are hidden here because all individual estimates
    # are explicitly plotted in the next layer.
    boxplot = ax.boxplot(
        [estimate],
        positions=[0.0],
        widths=0.22,
        vert=True,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={
            "color": NAVY,
            "linewidth": 1.8,
        },
        boxprops={
            "facecolor": WHITE,
            "edgecolor": NAVY,
            "linewidth": 1.1,
            "alpha": 0.92,
        },
        whiskerprops={
            "color": NAVY,
            "linewidth": 1.0,
        },
        capprops={
            "color": NAVY,
            "linewidth": 1.0,
        },
    )

    # Keep box elements above the violin.
    for component in [
        "boxes",
        "whiskers",
        "caps",
        "medians",
    ]:
        for artist in boxplot[component]:
            artist.set_zorder(2)

    # ------------------------------------------------------------
    # Every replicate estimate, including outliers
    # ------------------------------------------------------------
    x_values = deterministic_jitter(
        replicate,
        width=0.15,
    )

    ax.scatter(
        x_values,
        estimate,
        s=31,
        facecolor=WHITE,
        edgecolor=BLUE,
        linewidth=0.9,
        alpha=0.90,
        zorder=3,
    )

    # ------------------------------------------------------------
    # Fixed generative truth
    # ------------------------------------------------------------
    ax.axhline(
        truth,
        color=RED,
        linewidth=1.5,
        linestyle="--",
        zorder=2,
    )

    # Place the truth label near the right edge.
    ax.text(
        0.97,
        truth,
        f"Truth = {truth:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.8,
        color=RED,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": WHITE,
            "edgecolor": "none",
            "alpha": 0.82,
        },
        zorder=6,
    )

    # ------------------------------------------------------------
    # Mean estimate ± 1 SD
    # ------------------------------------------------------------
    if np.isfinite(estimate_sd):
        ax.errorbar(
            [0.0],
            [estimate_mean],
            yerr=[[estimate_sd], [estimate_sd]],
            fmt="none",
            ecolor=ORANGE,
            elinewidth=1.8,
            capsize=5,
            capthick=1.4,
            zorder=5,
        )

    ax.scatter(
        [0.0],
        [estimate_mean],
        s=105,
        marker="D",
        facecolor=ORANGE,
        edgecolor=NAVY,
        linewidth=0.9,
        zorder=6,
    )

    # ------------------------------------------------------------
    # Panel formatting
    # ------------------------------------------------------------
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(axis_min, axis_max)

    ax.set_xticks([0.0])
    ax.set_xticklabels(
        [f"Replicate estimates of {summary['symbol']}"],
        fontsize=9.5,
    )

    ax.set_title(
        str(summary["panel_title"]),
        fontsize=13.5,
        fontweight="bold",
        color=DARK,
        pad=9,
    )

    ax.text(
        -0.10,
        1.04,
        str(summary["panel"]),
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        va="top",
    )

    ax.set_ylabel(
        f"Estimated {summary['symbol']}",
        fontsize=10.5,
    )
    ax.set_xlabel("")

    ax.text(
        0.04,
        0.96,
        (
            f"n={int(summary['n'])}\n"
            f"Mean={estimate_mean:.3f}\n"
            f"Bias={float(summary['bias']):.3f}\n"
            f"RMSE={float(summary['rmse']):.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color=DARK,
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": WHITE,
            "edgecolor": LIGHT_GRAY,
            "alpha": 0.94,
        },
        zorder=7,
    )

    boundary_fraction = summary.get(
        "boundary_hit_fraction",
        np.nan,
    )

    if pd.notna(boundary_fraction):
        ax.text(
            0.96,
            0.04,
            (
                "Boundary hits = "
                f"{100.0 * float(boundary_fraction):.1f}%"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.8,
            color=RED,
        )

    ax.grid(
        axis="y",
        color=LIGHT_GRAY,
        linewidth=0.7,
        alpha=0.75,
    )
    ax.tick_params(
        axis="both",
        labelsize=9.3,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser()
    data_dir = root / "figures/data"
    outdir = root / "figures"

    points_path = data_dir / "Figure3_parameter_recovery_points.csv"
    summary_path = data_dir / "Figure3_parameter_recovery_summary.csv"
    metadata_path = data_dir / "Figure3_parameter_recovery_metadata.json"

    for path in [points_path, summary_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    points = pd.read_csv(points_path)
    summary = pd.read_csv(summary_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, constrained_layout=False)

    for ax, panel in zip(axes.flatten(), PANEL_ORDER):
        panel_points = points[points["panel"] == panel].sort_values("replicate")
        panel_summary = summary[summary["panel"] == panel]
        if panel_points.empty:
            raise ValueError(f"No archived points for panel {panel}")
        if len(panel_summary) != 1:
            raise ValueError(
                f"Expected one summary row for panel {panel}; "
                f"found {len(panel_summary)}"
            )
        plot_panel(ax, panel_points, panel_summary.iloc[0])

    fig.text(
        0.012,
        0.50,
        (
            "Violin and box: sampling distribution; open circles: all replicates; "
            "orange diamond and error bar: mean ± 1 SD"
        ),
        rotation=90,
        va="center",
        fontsize=9.2,
        color=GRAY,
    )
    fig.text(
        0.99,
        0.50,
        f"Archived recovery rows: {metadata['archived_rows']['points']}",
        rotation=270,
        va="center",
        ha="right",
        fontsize=8.8,
        color=GRAY,
    )

    fig.tight_layout(
        rect=[0.03, 0.065, 0.97, 0.925],
        h_pad=2.0,
        w_pad=1.5,
    )

    stem = outdir / "Figure3_continuous_parameter_recovery"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {stem.with_suffix('.svg')}")
    print(f"[SAVED] {stem.with_suffix('.pdf')}")
    print(f"[SAVED] {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
