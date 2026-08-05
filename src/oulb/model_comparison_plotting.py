from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

COLOR_STABLE = "#4C9ED9"
COLOR_SWITCHING = "#F28E2B"
COLOR_GAUSSIAN = "#6E6E6E"
COLOR_HEAVY = "#B22222"
COLOR_OTHER = "#9A9A9A"
COLOR_REF = "#7A7A7A"


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="top", ha="left")


def _draw_box(ax, xy, w, h, title, body):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                         linewidth=1.3, edgecolor=COLOR_REF, facecolor="white")
    ax.add_patch(box)
    ax.text(xy[0] + 0.03, xy[1] + h - 0.08, title, fontsize=12,
            fontweight="bold", ha="left", va="top")
    ax.text(xy[0] + 0.03, xy[1] + h - 0.18, body, fontsize=11,
            ha="left", va="top")


def render_model_comparison_figure(
    comparison: pd.DataFrame,
    tail_grid: pd.DataFrame,
    casewise: pd.DataFrame,
    output_prefix: Path,
    *,
    top_annotate: int = 5,
    figsize: Tuple[float, float] = (16.0, 10.0),
    dpi: int = 600,
) -> tuple[Path, Path]:
    """Render Figure S6E-H without fitting or threshold selection."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    ax_a.set_axis_off()
    ax_a.set_title("Nested Gaussian and heavy-tail model ladder", fontsize=15)
    _draw_box(ax_a, (0.03, 0.56), 0.42, 0.32, "M0  Gaussian pooled",
              r"$\Delta_i \sim \mathcal{N}(\mu,\sigma^2)$")
    _draw_box(ax_a, (0.53, 0.56), 0.42, 0.32, "M1  Gaussian branch-aware",
              r"$\Delta_i \sim \mathcal{N}(\mu_g,\sigma_g^2)$")
    _draw_box(ax_a, (0.03, 0.12), 0.42, 0.32, "M2  Student-t pooled",
              r"$\Delta_i \sim t_{\nu}(\mu,\sigma)$")
    _draw_box(ax_a, (0.53, 0.12), 0.42, 0.32, "M3  Student-t branch-aware",
              r"$\Delta_i \sim t_{\nu}(\mu_g,\sigma_g)$")

    plot_df = comparison.sort_values("delta_aicc").reset_index(drop=True)
    y = np.arange(len(plot_df))[::-1]
    colors = [COLOR_HEAVY if np.isclose(v, 0.0) else COLOR_OTHER
              for v in plot_df["delta_aicc"]]
    ax_b.barh(y, plot_df["delta_aicc"], color=colors, edgecolor="white")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(plot_df["model_label"])
    ax_b.set_xlabel(r"$\Delta$AICc  (lower is better)", fontsize=12)
    ax_b.set_title("Archived model comparison", fontsize=15)
    for yi, row in zip(y, plot_df.to_dict("records")):
        ax_b.text(row["delta_aicc"] + 0.08, yi, f"AICc={row['aicc']:.2f}",
                  fontsize=10, ha="left", va="center")
    ax_b.axvline(0, color=COLOR_REF, linestyle="--", linewidth=1.2)

    ax_c.plot(tail_grid["x"], tail_grid["observed_switching_survival"],
              color="black", linewidth=2.2, label="Observed Switching survival")
    ax_c.plot(tail_grid["x"], tail_grid["switching_gaussian_survival"],
              color=COLOR_GAUSSIAN, linewidth=2.0, linestyle="--",
              label="Archived Gaussian fit")
    ax_c.plot(tail_grid["x"], tail_grid["switching_student_t_survival"],
              color=COLOR_HEAVY, linewidth=2.0, label="Archived Student-t fit")
    positive = tail_grid[["observed_switching_survival", "switching_gaussian_survival",
                          "switching_student_t_survival"]].replace(0, np.nan).min().min()
    floor = float(positive) if np.isfinite(positive) else 1e-6
    ax_c.set_yscale("log")
    ax_c.set_ylim(bottom=max(floor * 0.8, 1e-6))
    ax_c.set_xlabel("DX→REL total displacement (6D)", fontsize=12)
    ax_c.set_ylabel(r"Switching survival  $P(\Delta \geq x)$", fontsize=12)
    ax_c.set_title("Archived switching-group tail fits", fontsize=15)
    ax_c.legend(frameon=False, fontsize=9)

    ranked = casewise.sort_values("delta_loglik", ascending=False).reset_index(drop=True)
    y = np.arange(len(ranked))[::-1]
    colors = ranked["stability_std"].map(
        {"Stable": COLOR_STABLE, "Switching": COLOR_SWITCHING}
    )
    for yi, delta, color in zip(y, ranked["delta_loglik"], colors):
        ax_d.hlines(yi, 0.0, delta, color=COLOR_REF, linewidth=1.4)
        ax_d.scatter(delta, yi, s=70, color=color, edgecolor="white", linewidth=0.7)
    ax_d.axvline(0, color=COLOR_REF, linestyle="--", linewidth=1.2)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(ranked["sample_std"])
    ax_d.set_xlabel(r"Archived case-wise log-likelihood gain  $\log p_i(M3)-\log p_i(M1)$",
                    fontsize=12)
    ax_d.set_title("Archived case-wise heavy-tail gain", fontsize=15)
    for row in ranked.head(min(top_annotate, len(ranked))).to_dict("records"):
        idx = int(ranked.index[ranked["sample_std"] == row["sample_std"]][0])
        yi = len(ranked) - 1 - idx
        ax_d.annotate(row["sample_std"], (row["delta_loglik"], yi), xytext=(6, 4),
                      textcoords="offset points", fontsize=10)

    for label, ax in zip(["E", "F", "G", "H"], axes.flatten()):
        _panel_label(ax, label)
    fig.tight_layout()
    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf
