from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_STABLE = "#4C9ED9"
COLOR_SWITCHING = "#F28E2B"
COLOR_EFFECT = "#B22222"
COLOR_REF = "#7A7A7A"
GROUP_DISPLAY = {"Stable": "Branch-continuous", "Switching": "Branch-switching"}


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="top", ha="left")


def _annotate(ax: plt.Axes, x, y, labels) -> None:
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(str(label), (xi, yi), xytext=(6, 6), textcoords="offset points",
                    fontsize=10, ha="left", va="bottom")


def render_non_gaussian_figure(
    ranked: pd.DataFrame,
    qq: pd.DataFrame,
    effects: pd.DataFrame,
    jumps: pd.DataFrame,
    metadata: dict,
    output_prefix: Path,
    *,
    top_annotate: int = 5,
    figsize: Tuple[float, float] = (15.5, 10.0),
    dpi: int = 600,
) -> tuple[Path, Path]:
    """Render Figure S6A-D using archived tables only."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    ax_a.plot(ranked["rank"], ranked["total_disp_std"], color=COLOR_REF,
              linewidth=1.2, alpha=0.7)
    for group, color in [("Stable", COLOR_STABLE), ("Switching", COLOR_SWITCHING)]:
        sub = ranked[ranked["stability_std"] == group]
        ax_a.scatter(sub["rank"], sub["total_disp_std"], s=50, color=color,
                     edgecolor="white", linewidth=0.6,
                     label=GROUP_DISPLAY[group], zorder=3)
    ax_a.axhline(float(metadata["stable_total_q95"]), color=COLOR_REF,
                 linestyle="--", linewidth=1.5,
                 label="Branch-continuous 95th percentile")
    top = ranked.nlargest(top_annotate, "total_disp_std")
    _annotate(ax_a, top["rank"], top["total_disp_std"], top["sample_std"])
    ax_a.set_title("Ranked total displacement reveals an upper-tail excess", fontsize=15)
    ax_a.set_xlabel("Sample rank", fontsize=12)
    ax_a.set_ylabel("DX→REL total displacement (6D)", fontsize=12)
    ax_a.legend(frameon=False, fontsize=10, loc="upper left")

    colors = qq["stability_std"].map({"Stable": COLOR_STABLE, "Switching": COLOR_SWITCHING})
    ax_b.scatter(qq["theoretical_q"], qq["z_total_std"], s=52, c=colors,
                 edgecolor="white", linewidth=0.6)
    lo = min(float(qq["theoretical_q"].min()), float(qq["z_total_std"].min())) - 0.3
    hi = max(float(qq["theoretical_q"].max()), float(qq["z_total_std"].max())) + 0.3
    ax_b.plot([lo, hi], [lo, hi], linestyle="--", color=COLOR_REF, linewidth=1.5)
    top = qq.nlargest(top_annotate, "z_total_std")
    _annotate(ax_b, top["theoretical_q"], top["z_total_std"], top["sample_std"])
    ax_b.set_title("Upper-tail departure from Gaussian expectation", fontsize=15)
    ax_b.set_xlabel("Theoretical normal quantile", fontsize=12)
    ax_b.set_ylabel("Observed robust z-score\n(vs branch-continuous baseline)", fontsize=12)

    order = ["Total", "Malignant", "TME"]
    effect_plot = effects.set_index("metric").loc[order].reset_index()
    y = np.arange(len(effect_plot))[::-1]
    for yi, row in zip(y, effect_plot.to_dict("records")):
        ax_c.hlines(yi, row["ci_lo"], row["ci_hi"], color=COLOR_REF, linewidth=2.4)
        ax_c.scatter(row["median_diff_switching_minus_stable"], yi, s=85,
                     color=COLOR_EFFECT, edgecolor="white", linewidth=0.8)
    ax_c.axvline(0, color=COLOR_REF, linestyle="--", linewidth=1.4)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(order)
    ax_c.set_xlabel("Median difference (Branch-switching − Branch-continuous)", fontsize=12)
    ax_c.set_title("Branch-switching intervals show larger displacement\nby robust effect size", fontsize=15)
    xmax = max(float(effect_plot["ci_hi"].max()), 0.0)
    for yi, row in zip(y, effect_plot.to_dict("records")):
        ax_c.text(xmax, yi, f"Cliff's δ={row['cliffs_delta']:.2f}", fontsize=9,
                  ha="right", va="center")

    display_jumps = jumps.sort_values("jump_score_std").reset_index(drop=True)
    yj = np.arange(len(display_jumps))
    colors = display_jumps["stability_std"].map(
        {"Stable": COLOR_STABLE, "Switching": COLOR_SWITCHING}
    )
    for yi, score, color in zip(yj, display_jumps["jump_score_std"], colors):
        ax_d.hlines(yi, 0.0, score, color=COLOR_REF, linewidth=1.6)
        ax_d.scatter(score, yi, s=70, color=color, edgecolor="white", linewidth=0.7)
    ax_d.set_yticks(yj)
    ax_d.set_yticklabels(display_jumps["sample_std"])
    ax_d.set_xlabel(r"Jump score  $-\log_{10}\{1-\Phi(z)\}$", fontsize=12)
    ax_d.set_title("Jump-candidate ranking identifies a small extreme subset", fontsize=15)
    max_score = float(display_jumps["jump_score_std"].max()) if len(display_jumps) else 1.0
    ax_d.set_xlim(0.0, max_score + max(3.0, 0.1 * max_score))
    for yi, row in zip(yj, display_jumps.to_dict("records")):
        label = row["transition_std"]
        tier = str(row.get("jump_tier_std", "")).strip()
        if tier and tier.lower() != "nan":
            label += f" | {tier}"
        ax_d.text(max_score + 0.6, yi, label, fontsize=9, ha="left", va="center",
                  clip_on=False)

    for label, ax in zip(["A", "B", "C", "D"], axes.flatten()):
        _panel_label(ax, label)

    fig.text(
        0.01, 0.01,
        f"Archived robust baseline: median={metadata['baseline_median']:.3f}, "
        f"scale={metadata['baseline_scale']:.3f}; bootstrap replicates={metadata['n_boot']}",
        fontsize=9, ha="left", va="bottom",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf
