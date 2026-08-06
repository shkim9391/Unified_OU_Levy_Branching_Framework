from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass(frozen=True)
class SimulationConfig:
    t_end: float = 10.0
    dt: float = 0.01
    x0: float = 0.0
    seed: int = 20260805

    @property
    def time(self) -> np.ndarray:
        return np.arange(0.0, self.t_end + self.dt, self.dt)


def brownian_motion(
    t: np.ndarray,
    x0: float,
    drift: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    dt = np.diff(t)
    x = np.empty_like(t)
    x[0] = x0
    for i, h in enumerate(dt, start=1):
        x[i] = x[i - 1] + drift * h + sigma * np.sqrt(h) * rng.normal()
    return x


def ou_process(
    t: np.ndarray,
    x0: float,
    theta: float,
    mu: np.ndarray | float,
    sigma: float,
    rng: np.random.Generator,
    jumps: dict[int, float] | None = None,
) -> np.ndarray:
    dt = np.diff(t)
    mu_arr = np.full_like(t, float(mu)) if np.isscalar(mu) else np.asarray(mu, dtype=float)
    if mu_arr.shape != t.shape:
        raise ValueError("mu must be scalar or have the same shape as t.")

    jumps = jumps or {}
    x = np.empty_like(t)
    x[0] = x0

    for i, h in enumerate(dt, start=1):
        x[i] = (
            x[i - 1]
            + theta * (mu_arr[i - 1] - x[i - 1]) * h
            + sigma * np.sqrt(h) * rng.normal()
        )
        if i in jumps:
            x[i] += jumps[i]
    return x


def branching_ou(
    t: np.ndarray,
    x0: float,
    branch_time: float,
    theta_pre: float,
    mu_pre: float,
    sigma_pre: float,
    branch_params: list[tuple[float, float, float]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    branch_idx = int(np.argmin(np.abs(t - branch_time)))

    ancestor = ou_process(
        t[: branch_idx + 1],
        x0=x0,
        theta=theta_pre,
        mu=mu_pre,
        sigma=sigma_pre,
        rng=rng,
    )

    descendants: list[np.ndarray] = []
    for theta, mu, sigma in branch_params:
        branch = ou_process(
            t[branch_idx:],
            x0=ancestor[-1],
            theta=theta,
            mu=mu,
            sigma=sigma,
            rng=rng,
        )
        descendants.append(branch)

    return ancestor, descendants, branch_idx


def style_axis(ax: Axes, panel: str, title: str) -> None:
    ax.text(
        -0.11, 1.07, panel,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=9)
    ax.set_xlabel("Time", fontsize=9.5)
    ax.set_ylabel("Latent state", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


#def add_observations(
#    ax: Axes,
#    t: np.ndarray,
#    x: np.ndarray,
#    observation_times: np.ndarray,
#    *,
#    label: str | None = None,
#    marker: str = "o",
#    zorder: int = 5,
#) -> None:
#    """No observation markers for Supplementary Figure S1."""
    return


def build_figure(config: SimulationConfig) -> Figure:
    t = config.time
    master = np.random.SeedSequence(config.seed)
    rngs = [np.random.default_rng(s) for s in master.spawn(10)]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.2, 8.6),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.09,
        top=0.89,
        wspace=0.22,
        hspace=0.32,
    )

    # Consistent observation schedule across the gallery.
    obs_times = np.array([0.0, 0.55, 1.25, 2.1, 3.0, 4.15, 5.2, 6.45, 7.15, 8.4, 9.1, 10.0])

    # A. Brownian motion
    ax = axes[0, 0]
    x_a = brownian_motion(t, config.x0, drift=0.05, sigma=0.55, rng=rngs[0])
    ax.plot(t, x_a, linewidth=1.7, label="Latent trajectory")
#    add_observations(ax, t, x_a, obs_times, label="Observed time points")
    style_axis(ax, "A", "Brownian motion")
    ax.text(
        0.03, 0.06,
        r"$dX_t=\beta\,dt+\sigma\,dW_t$",
        transform=ax.transAxes,
        fontsize=9.3,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75"),
    )

    # B. Standard OU process
    ax = axes[0, 1]
    mu_b = 1.1
    x_b = ou_process(t, config.x0, theta=0.85, mu=mu_b, sigma=0.48, rng=rngs[1])
    ax.plot(t, x_b, linewidth=1.7, label="Latent trajectory")
    ax.axhline(mu_b, linestyle="--", linewidth=1.25, label=r"Attractor $\mu$")
#    add_observations(ax, t, x_b, obs_times)
    style_axis(ax, "B", "Ornstein–Uhlenbeck process")
    ax.text(
        0.03, 0.06,
        r"$dX_t=\theta(\mu-X_t)\,dt+\sigma\,dW_t$",
        transform=ax.transAxes,
        fontsize=9.1,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75"),
    )

    # C. Therapy-shifted OU process
    ax = axes[0, 2]
    therapy_time = 4.6
    mu_c = np.where(t < therapy_time, 0.45, 1.75)
    x_c = ou_process(t, config.x0, theta=1.0, mu=mu_c, sigma=0.38, rng=rngs[2])
    ax.plot(t, x_c, linewidth=1.7, label="Latent trajectory")
    ax.plot(t, mu_c, linestyle="--", linewidth=1.35, label=r"Therapy-dependent attractor")
    ax.axvline(therapy_time, linestyle=":", linewidth=1.25, label="Therapy onset")
#    add_observations(ax, t, x_c, obs_times)
    style_axis(ax, "C", "Therapy-shifted OU")
    ax.text(
        0.03, 0.06,
        r"$\mu_t=\mu_{\mathrm{pre}}\mathbf{1}_{t<\tau_T}"
        r"+\mu_{\mathrm{post}}\mathbf{1}_{t\geq\tau_T}$",
        transform=ax.transAxes,
        fontsize=8.7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75"),
    )

    # D. OU with a Lévy jump
    ax = axes[1, 0]
    jump_time = 5.25
    jump_idx = int(np.argmin(np.abs(t - jump_time)))
    x_d = ou_process(
        t,
        config.x0,
        theta=0.75,
        mu=0.55,
        sigma=0.32,
        rng=rngs[3],
        jumps={jump_idx: 2.2},
    )
    ax.plot(t, x_d, linewidth=1.7, label="Latent trajectory")
    ax.axhline(0.55, linestyle="--", linewidth=1.25, label=r"Attractor $\mu$")
    ax.axvline(jump_time, linestyle=":", linewidth=1.3, label="Jump time")
    ax.annotate(
        "Lévy jump",
        xy=(jump_time, x_d[jump_idx]),
        xytext=(jump_time + 0.75, x_d[jump_idx] + 0.06),
        fontsize=8.8,
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
    )
#    add_observations(ax, t, x_d, obs_times)
    style_axis(ax, "D", "OU + Lévy jump")
    ax.text(
        0.03, 0.06,
        r"$dX_t=\theta(\mu-X_t)\,dt+\sigma\,dW_t+dJ_t$",
        transform=ax.transAxes,
        fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75"),
    )

    # E. OU with branching
    ax = axes[1, 1]
    branch_time = 4.4
    ancestor, descendants, branch_idx = branching_ou(
        t,
        x0=config.x0,
        branch_time=branch_time,
        theta_pre=0.8,
        mu_pre=0.35,
        sigma_pre=0.27,
        branch_params=[
            (0.95, 1.70, 0.30),
            (0.80, -1.05, 0.34),
        ],
        rng=rngs[4],
    )
    ax.plot(t[: branch_idx + 1], ancestor, linewidth=2.0, label="Shared ancestor")
    for j, branch in enumerate(descendants, start=1):
        ax.plot(t[branch_idx:], branch, linewidth=1.8, label=f"Descendant branch {j}")
    ax.axvline(branch_time, linestyle=":", linewidth=1.25, label="Branch point")
    ax.scatter(
        [t[branch_idx]],
        [ancestor[-1]],
        s=42,
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
    )
    style_axis(ax, "E", "OU + branching")
    ax.text(
        0.03, 0.06,
        r"$B_t\in\{1,2\},\quad"
        r"dX_t=\theta_{B_t}(\mu_{B_t}-X_t)\,dt+\sigma_{B_t}\,dW_t$",
        transform=ax.transAxes,
        fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75"),
    )

    # F. Full OULB process
    ax = axes[1, 2]
    full_branch_time = 5.7
    full_branch_idx = int(np.argmin(np.abs(t - full_branch_time)))
    full_therapy_time = 3.35
    full_jump_time = 4.55
    full_jump_idx = int(np.argmin(np.abs(t - full_jump_time)))

    mu_pre_full = np.where(t[: full_branch_idx + 1] < full_therapy_time, 0.15, 1.15)
    ancestor_full = ou_process(
        t[: full_branch_idx + 1],
        config.x0,
        theta=0.9,
        mu=mu_pre_full,
        sigma=0.28,
        rng=rngs[5],
        jumps={full_jump_idx: 1.35},
    )

    t_post = t[full_branch_idx:]
    branch1_mu = np.full_like(t_post, 1.75)
    branch2_mu = np.full_like(t_post, -0.65)
    branch1 = ou_process(
        t_post,
        ancestor_full[-1],
        theta=1.05,
        mu=branch1_mu,
        sigma=0.29,
        rng=rngs[6],
    )
    branch2 = ou_process(
        t_post,
        ancestor_full[-1],
        theta=0.72,
        mu=branch2_mu,
        sigma=0.36,
        rng=rngs[7],
    )

    ax.plot(t[: full_branch_idx + 1], ancestor_full, linewidth=2.0, label="Shared trajectory")
    ax.plot(t_post, branch1, linewidth=1.8, label="Branch 1")
    ax.plot(t_post, branch2, linewidth=1.8, label="Branch 2")
    ax.plot(
        t[: full_branch_idx + 1],
        mu_pre_full,
        linestyle="--",
        linewidth=1.2,
        label="Dynamic attractor",
    )
    ax.axvline(full_therapy_time, linestyle=":", linewidth=1.1, label="Therapy onset")
    ax.axvline(full_jump_time, linestyle="-.", linewidth=1.1, label="Jump event")
    ax.axvline(full_branch_time, linestyle=(0, (3, 2)), linewidth=1.1, label="Branch point")
    ax.scatter(
        [t[full_branch_idx]],
        [ancestor_full[-1]],
        s=42,
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
    )
    style_axis(ax, "F", "Full OULB process")
    ax.text(
        0.03, 0.06,
        r"$dX_t=\theta_{B_t}(\mu_{B_t,t}-X_t)\,dt"
        r"+\sigma_{B_t}\,dW_t+dJ_t$",
        transform=ax.transAxes,
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75"),
    )

    # Shared limits and subtle zero line.
    all_axes = axes.ravel()
    for ax in all_axes:
        ax.axhline(0.0, linewidth=0.65, alpha=0.25, zorder=0)
        ax.set_xlim(config.time[0], config.time[-1])
        ax.set_ylim(-2.25, 3.35)

    # Compact shared legend based on representative semantic elements.
    handles, labels = [], []
    for ax in all_axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)

    preferred = [
        "Latent trajectory",
        r"Attractor $\mu$",
        "Therapy onset",
        "Jump time",
        "Shared ancestor",
        "Descendant branch 1",
    ]
    selected_h, selected_l = [], []
    for item in preferred:
        if item in labels:
            idx = labels.index(item)
            selected_h.append(handles[idx])
            selected_l.append(labels[idx])

    fig.legend(
        selected_h,
        selected_l,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=6,
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
        columnspacing=1.4,
    )

    fig.suptitle(
        "Supplementary Figure S1. Representative simulated stochastic regimes",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.025,
        "All panels show one-dimensional latent trajectories generated under fixed seeds. "
        "Open circles denote irregular observation times where shown.",
        ha="center",
        fontsize=9,
    )

    return fig


def save_figure(fig: Figure, outdir: Path, stem: str, dpi: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure S1: representative OULB simulation regimes."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Output directory. Default: current directory.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="supplementary_figure_S1_simulated_regimes",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution. Default: 600.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260805,
        help="Master random seed. Default: 20260805.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(seed=args.seed)
    fig = build_figure(config)
    save_figure(fig, args.outdir, args.stem, args.dpi)
    plt.close(fig)

    print(f"Saved:")
    for ext in ("png", "pdf", "svg"):
        print(f"  {args.outdir / f'{args.stem}.{ext}'}")


if __name__ == "__main__":
    main()
