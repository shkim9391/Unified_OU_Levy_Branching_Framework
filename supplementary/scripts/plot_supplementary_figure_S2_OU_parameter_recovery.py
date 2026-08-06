from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import minimize


PARAMETERS = ("mu", "theta", "sigma")
PARAMETER_LABELS = {
    "mu": r"Attractor $\mu$",
    "theta": r"Mean reversion $\theta$",
    "sigma": r"Diffusion $\sigma$",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    replicates: int = 120
    seed: int = 20260805
    observation_counts: tuple[int, ...] = (25, 50, 100)
    t_end: float = 10.0
    x0: float = 0.0


def simulate_irregular_times(
    n_obs: int,
    t_end: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate sorted irregular observation times including 0 and t_end."""
    if n_obs < 3:
        raise ValueError("n_obs must be at least 3.")
    interior = np.sort(rng.uniform(0.0, t_end, size=n_obs - 2))
    return np.concatenate(([0.0], interior, [t_end]))


def simulate_exact_ou(
    times: np.ndarray,
    x0: float,
    mu: float,
    theta: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate the exact OU transition distribution at irregular times."""
    x = np.empty_like(times)
    x[0] = x0

    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        decay = np.exp(-theta * dt)
        mean = mu + (x[i - 1] - mu) * decay
        variance = sigma**2 * (1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta)
        x[i] = rng.normal(mean, np.sqrt(max(variance, 1e-12)))

    return x


def negative_log_likelihood(
    unconstrained: np.ndarray,
    times: np.ndarray,
    x: np.ndarray,
) -> float:
    """
    Exact OU negative log-likelihood.

    Parameters are represented as:
      mu = unconstrained[0]
      theta = exp(unconstrained[1])
      sigma = exp(unconstrained[2])
    """
    mu = unconstrained[0]
    theta = np.exp(unconstrained[1])
    sigma = np.exp(unconstrained[2])

    dt = np.diff(times)
    x_prev = x[:-1]
    x_next = x[1:]

    decay = np.exp(-theta * dt)
    mean = mu + (x_prev - mu) * decay
    variance = sigma**2 * (1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta)
    variance = np.maximum(variance, 1e-12)

    residual = x_next - mean
    return 0.5 * np.sum(
        np.log(2.0 * np.pi * variance) + residual**2 / variance
    )


def numerical_hessian(
    fun,
    x0: np.ndarray,
    rel_step: float = 2e-4,
) -> np.ndarray:
    """Finite-difference Hessian for a small parameter vector."""
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    h = rel_step * np.maximum(1.0, np.abs(x0))
    hessian = np.zeros((n, n), dtype=float)
    f0 = fun(x0)

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        hessian[i, i] = (fun(x0 + ei) - 2.0 * f0 + fun(x0 - ei)) / h[i]**2

        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h[j]
            value = (
                fun(x0 + ei + ej)
                - fun(x0 + ei - ej)
                - fun(x0 - ei + ej)
                + fun(x0 - ei - ej)
            ) / (4.0 * h[i] * h[j])
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian


def estimate_ou(
    times: np.ndarray,
    x: np.ndarray,
) -> dict[str, float | bool]:
    """Estimate OU parameters and approximate Wald intervals."""
    mu0 = float(np.mean(x))
    theta0 = 0.7
    sigma0 = max(float(np.std(np.diff(x))) / np.sqrt(max(np.mean(np.diff(times)), 1e-6)), 0.15)

    initial = np.array([mu0, np.log(theta0), np.log(sigma0)])

    objective = lambda p: negative_log_likelihood(p, times, x)

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (np.log(0.03), np.log(6.0)), (np.log(0.03), np.log(4.0))],
        options={"maxiter": 1000, "ftol": 1e-11},
    )

    if not result.success or not np.all(np.isfinite(result.x)):
        return {"success": False}

    mu_hat = float(result.x[0])
    theta_hat = float(np.exp(result.x[1]))
    sigma_hat = float(np.exp(result.x[2]))

    output: dict[str, float | bool] = {
        "success": True,
        "mu_hat": mu_hat,
        "theta_hat": theta_hat,
        "sigma_hat": sigma_hat,
    }

    try:
        hessian = numerical_hessian(objective, result.x)
        covariance_u = np.linalg.pinv(hessian)

        # Delta-method Jacobian from (mu, log theta, log sigma)
        # to (mu, theta, sigma).
        jacobian = np.diag([1.0, theta_hat, sigma_hat])
        covariance = jacobian @ covariance_u @ jacobian.T
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

        estimates = np.array([mu_hat, theta_hat, sigma_hat])
        lower = estimates - 1.96 * standard_errors
        upper = estimates + 1.96 * standard_errors

        lower[1:] = np.maximum(lower[1:], 0.0)

        for i, parameter in enumerate(PARAMETERS):
            output[f"{parameter}_se"] = float(standard_errors[i])
            output[f"{parameter}_lower"] = float(lower[i])
            output[f"{parameter}_upper"] = float(upper[i])

    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        for parameter in PARAMETERS:
            output[f"{parameter}_se"] = np.nan
            output[f"{parameter}_lower"] = np.nan
            output[f"{parameter}_upper"] = np.nan

    return output


def draw_true_parameters(rng: np.random.Generator) -> tuple[float, float, float]:
    """Sample parameters over a range appropriate for recovery validation."""
    mu = rng.uniform(-1.25, 1.75)
    theta = np.exp(rng.uniform(np.log(0.30), np.log(1.50)))
    sigma = rng.uniform(0.20, 0.90)
    return float(mu), float(theta), float(sigma)


def run_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    master = np.random.SeedSequence(config.seed)
    total = config.replicates * len(config.observation_counts)
    child_sequences = master.spawn(total)

    records: list[dict[str, float | int | bool]] = []
    sequence_index = 0

    for n_obs in config.observation_counts:
        for replicate in range(1, config.replicates + 1):
            rng = np.random.default_rng(child_sequences[sequence_index])
            sequence_index += 1

            mu, theta, sigma = draw_true_parameters(rng)

            times = simulate_irregular_times(
                n_obs,
                config.t_end,
                rng,
            )

            stationary_sd = sigma / np.sqrt(2.0 * theta)
            x0 = rng.normal(mu, stationary_sd)

            x = simulate_exact_ou(
                times,
                x0,
                mu,
                theta,
                sigma,
                rng,
            )

            estimate = estimate_ou(times, x)

            record: dict[str, float | int | bool] = {
                "n_obs": n_obs,
                "replicate": replicate,
                "mu_true": mu,
                "theta_true": theta,
                "sigma_true": sigma,
                **estimate,
            }
            records.append(record)

    results = pd.DataFrame.from_records(records)

    return (
        results.loc[results["success"] == True]
        .reset_index(drop=True)
    )


def bootstrap_metric_interval(
    true: np.ndarray,
    estimate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    metric: str,
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    """Return bootstrap 95% confidence limits for a recovery metric."""
    n = len(true)
    bootstrap_values = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)

        true_b = true[idx]
        estimate_b = estimate[idx]
        error_b = estimate_b - true_b

        if metric == "bias":
            bootstrap_values[b] = np.mean(error_b)

        elif metric == "rmse":
            bootstrap_values[b] = np.sqrt(np.mean(error_b**2))

        elif metric == "coverage":
            lower_b = lower[idx]
            upper_b = upper[idx]

            valid = np.isfinite(lower_b) & np.isfinite(upper_b)

            if np.any(valid):
                bootstrap_values[b] = np.mean(
                    (true_b[valid] >= lower_b[valid])
                    & (true_b[valid] <= upper_b[valid])
                )
            else:
                bootstrap_values[b] = np.nan

        else:
            raise ValueError(f"Unknown metric: {metric}")

    bootstrap_values = bootstrap_values[np.isfinite(bootstrap_values)]

    if len(bootstrap_values) == 0:
        return np.nan, np.nan

    ci_low, ci_high = np.percentile(
        bootstrap_values,
        [2.5, 97.5],
    )

    return float(ci_low), float(ci_high)


def style_axis(ax: Axes, panel: str, title: str) -> None:
    ax.text(
        -0.13,
        1.08,
        panel,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    ax.tick_params(labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def scatter_recovery_panel(
    ax: Axes,
    results: pd.DataFrame,
    parameter: str,
    panel: str,
) -> None:
    for n_obs, subset in results.groupby("n_obs", sort=True):
        ax.scatter(
            subset[f"{parameter}_true"],
            subset[f"{parameter}_hat"],
            s=15,
            alpha=0.35,
            label=f"{n_obs} observations",
        )

    values = np.concatenate(
        [
            results[f"{parameter}_true"].to_numpy(),
            results[f"{parameter}_hat"].to_numpy(),
        ]
    )
    low, high = np.nanpercentile(values, [1, 99])
    padding = 0.08 * (high - low if high > low else 1.0)
    low -= padding
    high += padding

    if parameter in {"theta", "sigma"}:
        low = max(0.0, low)

    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.8)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(f"True {PARAMETER_LABELS[parameter]}", fontsize=9.3)
    ax.set_ylabel(f"Estimated {PARAMETER_LABELS[parameter]}", fontsize=9.3)
    style_axis(ax, panel, f"Recovery of {PARAMETER_LABELS[parameter]}")


def grouped_metric_panel(
    ax: Axes,
    summary: pd.DataFrame,
    metric: str,
    panel: str,
    title: str,
    ylabel: str,
    reference: float | None = None,
) -> None:
    counts = sorted(summary["n_obs"].unique())

    markers = {
        "mu": "o",
        "theta": "s",
        "sigma": "^",
    }

    for parameter in PARAMETERS:
        subset = (
            summary.loc[summary["parameter"] == parameter]
            .set_index("n_obs")
            .loc[counts]
        )

        values = subset[metric].to_numpy()
        ci_low = subset[f"{metric}_ci_low"].to_numpy()
        ci_high = subset[f"{metric}_ci_high"].to_numpy()

        yerr = np.vstack(
            [
                np.maximum(values - ci_low, 0.0),
                np.maximum(ci_high - values, 0.0),
            ]
        )

        ax.errorbar(
            counts,
            values,
            yerr=yerr,
            marker=markers[parameter],
            linewidth=2.0,
            markersize=6,
            capsize=4,
            elinewidth=1.2,
            label=PARAMETER_LABELS[parameter],
        )

    if reference is not None:
        ax.axhline(
            reference,
            linestyle="--",
            linewidth=1.2,
            color="gray",
        )

    ax.set_xticks(counts)
    ax.set_xlabel(
        "Number of observations per trajectory",
        fontsize=9.3,
    )
    ax.set_ylabel(ylabel, fontsize=9.3)

    style_axis(ax, panel, title)


def summarize_metrics(
    results: pd.DataFrame,
    seed: int = 20260806,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    rows = []
    master_rng = np.random.default_rng(seed)

    for n_obs, subset in results.groupby("n_obs", sort=True):
        for parameter in PARAMETERS:
            true = subset[f"{parameter}_true"].to_numpy()
            estimate = subset[f"{parameter}_hat"].to_numpy()
            error = estimate - true

            lower = subset[f"{parameter}_lower"].to_numpy()
            upper = subset[f"{parameter}_upper"].to_numpy()

            valid_interval = np.isfinite(lower) & np.isfinite(upper)

            bias = float(np.mean(error))
            rmse = float(np.sqrt(np.mean(error**2)))

            coverage = (
                float(
                    np.mean(
                        (true[valid_interval] >= lower[valid_interval])
                        & (true[valid_interval] <= upper[valid_interval])
                    )
                )
                if np.any(valid_interval)
                else np.nan
            )

            metric_values = {
                "bias": bias,
                "rmse": rmse,
                "coverage": coverage,
            }

            row = {
                "n_obs": int(n_obs),
                "parameter": parameter,
                **metric_values,
                "n_success": int(len(subset)),
                "n_valid_interval": int(np.sum(valid_interval)),
            }

            for metric in ("bias", "rmse", "coverage"):
                bootstrap_seed = int(
                    master_rng.integers(0, np.iinfo(np.int32).max)
                )
                bootstrap_rng = np.random.default_rng(bootstrap_seed)

                ci_low, ci_high = bootstrap_metric_interval(
                    true=true,
                    estimate=estimate,
                    lower=lower,
                    upper=upper,
                    metric=metric,
                    rng=bootstrap_rng,
                    n_bootstrap=n_bootstrap,
                )

                row[f"{metric}_ci_low"] = ci_low
                row[f"{metric}_ci_high"] = ci_high

            rows.append(row)

    return pd.DataFrame(rows)

def build_figure(
    results: pd.DataFrame,
    summary: pd.DataFrame,
) -> Figure:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.2, 8.8),
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.10,
        top=0.88,
        wspace=0.28,
        hspace=0.36,
    )

    scatter_recovery_panel(axes[0, 0], results, "mu", "A")
    scatter_recovery_panel(axes[0, 1], results, "theta", "B")
    scatter_recovery_panel(axes[0, 2], results, "sigma", "C")

    grouped_metric_panel(
        axes[1, 0],
        summary,
        metric="bias",
        panel="D",
        title="Estimation bias",
        ylabel="Mean estimation error",
        reference=0.0,
    )
    
    grouped_metric_panel(
        axes[1, 1],
        summary,
        metric="rmse",
        panel="E",
        title="Root-mean-square error",
        ylabel="RMSE",
    )
    
    grouped_metric_panel(
        axes[1, 2],
        summary,
        metric="coverage",
        panel="F",
        title="Empirical interval coverage",
        ylabel="Coverage probability",
        reference=0.95,
    )
    axes[1, 2].set_ylim(0.70, 1.02)
    
    axes[1, 2].axhspan(
        0.925,
        0.975,
        alpha=0.08,
        zorder=0,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(labels),
        frameon=False,
        fontsize=8.8,
    )

    metric_handles, metric_labels = axes[1, 0].get_legend_handles_labels()
    axes[1, 1].legend(
        metric_handles,
        metric_labels,
        loc="upper right",
        frameon=False,
        fontsize=8.2,
    )

    fig.suptitle(
        "Supplementary Figure S2. Recovery of Ornstein–Uhlenbeck parameters",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    success_text = ", ".join(
        f"$n={int(n)}$: {int(count)} successful fits"
        for n, count in results.groupby("n_obs").size().items()
    )
    fig.text(
        0.5,
        0.03,
        "Trajectories were simulated at irregular observation times and fitted using "
        f"the exact OU transition likelihood. {success_text}.",
        ha="center",
        fontsize=8.8,
    )

    return fig


def save_outputs(
    fig: Figure,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path,
    stem: str,
    dpi: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig.savefig(outdir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")

    results.to_csv(outdir / "supplementary_figure_S2_OU_recovery_results.csv", index=False)
    summary.to_csv(outdir / "supplementary_figure_S2_OU_recovery_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure S2: OU parameter recovery."
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
        default="supplementary_figure_S2_OU_parameter_recovery",
        help="Output figure filename stem.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=120,
        help="Replicates per sampling-density condition. Default: 120.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260805,
        help="Master random seed. Default: 20260805.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution. Default: 600.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = BenchmarkConfig(
        replicates=args.replicates,
        seed=args.seed,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        results = run_benchmark(config)

    if results.empty:
        raise RuntimeError("No OU model fits converged.")

    summary = summarize_metrics(
        results,
        seed=args.seed + 1,
        n_bootstrap=1000,
    )
    fig = build_figure(results, summary)
    save_outputs(fig, results, summary, args.outdir, args.stem, args.dpi)
    plt.close(fig)

    print("Saved:")
    for ext in ("png", "pdf", "svg"):
        print(f"  {args.outdir / f'{args.stem}.{ext}'}")
    print(f"  {args.outdir / 'supplementary_figure_S2_OU_recovery_results.csv'}")
    print(f"  {args.outdir / 'supplementary_figure_S2_OU_recovery_summary.csv'}")


if __name__ == "__main__":
    main()
