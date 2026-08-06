from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.mixture import GaussianMixture


METRIC_LABELS = {
    "jump_recovery": "Jump recovery",
    "branch_accuracy": "Branch accuracy",
    "predictive_r2": r"Predictive $R^2$",
}


@dataclass(frozen=True)
class Config:
    seed: int = 20260805
    replicates: int = 40
    n_lineages: int = 120
    n_timepoints: int = 26
    latent_dim: int = 4
    t_end: float = 10.0
    jump_time: float = 3.8
    branch_time: float = 5.4
    holdout_points: int = 4
    theta: float = 0.85
    sigma: float = 0.28
    jump_size: float = 1.15
    branch_probs: tuple[float, ...] = (0.30, 0.40, 0.30)
    observation_noise: float = 0.15
    irregularity: float = 0.20
    missing_fraction: float = 0.05


def make_times(
    n_timepoints: int,
    t_end: float,
    irregularity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate increasing times with controllable interval irregularity."""
    if irregularity <= 0:
        return np.linspace(0.0, t_end, n_timepoints)

    mean_dt = t_end / (n_timepoints - 1)
    shape = max(1.0 / (irregularity**2), 1e-3)
    scale = mean_dt / shape
    intervals = rng.gamma(shape=shape, scale=scale, size=n_timepoints - 1)
    intervals *= t_end / intervals.sum()
    return np.concatenate(([0.0], np.cumsum(intervals)))


def branch_attractors(latent_dim: int) -> np.ndarray:
    """Create three separated branch attractors in q dimensions."""
    attractors = np.zeros((3, latent_dim), dtype=float)
    attractors[0, 0] = -1.15
    attractors[1, 0] = 0.25
    attractors[2, 0] = 1.55

    if latent_dim > 1:
        attractors[0, 1] = 0.75
        attractors[1, 1] = -0.65
        attractors[2, 1] = 0.45

    for d in range(2, latent_dim):
        attractors[:, d] = (
            np.array([-0.35, 0.10, 0.40]) * (0.75 ** (d - 2))
        )
    return attractors


def simulate_latent_oulb(
    cfg: Config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    times = make_times(
        cfg.n_timepoints,
        cfg.t_end,
        cfg.irregularity,
        rng,
    )
    jump_index = int(np.argmin(np.abs(times - cfg.jump_time)))
    branch_index = int(np.argmin(np.abs(times - cfg.branch_time)))
    attractors = branch_attractors(cfg.latent_dim)

    labels = rng.choice(
        3,
        size=cfg.n_lineages,
        p=np.asarray(cfg.branch_probs),
    )
    x = np.empty(
        (cfg.n_lineages, cfg.n_timepoints, cfg.latent_dim),
        dtype=float,
    )
    x[:, 0, :] = rng.normal(
        0.0,
        cfg.sigma / np.sqrt(2.0 * cfg.theta),
        size=(cfg.n_lineages, cfg.latent_dim),
    )

    jump_vector = np.zeros(cfg.latent_dim)
    jump_vector[0] = cfg.jump_size
    if cfg.latent_dim > 1:
        jump_vector[1] = 0.35 * cfg.jump_size

    for i in range(1, cfg.n_timepoints):
        dt = times[i] - times[i - 1]
        decay = np.exp(-cfg.theta * dt)
        variance = (
            cfg.sigma**2
            * (1.0 - np.exp(-2.0 * cfg.theta * dt))
            / (2.0 * cfg.theta)
        )

        if i < branch_index:
            mu = np.zeros((cfg.n_lineages, cfg.latent_dim))
        else:
            mu = attractors[labels]

        mean = mu + (x[:, i - 1, :] - mu) * decay
        x[:, i, :] = rng.normal(
            mean,
            np.sqrt(max(variance, 1e-12)),
        )

        if i == jump_index:
            x[:, i, :] += jump_vector

    return times, x, labels, jump_index, branch_index


def add_observation_process(
    latent: np.ndarray,
    noise: float,
    missing_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    observed = latent + rng.normal(0.0, noise, size=latent.shape)

    if missing_fraction > 0:
        mask = rng.random(observed.shape) < missing_fraction
        mask[:, 0, :] = False
        mask[:, -1, :] = False
        observed[mask] = np.nan

    return observed


def interpolate_missing(values: np.ndarray) -> np.ndarray:
    """Linear interpolation along time for each lineage and dimension."""
    out = values.copy()
    n_lineages, n_times, n_dim = out.shape
    index = np.arange(n_times)

    for n in range(n_lineages):
        for d in range(n_dim):
            y = out[n, :, d]
            valid = np.isfinite(y)
            if np.sum(valid) < 2:
                fill = np.nanmean(y)
                out[n, :, d] = 0.0 if not np.isfinite(fill) else fill
            else:
                out[n, :, d] = np.interp(
                    index,
                    index[valid],
                    y[valid],
                )
    return out


def detect_shared_jump(
    observed: np.ndarray,
    branch_index: int,
) -> int:
    """Locate the largest shared multivariate displacement before branching."""
    differences = np.diff(observed[:, :branch_index, :], axis=1)
    norms = np.linalg.norm(differences, axis=2)
    score = np.median(norms, axis=0)
    return int(np.argmax(score)) + 1


def make_branch_features(
    observed: np.ndarray,
    branch_index: int,
    training_end: int,
) -> np.ndarray:
    post = observed[:, branch_index:training_end, :]
    mean = post.mean(axis=1)
    terminal = post[:, -1, :]
    displacement = terminal - observed[:, branch_index, :]
    return np.concatenate([mean, terminal, displacement], axis=1)


def align_labels(
    true_labels: np.ndarray,
    raw_labels: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    cm = confusion_matrix(
        true_labels,
        raw_labels,
        labels=np.arange(3),
    )
    true_idx, component_idx = linear_sum_assignment(-cm)
    mapping = {
        int(component): int(branch)
        for branch, component in zip(true_idx, component_idx)
    }
    aligned = np.array(
        [mapping[int(label)] for label in raw_labels],
        dtype=int,
    )
    return aligned, mapping


def infer_branches(
    observed: np.ndarray,
    true_labels: np.ndarray,
    branch_index: int,
    training_end: int,
    seed: int,
) -> np.ndarray:
    features = make_branch_features(
        observed,
        branch_index,
        training_end,
    )
    model = GaussianMixture(
        n_components=3,
        covariance_type="diag",
        n_init=3,
        reg_covar=1e-5,
        random_state=seed,
    )
    raw = model.fit_predict(features)
    aligned, _ = align_labels(true_labels, raw)
    return aligned


def estimate_branch_ar1(
    observed: np.ndarray,
    inferred_labels: np.ndarray,
    branch_index: int,
    training_end: int,
) -> dict[int, tuple[np.ndarray, float]]:
    """
    Estimate branch-specific attractors and one shared AR coefficient per branch.
    """
    estimates: dict[int, tuple[np.ndarray, float]] = {}

    for branch in range(3):
        values = observed[
            inferred_labels == branch,
            branch_index:training_end,
            :,
        ]
        mu_hat = values.mean(axis=(0, 1))

        previous = values[:, :-1, :] - mu_hat
        current = values[:, 1:, :] - mu_hat
        denominator = float(np.sum(previous**2))
        rho = (
            float(np.sum(previous * current)) / denominator
            if denominator > 1e-12
            else 0.0
        )
        rho = float(np.clip(rho, 0.0, 0.995))
        estimates[branch] = (mu_hat, rho)

    return estimates


def predict_holdout(
    observed: np.ndarray,
    inferred_labels: np.ndarray,
    branch_estimates: dict[int, tuple[np.ndarray, float]],
    training_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = []
    actual = []

    for lineage in range(observed.shape[0]):
        branch = int(inferred_labels[lineage])
        mu_hat, rho = branch_estimates[branch]
        state = observed[lineage, training_end - 1, :].copy()

        for i in range(training_end, observed.shape[1]):
            state = mu_hat + rho * (state - mu_hat)
            predicted.append(state.copy())
            actual.append(observed[lineage, i, :].copy())

    return np.asarray(actual).ravel(), np.asarray(predicted).ravel()


def evaluate_once(
    cfg: Config,
    rng: np.random.Generator,
    seed: int,
) -> dict[str, float]:
    times, latent, true_labels, jump_index, branch_index = simulate_latent_oulb(
        cfg,
        rng,
    )
    observed = add_observation_process(
        latent,
        cfg.observation_noise,
        cfg.missing_fraction,
        rng,
    )
    observed = interpolate_missing(observed)
    training_end = cfg.n_timepoints - cfg.holdout_points

    detected_jump = detect_shared_jump(
        observed,
        branch_index,
    )
    inferred = infer_branches(
        observed,
        true_labels,
        branch_index,
        training_end,
        seed,
    )
    estimates = estimate_branch_ar1(
        observed,
        inferred,
        branch_index,
        training_end,
    )
    actual, predicted = predict_holdout(
        observed,
        inferred,
        estimates,
        training_end,
    )

    branch_accuracy = float(np.mean(inferred == true_labels))
    predictive_r2 = float(r2_score(actual, predicted))
    predictive_r2 = float(np.clip(predictive_r2, -0.25, 1.0))

    return {
        "jump_recovery": float(detected_jump == jump_index),
        "branch_accuracy": branch_accuracy,
        "predictive_r2": predictive_r2,
    }


def run_condition_grid(
    base: Config,
    factor: str,
    values: list[float | int],
) -> pd.DataFrame:
    master = np.random.SeedSequence(
        base.seed + sum(ord(c) for c in factor)
    )
    rngs = [
        np.random.default_rng(s)
        for s in master.spawn(len(values) * base.replicates)
    ]
    rows = []
    idx = 0

    for value in values:
        for replicate in range(base.replicates):
            kwargs = {}
            if factor == "observation_noise":
                kwargs["observation_noise"] = float(value)
            elif factor == "irregularity":
                kwargs["irregularity"] = float(value)
            elif factor == "missing_fraction":
                kwargs["missing_fraction"] = float(value)
            elif factor == "n_lineages":
                kwargs["n_lineages"] = int(value)
            elif factor == "latent_dim":
                kwargs["latent_dim"] = int(value)
            else:
                raise ValueError(f"Unsupported factor: {factor}")

            cfg = replace(base, **kwargs)
            metrics = evaluate_once(
                cfg,
                rngs[idx],
                seed=base.seed + idx,
            )
            idx += 1
            for metric, metric_value in metrics.items():
                rows.append({
                    "factor": factor,
                    "condition": value,
                    "replicate": replicate + 1,
                    "metric": metric,
                    "value": metric_value,
                })

    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 500,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means[b] = np.mean(sample)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(np.mean(values)), float(low), float(high)


def summarize_results(
    results: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for keys, subset in results.groupby(
        ["factor", "condition", "metric"],
        sort=True,
    ):
        factor, condition, metric = keys
        mean, low, high = bootstrap_mean_ci(
            subset["value"].to_numpy(),
            rng,
        )
        rows.append({
            "factor": factor,
            "condition": condition,
            "metric": metric,
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
        })
    return pd.DataFrame(rows)


def measure_runtime(
    base: Config,
    sample_sizes: list[int],
    dimensions: list[int],
    repeats: int = 4,
) -> pd.DataFrame:
    rows = []
    master = np.random.SeedSequence(base.seed + 9000)
    total = len(sample_sizes) * len(dimensions) * repeats
    rngs = [
        np.random.default_rng(s)
        for s in master.spawn(total)
    ]
    idx = 0

    for q in dimensions:
        for n in sample_sizes:
            for repeat_index in range(repeats):
                cfg = replace(
                    base,
                    n_lineages=n,
                    latent_dim=q,
                    replicates=1,
                )
                start = time.perf_counter()
                evaluate_once(
                    cfg,
                    rngs[idx],
                    seed=base.seed + 10000 + idx,
                )
                elapsed = time.perf_counter() - start
                idx += 1
                rows.append({
                    "latent_dim": q,
                    "n_lineages": n,
                    "repeat": repeat_index + 1,
                    "runtime_seconds": elapsed,
                })

    return pd.DataFrame(rows)


def style_axis(
    ax: Axes,
    panel: str,
    title: str,
) -> None:
    ax.text(
        -0.13,
        1.08,
        panel,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    ax.set_title(
        title,
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax.tick_params(labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_sensitivity_panel(
    ax: Axes,
    summary: pd.DataFrame,
    factor: str,
    panel: str,
    title: str,
    xlabel: str,
    x_as_percent: bool = False,
) -> None:
    subset = summary.loc[summary["factor"] == factor]
    markers = {
        "jump_recovery": "o",
        "branch_accuracy": "s",
        "predictive_r2": "^",
    }

    for metric in METRIC_LABELS:
        metric_data = (
            subset.loc[subset["metric"] == metric]
            .sort_values("condition")
        )
        x = metric_data["condition"].to_numpy(dtype=float)
        if x_as_percent:
            x = x * 100.0

        mean = metric_data["mean"].to_numpy()
        low = metric_data["ci_low"].to_numpy()
        high = metric_data["ci_high"].to_numpy()

        ax.plot(
            x,
            mean,
            marker=markers[metric],
            linewidth=2.0,
            markersize=5.5,
            label=METRIC_LABELS[metric],
        )
        ax.fill_between(
            x,
            low,
            high,
            alpha=0.10,
        )

    ax.set_ylim(-0.05, 1.03)
    ax.axhline(0.8, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Performance")
    style_axis(ax, panel, title)


def build_figure(
    summary: pd.DataFrame,
    runtime: pd.DataFrame,
) -> Figure:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.2, 8.8),
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.10,
        top=0.88,
        wspace=0.29,
        hspace=0.38,
    )

    plot_sensitivity_panel(
        axes[0, 0],
        summary,
        "observation_noise",
        "A",
        "Sensitivity to observation noise",
        "Observation-noise SD",
    )
    plot_sensitivity_panel(
        axes[0, 1],
        summary,
        "irregularity",
        "B",
        "Sensitivity to irregular sampling",
        "Sampling-interval CV",
    )
    plot_sensitivity_panel(
        axes[0, 2],
        summary,
        "missing_fraction",
        "C",
        "Sensitivity to missing data",
        "Missing observations (%)",
        x_as_percent=True,
    )
    plot_sensitivity_panel(
        axes[1, 0],
        summary,
        "n_lineages",
        "D",
        "Sensitivity to sample size",
        "Number of lineages",
    )
    plot_sensitivity_panel(
        axes[1, 1],
        summary,
        "latent_dim",
        "E",
        "Sensitivity to latent dimension",
        "Latent dimension",
    )

    # F. Runtime.
    ax = axes[1, 2]
    runtime_summary = (
        runtime.groupby(
            ["latent_dim", "n_lineages"],
            as_index=False,
        )["runtime_seconds"]
        .agg(["mean", "std"])
        .reset_index()
    )
    for q, subset in runtime_summary.groupby(
        "latent_dim",
        sort=True,
    ):
        subset = subset.sort_values("n_lineages")
        ax.errorbar(
            subset["n_lineages"],
            subset["mean"],
            yerr=subset["std"].fillna(0.0),
            marker="o",
            linewidth=2.0,
            capsize=3,
            label=f"$q={int(q)}$",
        )
    ax.set_xlabel("Number of lineages")
    ax.set_ylabel("Runtime (seconds)")
    ax.legend(
        frameon=False,
        fontsize=8,
        title="Latent dimension",
        title_fontsize=8,
    )
    style_axis(
        ax,
        "F",
        "Runtime scaling",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=8.8,
    )

    fig.suptitle(
        "Supplementary Figure S6. Sensitivity analyses of the OULB framework",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.03,
        "Lines show mean performance across simulation replicates; shaded regions "
        "denote bootstrap 95% confidence intervals. Runtime includes simulation, "
        "jump localization, branch reconstruction, and held-out prediction.",
        ha="center",
        fontsize=8.8,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure S6: sensitivity analyses."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--stem",
        default="supplementary_figure_S6_sensitivity_analyses",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260805,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Config(
        seed=args.seed,
        replicates=args.replicates,
    )

    grids = {
        "observation_noise": [0.00, 0.10, 0.20, 0.35, 0.50, 0.70],
        "irregularity": [0.00, 0.20, 0.40, 0.60, 0.80, 1.00],
        "missing_fraction": [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
        "n_lineages": [30, 60, 120, 180, 240],
        "latent_dim": [1, 2, 4, 8, 12, 16],
    }

    all_results = []
    for factor, values in grids.items():
        all_results.append(
            run_condition_grid(
                base,
                factor,
                values,
            )
        )
    results = pd.concat(
        all_results,
        ignore_index=True,
    )
    summary = summarize_results(
        results,
        seed=args.seed + 1,
    )

    runtime = measure_runtime(
        base,
        sample_sizes=[30, 60, 120, 240],
        dimensions=[1, 4, 8, 16],
        repeats=4,
    )

    fig = build_figure(
        summary,
        runtime,
    )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            args.outdir / f"{args.stem}.{ext}",
            dpi=args.dpi if ext == "png" else None,
            bbox_inches="tight",
        )

    results.to_csv(
        args.outdir
        / "supplementary_figure_S6_sensitivity_results.csv",
        index=False,
    )
    summary.to_csv(
        args.outdir
        / "supplementary_figure_S6_sensitivity_summary.csv",
        index=False,
    )
    runtime.to_csv(
        args.outdir
        / "supplementary_figure_S6_runtime_results.csv",
        index=False,
    )
    plt.close(fig)

    print(
        "Saved Figure S6 outputs to",
        args.outdir.resolve(),
    )


if __name__ == "__main__":
    main()
