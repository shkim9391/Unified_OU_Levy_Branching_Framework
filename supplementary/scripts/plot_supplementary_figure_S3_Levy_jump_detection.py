from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import auc, precision_recall_curve, roc_curve


@dataclass(frozen=True)
class Config:
    replicates: int = 250
    null_replicates: int = 250
    seed: int = 20260805
    n_obs: int = 60
    t_end: float = 10.0
    mu: float = 0.5
    theta: float = 0.85
    sigma: float = 0.35
    jump_sizes: tuple[float, ...] = (0.5, 0.8, 1.1, 1.5, 2.0)
    prior_jump: float = 0.05
    jump_scale_multiplier: float = 4.0


def irregular_times(n_obs, t_end, rng):
    interior = np.sort(rng.uniform(0, t_end, n_obs - 2))
    return np.concatenate(([0.0], interior, [t_end]))


def simulate_ou(times, x0, mu, theta, sigma, rng, jump_index=None, jump_size=0.0):
    x = np.empty_like(times)
    x[0] = x0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        decay = np.exp(-theta * dt)
        mean = mu + (x[i - 1] - mu) * decay
        var = sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
        x[i] = rng.normal(mean, np.sqrt(max(var, 1e-12)))
        if jump_index is not None and i == jump_index:
            x[i] += jump_size
    return x


def innovations(times, x, mu, theta, sigma):
    dt = np.diff(times)
    decay = np.exp(-theta * dt)
    mean = mu + (x[:-1] - mu) * decay
    var = sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
    return x[1:] - mean, np.maximum(var, 1e-12)


def log_normal_density(values, variance):
    return -0.5 * (np.log(2 * np.pi * variance) + values**2 / variance)


def jump_probabilities(residuals, variance, prior_jump, jump_scale_multiplier):
    tau2 = jump_scale_multiplier**2 * np.median(variance)
    log0 = log_normal_density(residuals, variance)
    log1 = log_normal_density(residuals, variance + tau2)
    log_prior_odds = np.log(prior_jump / (1 - prior_jump))
    return expit(log_prior_odds + log1 - log0)


def detect(times, x, cfg):
    residuals, variance = innovations(times, x, cfg.mu, cfg.theta, cfg.sigma)
    probs = jump_probabilities(
        residuals, variance, cfg.prior_jump, cfg.jump_scale_multiplier
    )
    k = int(np.argmax(probs))
    return {
        "probs": probs,
        "index": k + 1,
        "time": float(times[k + 1]),
        "size": float(residuals[k]),
        "max_prob": float(probs[k]),
    }


def run_benchmark(cfg):
    master = np.random.SeedSequence(cfg.seed)
    rngs = [np.random.default_rng(s) for s in master.spawn(
        cfg.replicates + cfg.null_replicates + 1
    )]

    # Representative example
    rng = rngs[0]
    times = irregular_times(cfg.n_obs, cfg.t_end, rng)
    jump_index = int(np.argmin(np.abs(times - 5.2)))
    x0 = rng.normal(cfg.mu, cfg.sigma / np.sqrt(2 * cfg.theta))
    x = simulate_ou(times, x0, cfg.mu, cfg.theta, cfg.sigma, rng,
                    jump_index=jump_index, jump_size=1.5)
    example = {
        "times": times, "x": x, "jump_index": jump_index,
        "jump_time": float(times[jump_index]), "jump_size": 1.5,
        **detect(times, x, cfg),
    }

    event_rows = []
    trajectory_rows = []

    for r in range(cfg.replicates):
        rng = rngs[r + 1]
        times = irregular_times(cfg.n_obs, cfg.t_end, rng)
        target = rng.uniform(2, 8)
        j = int(np.argmin(np.abs(times - target)))
        j = min(max(j, 2), len(times) - 2)
        size = float(rng.choice(cfg.jump_sizes) * rng.choice([-1, 1]))
        x0 = rng.normal(cfg.mu, cfg.sigma / np.sqrt(2 * cfg.theta))
        x = simulate_ou(times, x0, cfg.mu, cfg.theta, cfg.sigma, rng,
                        jump_index=j, jump_size=size)
        d = detect(times, x, cfg)

        trajectory_rows.append({
            "dataset": "jump", "replicate": r + 1,
            "true_time": times[j], "detected_time": d["time"],
            "true_size": size, "detected_size": d["size"],
            "max_probability": d["max_prob"],
        })
        for k, p in enumerate(d["probs"], start=1):
            event_rows.append({
                "dataset": "jump", "replicate": r + 1,
                "is_true_jump": int(k == j), "probability": float(p),
            })

    null_rows = []
    for r in range(cfg.null_replicates):
        rng = rngs[1 + cfg.replicates + r]
        times = irregular_times(cfg.n_obs, cfg.t_end, rng)
        x0 = rng.normal(cfg.mu, cfg.sigma / np.sqrt(2 * cfg.theta))
        x = simulate_ou(times, x0, cfg.mu, cfg.theta, cfg.sigma, rng)
        d = detect(times, x, cfg)
        null_rows.append({
            "replicate": r + 1,
            "max_probability": d["max_prob"],
        })
        for p in d["probs"]:
            event_rows.append({
                "dataset": "null", "replicate": r + 1,
                "is_true_jump": 0, "probability": float(p),
            })

    return (
        example,
        pd.DataFrame(trajectory_rows),
        pd.DataFrame(event_rows),
        pd.DataFrame(null_rows),
    )


def style(ax, letter, title):
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)


def build_figure(example, trajectories, events, nulls):
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.8))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10,
                        top=0.88, wspace=0.30, hspace=0.38)

    # A
    ax = axes[0, 0]
    t, x = example["times"], example["x"]
    j = example["jump_index"]
    ax.plot(t, x, marker="o", markersize=3.2, linewidth=1.6)
    ax.axvline(example["jump_time"], linestyle="--", linewidth=1.4)
    ax.scatter([t[j]], [x[j]], s=55, facecolors="white",
               edgecolors="black", zorder=5)
    ax.annotate("True jump", xy=(t[j], x[j]),
                xytext=(t[j] + 0.7, x[j] + 0.06),
                arrowprops=dict(arrowstyle="->", linewidth=1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Latent state")
    style(ax, "A", "Representative trajectory with a Lévy jump")

    # B
    ax = axes[0, 1]
    tt = t[1:]
    p = example["probs"]
    ax.plot(tt, p, marker="o", markersize=3.2, linewidth=1.6)
    ax.axvline(example["jump_time"], linestyle="--", linewidth=1.4)
    ax.axhline(0.5, linestyle=":", linewidth=1.2)
    ax.fill_between(tt, 0.5, p, where=p >= 0.5, alpha=0.15)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time")
    ax.set_ylabel("Posterior jump probability")
    style(ax, "B", "Jump probability across time")

    # C
    ax = axes[0, 2]
    time_error = trajectories["detected_time"] - trajectories["true_time"]
    lim = max(0.5, np.percentile(np.abs(time_error), 98) * 1.1)
    ax.hist(time_error, bins=np.linspace(-lim, lim, 25),
            alpha=0.8, edgecolor="white")
    ax.axvline(0, linestyle="--", linewidth=1.4)
    med = np.median(np.abs(time_error))
    ax.text(0.97, 0.94, f"Median |error| = {med:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="0.75"))
    ax.set_xlabel("Detected time − true time")
    ax.set_ylabel("Number of trajectories")
    style(ax, "C", "Jump-time recovery")

    # D
    ax = axes[1, 0]
    ax.scatter(trajectories["true_size"], trajectories["detected_size"],
               s=20, alpha=0.55)
    vals = np.concatenate([trajectories["true_size"],
                           trajectories["detected_size"]])
    lo, hi = np.percentile(vals, [1, 99])
    pad = 0.1 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    r = np.corrcoef(trajectories["true_size"],
                    trajectories["detected_size"])[0, 1]
    rmse = np.sqrt(np.mean(
        (trajectories["detected_size"] - trajectories["true_size"])**2
    ))
    ax.text(0.05, 0.94, f"$r$ = {r:.2f}\nRMSE = {rmse:.2f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor="0.75"))
    ax.set_xlabel("True jump size")
    ax.set_ylabel("Estimated jump size")
    style(ax, "D", "Jump-size recovery")

    # E
    ax = axes[1, 1]
    y = events["is_true_jump"].to_numpy()
    score = events["probability"].to_numpy()
    fpr, tpr, _ = roc_curve(y, score)
    precision, recall, _ = precision_recall_curve(y, score)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot(recall, precision, linewidth=2,
            label=f"Precision–recall (AUC = {pr_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
    ax.axhline(np.mean(y), linestyle="--", linewidth=1,
               label=f"PR baseline = {np.mean(y):.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-positive rate or recall")
    ax.set_ylabel("True-positive rate or precision")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, "E", "Jump-classification performance")

    # F
    ax = axes[1, 2]
    thresholds = np.array([0.25, 0.50, 0.75, 0.90])
    fpr_traj = np.array([
        np.mean(nulls["max_probability"] >= q) for q in thresholds
    ])
    n = len(nulls)
    se = np.sqrt(fpr_traj * (1 - fpr_traj) / n)
    yerr = 1.96 * se
    ax.errorbar(thresholds, fpr_traj, yerr=yerr,
                marker="o", linewidth=2, capsize=4)
    ax.axhline(0.05, linestyle="--", linewidth=1.2)
    ax.set_xticks(thresholds)
    ax.set_ylim(0, min(1, max(0.2, np.max(fpr_traj + yerr) + 0.08)))
    ax.set_xlabel("Posterior-probability threshold")
    ax.set_ylabel("Trajectory-level false-positive rate")
    style(ax, "F", "False-positive rate under pure OU dynamics")

    fig.suptitle(
        "Supplementary Figure S3. Validation of Lévy jump detection",
        fontsize=15, fontweight="bold", y=0.985
    )
    fig.text(
        0.5, 0.03,
        "Jump-containing and pure-OU trajectories were simulated at irregular times. "
        "Detection used exact OU innovations and a spike-and-slab-inspired event model.",
        ha="center", fontsize=8.8
    )

    summary = pd.DataFrame({
        "metric": [
            "median_absolute_time_error", "jump_size_correlation",
            "jump_size_rmse", "roc_auc", "precision_recall_auc",
            "false_positive_rate_threshold_0.5",
        ],
        "value": [
            med, r, rmse, roc_auc, pr_auc,
            fpr_traj[1],
        ],
    })
    return fig, summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("."))
    p.add_argument("--stem", default="supplementary_figure_S3_Levy_jump_detection")
    p.add_argument("--replicates", type=int, default=250)
    p.add_argument("--null-replicates", type=int, default=250)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        replicates=args.replicates,
        null_replicates=args.null_replicates,
        seed=args.seed,
    )
    example, trajectories, events, nulls = run_benchmark(cfg)
    fig, summary = build_figure(example, trajectories, events, nulls)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.outdir / f"{args.stem}.{ext}",
                    dpi=args.dpi if ext == "png" else None,
                    bbox_inches="tight")
    trajectories.to_csv(
        args.outdir / "supplementary_figure_S3_jump_recovery.csv",
        index=False
    )
    events.to_csv(
        args.outdir / "supplementary_figure_S3_event_scores.csv",
        index=False
    )
    nulls.to_csv(
        args.outdir / "supplementary_figure_S3_null_results.csv",
        index=False
    )
    summary.to_csv(
        args.outdir / "supplementary_figure_S3_summary.csv",
        index=False
    )
    plt.close(fig)

    print("Saved Figure S3 outputs to", args.outdir.resolve())


if __name__ == "__main__":
    main()
