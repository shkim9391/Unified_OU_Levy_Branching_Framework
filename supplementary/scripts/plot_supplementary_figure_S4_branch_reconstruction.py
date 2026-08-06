from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    adjusted_rand_score,
)

@dataclass(frozen=True)
class Config:
    seed: int = 20260805
    n_lineages: int = 240
    n_branches: int = 3
    n_timepoints: int = 24
    branch_time_index: int = 8
    theta: float = 0.9
    sigma: float = 0.32
    ancestral_mu: float = 0.0
    branch_attractors: tuple[float, ...] = (-1.3, 0.25, 1.6)
    branch_probs: tuple[float, ...] = (0.30, 0.40, 0.30)
    replicates: int = 200
    separation_grid: tuple[float, ...] = (0.4, 0.7, 1.0, 1.3, 1.6, 2.0)


def simulate_exact_ou_segment(
    times: np.ndarray,
    x0: float,
    mu: float,
    theta: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.empty_like(times)
    x[0] = x0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        decay = np.exp(-theta * dt)
        mean = mu + (x[i - 1] - mu) * decay
        variance = sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
        x[i] = rng.normal(mean, np.sqrt(max(variance, 1e-12)))
    return x


def simulate_dataset(
    cfg: Config,
    rng: np.random.Generator,
    separation_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.linspace(0.0, 10.0, cfg.n_timepoints)
    branch_time = cfg.branch_time_index

    labels = rng.choice(
        cfg.n_branches,
        size=cfg.n_lineages,
        p=np.asarray(cfg.branch_probs),
    )

    base = np.asarray(cfg.branch_attractors)
    centered = base - np.mean(base)
    attractors = np.mean(base) + separation_scale * centered

    trajectories = np.empty((cfg.n_lineages, cfg.n_timepoints), dtype=float)

    for i, label in enumerate(labels):
        x0 = rng.normal(
            cfg.ancestral_mu,
            cfg.sigma / np.sqrt(2 * cfg.theta),
        )
        ancestor = simulate_exact_ou_segment(
            times[: branch_time + 1],
            x0,
            cfg.ancestral_mu,
            cfg.theta,
            cfg.sigma,
            rng,
        )
        descendant = simulate_exact_ou_segment(
            times[branch_time:],
            ancestor[-1],
            float(attractors[label]),
            cfg.theta,
            cfg.sigma,
            rng,
        )
        trajectories[i, : branch_time + 1] = ancestor
        trajectories[i, branch_time:] = descendant

    return times, trajectories, labels, attractors


def trajectory_features(
    trajectories: np.ndarray,
    branch_time_index: int,
) -> np.ndarray:
    post = trajectories[:, branch_time_index:]
    terminal = post[:, -1]
    post_mean = post.mean(axis=1)
    post_slope = np.polyfit(
        np.arange(post.shape[1]),
        post.T,
        deg=1,
    )[0]
    displacement = terminal - trajectories[:, branch_time_index]
    return np.column_stack([post_mean, terminal, post_slope, displacement])


def align_components(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    n_branches: int,
) -> tuple[np.ndarray, dict[int, int]]:
    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(n_branches),
    )
    rows, cols = linear_sum_assignment(-cm)
    mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    aligned = np.array([mapping[int(v)] for v in predicted_labels], dtype=int)
    return aligned, mapping


def align_probabilities(
    raw_probabilities: np.ndarray,
    mapping: dict[int, int],
    n_branches: int,
) -> np.ndarray:
    aligned = np.zeros_like(raw_probabilities)
    for component, branch in mapping.items():
        aligned[:, branch] = raw_probabilities[:, component]
    return aligned


def infer_branches(
    trajectories: np.ndarray,
    true_labels: np.ndarray,
    cfg: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, GaussianMixture]:
    features = trajectory_features(
        trajectories,
        cfg.branch_time_index,
    )
    model = GaussianMixture(
        n_components=cfg.n_branches,
        covariance_type="full",
        random_state=seed,
        n_init=3,
        reg_covar=1e-6,
    )
    raw_labels = model.fit_predict(features)
    raw_probs = model.predict_proba(features)

    labels, mapping = align_components(
        true_labels,
        raw_labels,
        cfg.n_branches,
    )
    probabilities = align_probabilities(
        raw_probs,
        mapping,
        cfg.n_branches,
    )
    return labels, probabilities, model


def true_branch_probabilities(
    trajectories: np.ndarray,
    attractors: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    Construct soft true branch probabilities from terminal-state proximity.

    These probabilities provide a continuous target for calibration rather
    than using only one-hot branch labels.
    """
    terminal = trajectories[:, -1][:, None]
    scale = 0.45
    logits = -0.5 * ((terminal - attractors[None, :]) / scale) ** 2
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def transition_matrix_from_labels(
    labels: np.ndarray,
    n_branches: int,
) -> np.ndarray:
    """
    Construct ancestral-to-descendant transition probabilities.

    Row 0 is the shared ancestor. Rows 1..K are absorbing descendant states.
    """
    matrix = np.zeros((n_branches + 1, n_branches + 1), dtype=float)
    counts = np.bincount(labels, minlength=n_branches)
    matrix[0, 1:] = counts / counts.sum()
    for k in range(n_branches):
        matrix[k + 1, k + 1] = 1.0
    return matrix


def run_accuracy_benchmark(
    cfg: Config,
) -> pd.DataFrame:
    master = np.random.SeedSequence(cfg.seed + 1000)
    total = cfg.replicates * len(cfg.separation_grid)
    child_sequences = master.spawn(total)
    rows = []
    idx = 0

    benchmark_cfg = Config(
        seed=cfg.seed,
        n_lineages=min(cfg.n_lineages, 90),
        n_branches=cfg.n_branches,
        n_timepoints=cfg.n_timepoints,
        branch_time_index=cfg.branch_time_index,
        theta=cfg.theta,
        sigma=cfg.sigma,
        ancestral_mu=cfg.ancestral_mu,
        branch_attractors=cfg.branch_attractors,
        branch_probs=cfg.branch_probs,
        replicates=cfg.replicates,
        separation_grid=cfg.separation_grid,
    )

    for separation in cfg.separation_grid:
        for replicate in range(cfg.replicates):
            rng = np.random.default_rng(child_sequences[idx])
            idx += 1
            _, trajectories, labels, _ = simulate_dataset(
                benchmark_cfg,
                rng,
                separation_scale=separation,
            )
            inferred, _, _ = infer_branches(
                trajectories,
                labels,
                benchmark_cfg,
                seed=cfg.seed + replicate,
            )
            rows.append(
                {
                    "separation": separation,
                    "replicate": replicate + 1,
                    "accuracy": float(np.mean(inferred == labels)),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[b] = np.mean(sample)
    return tuple(np.percentile(boot, [2.5, 97.5]))


def style_axis(
    ax: Axes,
    panel: str,
    title: str,
    panel_x: float = -0.13,
):
    ax.text(
        panel_x,
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


def draw_tree(
    ax: Axes,
    branch_values: np.ndarray,
    branch_labels: list[str],
    title_letter: str,
    title: str,
    inferred: bool = False,
) -> None:
    root_x, root_y = 0.12, 0.50
    split_x = 0.45
    end_x = 0.90
    ys = np.linspace(0.18, 0.82, len(branch_values))

    ax.plot([root_x, split_x], [root_y, root_y], linewidth=2.2)
    ax.scatter([root_x], [root_y], s=55, zorder=5)
    ax.scatter([split_x], [root_y], s=65, facecolors="white",
               edgecolors="black", zorder=5)

    for y, value, label in zip(ys, branch_values, branch_labels):
        ax.plot([split_x, end_x], [root_y, y], linewidth=2.0)
        ax.scatter([end_x], [y], s=55, zorder=5)
        ax.text(
            end_x - 0.01,
            y + 0.07,
            f"{label}\n$n={int(value)}$",
            ha="center",
            fontsize=8.6,
        )

    ax.text(root_x, root_y - 0.10, "Shared ancestor", ha="center", fontsize=8.5)
    ax.text(split_x, root_y - 0.10, "Branch point", ha="center", fontsize=8.5)

    if inferred:
        ax.text(
            0.50,
            0.04,
            "Topology reconstructed from trajectory-level features",
            ha="center",
            fontsize=8.2,
            style="italic",
        )
    else:
        ax.text(
            0.50,
            0.04,
            "Known simulated lineage topology",
            ha="center",
            fontsize=8.2,
            style="italic",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    style_axis(ax, title_letter, title)


def build_figure(
    cfg: Config,
    times: np.ndarray,
    trajectories: np.ndarray,
    true_labels: np.ndarray,
    inferred_labels: np.ndarray,
    inferred_probs: np.ndarray,
    attractors: np.ndarray,
    accuracy_results: pd.DataFrame,
) -> tuple[Figure, pd.DataFrame, pd.DataFrame]:
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.8))
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.10,
        top=0.88,
        wspace=0.30,
        hspace=0.38,
    )

    true_counts = np.bincount(true_labels, minlength=cfg.n_branches)
    inferred_counts = np.bincount(inferred_labels, minlength=cfg.n_branches)
    branch_names = [f"Branch {i + 1}" for i in range(cfg.n_branches)]

    # A. Simulated lineage.
    draw_tree(
        axes[0, 0],
        true_counts,
        branch_names,
        "A",
        "Simulated lineage",
        inferred=False,
    )

    # B. Inferred lineage.
    draw_tree(
        axes[0, 1],
        inferred_counts,
        branch_names,
        "B",
        "Inferred lineage",
        inferred=True,
    )

    # C. Confusion matrix.
    ax = axes[0, 2]
    cm = confusion_matrix(
        true_labels,
        inferred_labels,
        labels=np.arange(cfg.n_branches),
    )
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)
    image = ax.imshow(cm_norm, vmin=0, vmax=1, aspect="equal")
    for i in range(cfg.n_branches):
        for j in range(cfg.n_branches):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set_xticks(range(cfg.n_branches))
    ax.set_yticks(range(cfg.n_branches))
    ax.set_xticklabels(branch_names, rotation=25, ha="right")
    ax.set_yticklabels(branch_names)
    ax.set_xlabel("Inferred branch")
    ax.set_ylabel("True branch")
    style_axis(
        ax,
        "C",
        "Branch-assignment confusion matrix",
        panel_x=-0.32,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                 label="Row-normalized proportion")

    # D. Branch-probability recovery.
    ax = axes[1, 0]
    true_probs = true_branch_probabilities(
        trajectories,
        attractors,
        cfg,
    )
    for branch in range(cfg.n_branches):
        ax.scatter(
            true_probs[:, branch],
            inferred_probs[:, branch],
            s=18,
            alpha=0.45,
            label=branch_names[branch],
        )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.6)
    brier = float(np.mean((true_probs - inferred_probs) ** 2))
    corr = float(
        np.corrcoef(true_probs.ravel(), inferred_probs.ravel())[0, 1]
    )
    ax.text(
        0.05,
        0.95,
        f"$r$ = {corr:.2f}\nBrier = {brier:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", edgecolor="0.75"),
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("True branch probability")
    ax.set_ylabel("Inferred branch probability")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(ax, "D", "Branch-probability recovery")

    # E. Transition matrices.
    ax = axes[1, 1]
    true_matrix = transition_matrix_from_labels(
        true_labels,
        cfg.n_branches,
    )
    inferred_matrix = transition_matrix_from_labels(
        inferred_labels,
        cfg.n_branches,
    )
    spacer = np.full((cfg.n_branches + 1, 1), np.nan)
    combined = np.concatenate([true_matrix, spacer, inferred_matrix], axis=1)
    masked = np.ma.masked_invalid(combined)
    image = ax.imshow(masked, vmin=0, vmax=1, aspect="auto")

    labels = ["Ancestor"] + branch_names
    ax.set_yticks(range(cfg.n_branches + 1))
    ax.set_yticklabels(labels)
    x_positions = list(range(cfg.n_branches + 1)) + \
        list(range(cfg.n_branches + 2, 2 * cfg.n_branches + 3))
    x_labels = labels + labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7.5)

    for i in range(cfg.n_branches + 1):
        for j in range(cfg.n_branches + 1):
            ax.text(j, i, f"{true_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=7.6)
            ax.text(j + cfg.n_branches + 2, i,
                    f"{inferred_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=7.6)

    ax.text(
        (cfg.n_branches) / 2,
        -0.53,
        "True",
        ha="center",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        cfg.n_branches + 2 + cfg.n_branches / 2,
        -0.53,
        "Inferred",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.set_xlabel("Destination state")
    ax.set_ylabel("Origin state")
    style_axis(ax, "E", "Branch-transition matrix recovery")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                 label="Transition probability")

    overall_accuracy = accuracy_score(
        true_labels,
        inferred_labels,
    )
    
    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        inferred_labels,
    )
    
    macro_f1 = f1_score(
        true_labels,
        inferred_labels,
        average="macro",
    )
    
    ari = adjusted_rand_score(
        true_labels,
        inferred_labels,
    )
    
    # F. Accuracy versus branch separation.
    ax = axes[1, 2]
    summary_rows = []
    ci_rng = np.random.default_rng(cfg.seed + 5000)

    for separation, subset in accuracy_results.groupby("separation", sort=True):
        values = subset["accuracy"].to_numpy()
        mean = float(np.mean(values))
        low, high = bootstrap_ci(values, ci_rng)
        summary_rows.append(
            {
                "separation": separation,
                "mean_accuracy": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )

    acc_summary = pd.DataFrame(summary_rows)
    values = acc_summary["mean_accuracy"].to_numpy()
    yerr = np.vstack([
        values - acc_summary["ci_low"].to_numpy(),
        acc_summary["ci_high"].to_numpy() - values,
    ])
    ax.errorbar(
        acc_summary["separation"],
        values,
        yerr=yerr,
        marker="o",
        linewidth=2.1,
        markersize=6,
        capsize=4,
    )
    ax.axhline(
        1.0 / cfg.n_branches,
        linestyle="--",
        linewidth=1.2,
        label="Chance level",
    )
    ax.set_ylim(0.25, 1.02)
    ax.set_xlabel("Branch separation scale")
    ax.set_ylabel("Branch-assignment accuracy")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(ax, "F", "Branch accuracy across separation")
    
    ax.text(
        0.04,
        0.18,
        (
            f"Overall accuracy = {overall_accuracy:.2f}\n"
            f"Balanced accuracy = {balanced_accuracy:.2f}\n"
            f"Macro-F1 = {macro_f1:.2f}\n"
            f"ARI = {ari:.2f}"
        ),
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.95,
        ),
    )

    overall_accuracy = float(np.mean(true_labels == inferred_labels))
    fig.suptitle(
        "Supplementary Figure S4. Validation of branch reconstruction",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )


    # F. Accuracy versus branch separation.
    ax = axes[1, 2]
    summary_rows = []
    ci_rng = np.random.default_rng(cfg.seed + 5000)

    for separation, subset in accuracy_results.groupby("separation", sort=True):
        values = subset["accuracy"].to_numpy()
        mean = float(np.mean(values))
        low, high = bootstrap_ci(values, ci_rng)
        summary_rows.append(
            {
                "separation": separation,
                "mean_accuracy": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )

    acc_summary = pd.DataFrame(summary_rows)
    values = acc_summary["mean_accuracy"].to_numpy()
    yerr = np.vstack([
        values - acc_summary["ci_low"].to_numpy(),
        acc_summary["ci_high"].to_numpy() - values,
    ])
    ax.errorbar(
        acc_summary["separation"],
        values,
        yerr=yerr,
        marker="o",
        linewidth=2.1,
        markersize=6,
        capsize=4,
    )
    ax.set_ylim(0.25, 1.02)
    ax.set_xlabel("Branch separation scale")
    ax.set_ylabel("Branch-assignment accuracy")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(ax, "F", "Branch accuracy across separation")

    overall_accuracy = float(np.mean(true_labels == inferred_labels))
    fig.suptitle(
        "Supplementary Figure S4. Validation of branch reconstruction",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
#    fig.text(
#        0.5,
#        0.03,
#        f"Three descendant branches were simulated from a shared ancestor and "
#        f"reconstructed from trajectory-level features. Overall aligned accuracy "
#        f"at the reference separation was {overall_accuracy:.3f}.",
#        ha="center",
#        fontsize=8.8,
#    )

    assignments = pd.DataFrame({
        "lineage_id": np.arange(1, cfg.n_lineages + 1),
        "true_branch": true_labels + 1,
        "inferred_branch": inferred_labels + 1,
        "correct": true_labels == inferred_labels,
    })
    for branch in range(cfg.n_branches):
        assignments[f"prob_branch_{branch + 1}"] = inferred_probs[:, branch]

    matrices = []
    for matrix_name, matrix in [
        ("true", true_matrix),
        ("inferred", inferred_matrix),
    ]:
        for i, origin in enumerate(labels):
            for j, destination in enumerate(labels):
                matrices.append({
                    "matrix": matrix_name,
                    "origin": origin,
                    "destination": destination,
                    "probability": matrix[i, j],
                })

    return fig, assignments, pd.DataFrame(matrices), acc_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure S4: branch reconstruction."
    )
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument(
        "--stem",
        default="supplementary_figure_S4_branch_reconstruction",
    )
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(seed=args.seed, replicates=args.replicates)
    rng = np.random.default_rng(cfg.seed)

    times, trajectories, true_labels, attractors = simulate_dataset(
        cfg,
        rng,
        separation_scale=1.0,
    )
    inferred_labels, inferred_probs, _ = infer_branches(
        trajectories,
        true_labels,
        cfg,
        seed=cfg.seed,
    )
    accuracy_results = run_accuracy_benchmark(cfg)

    fig, assignments, matrices, acc_summary = build_figure(
        cfg,
        times,
        trajectories,
        true_labels,
        inferred_labels,
        inferred_probs,
        attractors,
        accuracy_results,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            args.outdir / f"{args.stem}.{ext}",
            dpi=args.dpi if ext == "png" else None,
            bbox_inches="tight",
        )

    assignments.to_csv(
        args.outdir / "supplementary_figure_S4_branch_assignments.csv",
        index=False,
    )
    accuracy_results.to_csv(
        args.outdir / "supplementary_figure_S4_accuracy_replicates.csv",
        index=False,
    )
    acc_summary.to_csv(
        args.outdir / "supplementary_figure_S4_accuracy_summary.csv",
        index=False,
    )
    matrices.to_csv(
        args.outdir / "supplementary_figure_S4_transition_matrices.csv",
        index=False,
    )
    plt.close(fig)
    print("Saved Figure S4 outputs to", args.outdir.resolve())


if __name__ == "__main__":
    main()
