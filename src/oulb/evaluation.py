from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import norm, t

EPS = 1e-8


@dataclass(frozen=True)
class DisplacementModelSpec:
    """One pooled or binary-group-aware distribution model."""

    model_id: str
    model_label: str
    family: str
    branch_aware: bool

    def __post_init__(self) -> None:
        model_id = str(self.model_id).strip()
        model_label = str(self.model_label).strip()
        family = str(self.family).strip().lower()
        if not model_id or not model_label:
            raise ValueError("model_id and model_label must be non-empty.")
        if family not in {"gaussian", "student_t"}:
            raise ValueError("family must be 'gaussian' or 'student_t'.")
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "model_label", model_label)
        object.__setattr__(self, "family", family)


DEFAULT_DISPLACEMENT_MODEL_SPECS: tuple[DisplacementModelSpec, ...] = (
    DisplacementModelSpec("M0", "Gaussian pooled", "gaussian", False),
    DisplacementModelSpec("M1", "Gaussian branch-aware", "gaussian", True),
    DisplacementModelSpec("M2", "Student-t pooled", "student_t", False),
    DisplacementModelSpec("M3", "Student-t branch-aware", "student_t", True),
)

# General framework-level aliases.
DistributionModelSpec = DisplacementModelSpec
DEFAULT_MODEL_SPECS = DEFAULT_DISPLACEMENT_MODEL_SPECS


@dataclass
class FitResult:
    """Maximum-likelihood result and per-observation log densities."""

    model_id: str
    model_label: str
    family: str
    branch_aware: bool
    k: int
    n: int
    success: bool
    message: str
    loglik: float
    aic: float
    aicc: float
    bic: float
    params: Dict[str, float]
    logpdf: np.ndarray


@dataclass
class ModelComparisonArtifacts:
    """All non-graphical products from the default model comparison."""

    results: Dict[str, FitResult]
    comparison_table: pd.DataFrame
    parameter_table: pd.DataFrame
    tail_fit_grid: pd.DataFrame
    casewise_loglik_gain: pd.DataFrame
    best_gaussian_id: str
    best_student_t_id: str

    @property
    def best_gaussian(self) -> FitResult:
        return self.results[self.best_gaussian_id]

    @property
    def best_student_t(self) -> FitResult:
        return self.results[self.best_student_t_id]


def standardize_stability(value: object) -> Optional[str]:
    """Map common stable/switching encodings to canonical labels.

    The substring-matching order intentionally preserves the validated script.
    """
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return None

    if "stable" in text or "same" in text or "no_switch" in text:
        return "Stable"
    if "switch" in text or "chang" in text or "different" in text:
        return "Switching"

    if text in {"0", "false", "no"}:
        return "Stable"
    if text in {"1", "true", "yes"}:
        return "Switching"

    try:
        numeric = float(text)
        return "Switching" if numeric != 0 else "Stable"
    except ValueError:
        return None


def load_table(path: Path) -> pd.DataFrame:
    """Load a delimited, spreadsheet, or Parquet table by suffix."""
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def prepare_displacement_data(
    table: pd.DataFrame,
    *,
    metric_column: str,
    sample_column: str,
    dx_branch_column: str,
    rel_branch_column: str,
    stability_column: Optional[str] = None,
    minimum_per_group: int = 2,
) -> pd.DataFrame:
    """Standardize a per-case displacement table for binary-group fitting."""
    required = [metric_column, sample_column, dx_branch_column, rel_branch_column]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if minimum_per_group < 1:
        raise ValueError("minimum_per_group must be at least 1.")

    out = table.copy()
    out["sample_std"] = out[sample_column].astype(str).str.strip()
    out["dx_branch_std"] = out[dx_branch_column].astype(str).str.strip()
    out["rel_branch_std"] = out[rel_branch_column].astype(str).str.strip()
    out["disp_std"] = pd.to_numeric(out[metric_column], errors="coerce")
    out["transition_std"] = (
        out["dx_branch_std"] + "→" + out["rel_branch_std"]
    )

    if stability_column and stability_column in out.columns:
        out["stability_std"] = out[stability_column].map(standardize_stability)
    else:
        out["stability_std"] = None

    fallback = np.where(
        out["dx_branch_std"] == out["rel_branch_std"],
        "Stable",
        "Switching",
    )
    out["stability_std"] = out["stability_std"].fillna(
        pd.Series(fallback, index=out.index)
    )
    out["stability_std"] = (
        out["stability_std"].astype(str).str.strip().str.title()
    )

    out = out.dropna(
        subset=[
            "sample_std",
            "disp_std",
            "dx_branch_std",
            "rel_branch_std",
            "stability_std",
        ]
    ).copy()
    out = out[out["stability_std"].isin(["Stable", "Switching"])].copy()
    if out.empty:
        raise ValueError("No valid rows remain after preprocessing.")

    n_stable = int((out["stability_std"] == "Stable").sum())
    n_switching = int((out["stability_std"] == "Switching").sum())
    if n_stable < minimum_per_group or n_switching < minimum_per_group:
        raise ValueError(
            f"Need at least {minimum_per_group} Stable and {minimum_per_group} "
            f"Switching rows. Found Stable={n_stable}, Switching={n_switching}."
        )

    out["group_code"] = np.where(out["stability_std"] == "Stable", 0, 1)
    return out


def prepare_transition_metric_data(
    table: pd.DataFrame,
    *,
    metric_column: str,
    sample_column: str,
    start_branch_column: str,
    end_branch_column: str,
    stability_column: Optional[str] = None,
    minimum_per_group: int = 2,
) -> pd.DataFrame:
    """General-name alias for :func:`prepare_displacement_data`."""
    return prepare_displacement_data(
        table,
        metric_column=metric_column,
        sample_column=sample_column,
        dx_branch_column=start_branch_column,
        rel_branch_column=end_branch_column,
        stability_column=stability_column,
        minimum_per_group=minimum_per_group,
    )


def aicc_from_loglik(loglik: float, k: int, n: int) -> float:
    """Small-sample corrected Akaike information criterion."""
    aic = 2 * k - 2 * loglik
    if n <= k + 1:
        return np.inf
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def compute_information_criteria(
    loglik: float,
    k: int,
    n: int,
) -> Tuple[float, float, float]:
    """Return AIC, AICc, and BIC from a maximized log likelihood."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if k <= 0:
        raise ValueError("k must be positive.")
    aic = 2 * k - 2 * loglik
    aicc = aicc_from_loglik(loglik, k, n)
    bic = math.log(n) * k - 2 * loglik
    return aic, aicc, bic


def safe_std(values: np.ndarray, *, floor: float = 1e-4) -> float:
    """Sample standard deviation with the validated positive floor."""
    values = np.asarray(values, dtype=float)
    standard_deviation = (
        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    )
    return max(standard_deviation, float(floor))


# Branch-aware models have separate means and scales for the two groups.
# Student-t models share one degrees-of-freedom parameter across groups.
def gaussian_pooled_logpdf(
    theta: np.ndarray,
    x: np.ndarray,
    group: np.ndarray,
) -> np.ndarray:
    del group
    mu, log_sigma = theta
    return norm.logpdf(x, loc=mu, scale=np.exp(log_sigma))


def gaussian_branch_logpdf(
    theta: np.ndarray,
    x: np.ndarray,
    group: np.ndarray,
) -> np.ndarray:
    mu_stable, mu_switching, log_sigma_stable, log_sigma_switching = theta
    mu = np.where(group == 0, mu_stable, mu_switching)
    sigma = np.where(
        group == 0,
        np.exp(log_sigma_stable),
        np.exp(log_sigma_switching),
    )
    return norm.logpdf(x, loc=mu, scale=sigma)


def student_pooled_logpdf(
    theta: np.ndarray,
    x: np.ndarray,
    group: np.ndarray,
) -> np.ndarray:
    del group
    mu, log_sigma, log_nu_minus_2 = theta
    sigma = np.exp(log_sigma)
    nu = 2.0 + np.exp(log_nu_minus_2)
    return t.logpdf(x, df=nu, loc=mu, scale=sigma)


def student_branch_logpdf(
    theta: np.ndarray,
    x: np.ndarray,
    group: np.ndarray,
) -> np.ndarray:
    (
        mu_stable,
        mu_switching,
        log_sigma_stable,
        log_sigma_switching,
        log_nu_minus_2,
    ) = theta
    mu = np.where(group == 0, mu_stable, mu_switching)
    sigma = np.where(
        group == 0,
        np.exp(log_sigma_stable),
        np.exp(log_sigma_switching),
    )
    nu = 2.0 + np.exp(log_nu_minus_2)
    return t.logpdf(x, df=nu, loc=mu, scale=sigma)


LogPDF = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def nll_from_logpdf_fn(
    logpdf_fn: LogPDF,
    theta: np.ndarray,
    x: np.ndarray,
    group: np.ndarray,
) -> float:
    """Negative summed log likelihood with a finite-value guard."""
    logpdf = logpdf_fn(theta, x, group)
    if np.any(~np.isfinite(logpdf)):
        return 1e12
    return -float(np.sum(logpdf))


def optimize_model(
    logpdf_fn: LogPDF,
    theta0: np.ndarray,
    x: np.ndarray,
    group: np.ndarray,
) -> OptimizeResult:
    """Fit with L-BFGS-B and preserve the Powell fallback."""

    def objective(theta: np.ndarray) -> float:
        return nll_from_logpdf_fn(logpdf_fn, theta, x, group)

    result = minimize(objective, theta0, method="L-BFGS-B")
    if result.success:
        return result
    return minimize(objective, theta0, method="Powell")


def fit_displacement_model(
    spec: DisplacementModelSpec,
    x: np.ndarray,
    group: np.ndarray,
) -> FitResult:
    """Fit one pooled or branch-aware Gaussian/Student-t model."""
    x = np.asarray(x, dtype=float)
    group = np.asarray(group, dtype=int)
    if x.ndim != 1 or group.ndim != 1 or len(x) != len(group):
        raise ValueError("x and group must be one-dimensional arrays of equal length.")
    if len(x) == 0:
        raise ValueError("At least one observation is required.")
    if np.any(~np.isfinite(x)):
        raise ValueError("x contains non-finite values.")
    if np.any(~np.isin(group, [0, 1])):
        raise ValueError("group must contain only 0 and 1.")

    stable = x[group == 0]
    switching = x[group == 1]
    if spec.branch_aware and (stable.size == 0 or switching.size == 0):
        raise ValueError("Branch-aware models require both groups.")

    if spec.family == "gaussian" and not spec.branch_aware:
        theta0 = np.array([float(np.mean(x)), np.log(safe_std(x))])
        logpdf_fn = gaussian_pooled_logpdf
        k = 2
    elif spec.family == "gaussian" and spec.branch_aware:
        theta0 = np.array(
            [
                float(np.mean(stable)),
                float(np.mean(switching)),
                np.log(safe_std(stable)),
                np.log(safe_std(switching)),
            ]
        )
        logpdf_fn = gaussian_branch_logpdf
        k = 4
    elif spec.family == "student_t" and not spec.branch_aware:
        nu0 = 8.0
        theta0 = np.array(
            [
                float(np.median(x)),
                np.log(safe_std(x)),
                np.log(nu0 - 2.0),
            ]
        )
        logpdf_fn = student_pooled_logpdf
        k = 3
    elif spec.family == "student_t" and spec.branch_aware:
        nu0 = 8.0
        theta0 = np.array(
            [
                float(np.median(stable)),
                float(np.median(switching)),
                np.log(safe_std(stable)),
                np.log(safe_std(switching)),
                np.log(nu0 - 2.0),
            ]
        )
        logpdf_fn = student_branch_logpdf
        k = 5
    else:  # pragma: no cover - guarded by specification validation
        raise ValueError("Unsupported model specification.")

    optimization = optimize_model(logpdf_fn, theta0, x, group)
    theta_hat = np.asarray(optimization.x, dtype=float)
    logpdf = logpdf_fn(theta_hat, x, group)
    loglik = float(np.sum(logpdf))
    aic, aicc, bic = compute_information_criteria(loglik, k, len(x))

    if spec.family == "gaussian" and not spec.branch_aware:
        params = {
            "mu": float(theta_hat[0]),
            "sigma": float(np.exp(theta_hat[1])),
        }
    elif spec.family == "gaussian" and spec.branch_aware:
        params = {
            "mu_stable": float(theta_hat[0]),
            "mu_switching": float(theta_hat[1]),
            "sigma_stable": float(np.exp(theta_hat[2])),
            "sigma_switching": float(np.exp(theta_hat[3])),
        }
    elif spec.family == "student_t" and not spec.branch_aware:
        params = {
            "mu": float(theta_hat[0]),
            "sigma": float(np.exp(theta_hat[1])),
            "nu": float(2.0 + np.exp(theta_hat[2])),
        }
    else:
        params = {
            "mu_stable": float(theta_hat[0]),
            "mu_switching": float(theta_hat[1]),
            "sigma_stable": float(np.exp(theta_hat[2])),
            "sigma_switching": float(np.exp(theta_hat[3])),
            "nu": float(2.0 + np.exp(theta_hat[4])),
        }

    return FitResult(
        model_id=spec.model_id,
        model_label=spec.model_label,
        family=spec.family,
        branch_aware=spec.branch_aware,
        k=k,
        n=len(x),
        success=bool(optimization.success),
        message=str(optimization.message),
        loglik=loglik,
        aic=aic,
        aicc=aicc,
        bic=bic,
        params=params,
        logpdf=logpdf,
    )


def fit_displacement_model_ladder(
    x: np.ndarray,
    group: np.ndarray,
    *,
    specs: Sequence[DisplacementModelSpec] = DEFAULT_DISPLACEMENT_MODEL_SPECS,
) -> Dict[str, FitResult]:
    """Fit an ordered collection of displacement models."""
    results: Dict[str, FitResult] = {}
    for spec in specs:
        if spec.model_id in results:
            raise ValueError(f"Duplicate model_id: {spec.model_id}")
        results[spec.model_id] = fit_displacement_model(spec, x, group)
    return results


def results_to_tables(
    results: Mapping[str, FitResult],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert fits to comparison and long-form parameter tables."""
    comparison_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    for model_id, fit in results.items():
        comparison_rows.append(
            {
                "model_id": model_id,
                "model_label": fit.model_label,
                "family": fit.family,
                "branch_aware": fit.branch_aware,
                "k": fit.k,
                "n": fit.n,
                "success": fit.success,
                "message": fit.message,
                "loglik": fit.loglik,
                "aic": fit.aic,
                "aicc": fit.aicc,
                "bic": fit.bic,
            }
        )
        for parameter, value in fit.params.items():
            parameter_rows.append(
                {
                    "model_id": model_id,
                    "model_label": fit.model_label,
                    "parameter": parameter,
                    "value": value,
                }
            )

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values("aicc", ascending=True)
        .reset_index(drop=True)
    )
    best_aicc = float(comparison["aicc"].min())
    comparison["delta_aicc"] = comparison["aicc"] - best_aicc
    weights = np.exp(-0.5 * comparison["delta_aicc"].to_numpy(dtype=float))
    comparison["aicc_weight"] = weights / np.sum(weights)
    return comparison, pd.DataFrame(parameter_rows)


def best_model_by_family(comparison: pd.DataFrame, family: str) -> str:
    """Model ID with the smallest AICc within one family."""
    subset = comparison[comparison["family"] == family].sort_values(
        "aicc", ascending=True
    )
    if subset.empty:
        raise ValueError(f"No models found for family={family}")
    return str(subset.iloc[0]["model_id"])


def fitted_survival_curve(
    fit: FitResult,
    x_grid: np.ndarray,
    *,
    p_stable: float,
    p_switching: float,
) -> np.ndarray:
    """Evaluate a fitted marginal survival curve."""
    x_grid = np.asarray(x_grid, dtype=float)
    if fit.family == "gaussian" and not fit.branch_aware:
        return norm.sf(x_grid, loc=fit.params["mu"], scale=fit.params["sigma"])
    if fit.family == "gaussian" and fit.branch_aware:
        return (
            p_stable
            * norm.sf(
                x_grid,
                loc=fit.params["mu_stable"],
                scale=fit.params["sigma_stable"],
            )
            + p_switching
            * norm.sf(
                x_grid,
                loc=fit.params["mu_switching"],
                scale=fit.params["sigma_switching"],
            )
        )
    if fit.family == "student_t" and not fit.branch_aware:
        return t.sf(
            x_grid,
            df=fit.params["nu"],
            loc=fit.params["mu"],
            scale=fit.params["sigma"],
        )
    if fit.family == "student_t" and fit.branch_aware:
        return (
            p_stable
            * t.sf(
                x_grid,
                df=fit.params["nu"],
                loc=fit.params["mu_stable"],
                scale=fit.params["sigma_stable"],
            )
            + p_switching
            * t.sf(
                x_grid,
                df=fit.params["nu"],
                loc=fit.params["mu_switching"],
                scale=fit.params["sigma_switching"],
            )
        )
    raise ValueError("Unsupported fit for survival curve.")


def empirical_survival(x: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Empirical right-tail survival using the inclusive legacy rule."""
    x = np.asarray(x, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    return np.array([(x >= value).mean() for value in x_grid], dtype=float)


def build_switching_tail_grid(
    prepared_data: pd.DataFrame,
    best_gaussian: FitResult,
    best_student_t: FitResult,
    *,
    grid_size: int = 300,
) -> pd.DataFrame:
    """Switching-group empirical and fitted survival table."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    switching = prepared_data[
        prepared_data["stability_std"] == "Switching"
    ].copy()
    if switching.empty:
        raise ValueError("No Switching observations are available.")
    x = switching["disp_std"].to_numpy(dtype=float)
    x_grid = np.linspace(
        max(0.0, float(np.min(x)) * 0.95),
        float(np.max(x)) * 1.15,
        grid_size,
    )
    observed = empirical_survival(x, x_grid)

    if best_gaussian.branch_aware:
        gaussian = norm.sf(
            x_grid,
            loc=best_gaussian.params["mu_switching"],
            scale=best_gaussian.params["sigma_switching"],
        )
    else:
        gaussian = fitted_survival_curve(
            best_gaussian, x_grid, p_stable=0.0, p_switching=1.0
        )

    if best_student_t.branch_aware:
        student = t.sf(
            x_grid,
            df=best_student_t.params["nu"],
            loc=best_student_t.params["mu_switching"],
            scale=best_student_t.params["sigma_switching"],
        )
    else:
        student = fitted_survival_curve(
            best_student_t, x_grid, p_stable=0.0, p_switching=1.0
        )

    return pd.DataFrame(
        {
            "x": x_grid,
            "observed_switching_survival": observed,
            "switching_gaussian_survival": gaussian,
            "switching_student_t_survival": student,
            "best_gaussian_model": best_gaussian.model_id,
            "best_student_t_model": best_student_t.model_id,
        }
    )


def build_casewise_loglik_gain(
    prepared_data: pd.DataFrame,
    results: Mapping[str, FitResult],
    *,
    gaussian_branch_model_id: str = "M1",
    student_branch_model_id: str = "M3",
) -> pd.DataFrame:
    """Per-case M3-minus-M1 log-likelihood gain table."""
    if gaussian_branch_model_id not in results:
        raise KeyError(f"Missing model: {gaussian_branch_model_id}")
    if student_branch_model_id not in results:
        raise KeyError(f"Missing model: {student_branch_model_id}")
    gaussian = results[gaussian_branch_model_id]
    student = results[student_branch_model_id]
    if len(gaussian.logpdf) != len(prepared_data) or len(student.logpdf) != len(
        prepared_data
    ):
        raise ValueError("Per-case log-likelihood arrays do not match data rows.")

    casewise = prepared_data[
        [
            "sample_std",
            "stability_std",
            "dx_branch_std",
            "rel_branch_std",
            "transition_std",
            "disp_std",
        ]
    ].copy()
    casewise["loglik_m1_gaussian_branch"] = gaussian.logpdf
    casewise["loglik_m3_student_t_branch"] = student.logpdf
    casewise["delta_loglik"] = (
        casewise["loglik_m3_student_t_branch"]
        - casewise["loglik_m1_gaussian_branch"]
    )
    return casewise.sort_values("delta_loglik", ascending=False).reset_index(
        drop=True
    )


def run_default_model_comparison(
    prepared_data: pd.DataFrame,
    *,
    model_specs: Sequence[DisplacementModelSpec] = DEFAULT_DISPLACEMENT_MODEL_SPECS,
    tail_grid_points: int = 300,
) -> ModelComparisonArtifacts:
    """Fit M0-M3 and construct all non-graphical artifacts."""
    required = ["disp_std", "group_code", "stability_std"]
    missing = [column for column in required if column not in prepared_data.columns]
    if missing:
        raise ValueError(f"Prepared data missing required columns: {missing}")

    results = fit_displacement_model_ladder(
        prepared_data["disp_std"].to_numpy(dtype=float),
        prepared_data["group_code"].to_numpy(dtype=int),
        specs=model_specs,
    )
    comparison, parameters = results_to_tables(results)
    best_gaussian_id = best_model_by_family(comparison, "gaussian")
    best_student_t_id = best_model_by_family(comparison, "student_t")
    tail_grid = build_switching_tail_grid(
        prepared_data,
        results[best_gaussian_id],
        results[best_student_t_id],
        grid_size=tail_grid_points,
    )
    casewise = build_casewise_loglik_gain(prepared_data, results)
    return ModelComparisonArtifacts(
        results=results,
        comparison_table=comparison,
        parameter_table=parameters,
        tail_fit_grid=tail_grid,
        casewise_loglik_gain=casewise,
        best_gaussian_id=best_gaussian_id,
        best_student_t_id=best_student_t_id,
    )


def build_fit_summary_text(
    artifacts: ModelComparisonArtifacts,
    *,
    heading: str = "Supplementary Figure S6E-H model comparison summary",
    title: str | None = None,
) -> str:
    """Return the validated plain-text fit summary."""
    if title is not None:
        heading = str(title)
    lines = [
        heading,
        "=" * 40,
        "",
        (
            f"Best Gaussian model: {artifacts.best_gaussian.model_label} "
            f"({artifacts.best_gaussian.model_id})"
        ),
        (
            f"Best heavy-tail model: {artifacts.best_student_t.model_label} "
            f"({artifacts.best_student_t.model_id})"
        ),
        "",
        "Model ranking by AICc:",
    ]
    for _, row in artifacts.comparison_table.iterrows():
        lines.append(
            f"  {row['model_id']} | {row['model_label']} | "
            f"AICc={row['aicc']:.4f} | ΔAICc={row['delta_aicc']:.4f} | "
            f"weight={row['aicc_weight']:.4f}"
        )
    lines.extend(["", "Parameters:"])
    for _, row in artifacts.parameter_table.iterrows():
        lines.append(
            f"  {row['model_id']} | {row['parameter']} = {row['value']:.6f}"
        )
    return "\n".join(lines) + "\n"


def write_fit_summary(
    path: Path,
    comparison: pd.DataFrame,
    parameters: pd.DataFrame,
    *,
    best_gaussian: FitResult,
    best_student_t: FitResult,
    title: str = "Supplementary Figure S6E-H model comparison summary",
) -> None:
    """Write a fit summary from explicit comparison components."""
    artifacts = ModelComparisonArtifacts(
        results={
            best_gaussian.model_id: best_gaussian,
            best_student_t.model_id: best_student_t,
        },
        comparison_table=comparison,
        parameter_table=parameters,
        tail_fit_grid=pd.DataFrame(),
        casewise_loglik_gain=pd.DataFrame(),
        best_gaussian_id=best_gaussian.model_id,
        best_student_t_id=best_student_t.model_id,
    )
    Path(path).write_text(
        build_fit_summary_text(artifacts, heading=title),
        encoding="utf-8",
    )
