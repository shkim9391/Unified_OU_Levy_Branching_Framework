from __future__ import annotations
from itertools import product
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_brownian_mle(times, x):
    t=np.asarray(times,float); y=np.asarray(x,float).reshape(-1); dt=np.diff(t); dy=np.diff(y)
    def nll(z):
        drift=z[0]; sigma=np.exp(z[1]); var=sigma*sigma*dt
        return .5*np.sum(np.log(2*np.pi*var)+(dy-drift*dt)**2/var)
    init=[np.sum(dy)/np.sum(dt), np.log(max(np.std(dy/np.sqrt(dt)),1e-4))]
    r=minimize(nll,init,method="L-BFGS-B")
    return {"drift":float(r.x[0]),"sigma":float(np.exp(r.x[1])),"loglik":float(-r.fun),"success":bool(r.success)}


def _validate_transition_inputs(times, values):
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float).reshape(-1)

    if t.ndim != 1 or y.ndim != 1:
        raise ValueError("times and values must be one-dimensional.")
    if len(t) != len(y):
        raise ValueError("times and values must have equal length.")
    if len(t) < 2:
        raise ValueError("At least two observations are required.")
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        raise ValueError("times and values must be finite.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Observation times must be strictly increasing.")

    return t, y


def _ou_transition_variance(theta, sigma, dt):
    """
    Numerically stable OU transition variance.

    sigma^2 * [1 - exp(-2 theta dt)] / (2 theta)

    Uses expm1 to avoid cancellation near theta = 0 and explicitly
    applies the Brownian limiting variance sigma^2 * dt.
    """
    theta = float(theta)
    sigma = float(sigma)
    dt = np.asarray(dt, dtype=float)

    if theta < 1e-7:
        variance = sigma * sigma * dt
    else:
        variance = (
            sigma * sigma
            * (-np.expm1(-2.0 * theta * dt))
            / (2.0 * theta)
        )

    return np.maximum(variance, 1e-12)


def fit_ou_mle(times, x):
    t, y = _validate_transition_inputs(times, x)

    dt = np.diff(t)
    x0 = y[:-1]
    x1 = y[1:]

    # Broad but finite bounds prevent meaningless overflow and underflow.
    theta_bounds = (1e-4, 20.0)
    sigma_bounds = (1e-6, 20.0)

    def nll(z):
        mu = float(z[0])
        theta = float(np.exp(z[1]))
        sigma = float(np.exp(z[2]))

        phi = np.exp(-theta * dt)
        mean = mu + (x0 - mu) * phi
        variance = _ou_transition_variance(theta, sigma, dt)

        terms = (
            np.log(2.0 * np.pi * variance)
            + ((x1 - mean) ** 2) / variance
        )

        if np.any(~np.isfinite(terms)):
            return 1e100

        return 0.5 * float(np.sum(terms))

    initial_sigma = max(
        float(np.std(np.diff(y), ddof=1)) if len(y) > 2 else 1e-3,
        1e-3,
    )

    initial = np.array([
        float(np.mean(y)),
        np.log(0.8),
        np.log(initial_sigma),
    ])

    bounds = [
        (None, None),
        (np.log(theta_bounds[0]), np.log(theta_bounds[1])),
        (np.log(sigma_bounds[0]), np.log(sigma_bounds[1])),
    ]

    result = minimize(
        nll,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
    )

    theta_hat = float(np.exp(result.x[1]))
    sigma_hat = float(np.exp(result.x[2]))

    at_theta_bound = bool(
        np.isclose(theta_hat, theta_bounds[0], rtol=0, atol=1e-6)
        or np.isclose(theta_hat, theta_bounds[1], rtol=0, atol=1e-6)
    )
    at_sigma_bound = bool(
        np.isclose(sigma_hat, sigma_bounds[0], rtol=0, atol=1e-8)
        or np.isclose(sigma_hat, sigma_bounds[1], rtol=0, atol=1e-6)
    )

    finite_fit = bool(
        np.isfinite(result.fun)
        and np.isfinite(theta_hat)
        and np.isfinite(sigma_hat)
    )

    return {
        "mu": float(result.x[0]),
        "theta": theta_hat,
        "sigma": sigma_hat,
        "loglik": float(-result.fun) if np.isfinite(result.fun) else np.nan,
        "success": bool(result.success and finite_fit),
        "optimizer_success": bool(result.success),
        "at_theta_bound": at_theta_bound,
        "at_sigma_bound": at_sigma_bound,
        "message": str(result.message),
    }


def fit_shifted_ou_mle(times, x, treatment):
    t, y = _validate_transition_inputs(times, x)
    u = np.asarray(treatment, dtype=float).reshape(-1)

    if len(u) != len(y):
        raise ValueError("treatment and observations must have equal length.")
    if np.any(~np.isfinite(u)):
        raise ValueError("treatment must contain finite values.")

    dt = np.diff(t)
    x0 = y[:-1]
    x1 = y[1:]
    u1 = u[1:]

    theta_bounds = (1e-4, 20.0)
    sigma_bounds = (1e-6, 20.0)

    def nll(z):
        mu0 = float(z[0])
        delta = float(z[1])
        theta = float(np.exp(z[2]))
        sigma = float(np.exp(z[3]))

        attractor = mu0 + delta * u1
        phi = np.exp(-theta * dt)
        mean = attractor + (x0 - attractor) * phi
        variance = _ou_transition_variance(theta, sigma, dt)

        terms = (
            np.log(2.0 * np.pi * variance)
            + ((x1 - mean) ** 2) / variance
        )

        if np.any(~np.isfinite(terms)):
            return 1e100

        return 0.5 * float(np.sum(terms))

    initial_sigma = max(
        float(np.std(np.diff(y), ddof=1)) if len(y) > 2 else 1e-3,
        1e-3,
    )

    initial = np.array([
        float(np.mean(y)),
        0.0,
        np.log(0.8),
        np.log(initial_sigma),
    ])

    bounds = [
        (None, None),
        (None, None),
        (np.log(theta_bounds[0]), np.log(theta_bounds[1])),
        (np.log(sigma_bounds[0]), np.log(sigma_bounds[1])),
    ]

    result = minimize(
        nll,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
    )

    theta_hat = float(np.exp(result.x[2]))
    sigma_hat = float(np.exp(result.x[3]))

    at_theta_bound = bool(
        np.isclose(theta_hat, theta_bounds[0], rtol=0, atol=1e-6)
        or np.isclose(theta_hat, theta_bounds[1], rtol=0, atol=1e-6)
    )
    at_sigma_bound = bool(
        np.isclose(sigma_hat, sigma_bounds[0], rtol=0, atol=1e-8)
        or np.isclose(sigma_hat, sigma_bounds[1], rtol=0, atol=1e-6)
    )

    finite_fit = bool(
        np.isfinite(result.fun)
        and np.isfinite(theta_hat)
        and np.isfinite(sigma_hat)
    )

    return {
        "mu0": float(result.x[0]),
        "delta": float(result.x[1]),
        "theta": theta_hat,
        "sigma": sigma_hat,
        "loglik": float(-result.fun) if np.isfinite(result.fun) else np.nan,
        "success": bool(result.success and finite_fit),
        "optimizer_success": bool(result.success),
        "at_theta_bound": at_theta_bound,
        "at_sigma_bound": at_sigma_bound,
        "message": str(result.message),
    }


def standardized_ou_innovations(times, x, mu, theta, sigma):
    t, y = _validate_transition_inputs(times, x)

    theta = float(theta)
    sigma = float(sigma)

    if theta < 0:
        raise ValueError("theta must be nonnegative.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    dt = np.diff(t)
    phi = np.exp(-theta * dt)
    mean = mu + (y[:-1] - mu) * phi
    variance = _ou_transition_variance(theta, sigma, dt)

    return (y[1:] - mean) / np.sqrt(variance)


def detect_jump_intervals(z, threshold=3.0): return np.abs(np.asarray(z,float))>threshold


def binary_metrics(truth,pred):
    truth=np.asarray(truth,bool); pred=np.asarray(pred,bool); tp=np.sum(truth&pred); fp=np.sum(~truth&pred); fn=np.sum(truth&~pred); tn=np.sum(~truth&~pred)
    return {"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),"precision":float(tp/(tp+fp)) if tp+fp else np.nan,"recall":float(tp/(tp+fn)) if tp+fn else np.nan,"false_positive_rate":float(fp/(fp+tn)) if fp+tn else np.nan}


def adjusted_rand_index(labels_true,labels_pred):
    a=np.asarray(labels_true); b=np.asarray(labels_pred); tab=pd.crosstab(a,b).to_numpy(); n=tab.sum()
    c2=lambda x: x*(x-1)/2
    sum_comb=c2(tab).sum(); row=c2(tab.sum(axis=1)).sum(); col=c2(tab.sum(axis=0)).sum(); expected=row*col/c2(n) if n>1 else 0; denom=.5*(row+col)-expected
    return float((sum_comb-expected)/denom) if denom else 1.0


def parameter_recovery_table(records):
    df=pd.DataFrame(records)
    df["error"]=df["estimate"]-df["truth"]; df["abs_error"]=df["error"].abs(); df["sq_error"]=df["error"]**2
    return df


def summarize_recovery(df):
    return df.groupby([c for c in ["scenario","parameter"] if c in df.columns],dropna=False).agg(n=("error","size"),bias=("error","mean"),mae=("abs_error","mean"),rmse=("sq_error",lambda x:float(np.sqrt(np.mean(x))))).reset_index()


def cartesian_design(grid: dict) -> pd.DataFrame:
    keys=list(grid); return pd.DataFrame([dict(zip(keys,v)) for v in product(*(grid[k] for k in keys))])


def fit_ou_mle_measurement_aware(
    times,
    x,
    measurement_sd,
):
    """
    Fit an OU model using an approximate measurement-error-aware
    transition likelihood.

    Observation model
    -----------------
    Y_i = X_i + epsilon_i
    epsilon_i ~ Normal(0, measurement_sd_i**2)

    Approximate conditional variance
    --------------------------------
    Var(Y_{i+1} | Y_i) =
        Q_i
        + measurement_var_{i+1}
        + phi_i**2 * measurement_var_i

    where
        phi_i = exp(-theta * dt_i)
        Q_i is the OU process transition variance.

    This is a conditional approximation for benchmarking rather than
    a full Kalman-filter state-space likelihood.
    """
    t, y = _validate_transition_inputs(times, x)

    measurement_sd = np.asarray(measurement_sd, dtype=float)

    if measurement_sd.ndim == 0:
        measurement_sd = np.full(
            len(y),
            float(measurement_sd),
            dtype=float,
        )

    measurement_sd = measurement_sd.reshape(-1)

    if len(measurement_sd) != len(y):
        raise ValueError(
            "measurement_sd must be scalar or contain one value "
            "per observation."
        )

    if np.any(~np.isfinite(measurement_sd)):
        raise ValueError(
            "measurement_sd must contain finite values."
        )

    if np.any(measurement_sd < 0):
        raise ValueError(
            "measurement_sd must be nonnegative."
        )

    measurement_var = measurement_sd**2

    dt = np.diff(t)
    y0 = y[:-1]
    y1 = y[1:]
    var0 = measurement_var[:-1]
    var1 = measurement_var[1:]

    theta_bounds = (1e-4, 20.0)
    sigma_bounds = (1e-6, 20.0)

    def nll(z):
        mu = float(z[0])
        theta = float(np.exp(z[1]))
        sigma = float(np.exp(z[2]))

        phi = np.exp(-theta * dt)
        mean = mu + (y0 - mu) * phi

        process_var = _ou_transition_variance(
            theta,
            sigma,
            dt,
        )

        total_var = (
            process_var
            + var1
            + (phi**2) * var0
        )
        total_var = np.maximum(total_var, 1e-12)

        terms = (
            np.log(2.0 * np.pi * total_var)
            + ((y1 - mean) ** 2) / total_var
        )

        if np.any(~np.isfinite(terms)):
            return 1e100

        return 0.5 * float(np.sum(terms))

    initial_sigma = max(
        float(np.std(np.diff(y), ddof=1))
        if len(y) > 2
        else 1e-3,
        1e-3,
    )

    initial = np.array(
        [
            float(np.mean(y)),
            np.log(0.8),
            np.log(initial_sigma),
        ],
        dtype=float,
    )

    bounds = [
        (None, None),
        (
            np.log(theta_bounds[0]),
            np.log(theta_bounds[1]),
        ),
        (
            np.log(sigma_bounds[0]),
            np.log(sigma_bounds[1]),
        ),
    ]

    result = minimize(
        nll,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
    )

    theta_hat = float(np.exp(result.x[1]))
    sigma_hat = float(np.exp(result.x[2]))

    at_theta_bound = bool(
        np.isclose(
            theta_hat,
            theta_bounds[0],
            rtol=0,
            atol=1e-6,
        )
        or np.isclose(
            theta_hat,
            theta_bounds[1],
            rtol=0,
            atol=1e-6,
        )
    )

    at_sigma_bound = bool(
        np.isclose(
            sigma_hat,
            sigma_bounds[0],
            rtol=0,
            atol=1e-8,
        )
        or np.isclose(
            sigma_hat,
            sigma_bounds[1],
            rtol=0,
            atol=1e-6,
        )
    )

    finite_fit = bool(
        np.isfinite(result.fun)
        and np.isfinite(theta_hat)
        and np.isfinite(sigma_hat)
    )

    return {
        "mu": float(result.x[0]),
        "theta": theta_hat,
        "sigma": sigma_hat,
        "loglik": (
            float(-result.fun)
            if np.isfinite(result.fun)
            else np.nan
        ),
        "success": bool(
            result.success
            and finite_fit
        ),
        "optimizer_success": bool(result.success),
        "at_theta_bound": at_theta_bound,
        "at_sigma_bound": at_sigma_bound,
        "message": str(result.message),
    }
