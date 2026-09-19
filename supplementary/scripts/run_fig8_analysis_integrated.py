from __future__ import annotations

import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from scipy.optimize import minimize
from scipy.special import logsumexp, gammaln
from scipy.stats import norm


# ============================================================
# Paths and reproducibility
# ============================================================

BASE_DIR = Path(
    "/Figure8"
)

DATA_DIR = BASE_DIR / "data"
SUMMARY_DIR = BASE_DIR / "summaries"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

PNG = BASE_DIR / "Figure_8.png"
PDF = BASE_DIR / "Figure_8.pdf"

SEED = 20260916
rng = np.random.default_rng(SEED)


# ============================================================
# Frozen validated analysis configuration
# ============================================================

N_TRAIN_TRAJ = int(
    os.getenv(
        "FIG8_TRAIN_TRAJ",
        "10",
    )
)

N_TEST_TRAJ = int(
    os.getenv(
        "FIG8_TEST_TRAJ",
        "25",
    )
)

N_TEMPORAL_REPLICATES = int(
    os.getenv(
        "FIG8_TEMP_REPS",
        "25",
    )
)

N_MISSPEC_REPLICATES = int(
    os.getenv(
        "FIG8_MISSPEC_REPS",
        "30",
    )
)

N_STARTS = int(
    os.getenv(
        "FIG8_N_STARTS",
        "3",
    )
)

FORCE_ANALYSIS = (
    os.getenv(
        "FIG8_FORCE",
        "0",
    )
    == "1"
)

RERUN_MISSPEC_ONLY = (
    os.getenv(
        "FIG8_RERUN_MISSPEC_ONLY",
        "0",
    )
    == "1"
)

TRAIN_FRACTION = 0.80
T_END = 10.0
T_TX = 5.0
N_JUMP_MIXTURE_MAX = 15

MODELS = ["OU", "OU+T", "OU+L", "OU+B", "OULB"]
GENERATORS = ["OU", "OU+T", "OU+L", "OU+B", "OULB", "Nonlinear"]
DIFFICULTIES = ["Easy", "Intermediate", "Difficult"]

# Mechanisms are intentionally made progressively harder to separate.
DIFFICULTY_CONFIG = {
    "Easy": {
        "n_obs": 60,
        "theta": 0.65,
        "sigma": 0.16,
        "obs_noise": 0.05,
        "treatment_shift": 1.20,
        "jump_scale": 1.20,
        "jump_lambda": 0.30,
        "branch_sep": 1.40,
        "branch_q": 0.12,
    },

    "Intermediate": {
        "n_obs": 24,
        "theta": 0.65,
        "sigma": 0.22,
        "obs_noise": 0.22,
        "treatment_shift": 0.65,
        "jump_scale": 0.65,
        "jump_lambda": 0.20,
        "branch_sep": 0.70,
        "branch_q": 0.12,
    },

    "Difficult": {
        "n_obs": 12,
        "theta": 0.65,
        "sigma": 0.32,
        "obs_noise": 0.45,
        "treatment_shift": 0.40,
        "jump_scale": 0.40,
        "jump_lambda": 0.12,
        "branch_sep": 0.40,
        "branch_q": 0.12,
    },
}


# ============================================================
# Global plotting style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# Utilities
# ============================================================

def clamp(x, lo, hi):
    return float(np.clip(x, lo, hi))


def irregular_times(n_obs: int, local_rng: np.random.Generator):
    if n_obs < 3:
        raise ValueError("n_obs must be >= 3")
    inner = np.sort(local_rng.uniform(0.0, T_END, n_obs - 2))
    return np.concatenate(([0.0], inner, [T_END]))


def exact_ou_step(x, dt, theta, mu, sigma, local_rng):
    a = math.exp(-theta * dt)
    var = (sigma**2 / (2.0 * theta)) * (1.0 - a * a)
    return mu + (x - mu) * a + math.sqrt(max(var, 1e-12)) * local_rng.normal()


def symmetric_ctmc_switch_prob(q, dt):
    return 0.5 * (1.0 - math.exp(-2.0 * q * dt))


def poisson_logpmf_array(lam_dt, nmax=N_JUMP_MIXTURE_MAX):
    n = np.arange(nmax + 1, dtype=float)
    lam_dt = max(float(lam_dt), 1e-12)
    lw = -lam_dt + n * math.log(lam_dt) - gammaln(n + 1.0)
    return lw - logsumexp(lw)


def bootstrap_mean_ci(values, n_boot=2000, seed=SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    brng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = brng.choice(values, size=len(values), replace=True).mean()
    return (
        float(values.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


# ============================================================
# Generators
# ============================================================

def simulate_in_family(
    generator,
    difficulty,
    replicate,
    event_distance=None,
    temporal_mode=None,
):
    cfg = DIFFICULTY_CONFIG[difficulty]

    local_rng = np.random.default_rng(
        SEED
        + 100003 * replicate
        + 1009 * DIFFICULTIES.index(difficulty)
        + 37 * (
            GENERATORS.index(generator)
            if generator in GENERATORS
            else 0
        )
        + (
            0
            if event_distance is None
            else int(round(event_distance * 100))
        )
        + (
            0
            if temporal_mode is None
            else {
                "Treatment": 11,
                "Jump": 17,
                "Branch": 23,
            }[temporal_mode]
        )
    )

    # --------------------------------------------------------
    # Observation times and generating parameters
    # --------------------------------------------------------

    times = irregular_times(
        cfg["n_obs"],
        local_rng,
    )

    n = len(times)

    theta = cfg["theta"]
    sigma = cfg["sigma"]
    obs_noise = cfg["obs_noise"]

    treatment_shift = cfg["treatment_shift"]

    jump_scale = cfg["jump_scale"]
    jump_lambda = cfg["jump_lambda"]

    branch_sep = cfg["branch_sep"]
    branch_q = cfg["branch_q"]

    base_mu = 0.0

    # --------------------------------------------------------
    # Active mechanisms
    # --------------------------------------------------------

    treatment_active = (
        generator in {"OU+T", "OULB"}
    )

    jump_active = (
        generator in {"OU+L", "OULB"}
    )

    branch_active = (
        generator in {"OU+B", "OULB"}
    )

    # --------------------------------------------------------
    # Controlled temporal-confounding experiment
    # --------------------------------------------------------

    if temporal_mode is not None:

        treatment_active = True

        if temporal_mode == "Treatment":

            # Treatment is always active.
            # Add either jump or branch as a nearby competing
            # endogenous mechanism.
            jump_active = (
                replicate % 2 == 0
            )

            branch_active = (
                not jump_active
            )

        elif temporal_mode == "Jump":

            jump_active = True
            branch_active = False

        elif temporal_mode == "Branch":

            jump_active = False
            branch_active = True

    # --------------------------------------------------------
    # Jump timing
    # --------------------------------------------------------

    if event_distance is None:

        if difficulty == "Easy":

            # Well separated from treatment onset
            jump_time = 3.5

        elif difficulty == "Intermediate":

            # Partial temporal overlap
            jump_time = (
                T_TX
                + local_rng.uniform(
                    -0.75,
                    0.75,
                )
            )

        else:

            # Deliberately close to treatment onset
            jump_time = (
                T_TX
                + local_rng.uniform(
                    -0.25,
                    0.25,
                )
            )

    else:

        # Controlled event timing for Panel E
        jump_time = (
            T_TX
            + event_distance
        )

    jump_time = clamp(
        jump_time,
        0.25,
        T_END - 0.25,
    )

    # Realized jump magnitude is defined only for the
    # controlled single-event experiment in Panel E.
    controlled_jump_size = np.nan

    # --------------------------------------------------------
    # State arrays
    # --------------------------------------------------------

    latent = np.zeros(n)

    latent[0] = local_rng.normal(
        0.0,
        0.08,
    )

    branch = np.zeros(
        n,
        dtype=int,
    )

    # Match fitted model:
    # stationary initial distribution = (0.5, 0.5)
    if branch_active:
        branch[0] = int(
            local_rng.integers(
                0,
                2,
            )
        )
    else:
        branch[0] = 0

    jump_event = np.zeros(
        n,
        dtype=int,
    )

    # --------------------------------------------------------
    # Simulate latent process
    # --------------------------------------------------------

    for i in range(1, n):

        dt = (
            times[i]
            - times[i - 1]
        )

        midpoint = (
            0.5
            * (
                times[i]
                + times[i - 1]
            )
        )

    # ----------------------------------------------------
    # Branch-state evolution
    # ----------------------------------------------------
    
    if branch_active:
    
        # Panel E uses a controlled single branch transition.
        # This applies both when Branch itself is the component
        # being tested and when branching is the competing
        # endogenous event in the Treatment experiment.
        controlled_branch = (
            event_distance is not None
            and (
                temporal_mode == "Branch"
                or (
                    temporal_mode == "Treatment"
                    and not jump_active
                )
            )
        )
    
        if controlled_branch:
    
            controlled_branch_time = clamp(
                T_TX + event_distance,
                0.25,
                T_END - 0.25,
            )
    
            if (
                times[i - 1]
                < controlled_branch_time
                <= times[i]
            ):
                branch[i] = (
                    1 - branch[i - 1]
                )
            else:
                branch[i] = (
                    branch[i - 1]
                )
    
        else:
    
            # Main Panels B-D and in-family Panel F:
            # symmetric two-state CTMC matching the
            # fitted OU+B / OULB transition model.
            p_switch = (
                symmetric_ctmc_switch_prob(
                    branch_q,
                    dt,
                )
            )
    
            if (
                local_rng.random()
                < p_switch
            ):
                branch[i] = (
                    1 - branch[i - 1]
                )
            else:
                branch[i] = (
                    branch[i - 1]
                )
    
    else:
    
        branch[i] = 0

        # ----------------------------------------------------
        # Branch-specific attractor
        # Match fitted likelihood:
        #
        # mu_0 = base_mu - branch_sep/2
        # mu_1 = base_mu + branch_sep/2
        # ----------------------------------------------------

        if branch_active:

            if branch[i] == 0:

                mu = (
                    base_mu
                    - 0.5 * branch_sep
                )

            else:

                mu = (
                    base_mu
                    + 0.5 * branch_sep
                )

        else:

            mu = base_mu

        # ----------------------------------------------------
        # Treatment-induced attractor shift
        # ----------------------------------------------------

        if (
            treatment_active
            and midpoint >= T_TX
        ):

            mu += treatment_shift

        # ----------------------------------------------------
        # Exact OU propagation
        # ----------------------------------------------------

        latent[i] = exact_ou_step(
            latent[i - 1],
            dt,
            theta,
            mu,
            sigma,
            local_rng,
        )

        # ----------------------------------------------------
        # Lévy / compound-Poisson jump component
        # ----------------------------------------------------

        if jump_active:

            if event_distance is None:

                # Main Panels B-D and in-family Panel F:
                # compound-Poisson Gaussian jumps matching
                # the fitted OU+L / OULB transition model.
                n_jumps = local_rng.poisson(
                    jump_lambda * dt
                )

                if n_jumps > 0:

                    jump_increment = local_rng.normal(
                        loc=0.0,
                        scale=jump_scale,
                        size=n_jumps,
                    ).sum()

                    latent[i] += jump_increment

                    # Store number of jumps occurring in
                    # this observation interval.
                    jump_event[i] = n_jumps

            else:

                # Panel E only:
                # retain one controlled event because its
                # temporal distance from treatment onset is
                # the experimental variable.
                if (
                    times[i - 1]
                    < jump_time
                    <= times[i]
                ):

                    jump_increment = local_rng.normal(
                        loc=0.0,
                        scale=jump_scale,
                    )

                    controlled_jump_size = float(
                        jump_increment
                    )

                    latent[i] += jump_increment
                    jump_event[i] = 1
                    
    # --------------------------------------------------------
    # Observation process
    # --------------------------------------------------------

    observed = (
        latent
        + local_rng.normal(
            0.0,
            obs_noise,
            size=n,
        )
    )

    # --------------------------------------------------------
    # Training / held-out split
    # --------------------------------------------------------

    split_index = max(
        3,
        int(
            math.floor(
                TRAIN_FRACTION * n
            )
        ),
    )

    split_index = min(
        split_index,
        n - 2,
    )

    # --------------------------------------------------------
    # Branch-switch truth
    # --------------------------------------------------------

    n_branch_switches = int(
        np.sum(
            branch[1:]
            != branch[:-1]
        )
    )

    # --------------------------------------------------------
    # Truth metadata
    # --------------------------------------------------------

    truth = {
        "theta":
            theta,

        "sigma":
            sigma,

        "obs_noise":
            obs_noise,

        "treatment_shift":
            treatment_shift
            if treatment_active
            else 0.0,

        # A unique jump time/size exists only in the
        # controlled temporal-confounding experiment.
        "jump_time":
            (
                jump_time
                if (
                    jump_active
                    and event_distance is not None
                )
                else np.nan
            ),

        "jump_size":
            (
                controlled_jump_size
                if (
                    jump_active
                    and event_distance is not None
                )
                else np.nan
            ),

        "jump_lambda":
            (
                jump_lambda
                if jump_active
                else 0.0
            ),

        "jump_scale":
            (
                jump_scale
                if jump_active
                else 0.0
            ),

        "branch_time":
            (
                clamp(
                    T_TX + event_distance,
                    0.25,
                    T_END - 0.25,
                )
                if (
                    branch_active
                    and event_distance is not None
                    and (
                        temporal_mode == "Branch"
                        or (
                            temporal_mode == "Treatment"
                            and not jump_active
                        )
                    )
                )
                else np.nan
            ),

        "branch_q":
            branch_q
            if branch_active
            else 0.0,

        "branch_sep":
            branch_sep
            if branch_active
            else 0.0,

        "n_branch_switches":
            n_branch_switches,

        "treatment_active":
            treatment_active,

        "jump_active":
            jump_active,

        "branch_active":
            branch_active,
    }

    return {
        "times":
            times,

        "latent":
            latent,

        "observed":
            observed,

        "branch":
            branch,

        "jump_event":
            jump_event,

        "split_index":
            split_index,

        "truth":
            truth,
    }


def simulate_nonlinear(difficulty, replicate):
    """
    Out-of-family nonlinear bistable process with weak
    periodic forcing:

        dX =
        [-4 a X (X^2 - b^2)
         + A sin(omega t)] dt
        + sigma dW

    This process is deliberately outside the OULB hierarchy:
    the restoring force is nonlinear, the system contains two
    stable basins, and the landscape is continuously modulated.
    """

    cfg = DIFFICULTY_CONFIG[difficulty]

    local_rng = np.random.default_rng(
        SEED
        + 700000
        + 100003 * replicate
        + 1009 * DIFFICULTIES.index(difficulty)
    )

    times = irregular_times(
        cfg["n_obs"],
        local_rng
    )

    latent = np.zeros(len(times))

    # Begin near the negative stable basin
    latent[0] = -0.80

    # Double-well parameters
    a_dw = 1.25
    b_dw = 0.80

    # Continuous nonlinear forcing
    forcing_amp = 0.75
    forcing_period = 4.0

    omega = (
        2.0 * np.pi / forcing_period
    )

    # Enough stochasticity to permit recurrent transitions
    sigma_dw = 0.65

    # Fine integration grid for nonlinear dynamics
    internal_dt = 0.005

    current_time = times[0]

    for i in range(1, len(times)):

        x = latent[i - 1]

        target_time = times[i]

        while current_time < target_time - 1e-12:

            dt = min(
                internal_dt,
                target_time - current_time
            )

            nonlinear_drift = (
                -4.0
                * a_dw
                * x
                * (
                    x * x
                    - b_dw * b_dw
                )
            )

            forcing = (
                forcing_amp
                * np.sin(
                    omega * current_time
                )
            )

            drift = (
                nonlinear_drift
                + forcing
            )

            x = (
                x
                + drift * dt
                + sigma_dw
                * np.sqrt(dt)
                * local_rng.normal()
            )

            current_time += dt

        latent[i] = x

    observed = (
        latent
        + local_rng.normal(
            0.0,
            cfg["obs_noise"],
            size=len(times),
        )
    )

    split_index = max(
        3,
        int(
            np.floor(
                TRAIN_FRACTION
                * len(times)
            )
        ),
    )

    split_index = min(
        split_index,
        len(times) - 2,
    )

    return {
        "times": times,
        "latent": latent,
        "observed": observed,
        "branch": np.zeros(
            len(times),
            dtype=int,
        ),
        "jump_event": np.zeros(
            len(times),
            dtype=int,
        ),
        "split_index": split_index,

        "truth": {
            "theta": np.nan,
            "sigma": sigma_dw,
            "obs_noise":
                cfg["obs_noise"],

            "treatment_shift": 0.0,
            "jump_time": np.nan,
            "jump_size": 0.0,
            "branch_time": np.nan,
            "branch_sep": np.nan,

            "treatment_active": False,
            "jump_active": False,
            "branch_active": False,

            # Explicit misspecification metadata
            "nonlinear_a": a_dw,
            "nonlinear_b": b_dw,
            "forcing_amp":
                forcing_amp,
            "forcing_period":
                forcing_period,
        },
    }


def simulate_trajectory(generator, difficulty, replicate, **kwargs):
    if generator == "Nonlinear":
        return simulate_nonlinear(difficulty, replicate)
    return simulate_in_family(generator, difficulty, replicate, **kwargs)


# ============================================================
# Candidate-model likelihoods
# ============================================================

PARAMETER_SPECS = {
    "OU": {
        "names": ["theta", "mu", "sigma", "obs_sd"],
        "bounds": [(0.05, 2.0), (-2.0, 2.0), (0.05, 1.2), (0.01, 1.0)],
    },
    "OU+T": {
        "names": ["theta", "mu", "delta", "sigma", "obs_sd"],
        "bounds": [(0.05, 2.0), (-2.0, 2.0), (-1.5, 1.5), (0.05, 1.2), (0.01, 1.0)],
    },
    "OU+L": {
        "names": ["theta", "mu", "sigma", "lambda", "sJ", "obs_sd"],
        "bounds": [(0.05, 2.0), (-2.0, 2.0), (0.05, 1.2), (0.005, 1.0), (0.05, 2.0), (0.01, 1.0)],
    },
    "OU+B": {
        "names": ["theta", "base_mu", "branch_sep", "sigma", "q", "obs_sd"],
        "bounds": [(0.05, 2.0), (-2.0, 2.0), (0.05, 2.0), (0.05, 1.2), (0.005, 1.0), (0.01, 1.0)],
    },
    "OULB": {
        "names": ["theta", "base_mu", "branch_sep", "delta", "sigma", "q", "lambda", "sJ", "obs_sd"],
        "bounds": [
            (0.05, 2.0), (-2.0, 2.0), (0.05, 2.0), (-1.5, 1.5),
            (0.05, 1.2), (0.005, 1.0), (0.005, 1.0), (0.05, 2.0), (0.01, 1.0)
        ],
    },
}


def unpack_params(model, p):
    names = PARAMETER_SPECS[model]["names"]
    return dict(zip(names, map(float, p)))


def ou_transition_moments(y_prev, dt, theta, mu, sigma, obs_sd):
    a = math.exp(-theta * dt)
    mean = mu + (y_prev - mu) * a
    proc_var = (sigma**2 / (2.0 * theta)) * (1.0 - a * a)
    meas_var = obs_sd**2 * (1.0 + a * a)
    return mean, max(proc_var + meas_var, 1e-10)


def jump_log_emission(y_next, mean, base_var, lam, sJ, dt):
    lw = poisson_logpmf_array(lam * dt)
    n = np.arange(len(lw), dtype=float)
    vars_n = base_var + n * sJ * sJ
    lpdf = -0.5 * (
        np.log(2.0 * np.pi * vars_n)
        + (y_next - mean) ** 2 / vars_n
    )
    return float(logsumexp(lw + lpdf))


def jump_predictive_variance(base_var, lam, sJ, dt):
    return base_var + lam * dt * sJ * sJ


def nonbranch_step(model, params, t0, t1, y0):
    dt = t1 - t0
    midpoint = 0.5 * (t0 + t1)

    theta = params["theta"]
    sigma = params["sigma"]
    obs_sd = params["obs_sd"]

    mu = params["mu"]
    if model == "OU+T" and midpoint >= T_TX:
        mu += params["delta"]

    mean, base_var = ou_transition_moments(
        y0, dt, theta, mu, sigma, obs_sd
    )
    return mean, base_var


def loglik_nonbranch(model, params, times, y, return_steps=False):
    step_ll = []
    for i in range(len(y) - 1):
        mean, base_var = nonbranch_step(
            model, params, times[i], times[i + 1], y[i]
        )
        dt = times[i + 1] - times[i]

        if model == "OU+L":
            ll = jump_log_emission(
                y[i + 1], mean, base_var,
                params["lambda"], params["sJ"], dt
            )
        else:
            ll = norm.logpdf(y[i + 1], loc=mean, scale=math.sqrt(base_var))
        step_ll.append(float(ll))

    step_ll = np.asarray(step_ll)
    return step_ll if return_steps else float(step_ll.sum())


def branch_emission_logpdf(model, params, branch_k, t0, t1, y0, y1):
    dt = t1 - t0
    midpoint = 0.5 * (t0 + t1)

    theta = params["theta"]
    sigma = params["sigma"]
    obs_sd = params["obs_sd"]

    base_mu = params["base_mu"]
    sep = params["branch_sep"]
    mu = base_mu + (-0.5 * sep if branch_k == 0 else 0.5 * sep)

    if model == "OULB" and midpoint >= T_TX:
        mu += params["delta"]

    mean, base_var = ou_transition_moments(
        y0, dt, theta, mu, sigma, obs_sd
    )

    if model == "OULB":
        ll = jump_log_emission(
            y1, mean, base_var,
            params["lambda"], params["sJ"], dt
        )
    else:
        ll = norm.logpdf(y1, loc=mean, scale=math.sqrt(base_var))

    return float(ll), float(mean), float(base_var)


def forward_branch(model, params, times, y, return_steps=False, return_predictions=False):
    log_alpha = np.log(np.array([0.5, 0.5]))
    step_ll = []
    pred_means = []
    pred_vars = []

    for i in range(len(y) - 1):
        dt = times[i + 1] - times[i]
        q = params["q"]
        psw = symmetric_ctmc_switch_prob(q, dt)
        P = np.array([
            [1.0 - psw, psw],
            [psw, 1.0 - psw],
        ])
        logP = np.log(np.clip(P, 1e-300, 1.0))

        alpha = np.exp(log_alpha)
        alpha = alpha / alpha.sum()
        pred_branch = alpha @ P

        branch_means = np.zeros(2)
        branch_vars = np.zeros(2)
        emission_ll = np.zeros(2)

        for k in range(2):
            ll, mean_k, base_var_k = branch_emission_logpdf(
                model, params, k, times[i], times[i + 1], y[i], y[i + 1]
            )
            emission_ll[k] = ll
            branch_means[k] = mean_k

            if model == "OULB":
                branch_vars[k] = jump_predictive_variance(
                    base_var_k, params["lambda"], params["sJ"], dt
                )
            else:
                branch_vars[k] = base_var_k

        mix_mean = float(np.sum(pred_branch * branch_means))
        mix_second = float(np.sum(pred_branch * (branch_vars + branch_means**2)))
        mix_var = max(mix_second - mix_mean**2, 1e-10)
        pred_means.append(mix_mean)
        pred_vars.append(mix_var)

        next_log_alpha = np.empty(2)
        for k in range(2):
            next_log_alpha[k] = logsumexp(log_alpha + logP[:, k]) + emission_ll[k]

        ll_step = float(logsumexp(next_log_alpha))
        step_ll.append(ll_step)
        log_alpha = next_log_alpha - ll_step

    step_ll = np.asarray(step_ll)

    if return_predictions:
        return step_ll, np.asarray(pred_means), np.asarray(pred_vars)
    if return_steps:
        return step_ll
    return float(step_ll.sum())


def model_loglik(model, params, times, y, return_steps=False):
    if model in {"OU", "OU+T", "OU+L"}:
        return loglik_nonbranch(model, params, times, y, return_steps=return_steps)
    return forward_branch(model, params, times, y, return_steps=return_steps)


# ============================================================
# Fitting
# ============================================================

def heuristic_start(model, times, y):
    dy = np.diff(y)
    dt = np.diff(times)
    med_dt = max(float(np.median(dt)), 1e-3)

    mu0 = float(np.mean(y))
    sigma0 = clamp(np.std(dy) / math.sqrt(med_dt + 1e-9), 0.08, 0.8)
    obs0 = clamp(0.25 * np.std(y) + 0.05, 0.05, 0.5)

    pre = y[times < T_TX]
    post = y[times >= T_TX]
    delta0 = 0.0
    if len(pre) >= 2 and len(post) >= 2:
        delta0 = clamp(float(np.mean(post) - np.mean(pre)), -1.0, 1.0)

    sep0 = clamp(max(0.20, 1.2 * np.std(y)), 0.10, 1.5)
    sj0 = clamp(max(0.15, np.quantile(np.abs(dy), 0.8)), 0.08, 1.5)

    starts = {
        "OU": [0.6, mu0, sigma0, obs0],
        "OU+T": [0.6, mu0, delta0, sigma0, obs0],
        "OU+L": [0.6, mu0, sigma0, 0.15, sj0, obs0],
        "OU+B": [0.6, mu0, sep0, sigma0, 0.10, obs0],
        "OULB": [0.6, mu0, sep0, delta0, sigma0, 0.10, 0.15, sj0, obs0],
    }
    return np.asarray(starts[model], dtype=float)


def fit_model(model, times_train, y_train, fit_seed):
    spec = PARAMETER_SPECS[model]
    bounds = spec["bounds"]

    def objective(p):
        params = unpack_params(model, p)
        ll = model_loglik(model, params, times_train, y_train)
        if not np.isfinite(ll):
            return 1e20
        return -ll

    starts = [heuristic_start(model, times_train, y_train)]
    srng = np.random.default_rng(fit_seed)

    for _ in range(max(0, N_STARTS - 1)):
        random_start = np.array([srng.uniform(lo, hi) for lo, hi in bounds])
        starts.append(random_start)

    best = None
    for x0 in starts:
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res

    success = bool(best is not None and np.isfinite(best.fun))
    if not success:
        return {
            "fit_success": False,
            "params": {},
            "train_loglik": np.nan,
            "n_iter": np.nan,
            "message": "No finite fit",
        }

    return {
        "fit_success": bool(best.success or np.isfinite(best.fun)),
        "params": unpack_params(model, best.x),
        "train_loglik": float(-best.fun),
        "n_iter": getattr(best, "nit", np.nan),
        "message": str(getattr(best, "message", "")),
    }


def fit_all_models_for_trajectory(generator, difficulty, replicate, traj):
    times = traj["times"]
    y = traj["observed"]
    split = traj["split_index"]

    times_train = times[:split]
    y_train = y[:split]

    rows = []
    for m_idx, model in enumerate(MODELS):
        fit_seed = (
            SEED + 9000000
            + replicate * 1009
            + DIFFICULTIES.index(difficulty) * 101
            + (GENERATORS.index(generator) if generator in GENERATORS else 0) * 17
            + m_idx
        )
        fit = fit_model(model, times_train, y_train, fit_seed)
        metrics = evaluate_fitted_model(model, fit, times, y, split)

        params = fit.get("params", {})
        row = {
            "generator": generator,
            "difficulty": difficulty,
            "replicate": replicate,
            "fitted_model": model,
            "fit_success": bool(fit["fit_success"]),
            "train_loglik": fit.get("train_loglik", np.nan),
            "heldout_elpd": metrics["heldout_elpd"],
            "heldout_elpd_per_obs": metrics["heldout_elpd_per_obs"],
            "heldout_rmse": metrics["heldout_rmse"],
            "coverage95": metrics["coverage95"],
            "residual_acf1": metrics["residual_acf1"],
            "n_iter": fit.get("n_iter", np.nan),
            "optimizer_message": fit.get("message", ""),
        }

        for name in [
            "theta", "mu", "base_mu", "branch_sep", "delta",
            "sigma", "q", "lambda", "sJ", "obs_sd"
        ]:
            row[f"{name}_hat"] = params.get(name, np.nan)

        rows.append(row)

    return rows


def pooled_heuristic_start(
    model,
    trajectories,
):
    """
    Construct one heuristic starting point from all pooled
    training trajectories.
    """

    all_y = []
    all_dy = []
    all_dt = []

    pre_y = []
    post_y = []

    for traj in trajectories:

        times = traj["times"]
        y = traj["observed"]

        all_y.extend(y)

        if len(y) > 1:
            all_dy.extend(np.diff(y))
            all_dt.extend(np.diff(times))

        pre_y.extend(
            y[times < T_TX]
        )

        post_y.extend(
            y[times >= T_TX]
        )

    all_y = np.asarray(
        all_y,
        dtype=float,
    )

    all_dy = np.asarray(
        all_dy,
        dtype=float,
    )

    all_dt = np.asarray(
        all_dt,
        dtype=float,
    )

    med_dt = max(
        float(np.median(all_dt)),
        1e-3,
    )

    mu0 = float(
        np.mean(all_y)
    )

    sigma0 = clamp(
        np.std(all_dy)
        / np.sqrt(med_dt + 1e-9),
        0.08,
        0.8,
    )

    obs0 = clamp(
        0.25 * np.std(all_y) + 0.05,
        0.05,
        0.5,
    )

    if (
        len(pre_y) >= 2
        and len(post_y) >= 2
    ):

        delta0 = clamp(
            float(
                np.mean(post_y)
                - np.mean(pre_y)
            ),
            -1.0,
            1.0,
        )

    else:

        delta0 = 0.0

    sep0 = clamp(
        max(
            0.20,
            1.2 * np.std(all_y),
        ),
        0.10,
        1.5,
    )

    sj0 = clamp(
        max(
            0.15,
            np.quantile(
                np.abs(all_dy),
                0.80,
            ),
        ),
        0.08,
        1.5,
    )

    starts = {
        "OU": [
            0.6,
            mu0,
            sigma0,
            obs0,
        ],

        "OU+T": [
            0.6,
            mu0,
            delta0,
            sigma0,
            obs0,
        ],

        "OU+L": [
            0.6,
            mu0,
            sigma0,
            0.15,
            sj0,
            obs0,
        ],

        "OU+B": [
            0.6,
            mu0,
            sep0,
            sigma0,
            0.10,
            obs0,
        ],

        "OULB": [
            0.6,
            mu0,
            sep0,
            delta0,
            sigma0,
            0.10,
            0.15,
            sj0,
            obs0,
        ],
    }

    return np.asarray(
        starts[model],
        dtype=float,
    )


def fit_model_pooled(
    model,
    trajectories,
    fit_seed,
):
    """
    Fit one shared parameter vector to a collection of
    independent training trajectories.

    Each trajectory contributes its full likelihood.
    Trajectories remain independent; likelihoods are summed.
    """

    spec = PARAMETER_SPECS[model]
    bounds = spec["bounds"]

    def objective(p):

        params = unpack_params(
            model,
            p,
        )

        total_ll = 0.0

        for traj in trajectories:

            times = traj["times"]
            y = traj["observed"]

            ll = model_loglik(
                model,
                params,
                times,
                y,
            )

            if not np.isfinite(ll):
                return 1e20

            total_ll += ll

        return -total_ll

    starts = [
        pooled_heuristic_start(
            model,
            trajectories,
        )
    ]

    srng = np.random.default_rng(
        fit_seed
    )

    for _ in range(
        max(0, N_STARTS - 1)
    ):

        random_start = np.array([
            srng.uniform(lo, hi)
            for lo, hi in bounds
        ])

        starts.append(
            random_start
        )

    best = None

    for x0 in starts:

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 800,
            },
        )

        if (
            best is None
            or (
                np.isfinite(res.fun)
                and res.fun < best.fun
            )
        ):
            best = res

    success = bool(
        best is not None
        and np.isfinite(best.fun)
    )

    if not success:

        return {
            "fit_success": False,
            "params": {},
            "train_loglik": np.nan,
            "n_iter": np.nan,
            "message":
                "No finite pooled fit",
        }

    return {
        "fit_success":
            bool(
                best.success
                or np.isfinite(best.fun)
            ),

        "params":
            unpack_params(
                model,
                best.x,
            ),

        "train_loglik":
            float(-best.fun),

        "n_iter":
            getattr(
                best,
                "nit",
                np.nan,
            ),

        "message":
            str(
                getattr(
                    best,
                    "message",
                    "",
                )
            ),
    }


def predict_nonbranch(model, params, times, y):
    means = []
    vars_ = []
    for i in range(len(y) - 1):
        mean, base_var = nonbranch_step(model, params, times[i], times[i + 1], y[i])
        dt = times[i + 1] - times[i]
        var = base_var
        if model == "OU+L":
            var = jump_predictive_variance(base_var, params["lambda"], params["sJ"], dt)
        means.append(mean)
        vars_.append(var)
    return np.asarray(means), np.asarray(vars_)


def evaluate_fitted_model(model, fit, times, y, split_index):
    if not fit["fit_success"]:
        return {
            "heldout_elpd": np.nan,
            "heldout_elpd_per_obs": np.nan,
            "heldout_rmse": np.nan,
            "coverage95": np.nan,
            "residual_acf1": np.nan,
        }

    params = fit["params"]

    if model in {"OU", "OU+T", "OU+L"}:
        step_ll = model_loglik(model, params, times, y, return_steps=True)
        pred_mean, pred_var = predict_nonbranch(model, params, times, y)
    else:
        step_ll, pred_mean, pred_var = forward_branch(
            model, params, times, y, return_predictions=True
        )

    heldout_step_start = max(0, split_index - 1)
    mask = np.arange(len(step_ll)) >= heldout_step_start

    heldout_ll = step_ll[mask]
    heldout_y = y[1:][mask]
    heldout_mean = pred_mean[mask]
    heldout_var = pred_var[mask]

    rmse = float(np.sqrt(np.mean((heldout_y - heldout_mean) ** 2)))
    sd = np.sqrt(np.maximum(heldout_var, 1e-12))
    covered = (
        (heldout_y >= heldout_mean - 1.96 * sd)
        & (heldout_y <= heldout_mean + 1.96 * sd)
    )

    residuals = (
        heldout_y
        - heldout_mean
    )
    
    if len(residuals) >= 3:
    
        residual_acf1 = np.corrcoef(
            residuals[:-1],
            residuals[1:]
        )[0, 1]
    
    else:
        residual_acf1 = np.nan

    return {
        "heldout_elpd":
            float(np.sum(heldout_ll)),
    
        "heldout_elpd_per_obs":
            float(np.mean(heldout_ll)),
    
        "heldout_rmse":
            rmse,
    
        "coverage95":
            float(np.mean(covered)),
    
        "residual_acf1":
            residual_acf1,
    }


def evaluate_independent_trajectory(
    model,
    fit,
    traj,
):
    """
    Score an independent test trajectory using parameters
    estimated from pooled training trajectories.

    The first observation initializes prediction; all
    subsequent observations are scored sequentially.
    """

    if not fit["fit_success"]:

        return {
            "test_elpd":
                np.nan,

            "test_elpd_per_obs":
                np.nan,

            "test_rmse":
                np.nan,

            "coverage95":
                np.nan,

            "residual_acf1":
                np.nan,
        }

    times = traj["times"]
    y = traj["observed"]

    params = fit["params"]

    if model in {
        "OU",
        "OU+T",
        "OU+L",
    }:

        step_ll = model_loglik(
            model,
            params,
            times,
            y,
            return_steps=True,
        )

        pred_mean, pred_var = (
            predict_nonbranch(
                model,
                params,
                times,
                y,
            )
        )

    else:

        (
            step_ll,
            pred_mean,
            pred_var,
        ) = forward_branch(
            model,
            params,
            times,
            y,
            return_predictions=True,
        )

    test_y = y[1:]

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    test_y
                    - pred_mean
                ) ** 2
            )
        )
    )

    sd = np.sqrt(
        np.maximum(
            pred_var,
            1e-12,
        )
    )

    covered = (
        (
            test_y
            >= pred_mean
            - 1.96 * sd
        )
        &
        (
            test_y
            <= pred_mean
            + 1.96 * sd
        )
    )

    residuals = (
        test_y
        - pred_mean
    )

    if (
        len(residuals) >= 3
        and np.std(residuals[:-1]) > 0
        and np.std(residuals[1:]) > 0
    ):

        residual_acf1 = float(
            np.corrcoef(
                residuals[:-1],
                residuals[1:],
            )[0, 1]
        )

    else:

        residual_acf1 = np.nan

    return {
        "test_elpd":
            float(
                np.sum(step_ll)
            ),

        "test_elpd_per_obs":
            float(
                np.mean(step_ll)
            ),

        "test_rmse":
            rmse,

        "coverage95":
            float(
                np.mean(covered)
            ),

        "residual_acf1":
            residual_acf1,
    }


def append_trajectory_records(
    records,
    traj,
    generator,
    difficulty,
    trajectory_id,
    dataset_role,
):
    """
    Append one simulated trajectory to the raw trajectory table.
    """

    times = traj["times"]
    truth = traj["truth"]

    for i in range(len(times)):

        records.append({
            "generator": generator,
            "difficulty": difficulty,
            "dataset_role": dataset_role,
            "trajectory_id": trajectory_id,
            "obs_index": i,
            "time": times[i],
            "latent_state": traj["latent"][i],
            "observed_state": traj["observed"][i],

            "treatment": (
                int(times[i] >= T_TX)
                if truth["treatment_active"]
                else 0
            ),

            "true_branch": int(
                traj["branch"][i]
            ),

            "jump_event": int(
                traj["jump_event"][i]
            ),

            "true_theta": truth["theta"],
            "true_sigma": truth["sigma"],
            "true_obs_noise": truth["obs_noise"],

            "true_treatment_shift":
                truth["treatment_shift"],

            "true_jump_time":
                truth["jump_time"],

            "true_jump_size":
                truth["jump_size"],
                
            "true_jump_lambda":
                truth.get(
                    "jump_lambda",
                    np.nan,
                ),
        
            "true_jump_scale":
                truth.get(
                    "jump_scale",
                    np.nan,
                ),

            "true_branch_time":
                truth["branch_time"],

            "true_branch_sep":
                truth["branch_sep"],

            "true_branch_q":
                truth.get(
                    "branch_q",
                    np.nan,
                ),

            "true_n_branch_switches":
                truth.get(
                    "n_branch_switches",
                    0,
                ),
        })


def run_main_analysis():
    """
    Pooled-training / independent-test experiment
    for Panels B-D.

    For every generator x difficulty condition:

        1. simulate N_TRAIN_TRAJ training trajectories;
        2. simulate N_TEST_TRAJ independent test trajectories;
        3. fit each of the five candidate models once to the
           pooled training trajectories;
        4. score every candidate on every independent test
           trajectory;
        5. select the model with maximum test ELPD separately
           for each test trajectory.
    """

    trajectory_records = []
    model_fit_records = []
    test_score_records = []

    condition_counter = 0

    total_conditions = (
        len(GENERATORS)
        * len(DIFFICULTIES)
    )

    for generator in GENERATORS:

        for difficulty in DIFFICULTIES:

            condition_counter += 1

            print(
                f"\n[pooled main] "
                f"{condition_counter}/"
                f"{total_conditions}: "
                f"{generator}, "
                f"{difficulty}"
            )

            # --------------------------------------------
            # Independent trajectory pools
            # --------------------------------------------

            train_trajectories = []

            test_trajectories = []

            # Training replicates use IDs 0 ...
            for replicate in range(
                N_TRAIN_TRAJ
            ):

                traj = simulate_trajectory(
                    generator,
                    difficulty,
                    replicate,
                )

                train_trajectories.append(
                    traj
                )

                append_trajectory_records(
                    trajectory_records,
                    traj,
                    generator,
                    difficulty,
                    replicate,
                    dataset_role="train",
                )

            # Use a non-overlapping replicate namespace.
            for test_id in range(
                N_TEST_TRAJ
            ):

                replicate = (
                    100000
                    + test_id
                )

                traj = simulate_trajectory(
                    generator,
                    difficulty,
                    replicate,
                )

                test_trajectories.append(
                    traj
                )

                append_trajectory_records(
                    trajectory_records,
                    traj,
                    generator,
                    difficulty,
                    test_id,
                    dataset_role="test",
                )

            # --------------------------------------------
            # Fit candidate models once per condition
            # --------------------------------------------

            fitted_models = {}

            for m_idx, model in enumerate(
                MODELS
            ):

                print(
                    f"    fitting {model}"
                )

                fit_seed = (
                    SEED
                    + 9000000
                    + 10000
                    * GENERATORS.index(
                        generator
                    )
                    + 1000
                    * DIFFICULTIES.index(
                        difficulty
                    )
                    + m_idx
                )

                fit = fit_model_pooled(
                    model,
                    train_trajectories,
                    fit_seed,
                )

                fitted_models[model] = fit

                params = fit.get(
                    "params",
                    {},
                )

                row = {
                    "generator":
                        generator,

                    "difficulty":
                        difficulty,

                    "fitted_model":
                        model,

                    "n_train_trajectories":
                        N_TRAIN_TRAJ,

                    "fit_success":
                        bool(
                            fit[
                                "fit_success"
                            ]
                        ),

                    "train_loglik":
                        fit.get(
                            "train_loglik",
                            np.nan,
                        ),

                    "n_iter":
                        fit.get(
                            "n_iter",
                            np.nan,
                        ),

                    "optimizer_message":
                        fit.get(
                            "message",
                            "",
                        ),
                }

                for name in [
                    "theta",
                    "mu",
                    "base_mu",
                    "branch_sep",
                    "delta",
                    "sigma",
                    "q",
                    "lambda",
                    "sJ",
                    "obs_sd",
                ]:

                    row[
                        f"{name}_hat"
                    ] = params.get(
                        name,
                        np.nan,
                    )

                model_fit_records.append(
                    row
                )

            # --------------------------------------------
            # Score independent test trajectories
            # --------------------------------------------

            for test_id, traj in enumerate(
                test_trajectories
            ):

                for model in MODELS:

                    fit = (
                        fitted_models[
                            model
                        ]
                    )

                    metrics = (
                        evaluate_independent_trajectory(
                            model,
                            fit,
                            traj,
                        )
                    )

                    test_score_records.append({
                        "generator":
                            generator,

                        "difficulty":
                            difficulty,

                        "test_id":
                            test_id,

                        "fitted_model":
                            model,

                        "fit_success":
                            bool(
                                fit[
                                    "fit_success"
                                ]
                            ),

                        "test_elpd":
                            metrics[
                                "test_elpd"
                            ],

                        "test_elpd_per_obs":
                            metrics[
                                "test_elpd_per_obs"
                            ],

                        "test_rmse":
                            metrics[
                                "test_rmse"
                            ],

                        "coverage95":
                            metrics[
                                "coverage95"
                            ],

                        "residual_acf1":
                            metrics[
                                "residual_acf1"
                            ],
                    })

    # ====================================================
    # Save raw trajectories and pooled fits
    # ====================================================

    df_traj = pd.DataFrame(
        trajectory_records
    )

    df_fits = pd.DataFrame(
        model_fit_records
    )

    df_scores = pd.DataFrame(
        test_score_records
    )

    # --------------------------------------------
    # Test-trajectory Delta ELPD
    # --------------------------------------------

    df_scores["delta_elpd"] = np.nan

    valid = (
        df_scores["fit_success"]
        & np.isfinite(
            df_scores["test_elpd"]
        )
    )

    valid_scores = (
        df_scores.loc[
            valid
        ].copy()
    )

    df_scores.loc[
        valid,
        "delta_elpd",
    ] = (
        valid_scores[
            "test_elpd"
        ]
        -
        valid_scores.groupby([
            "generator",
            "difficulty",
            "test_id",
        ])[
            "test_elpd"
        ].transform("max")
    )

    df_traj.to_csv(
        DATA_DIR
        / "fig8_trajectories.csv",
        index=False,
    )

    df_fits.to_csv(
        DATA_DIR
        / "fig8_pooled_model_fits.csv",
        index=False,
    )

    df_scores.to_csv(
        DATA_DIR
        / "fig8_test_scores.csv",
        index=False,
    )

    # --------------------------------------------
    # Model selection per independent trajectory
    # --------------------------------------------

    successful = df_scores[
        df_scores["fit_success"]
        & np.isfinite(
            df_scores["test_elpd"]
        )
    ].copy()

    idx = (
        successful
        .groupby([
            "generator",
            "difficulty",
            "test_id",
        ])[
            "test_elpd"
        ]
        .idxmax()
    )

    df_sel = successful.loc[
        idx,
        [
            "generator",
            "difficulty",
            "test_id",
            "fitted_model",
            "test_elpd",
            "test_elpd_per_obs",
            "test_rmse",
            "coverage95",
            "residual_acf1",
        ],
    ].copy()

    df_sel = df_sel.rename(
        columns={
            "fitted_model":
                "selected_model"
        }
    )

    df_sel["correct"] = (
        df_sel["generator"]
        == df_sel["selected_model"]
    ).astype(float)

    df_sel.loc[
        df_sel["generator"]
        == "Nonlinear",
        "correct",
    ] = np.nan

    df_sel.to_csv(
        DATA_DIR
        / "fig8_model_selection.csv",
        index=False,
    )

    print(
        "\nSaved pooled main-analysis CSVs."
    )

    return (
        df_traj,
        df_scores,
        df_sel,
    )


# ============================================================
# Temporal-confounding analysis
# ============================================================

def model_contains_component(model, component):
    if component == "Treatment":
        return model in {"OU+T", "OULB"}
    if component == "Jump":
        return model in {"OU+L", "OULB"}
    if component == "Branch":
        return model in {"OU+B", "OULB"}
    raise ValueError(component)


def run_temporal_confounding():
    distances = np.arange(0.0, 3.01, 0.5)
    records = []

    total = 3 * len(distances) * N_TEMPORAL_REPLICATES
    counter = 0

    for component in ["Treatment", "Jump", "Branch"]:
        for distance in distances:
            for replicate in range(N_TEMPORAL_REPLICATES):
                counter += 1
                if counter == 1 or counter % 25 == 0 or counter == total:
                    print(f"[temporal] {counter}/{total}: {component}, distance={distance:.1f}, rep={replicate}")

                traj = simulate_in_family(
                    generator="OULB",
                    difficulty="Intermediate",
                    replicate=replicate,
                    event_distance=float(distance),
                    temporal_mode=component,
                )

                rows = fit_all_models_for_trajectory(
                    generator="OULB",
                    difficulty="Intermediate",
                    replicate=replicate,
                    traj=traj,
                )
                fit_df = pd.DataFrame(rows)
                fit_df = fit_df[
                    fit_df["fit_success"] & np.isfinite(fit_df["heldout_elpd"])
                ]

                if len(fit_df) == 0:
                    selected = None
                    success = np.nan
                    selected_elpd = np.nan
                else:
                    best_row = fit_df.loc[fit_df["heldout_elpd"].idxmax()]
                    selected = best_row["fitted_model"]
                    success = float(model_contains_component(selected, component))
                    selected_elpd = float(best_row["heldout_elpd"])

                records.append({
                    "component": component,
                    "event_distance": float(distance),
                    "replicate": replicate,
                    "treatment_time": T_TX,
                    "endogenous_event_time": T_TX + float(distance),
                    "selected_model": selected,
                    "component_identified": success,
                    "selected_heldout_elpd": selected_elpd,
                })

    df = pd.DataFrame(records)
    df.to_csv(DATA_DIR / "fig8_temporal_confounding.csv", index=False)
    print("Saved temporal-confounding CSV.")
    return df


# ============================================================
# Misspecification analysis
# ============================================================

def run_misspecification():
    """
    Panel F: pooled-training, independent-test robustness analysis.

    For each generating family (in-family OULB and out-of-family
    nonlinear bistable):

        1. Simulate N_TRAIN_TRAJ independent training trajectories.
        2. Fit OU and OULB jointly to exactly the same pooled
           training trajectories.
        3. Simulate N_MISSPEC_REPLICATES completely independent
           test trajectories.
        4. Evaluate both fitted models on every test trajectory.

    The resulting test scores permit paired, trajectory-level
    comparisons:

        delta_elpd_per_obs = ELPD_OULB - ELPD_OU
        delta_rmse         = RMSE_OU - RMSE_OULB

    Thus positive values indicate predictive improvement from OULB.

    OULB 95% predictive coverage is retained as an absolute
    calibration diagnostic.

    This design matches the pooled-training / independent-test
    architecture used in Panels B-D and avoids estimating the
    nine-parameter OULB model from a single short trajectory.
    """

    records = []

    models_F = [
        "OU",
        "OULB",
    ]

    for family in [
        "OULB",
        "Nonlinear",
    ]:

        print(
            f"\n[misspec] generating family: {family}"
        )

        # ====================================================
        # 1. Generate independent pooled training trajectories
        # ====================================================

        train_trajectories = []

        for train_id in range(
            N_TRAIN_TRAJ
        ):

            # Training namespace:
            # 200000, 200001, ...
            replicate = (
                200000
                + train_id
            )

            traj = simulate_trajectory(
                family,
                "Intermediate",
                replicate,
            )

            train_trajectories.append(
                traj
            )

        print(
            f"    generated "
            f"{N_TRAIN_TRAJ} training trajectories"
        )

        # ====================================================
        # 2. Fit OU and OULB once to the same pooled training set
        # ====================================================

        fitted_models = {}

        for m_idx, model in enumerate(
            models_F
        ):

            print(
                f"    fitting pooled {model}"
            )

            fit_seed = (
                SEED
                + 12000000
                + (
                    0
                    if family == "OULB"
                    else 500000
                )
                + 10000 * m_idx
            )

            fit = fit_model_pooled(
                model,
                train_trajectories,
                fit_seed=fit_seed,
            )

            fitted_models[
                model
            ] = fit

            if not fit[
                "fit_success"
            ]:

                print(
                    f"    WARNING: pooled "
                    f"{model} fit failed "
                    f"for {family}"
                )

        # ====================================================
        # 3. Generate completely independent test trajectories
        # ====================================================

        for test_id in range(
            N_MISSPEC_REPLICATES
        ):

            if (
                test_id == 0
                or (test_id + 1) % 10 == 0
                or test_id
                == N_MISSPEC_REPLICATES - 1
            ):

                print(
                    f"    [test] "
                    f"{test_id + 1}/"
                    f"{N_MISSPEC_REPLICATES}"
                )

            # Non-overlapping test namespace:
            # 300000, 300001, ...
            replicate = (
                300000
                + test_id
            )

            traj = simulate_trajectory(
                family,
                "Intermediate",
                replicate,
            )

            # =================================================
            # 4. Score both fitted models on exactly the same
            #    independent test trajectory
            # =================================================

            for model in models_F:

                fit = (
                    fitted_models[
                        model
                    ]
                )

                metrics = (
                    evaluate_independent_trajectory(
                        model,
                        fit,
                        traj,
                    )
                )

                records.append({
                    "generator_family":
                        family,

                    "replicate":
                        test_id,

                    "fitted_model":
                        model,

                    "fit_success":
                        bool(
                            fit[
                                "fit_success"
                            ]
                        ),

                    "test_elpd":
                        metrics[
                            "test_elpd"
                        ],

                    "test_elpd_per_obs":
                        metrics[
                            "test_elpd_per_obs"
                        ],

                    "test_rmse":
                        metrics[
                            "test_rmse"
                        ],

                    "coverage95":
                        metrics[
                            "coverage95"
                        ],

                    "residual_acf1":
                        metrics[
                            "residual_acf1"
                        ],
                })

    # ========================================================
    # Save trajectory-level test scores
    # ========================================================

    df = pd.DataFrame(
        records
    )

    df.to_csv(
        DATA_DIR
        / "fig8_misspecification.csv",
        index=False,
    )

    print(
        "Saved pooled paired "
        "misspecification CSV."
    )

    return df


# ============================================================
# Summary tables for Panels B-F
# ============================================================

def create_summary_tables(
    df_scores,
    df_sel,
    df_temporal,
    df_misspec,
):
    sel_mid = df_sel[df_sel["difficulty"] == "Intermediate"].copy()

    panelB = pd.crosstab(
        sel_mid["generator"],
        sel_mid["selected_model"],
        normalize="index",
    ).reindex(index=GENERATORS, columns=MODELS, fill_value=0.0)
    panelB.to_csv(SUMMARY_DIR / "panelB_confusion.csv")

    scores_mid = df_scores[
        (df_scores["difficulty"] == "Intermediate")
        & df_scores["fit_success"]
        & np.isfinite(
            df_scores["delta_elpd"]
        )
    ].copy()

    panelC = scores_mid.pivot_table(
        index="generator",
        columns="fitted_model",
        values="delta_elpd",
        aggfunc="mean",
    ).reindex(
        index=GENERATORS,
        columns=MODELS,
    )

    panelC.to_csv(
        SUMMARY_DIR
        / "panelC_delta_elpd.csv"
    )

    rows = []
    dsel = df_sel[df_sel["generator"].isin(["OU+T", "OU+L", "OU+B", "OULB"])].copy()

    for (generator, difficulty), g in dsel.groupby(["generator", "difficulty"]):
        vals = g["correct"].dropna().astype(float).values
        mean, lo, hi = bootstrap_mean_ci(
            vals,
            seed=SEED + 31 * DIFFICULTIES.index(difficulty)
                 + 7 * ["OU+T", "OU+L", "OU+B", "OULB"].index(generator),
        )
        rows.append({
            "generator": generator,
            "difficulty": difficulty,
            "accuracy": mean,
            "ci_low": lo,
            "ci_high": hi,
            "n": len(vals),
        })

    panelD = pd.DataFrame(rows)
    panelD.to_csv(SUMMARY_DIR / "panelD_difficulty.csv", index=False)

    rows = []
    for (component, distance), g in df_temporal.groupby(["component", "event_distance"]):
        vals = g["component_identified"].dropna().astype(float).values
        mean, lo, hi = bootstrap_mean_ci(
            vals,
            seed=SEED + int(distance * 100)
                 + {"Treatment": 1, "Jump": 2, "Branch": 3}[component],
        )
        rows.append({
            "component": component,
            "event_distance": distance,
            "identification_rate": mean,
            "ci_low": lo,
            "ci_high": hi,
            "n": len(vals),
        })

    panelE = pd.DataFrame(rows)
    panelE.to_csv(SUMMARY_DIR / "panelE_temporal.csv", index=False)

    # --------------------------------------------------------
    # Panel F: paired misspecification analysis
    # --------------------------------------------------------

    mm = df_misspec[
        df_misspec["fit_success"]
    ].copy()

    # Require both OU and OULB fits for the same trajectory.
    paired = mm.pivot_table(
        index=[
            "generator_family",
            "replicate",
        ],
        columns="fitted_model",
        values=[
            "test_elpd_per_obs",
            "test_rmse",
            "coverage95",
        ],
        aggfunc="first",
    )

    # Retain trajectories for which both models produced
    # finite predictive metrics.
    paired = paired.dropna(
        subset=[
            (
                "test_elpd_per_obs",
                "OU",
            ),
            (
                "test_elpd_per_obs",
                "OULB",
            ),
            (
                "test_rmse",
                "OU",
            ),
            (
                "test_rmse",
                "OULB",
            ),
        ]
    ).copy()

    # Positive means OULB is better for both quantities.
    paired[
        "delta_elpd_per_obs"
    ] = (
        paired[
            (
                "test_elpd_per_obs",
                "OULB",
            )
        ]
        -
        paired[
            (
                "test_elpd_per_obs",
                "OU",
            )
        ]
    )
    
    paired[
        "delta_rmse"
    ] = (
        paired[
            (
                "test_rmse",
                "OU",
            )
        ]
        -
        paired[
            (
                "test_rmse",
                "OULB",
            )
        ]
    )

    paired[
        "oulb_coverage95"
    ] = paired[
        (
            "coverage95",
            "OULB",
        )
    ]

    paired = (
        paired
        .reset_index()
    )

    rows = []

    metrics_F = [
        "delta_elpd_per_obs",
        "delta_rmse",
        "oulb_coverage95",
    ]

    for family in [
        "OULB",
        "Nonlinear",
    ]:

        gf = paired[
            paired[
                "generator_family"
            ]
            == family
        ]

        for metric in metrics_F:

            vals = (
                gf[metric]
                .dropna()
                .astype(float)
                .values
            )

            mean, lo, hi = (
                bootstrap_mean_ci(
                    vals,
                    seed=(
                        SEED
                        + (
                            0
                            if family == "OULB"
                            else 100
                        )
                        + metrics_F.index(
                            metric
                        )
                    ),
                )
            )

            rows.append({
                "generator_family":
                    family,

                "metric":
                    metric,

                "mean":
                    mean,

                "ci_low":
                    lo,

                "ci_high":
                    hi,

                "n":
                    len(vals),
            })

    panelF = pd.DataFrame(
        rows
    )

    panelF.to_csv(
        SUMMARY_DIR
        / "panelF_misspecification.csv",
        index=False,
    )

    print("Saved panel summary CSVs.")


# ============================================================
# Panel A schematic helpers
# ============================================================

def simulate_panelA_ou(
    times,
    theta=0.6,
    mu=0.0,
    sigma=0.22,
    x0=0.0,
    treatment_time=None,
    treatment_shift=0.0,
    jump_time=None,
    jump_size=0.0,
    branch_time=None,
    branch_mu=None,
    obs_noise=0.05,
):
    local_rng = np.random.default_rng(SEED + 4242)
    x = np.zeros(len(times))
    x[0] = x0

    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        midpoint = 0.5 * (times[i] + times[i - 1])
        current_mu = mu

        if treatment_time is not None and midpoint >= treatment_time:
            current_mu += treatment_shift

        if branch_time is not None and midpoint >= branch_time:
            current_mu = branch_mu

        x[i] = exact_ou_step(x[i - 1], dt, theta, current_mu, sigma, local_rng)

        if jump_time is not None and times[i - 1] < jump_time <= times[i]:
            x[i] += jump_size

    return x + local_rng.normal(0.0, obs_noise, len(x))


def simulate_panelA_bistable(times):
    local_rng = np.random.default_rng(SEED + 5252)

    x = np.zeros(len(times))
    x[0] = -0.80

    a_dw = 1.25
    b_dw = 0.80
    forcing_amp = 0.75
    forcing_period = 4.0
    omega = 2.0 * np.pi / forcing_period
    sigma_dw = 0.65

    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        t_mid = 0.5 * (times[i] + times[i - 1])

        nonlinear_drift = (
            -4.0
            * a_dw
            * x[i - 1]
            * (x[i - 1] ** 2 - b_dw ** 2)
        )

        forcing = (
            forcing_amp
            * np.sin(omega * t_mid)
        )

        x[i] = (
            x[i - 1]
            + (nonlinear_drift + forcing) * dt
            + sigma_dw
            * np.sqrt(dt)
            * local_rng.normal()
        )

    return x


# ============================================================
# Plot final S7 from CSVs only
# ============================================================

def plot_final_figure():
    panelB_df = pd.read_csv(SUMMARY_DIR / "panelB_confusion.csv", index_col=0).reindex(
        index=GENERATORS, columns=MODELS
    )
    panelC_df = pd.read_csv(SUMMARY_DIR / "panelC_delta_elpd.csv", index_col=0).reindex(
        index=GENERATORS, columns=MODELS
    )
    panelD = pd.read_csv(SUMMARY_DIR / "panelD_difficulty.csv")
    panelE = pd.read_csv(SUMMARY_DIR / "panelE_temporal.csv")
    panelF = pd.read_csv(SUMMARY_DIR / "panelF_misspecification.csv")

    confusion = panelB_df.values.astype(float)
    delta_elpd = panelC_df.values.astype(float)

    fig = plt.figure(figsize=(18, 10.5))
    gs = GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1.25, 1.0, 1.15],
        height_ratios=[1.0, 0.9],
        wspace=0.28,
        hspace=0.42,
    )

    fig.suptitle(
        "Mechanistic discrimination, "
        "predictive model comparison, and robustness to misspecification",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    # ---------------- Panel A ----------------
    axA = fig.add_subplot(gs[0, 0])
    axA.axis("off")
    axA.text(-0.04, 1.05, "A", transform=axA.transAxes, fontsize=16, fontweight="bold")
    axA.text(0.06, 1.05, "Integrated simulation design", transform=axA.transAxes,
             fontsize=12, fontweight="bold")

    names = [
        "OU",
        "OU + T\n(Treatment)",
        "OU + L\n(Lévy jump)",
        "OU + B\n(Branch)",
        "Full OULB",
        "Nonlinear\n(bistable, out-of-family)",
    ]

    t = np.linspace(0, 10, 90)
    schematic = [
        simulate_panelA_ou(t, mu=0),
        simulate_panelA_ou(t, treatment_time=5, treatment_shift=0.7),
        simulate_panelA_ou(t, jump_time=5, jump_size=1.0),
        simulate_panelA_ou(t, branch_time=5, branch_mu=0.8),
        simulate_panelA_ou(
            t, treatment_time=4.5, treatment_shift=0.35,
            jump_time=5.2, jump_size=0.7,
            branch_time=6.0, branch_mu=0.85
        ),
        simulate_panelA_bistable(t),
    ]

    lefts = np.linspace(0.01, 0.84, 6)
    for i, (name, yy, left) in enumerate(zip(names, schematic, lefts)):
        iax = axA.inset_axes([left, 0.66, 0.135, 0.20])
        iax.plot(t, yy, lw=1.4)
        iax.axhline(0, lw=0.5, alpha=0.35)
        iax.set_title(name, fontsize=7.5, pad=4)
        iax.set_xticks([])
        iax.set_yticks([])
        if i == 0:
            iax.set_ylabel("Latent state", fontsize=7)
            iax.set_xlabel("Time", fontsize=7)

    axA.plot([0.08, 0.92], [0.62, 0.62], transform=axA.transAxes, lw=1)

    def rounded_box(ax, xy, width, height, text):
        patch = FancyBboxPatch(
            xy, width, height,
            boxstyle="round,pad=0.015",
            transform=ax.transAxes,
            linewidth=0.8,
            facecolor="0.94",
            edgecolor="0.45",
        )
        ax.add_patch(patch)
        ax.text(
            xy[0] + width / 2,
            xy[1] + height / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
        )

    rounded_box(axA, (0.18, 0.43), 0.50, 0.10,
                "Simulate trajectories\n(with noise and irregular sampling)")
    rounded_box(axA, (0.18, 0.25), 0.50, 0.10,
                "Fit all candidate models\n(OU, OU+T, OU+L, OU+B, OULB)")
    rounded_box(axA, (0.18, 0.07), 0.50, 0.10,
                "Evaluate mechanistic classification,\npredictive performance, and robustness")

    for y1, y2 in [(0.62, 0.54), (0.43, 0.36), (0.25, 0.18)]:
        axA.annotate(
            "", xy=(0.43, y2), xytext=(0.43, y1), xycoords=axA.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.2),
        )

    difficulty_text = (
        "Three difficulty regimes\n"
        "• Easy: high signal, dense sampling\n"
        "• Intermediate\n"
        "• Difficult: sparse sampling, high noise"
    )
    axA.text(
        0.73, 0.06, difficulty_text,
        transform=axA.transAxes,
        fontsize=7.5,
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="0.97",
            edgecolor="0.5",
            linestyle="--",
        ),
    )

    # ---------------- Panel B ----------------
    axB = fig.add_subplot(gs[0, 1])
    axB.text(-0.18, 1.05, "B", transform=axB.transAxes, fontsize=16, fontweight="bold")
    axB.set_title(
        "Mechanism confusion matrix\n"
        "(model selected by independent-test ELPD)",
        fontweight="bold",
        pad=12,
    )

    imB = axB.imshow(confusion, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axB.set_xticks(range(len(MODELS)))
    axB.set_xticklabels(MODELS)
    axB.set_yticks(range(len(GENERATORS)))
    axB.set_yticklabels(GENERATORS)
    axB.set_xlabel("Selected model")
    axB.set_ylabel("True generating mechanism")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            val = confusion[i, j]
            if np.isfinite(val):
                axB.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if val > 0.55 else "black")

    cbB = fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)
    cbB.set_label("Selection proportion")

    # ---------------- Panel C ----------------
    axC = fig.add_subplot(gs[0, 2])
    axC.text(-0.18, 1.05, "C", transform=axC.transAxes, fontsize=16, fontweight="bold")
    axC.set_title(
        "Predictive performance across candidate models\n"
        "(mean trajectory-level ΔELPD on independent test data)",
        fontweight="bold",
        pad=12,
    )

    finite = delta_elpd[np.isfinite(delta_elpd)]
    lower = min(-1.0, float(np.nanpercentile(finite, 5))) if len(finite) else -5.0
    imC = axC.imshow(delta_elpd, cmap="coolwarm", vmin=lower, vmax=0.0, aspect="auto")
    axC.set_xticks(range(len(MODELS)))
    axC.set_xticklabels(MODELS)
    axC.set_yticks(range(len(GENERATORS)))
    axC.set_yticklabels(GENERATORS)
    axC.set_xlabel("Fitted model")
    axC.set_ylabel("True generating mechanism")

    for i in range(delta_elpd.shape[0]):
        for j in range(delta_elpd.shape[1]):
            val = delta_elpd[i, j]
            if np.isfinite(val):
                axC.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    cbC = fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04)
    cbC.set_label(r"$\Delta$ELPD (nats)")

    # ---------------- Panel D ----------------
    axD = fig.add_subplot(gs[1, 0])
    axD.text(-0.12, 1.08, "D", transform=axD.transAxes, fontsize=16, fontweight="bold")
    axD.set_title("Mechanistic discrimination across difficulty regimes",
                  fontweight="bold", pad=12)

    marker_map = {"OU+T": "o", "OU+L": "s", "OU+B": "^", "OULB": "D"}
    label_map = {
        "OU+T": "OU + T (Therapy)",
        "OU+L": "OU + L (Jump)",
        "OU+B": "OU + B (Branch)",
        "OULB": "OULB (Full)",
    }
    xpos = np.arange(len(DIFFICULTIES))

    for generator in ["OU+T", "OU+L", "OU+B", "OULB"]:
        sub = panelD[panelD["generator"] == generator].copy()
        sub["difficulty"] = pd.Categorical(
            sub["difficulty"], categories=DIFFICULTIES, ordered=True
        )
        sub = sub.sort_values("difficulty")
        yv = sub["accuracy"].to_numpy(float)
        lo = sub["ci_low"].to_numpy(float)
        hi = sub["ci_high"].to_numpy(float)
        yerr = np.vstack([yv - lo, hi - yv])

        axD.errorbar(
            xpos, yv, yerr=yerr,
            marker=marker_map[generator],
            lw=1.4, capsize=3,
            label=label_map[generator],
        )

    axD.set_xticks(xpos)
    axD.set_xticklabels(DIFFICULTIES)
    axD.set_ylim(0, 1.02)
    axD.set_ylabel("Classification accuracy")
    axD.set_xlabel("Difficulty regime")
    axD.legend(
    loc="upper left",
    frameon=True,
    fontsize=8,
    )

    # ---------------- Panel E ----------------
    axE = fig.add_subplot(gs[1, 1])
    axE.text(-0.14, 1.08, "E", transform=axE.transAxes, fontsize=16, fontweight="bold")
    axE.set_title(
        "Effect of temporal proximity to treatment\non component discrimination",
        fontweight="bold", pad=12,
    )

    component_markers = {"Treatment": "o", "Jump": "s", "Branch": "^"}

    for component in ["Treatment", "Jump", "Branch"]:
        sub = panelE[panelE["component"] == component].sort_values("event_distance")
        x = sub["event_distance"].to_numpy(float)
        yv = sub["identification_rate"].to_numpy(float)
        lo = sub["ci_low"].to_numpy(float)
        hi = sub["ci_high"].to_numpy(float)
        yerr = np.vstack([yv - lo, hi - yv])

        axE.errorbar(
            x, yv, yerr=yerr,
            marker=component_markers[component],
            lw=1.4, capsize=3,
            label=component,
        )

    axE.set_ylim(0, 1.02)
    axE.set_xlabel(r"$|T_{\mathrm{event}}-T_{\mathrm{tx}}|$ (time units)")
    axE.set_ylabel("Component identification rate")
    axE.legend(loc="upper left")

    # ---------------- Panel F ----------------
    axF = fig.add_subplot(gs[1, 2])
    axF.axis("off")
    axF.text(-0.14, 1.08, "F", transform=axF.transAxes, fontsize=16, fontweight="bold")
    axF.text(
        0.02, 1.03,
        "Predictive value and calibration under model misspecification",
        transform=axF.transAxes,
        fontsize=11,
        fontweight="bold",
    )

    metrics = [
        (
            "delta_elpd_per_obs",
            r"$\Delta$ELPD / observation"
            "\n(OULB - OU)"
        ),
        (
            "delta_rmse",
            r"$\Delta$RMSE"
            "\n(OU - OULB)"
        ),
        (
            "oulb_coverage95",
            "OULB 95% predictive\ncoverage"
        ),
    ]

    lefts = [0.02, 0.35, 0.68]
    for left, (metric, title) in zip(lefts, metrics):
        iax = axF.inset_axes([left, 0.12, 0.28, 0.72])
        vals = panelF[panelF["metric"] == metric].set_index(
            "generator_family"
        ).reindex(["OULB", "Nonlinear"])

        means = vals["mean"].to_numpy(float)
        lo = vals["ci_low"].to_numpy(float)
        hi = vals["ci_high"].to_numpy(float)
        err = np.vstack([means - lo, hi - means])

        xx = np.arange(2)
        bars = iax.bar(xx, means, yerr=err, capsize=3, alpha=0.85)
        iax.set_xticks(xx)
        iax.set_xticklabels(["In-family\nOULB", "Nonlinear"])
        iax.set_title(title, fontsize=8, fontweight="bold")
        iax.axhline(0, lw=0.6, alpha=0.4)

        if metric == "oulb_coverage95":
            iax.axhline(0.95, lw=0.8, ls="--", alpha=0.6)
            iax.set_ylim(0, 1.05)

        for bar, mean in zip(bars, means):
            iax.text(
                bar.get_x() + bar.get_width() / 2,
                mean,
                f"{mean:.2f}",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=7,
            )

    plt.subplots_adjust(
        left=0.055,
        right=0.975,
        top=0.90,
        bottom=0.10,
        wspace=0.30,
        hspace=0.45,
    )

    fig.savefig(PNG, dpi=600, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    print(f"Saved PNG: {PNG}")
    print(f"Saved PDF: {PDF}")
    plt.show()


# ============================================================
# Run
# ============================================================

def required_raw_files_exist():

    files = [
        DATA_DIR
        / "fig8_trajectories.csv",

        DATA_DIR
        / "fig8_pooled_model_fits.csv",

        DATA_DIR
        / "fig8_test_scores.csv",

        DATA_DIR
        / "fig8_model_selection.csv",

        DATA_DIR
        / "fig8_temporal_confounding.csv",

        DATA_DIR
        / "fig8_misspecification.csv",
    ]

    return all(
        p.exists()
        for p in files
    )


def load_raw_outputs():

    return (
        pd.read_csv(
            DATA_DIR
            / "fig8_trajectories.csv"
        ),

        pd.read_csv(
            DATA_DIR
            / "fig8_test_scores.csv"
        ),

        pd.read_csv(
            DATA_DIR
            / "fig8_model_selection.csv"
        ),

        pd.read_csv(
            DATA_DIR
            / "fig8_temporal_confounding.csv"
        ),

        pd.read_csv(
            DATA_DIR
            / "fig8_misspecification.csv"
        ),
    )


if __name__ == "__main__":

    print(
        f"Output directory: {BASE_DIR}"
    )

    print(
        f"N_TRAIN_TRAJ={N_TRAIN_TRAJ}"
    )

    print(
        f"N_TEST_TRAJ={N_TEST_TRAJ}"
    )

    print(
        "N_TEMPORAL_REPLICATES="
        f"{N_TEMPORAL_REPLICATES}"
    )

    print(
        "N_MISSPEC_REPLICATES="
        f"{N_MISSPEC_REPLICATES}"
    )

    print(
        f"N_STARTS={N_STARTS}"
    )

    print(
        f"FORCE_ANALYSIS={FORCE_ANALYSIS}"
    )

    print(
        f"RERUN_MISSPEC_ONLY="
        f"{RERUN_MISSPEC_ONLY}"
    )

    # ========================================================
    # Analysis execution
    # ========================================================

    if RERUN_MISSPEC_ONLY:

        print(
            "Reusing existing Panels A-E data; "
            "rerunning Panel F misspecification analysis only."
        )

        (
            df_traj,
            df_scores,
            df_sel,
            df_temporal,
            _,
        ) = load_raw_outputs()

        df_misspec = (
            run_misspecification()
        )

    elif (
        FORCE_ANALYSIS
        or not required_raw_files_exist()
    ):

        (
            df_traj,
            df_scores,
            df_sel,
        ) = run_main_analysis()

        df_temporal = (
            run_temporal_confounding()
        )

        df_misspec = (
            run_misspecification()
        )

    else:

        print(
            "Existing raw CSVs found; "
            "reusing them."
        )

        (
            df_traj,
            df_scores,
            df_sel,
            df_temporal,
            df_misspec,
        ) = load_raw_outputs()

    # ========================================================
    # Summaries and figure
    # ========================================================

    create_summary_tables(
        df_scores,
        df_sel,
        df_temporal,
        df_misspec,
    )

    plot_final_figure()

    print("\nGenerated raw data:")

    for name in [
        "fig8_trajectories.csv",
        "fig8_pooled_model_fits.csv",
        "fig8_test_scores.csv",
        "fig8_model_selection.csv",
        "fig8_temporal_confounding.csv",
        "fig8_misspecification.csv",
    ]:

        print(
            "  ",
            DATA_DIR / name,
        )
