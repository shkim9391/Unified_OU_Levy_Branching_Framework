from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ObservationSpec:
    noise_sd: float = 0.0
    missing_probability: float = 0.0
    retain_endpoints: bool = True


def regular_schedule(start: float, stop: float, n: int) -> np.ndarray:
    if n < 2 or stop <= start:
        raise ValueError("Require n>=2 and stop>start")
    return np.linspace(start, stop, n, dtype=float)


def irregular_schedule(start: float, stop: float, n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2 or stop <= start:
        raise ValueError("Require n>=2 and stop>start")
    interior = np.sort(rng.uniform(start, stop, size=max(0, n - 2)))
    return np.concatenate(([start], interior, [stop])).astype(float)


def thin_schedule(times: np.ndarray, keep_n: int, rng: np.random.Generator, retain_endpoints: bool = True) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if keep_n < 2 or keep_n > len(times):
        raise ValueError("keep_n must lie between 2 and len(times)")
    if retain_endpoints:
        interior = np.arange(1, len(times)-1)
        chosen = rng.choice(interior, size=keep_n-2, replace=False) if keep_n > 2 else np.array([], dtype=int)
        idx = np.sort(np.concatenate(([0], chosen, [len(times)-1])))
    else:
        idx = np.sort(rng.choice(np.arange(len(times)), size=keep_n, replace=False))
    return times[idx]


def observe_latent(times: np.ndarray, states: np.ndarray, spec: ObservationSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    if states.shape[0] != times.size:
        raise ValueError("states first dimension must match times")
    keep = rng.random(times.size) >= spec.missing_probability
    if spec.retain_endpoints and times.size:
        keep[0] = keep[-1] = True
    obs = states[keep].copy()
    if spec.noise_sd > 0:
        obs += rng.normal(0.0, spec.noise_sd, size=obs.shape)
    return times[keep], obs
