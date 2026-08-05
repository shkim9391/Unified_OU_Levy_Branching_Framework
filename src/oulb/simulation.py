from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from scipy.linalg import expm

Array = np.ndarray

@dataclass(frozen=True)
class JumpSpec:
    rate: float = 0.0
    scale: float = 0.0
    distribution: str = "normal"

@dataclass(frozen=True)
class BranchSpec:
    generator: Array
    theta: Array
    mu: Array
    sigma: Array

@dataclass
class SimulationResult:
    times: Array
    states: Array
    branches: Array
    event_ledger: list[dict] = field(default_factory=list)


def _vec(x, d: int) -> Array:
    a = np.asarray(x, dtype=float)
    if a.ndim == 0: a = np.repeat(a, d)
    if a.shape != (d,): raise ValueError(f"Expected scalar or shape ({d},)")
    return a


def brownian_transition(x: Array, dt: float, drift, sigma, rng: np.random.Generator) -> Array:
    x = np.asarray(x, dtype=float); d=x.size
    return x + _vec(drift,d)*dt + _vec(sigma,d)*np.sqrt(dt)*rng.normal(size=d)


def ou_transition(x: Array, dt: float, theta, mu, sigma, rng: np.random.Generator) -> Array:
    x=np.asarray(x,dtype=float); d=x.size
    th=_vec(theta,d); m=_vec(mu,d); sg=_vec(sigma,d)
    if np.any(th < 0): raise ValueError("theta must be nonnegative")
    phi=np.exp(-th*dt)
    var=sg**2*dt
    mask=th>1e-12
    var[mask]=sg[mask]**2*(1-np.exp(-2*th[mask]*dt))/(2*th[mask])
    return m+(x-m)*phi+np.sqrt(np.maximum(var,0))*rng.normal(size=d)


def draw_compound_poisson(dt: float, spec: JumpSpec, d: int, rng: np.random.Generator) -> tuple[Array,int]:
    n=int(rng.poisson(spec.rate*dt))
    if n==0 or spec.scale==0: return np.zeros(d), n
    if spec.distribution=="normal": marks=rng.normal(0,spec.scale,size=(n,d))
    elif spec.distribution=="laplace": marks=rng.laplace(0,spec.scale,size=(n,d))
    else: raise ValueError("distribution must be normal or laplace")
    return marks.sum(axis=0), n


def simulate_process(times: Array, x0, *, model: str="ou", drift=0.0, theta=1.0, mu=0.0, sigma=0.2,
                     treatment_shift: Callable[[float], Array|float] | None=None,
                     jump: JumpSpec | None=None, branch: BranchSpec | None=None,
                     initial_branch: int=0, seed: int|None=None) -> SimulationResult:
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0): raise ValueError("times must be strictly increasing")
    x=np.atleast_1d(np.asarray(x0,dtype=float)); d=x.size
    rng=np.random.default_rng(seed); states=[x.copy()]; b=int(initial_branch); branches=[b]; ledger=[]
    for i,dt in enumerate(np.diff(times)):
        t0,t1=times[i],times[i+1]
        if branch is not None:
            P=expm(np.asarray(branch.generator,dtype=float)*dt)
            b_new=int(rng.choice(P.shape[0],p=np.clip(P[b],0,1)/np.clip(P[b],0,1).sum()))
            if b_new!=b: ledger.append({"event":"branch_transition","interval":i,"time_start":t0,"time_end":t1,"from":b,"to":b_new})
            b=b_new; th=branch.theta[b]; m=branch.mu[b]; sg=branch.sigma[b]
        else: th,m,sg=theta,mu,sigma
        if treatment_shift is not None: m=_vec(m,d)+_vec(treatment_shift(t1),d)
        if model=="brownian": x=brownian_transition(x,dt,drift,sg,rng)
        elif model in {"ou","shifted_ou","ou_jump","ou_branching","full"}: x=ou_transition(x,dt,th,m,sg,rng)
        else: raise ValueError(f"Unknown model: {model}")
        if jump is not None:
            j,n=draw_compound_poisson(dt,jump,d,rng); x=x+j
            if n: ledger.append({"event":"jump","interval":i,"time_start":t0,"time_end":t1,"n_jumps":n,"magnitude":float(np.linalg.norm(j))})
        states.append(x.copy()); branches.append(b)
    return SimulationResult(times, np.vstack(states), np.asarray(branches,dtype=int), ledger)
