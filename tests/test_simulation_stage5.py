import numpy as np
from oulb.simulation import JumpSpec, BranchSpec, brownian_transition, ou_transition, simulate_process

def test_ou_zero_theta_matches_brownian_variance():
 r1=np.random.default_rng(1); r2=np.random.default_rng(1)
 a=ou_transition(np.array([0.]),1.,0.,0.,.2,r1); b=brownian_transition(np.array([0.]),1.,0.,.2,r2)
 assert np.allclose(a,b)

def test_deterministic_ou_mean():
 r=simulate_process(np.array([0.,1.]),[1.],model='ou',theta=1.,mu=0.,sigma=0.,seed=1)
 assert np.allclose(r.states[-1,0],np.exp(-1))

def test_treatment_shift_changes_attractor():
 f=lambda t: 1. if t>=1 else 0.
 r=simulate_process(np.array([0.,1.,2.]),[0.],model='shifted_ou',theta=2.,mu=0.,sigma=0.,treatment_shift=f,seed=1)
 assert r.states[-1,0]>.8

def test_jump_ledger():
 r=simulate_process(np.linspace(0,5,20),[0.],model='ou_jump',theta=1.,mu=0.,sigma=0.,jump=JumpSpec(10.,1.),seed=2)
 assert any(e['event']=='jump' for e in r.event_ledger)

def test_branch_states_valid():
 b=BranchSpec(np.array([[-1.,1.],[1.,-1.]]),np.array([1.,1.]),np.array([[0.],[1.]]),np.array([0.,0.]))
 r=simulate_process(np.linspace(0,2,10),[0.],model='ou_branching',branch=b,seed=3)
 assert set(r.branches)<= {0,1}
