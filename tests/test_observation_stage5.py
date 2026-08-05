import numpy as np
from oulb.observation import ObservationSpec, regular_schedule, irregular_schedule, thin_schedule, observe_latent

def test_schedules():
 rng=np.random.default_rng(1); a=regular_schedule(0,1,5); b=irregular_schedule(0,1,5,rng)
 assert np.all(np.diff(a)>0) and np.all(np.diff(b)>0) and b[0]==0 and b[-1]==1

def test_thinning_keeps_endpoints():
 x=np.arange(10.); y=thin_schedule(x,4,np.random.default_rng(2)); assert y[0]==0 and y[-1]==9

def test_observation_endpoints():
 t=np.arange(5.); x=t[:,None]; ot,ox=observe_latent(t,x,ObservationSpec(0,1,True),np.random.default_rng(3)); assert list(ot)==[0,4]
