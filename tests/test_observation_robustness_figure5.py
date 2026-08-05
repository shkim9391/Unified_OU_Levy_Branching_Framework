from __future__ import annotations
import numpy as np
import pandas as pd
from oulb.observation_robustness import improvement, rmse, summarize_condition

def test_rmse():
    assert np.isclose(rmse(np.array([1.,2.,3.]),np.array([1.,1.,1.])),np.sqrt(5./3.))

def test_improvement_sign():
    assert improvement(2.,1.)==1.
    assert improvement(1.,2.)==-1.

def test_condition_summary():
    frame=pd.DataFrame({
      "theta_naive":[1.,2.],"theta_measurement_aware":[1.,1.],"theta_true":[1.,1.],
      "sigma_naive":[.4,.6],"sigma_measurement_aware":[.5,.5],"sigma_true":[.5,.5],
      "theta_bound_naive":[True,False],"theta_bound_measurement_aware":[False,False]})
    result=summarize_condition(frame)
    assert result["theta_rmse_naive"]>result["theta_rmse_measurement_aware"]
    assert result["sigma_rmse_measurement_aware"]==0.
    assert result["theta_boundary_fraction_naive"]==.5
