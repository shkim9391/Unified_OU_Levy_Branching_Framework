from __future__ import annotations
import numpy as np
import pandas as pd

def rmse(values, truth) -> float:
    values=np.asarray(values,dtype=float); truth=np.asarray(truth,dtype=float)
    return float(np.sqrt(np.mean((values-truth)**2)))

def summarize_condition(group: pd.DataFrame) -> dict[str,float]:
    return {
        "theta_rmse_naive":rmse(group["theta_naive"],group["theta_true"]),
        "theta_rmse_measurement_aware":rmse(group["theta_measurement_aware"],group["theta_true"]),
        "sigma_rmse_naive":rmse(group["sigma_naive"],group["sigma_true"]),
        "sigma_rmse_measurement_aware":rmse(group["sigma_measurement_aware"],group["sigma_true"]),
        "theta_boundary_fraction_naive":float(group["theta_bound_naive"].astype(bool).mean()),
        "theta_boundary_fraction_measurement_aware":float(group["theta_bound_measurement_aware"].astype(bool).mean()),
    }

def improvement(naive: float, measurement_aware: float) -> float:
    return float(naive-measurement_aware)
