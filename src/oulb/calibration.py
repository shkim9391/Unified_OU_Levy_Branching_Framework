from __future__ import annotations
import numpy as np
import pandas as pd


def calibrate_from_interval_table(df: pd.DataFrame, displacement_col="displacement_hd", dt_col: str|None=None) -> dict:
    x=pd.to_numeric(df[displacement_col],errors="coerce").dropna().to_numpy(float)
    if x.size==0: raise ValueError("No finite displacement values")
    if dt_col and dt_col in df:
        dt=pd.to_numeric(df[dt_col],errors="coerce").dropna().to_numpy(float)
    else: dt=np.ones(len(df),float)
    return {"n_intervals":int(len(df)),"displacement_median":float(np.median(x)),"displacement_q90":float(np.quantile(x,.9)),"displacement_sd":float(np.std(x,ddof=1)),"dt_median":float(np.median(dt)),"dt_cv":float(np.std(dt,ddof=1)/np.mean(dt)) if len(dt)>1 and np.mean(dt)>0 else 0.0}
