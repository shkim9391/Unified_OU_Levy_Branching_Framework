from __future__ import annotations
import argparse,json,tomllib
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED=["n_observations","noise_sd","missing_probability","interval_pattern","replicate","observed_n","theta_true","sigma_true","theta_naive","sigma_naive","theta_abs_error_naive","sigma_abs_error_naive","theta_bound_naive","sigma_bound_naive","fit_success_naive","theta_measurement_aware","sigma_measurement_aware","theta_abs_error_measurement_aware","sigma_abs_error_measurement_aware","theta_bound_measurement_aware","sigma_bound_measurement_aware","fit_success_measurement_aware"]

def args():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=Path("configs/figure5_observation_robustness.toml")); p.add_argument("--overwrite",action="store_true"); return p.parse_args()
def read_config(path):
 with path.expanduser().open("rb") as h:return tomllib.load(h)
def boot_ci(values,n,seed):
 x=np.asarray(values,float); x=x[np.isfinite(x)]; rng=np.random.default_rng(seed); b=np.array([rng.choice(x,size=len(x),replace=True).mean() for _ in range(n)]); return np.percentile(b,[2.5,97.5])

def main():
 a=args(); c=read_config(a.config); root=Path(c["paths"]["project_root"]).expanduser(); src=root/c["paths"]["stress_results"]; out=root/c["paths"]["output_data_dir"]; out.mkdir(parents=True,exist_ok=True)
 paths=[out/"Figure5_observation_robustness_replicates.csv",out/"Figure5_observation_robustness_summary.csv",out/"Figure5_observation_robustness_overall.csv",out/"Figure5_observation_robustness_metadata.json"]
 if not a.overwrite and any(p.exists() for p in paths): raise FileExistsError("Figure 5 outputs exist; use --overwrite")
 raw=pd.read_csv(src); missing=[x for x in REQUIRED if x not in raw.columns]
 if missing: raise ValueError(f"Missing columns: {missing}")
 for col in ["n_observations","noise_sd","missing_probability","replicate","observed_n","theta_true","sigma_true","theta_naive","sigma_naive","theta_measurement_aware","sigma_measurement_aware"]: raw[col]=pd.to_numeric(raw[col],errors="coerce")
 raw=raw.dropna(subset=["n_observations","noise_sd","missing_probability","interval_pattern","theta_true","sigma_true","theta_naive","sigma_naive","theta_measurement_aware","sigma_measurement_aware"]).copy()
 for col in ["theta_bound_naive","sigma_bound_naive","theta_bound_measurement_aware","sigma_bound_measurement_aware","fit_success_naive","fit_success_measurement_aware"]: raw[col]=raw[col].astype(bool)
 raw["theta_squared_error_naive"]=(raw.theta_naive-raw.theta_true)**2; raw["theta_squared_error_measurement_aware"]=(raw.theta_measurement_aware-raw.theta_true)**2
 raw["sigma_squared_error_naive"]=(raw.sigma_naive-raw.sigma_true)**2; raw["sigma_squared_error_measurement_aware"]=(raw.sigma_measurement_aware-raw.sigma_true)**2
 raw.to_csv(paths[0],index=False)
 groups=["n_observations","noise_sd","missing_probability","interval_pattern"]; rows=[]
 for key,g in raw.groupby(groups,dropna=False):
  tn=float(np.sqrt(g.theta_squared_error_naive.mean())); ta=float(np.sqrt(g.theta_squared_error_measurement_aware.mean())); sn=float(np.sqrt(g.sigma_squared_error_naive.mean())); sa=float(np.sqrt(g.sigma_squared_error_measurement_aware.mean()))
  rows.append(dict(n_observations=int(key[0]),noise_sd=float(key[1]),missing_probability=float(key[2]),interval_pattern=str(key[3]),n_fits=len(g),observed_n_mean=float(g.observed_n.mean()),theta_rmse_naive=tn,theta_rmse_measurement_aware=ta,theta_rmse_improvement=tn-ta,sigma_rmse_naive=sn,sigma_rmse_measurement_aware=sa,sigma_rmse_improvement=sn-sa,theta_boundary_fraction_naive=float(g.theta_bound_naive.mean()),theta_boundary_fraction_measurement_aware=float(g.theta_bound_measurement_aware.mean()),theta_boundary_reduction=float(g.theta_bound_naive.mean()-g.theta_bound_measurement_aware.mean()),sigma_boundary_fraction_naive=float(g.sigma_bound_naive.mean()),sigma_boundary_fraction_measurement_aware=float(g.sigma_bound_measurement_aware.mean()),sigma_boundary_reduction=float(g.sigma_bound_naive.mean()-g.sigma_bound_measurement_aware.mean()),fit_success_fraction_naive=float(g.fit_success_naive.mean()),fit_success_fraction_measurement_aware=float(g.fit_success_measurement_aware.mean())))
 summary=pd.DataFrame(rows).sort_values(groups); summary.to_csv(paths[1],index=False)
 nboot=int(c["display"]["bootstrap_replicates"]); seed=int(c["display"]["bootstrap_seed"]); specs=[("theta","rmse","naive",np.sqrt(raw.theta_squared_error_naive)),("theta","rmse","measurement_aware",np.sqrt(raw.theta_squared_error_measurement_aware)),("sigma","rmse","naive",np.sqrt(raw.sigma_squared_error_naive)),("sigma","rmse","measurement_aware",np.sqrt(raw.sigma_squared_error_measurement_aware)),("theta","boundary_fraction","naive",raw.theta_bound_naive.astype(float)),("theta","boundary_fraction","measurement_aware",raw.theta_bound_measurement_aware.astype(float))]
 overall=[]
 for i,(parameter,metric,estimator,v) in enumerate(specs):
  lo,hi=boot_ci(v,nboot,seed+i); overall.append(dict(parameter=parameter,metric=metric,estimator=estimator,n=int(np.isfinite(v).sum()),mean=float(np.nanmean(v)),median=float(np.nanmedian(v)),ci_lower=float(lo),ci_upper=float(hi)))
 pd.DataFrame(overall).to_csv(paths[2],index=False)
 meta={"figure":"Figure 5","title":c["labels"]["title"],"subtitle":c["labels"]["subtitle"],"source_file":str(src),"source_rows":len(raw),"archived_rows":{"replicates":len(raw),"summary":len(summary),"overall":len(overall)},"tested_conditions":{"n_observations":sorted(summary.n_observations.unique().astype(int).tolist()),"noise_sd":sorted(summary.noise_sd.unique().tolist()),"missing_probability":sorted(summary.missing_probability.unique().tolist()),"interval_pattern":sorted(summary.interval_pattern.unique().tolist())},"improvement_sign_convention":"Positive values favor measurement-aware estimation."}
 paths[3].write_text(json.dumps(meta,indent=2)+"\n")
 for p in paths: print(f"[SAVED] {p}")
if __name__=="__main__":main()
