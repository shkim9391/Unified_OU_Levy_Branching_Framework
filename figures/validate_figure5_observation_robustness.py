from __future__ import annotations
import argparse,ast,json,tomllib
from pathlib import Path
import numpy as np
import pandas as pd
FORBIDDEN={"simulate_process","fit_ou_mle","fit_ou_mle_measurement_aware","minimize","default_rng","random"}
def args():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=Path("configs/figure5_observation_robustness.toml")); return p.parse_args()
def read_config(path):
 with path.expanduser().open("rb") as h:return tomllib.load(h)
def calls(path):
 tree=ast.parse(path.read_text()); out=set()
 for node in ast.walk(tree):
  if isinstance(node,ast.Call):
   if isinstance(node.func,ast.Name): out.add(node.func.id)
   elif isinstance(node.func,ast.Attribute): out.add(node.func.attr)
 return out
def main():
 a=args(); c=read_config(a.config); root=Path(c["paths"]["project_root"]).expanduser(); data=root/c["paths"]["output_data_dir"]; figdir=root/c["paths"]["output_figure_dir"]
 rp=data/"Figure5_observation_robustness_replicates.csv"; sp=data/"Figure5_observation_robustness_summary.csv"; op=data/"Figure5_observation_robustness_overall.csv"; mp=data/"Figure5_observation_robustness_metadata.json"; plot=root/"figures/plot_figure5_observation_robustness.py"; outs=[figdir/"Figure5_observation_model_robustness.svg",figdir/"Figure5_observation_model_robustness.pdf",figdir/"Figure5_observation_model_robustness.png"]
 for p in [rp,sp,op,mp,plot,*outs]:
  if not p.exists(): raise FileNotFoundError(p)
 r=pd.read_csv(rp); s=pd.read_csv(sp); o=pd.read_csv(op); m=json.loads(mp.read_text()); print("Figure 5 observation-robustness validation"); print("="*72)
 assert len(r)>0; print(f"[PASS] Replicate rows archived: {len(r)}")
 expected=r[["n_observations","noise_sd","missing_probability","interval_pattern"]].drop_duplicates().shape[0]; assert len(s)==expected; print(f"[PASS] Condition summary rows: {len(s)}")
 for col in ["theta_rmse_naive","theta_rmse_measurement_aware","sigma_rmse_naive","sigma_rmse_measurement_aware","theta_boundary_fraction_naive","theta_boundary_fraction_measurement_aware"]: assert np.isfinite(s[col]).all()
 print("[PASS] All primary metrics are finite")
 for col in ["theta_boundary_fraction_naive","theta_boundary_fraction_measurement_aware","sigma_boundary_fraction_naive","sigma_boundary_fraction_measurement_aware"]: assert s[col].between(0,1).all()
 print("[PASS] Boundary fractions lie in [0,1]")
 assert not (calls(plot)&FORBIDDEN); print("[PASS] Plotting layer contains no simulation, fitting, optimization, or RNG calls")
 assert m["archived_rows"]["replicates"]==len(r); print("[PASS] Metadata row counts agree")
 for p in outs: assert p.stat().st_size>0
 print("[PASS] SVG, PDF, and PNG outputs exist"); print("\nFigure 5 observation-model robustness validated successfully.")
if __name__=="__main__":main()
