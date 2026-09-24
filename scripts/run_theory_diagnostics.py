#!/usr/bin/env python
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from synapse_cut.synapse_bench import generate_instance
from synapse_cut.theory import diagnose

def main():
 p=argparse.ArgumentParser(); p.add_argument('--trials',type=int,default=100); p.add_argument('--seed',type=int,default=1234); p.add_argument('--out',default='results/theory'); a=p.parse_args(); rng=np.random.default_rng(a.seed); rows=[]
 for fam in ['chain','star','tree','grid','random_dag','clustered_dag','long_range_dag']:
  for t in range(a.trials):
   x=generate_instance(f'{fam}-{t}',fam,8,16,rng,.2,.1); d=diagnose(range(x.n_committed),x.violation_edges); rows.append({'family':fam,**d.__dict__})
 df=pd.DataFrame(rows); Path(a.out).mkdir(parents=True,exist_ok=True); df.to_csv(Path(a.out)/'diagnostics.csv',index=False); print(df.groupby('family')[['monotone','submodular','supermodular']].mean().to_string())
if __name__=='__main__': main()
