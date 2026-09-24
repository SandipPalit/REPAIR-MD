#!/usr/bin/env python
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from synapse_cut.synapse_bench import generate_benchmark,evaluate_instance
from synapse_cut.experiment import bootstrap_ci
FAMS=['chain','star','tree','grid','random_dag','clustered_dag','long_range_dag']
def main():
 p=argparse.ArgumentParser(); p.add_argument('--out',default='results/robustness'); p.add_argument('--trials',type=int,default=500); p.add_argument('--seeds',default='1,2,3,4,5'); p.add_argument('--corruption',default='.05,.10,.20,.30,.40'); p.add_argument('--noise',default='0,.05,.10,.20,.40'); p.add_argument('--budgets',default='1,2,4,6,8'); p.add_argument('--candidate-k',default='2,4,8,16,all'); p.add_argument('--families',default=','.join(FAMS)); a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); rows=[]
 for seed in map(int,a.seeds.split(',')):
  for c in map(float,a.corruption.split(',')):
   for noise in map(float,a.noise.split(',')):
    inst=generate_benchmark(a.families.split(','),a.trials,12,24,seed,c,noise)
    for B in map(int,a.budgets.split(',')):
     for kr in a.candidate_k.split(','):
      K=None if kr=='all' else int(kr)
      for method in ['greedy','beam','interaction','local','singleton','random','oracle']:
       vals=[evaluate_instance(x,method,B,K) for x in inst if len(x.oracle_repair)<=B]
       if not vals: continue
       d=pd.DataFrame([v.__dict__ for v in vals]); feasible=d.feasible.mean(); regrets=d.repair_regret.dropna(); rows.append({'seed':seed,'corruption_rate':c,'noise':noise,'budget':B,'candidate_k':kr,'method':method,'n':len(d),'feasible_rate':feasible,'infeasible_rate':1-feasible,'mean_regret':regrets.mean() if len(regrets) else np.nan,'mean_residual':d.residual_violation_weight.mean(),'mean_blocker_depth_reduction':d.blocker_depth_reduction.mean(),'mean_frontier_gain':d.frontier_gain.mean(),'mean_candidate_recall':d.candidate_recall.mean()})
 df=pd.DataFrame(rows); df.to_csv(out/'per_condition.csv',index=False)
 agg=df.groupby(['corruption_rate','noise','budget','candidate_k','method']).agg({'feasible_rate':['mean','std'],'mean_regret':['mean','std'],'mean_residual':['mean','std'],'mean_blocker_depth_reduction':['mean','std'],'mean_frontier_gain':['mean','std']}).reset_index(); agg.columns=['_'.join(x).strip('_') for x in agg.columns]; agg.to_csv(out/'summary.csv',index=False); print(agg.head(50).to_string(index=False))
if __name__=='__main__': main()
