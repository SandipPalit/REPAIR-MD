#!/usr/bin/env python
"""Budget x candidate-K solver ablation for the structural repair gate."""
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
from synapse_cut.synapse_bench import generate_benchmark, evaluate_instance

SOLVERS=['greedy','beam','interaction','local']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='results/repair_ablation'); ap.add_argument('--trials',type=int,default=500); ap.add_argument('--seed',type=int,default=20260920); ap.add_argument('--budgets',default='1,2,4,6,8'); ap.add_argument('--candidate-k',default='2,4,8,16,all'); ap.add_argument('--beam-width',type=int,default=16); args=ap.parse_args()
    inst=generate_benchmark(['chain','tree','star','grid','random_dag','clustered_dag','long_range_dag'],args.trials,12,24,args.seed,0.20,0.10)
    inst=[x for x in inst if len(x.oracle_repair)<=8]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for B in map(int,args.budgets.split(',')):
        for kraw in args.candidate_k.split(','):
            k=None if kraw=='all' else int(kraw)
            for solver in SOLVERS:
                for x in inst:
                    r=evaluate_instance(x,solver,B,k,beam_width=args.beam_width); d=r.__dict__.copy(); d.update(budget=B,candidate_k='all' if k is None else k); rows.append(d)
    df=pd.DataFrame(rows); df.to_csv(out/'results.csv',index=False)
    summary=df.groupby(['method','budget','candidate_k']).agg(feasible_rate=('feasible','mean'),mean_regret=('repair_regret','mean'),mean_residual=('residual_violation_weight','mean'),mean_bdr=('blocker_depth_reduction','mean'),mean_frontier_gain=('frontier_gain','mean'),mean_candidate_recall=('candidate_recall','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False); print(summary.to_string(index=False))

if __name__=='__main__': main()
