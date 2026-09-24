#!/usr/bin/env python
import argparse, json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
from synapse_cut.synapse_bench import generate_benchmark, evaluate_instance

DEFAULT_METHODS=['none','oracle','random','singleton','greedy','singleton_full','greedy_full','beam','beam_full','interaction','interaction_full','local','local_full']

def summarize(df):
    agg={'oracle_cost':'mean','selected_cost':'mean','repair_regret':'mean','oracle_precision':'mean','oracle_recall':'mean','corruption_precision':'mean','corruption_recall':'mean','initial_violation_weight':'mean','residual_violation_weight':'mean','depth_before':'mean','depth_after':'mean','depth_reduction':'mean','relative_depth_reduction':'mean','frontier_before':'mean','frontier_after':'mean','frontier_gain':'mean','relative_frontier_gain':'mean','candidate_recall':'mean','feasible':'mean','conditional_cost_regret':'mean','blocker_depth_before':'mean','blocker_depth_after':'mean','blocker_depth_reduction':'mean'}
    return df.groupby('method').agg(agg).reset_index()

def main():
    ap=argparse.ArgumentParser(description='SYNAPSE-Bench v2: interaction-aware structural repair benchmark')
    ap.add_argument('--out',default='results/synapse_bench')
    ap.add_argument('--families',default='chain,tree,star,grid,random_dag,clustered_dag,long_range_dag')
    ap.add_argument('--trials',type=int,default=200)
    ap.add_argument('--committed',type=int,default=12); ap.add_argument('--masked',type=int,default=24)
    ap.add_argument('--seed',type=int,default=20260920); ap.add_argument('--corruption-rate',type=float,default=0.20); ap.add_argument('--edge-noise',type=float,default=0.10)
    ap.add_argument('--budget',type=int,default=4); ap.add_argument('--candidate-k',type=int,default=8); ap.add_argument('--max-oracle-cost',type=int,default=4)
    ap.add_argument('--beam-width',type=int,default=16); ap.add_argument('--methods',default=','.join(DEFAULT_METHODS))
    ap.add_argument('--no-filter',action='store_true')
    args=ap.parse_args(); families=[x.strip() for x in args.families.split(',') if x.strip()]; methods=[x.strip() for x in args.methods.split(',') if x.strip()]
    instances=generate_benchmark(families,args.trials,args.committed,args.masked,args.seed,args.corruption_rate,args.edge_noise)
    if not args.no_filter and args.max_oracle_cost>0: instances=[x for x in instances if len(x.oracle_repair)<=args.max_oracle_cost]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    with (out/'instances.jsonl').open('w') as f:
        for x in instances: f.write(json.dumps(x.__dict__)+'\n')
    rows=[]
    for inst in instances:
        for method in methods:
            ck=args.candidate_k if method in ('singleton','greedy','beam','interaction','local') else None
            r=evaluate_instance(inst,method,args.budget,ck,beam_width=args.beam_width); rows.append(r.__dict__)
    df=pd.DataFrame(rows); df.to_csv(out/'results.csv',index=False); summary=summarize(df); summary.to_csv(out/'summary.csv',index=False)
    # Explicitly report infeasibility rather than treating it as negative regret.
    df['infeasible']=1-df['feasible']
    quality=df.groupby('method').agg(feasible_rate=('feasible','mean'),infeasible_rate=('infeasible','mean'),mean_repair_regret=('repair_regret','mean'),median_repair_regret=('repair_regret','median'),mean_residual=('residual_violation_weight','mean'),mean_frontier_gain=('frontier_gain','mean'),mean_blocker_depth_reduction=('blocker_depth_reduction','mean')).reset_index()
    quality.to_csv(out/'repair_quality.csv',index=False)
    print(summary.to_string(index=False)); print('\nRepair quality (regret only on feasible instances):\n',quality.to_string(index=False))
    g=df[df.method=='greedy'].copy()
    if not g.empty:
        corr=g[['repair_regret','depth_reduction','frontier_gain','candidate_recall']].corr(numeric_only=True); corr.to_csv(out/'correlations.csv'); print('\nCorrelations (greedy):\n',corr.to_string())

if __name__=='__main__': main()
