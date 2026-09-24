#!/usr/bin/env python
"""Controlled synthetic ablation of SYNAPSE-CUT components."""
import argparse,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synapse_cut.synapse_bench import generate_benchmark,evaluate_instance

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trials',type=int,default=1000);ap.add_argument('--out',default='results/ablations/final.csv');a=ap.parse_args();insts=generate_benchmark(['chain','star','tree','grid','random_dag','clustered_dag','long_range_dag'],a.trials,12,24,20260920,.15,.10); rows=[]
 variants=[('oracle', 'oracle',None),('singleton','singleton',8),('greedy','greedy',8),('beam','beam',8),('interaction','interaction',8),('local','local',8),('greedy_full','greedy_full',None),('beam_full','beam_full',None),('interaction_full','interaction_full',None),('local_full','local_full',None)]
 for label,method,k in variants:
  vals=[evaluate_instance(x,method,4,k,beam_width=16) for x in insts]; feas=sum(v.feasible for v in vals)/len(vals); residual=sum(v.residual_violation_weight for v in vals)/len(vals); regret=sum(v.repair_regret for v in vals if v.feasible)/max(1,sum(v.feasible for v in vals)); rows.append({'variant':label,'feasible_rate':feas,'mean_residual':residual,'mean_feasible_regret':regret,'mean_candidate_recall':sum(v.candidate_recall for v in vals)/len(vals),'mean_blocker_depth_reduction':sum(v.blocker_depth_reduction for v in vals)/len(vals),'mean_frontier_gain':sum(v.frontier_gain for v in vals)/len(vals)})
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(f'Wrote {a.out}')
if __name__=='__main__':main()
