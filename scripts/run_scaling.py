#!/usr/bin/env python
import argparse,csv,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synapse_cut.synapse_bench import generate_benchmark,evaluate_instance

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',default='results/scaling/final_scaling.csv'); ap.add_argument('--trials',type=int,default=200); ap.add_argument('--families',default='chain,star,tree,grid,random_dag,clustered_dag,long_range_dag'); ap.add_argument('--budgets',default='2,4,6'); ap.add_argument('--n-committed',default='8,12,16,20,32,48,64'); ap.add_argument('--n-masked',default='16,32,64,128,256'); ap.add_argument('--candidate-k',default='4,8,16,32,64'); a=ap.parse_args(); rows=[]
 for nc in map(int,a.n_committed.split(',')):
  nm=min(max(map(int,a.n_masked.split(','))),max(16,nc*2))
  for budget in map(int,a.budgets.split(',')):
   insts=generate_benchmark(a.families.split(','),a.trials,nc,nm,20260920,.15,.10)
   for k in map(int,a.candidate_k.split(',')):
    for method in ('beam','greedy','interaction','local','singleton'):
     vals=[]; t=time.perf_counter()
     for inst in insts: vals.append(evaluate_instance(inst,method,budget,k,beam_width=16))
     feasible=sum(v.feasible for v in vals)/len(vals); regret=sum(v.repair_regret for v in vals if v.feasible)/max(1,sum(v.feasible for v in vals)); recall=sum(v.candidate_recall for v in vals)/len(vals)
     rows.append({'n_committed':nc,'n_masked':nm,'budget':budget,'candidate_k':k,'method':method,'feasible_rate':feasible,'mean_feasible_regret':regret,'candidate_recall':recall,'runtime_sec':time.perf_counter()-t})
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(f'Wrote {len(rows)} scaling rows to {a.out}')
if __name__=='__main__':main()
