#!/usr/bin/env python
"""Enumerate small synthetic repair sets and emit the repair/depth/frontier Pareto frontier."""
import argparse,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synapse_cut.synapse_bench import generate_instance
from synapse_cut.objective import evaluate_state,pareto_frontier
import numpy as np

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--families',default='chain,star,tree,grid,random_dag'); ap.add_argument('--trials',type=int,default=100); ap.add_argument('--n-committed',type=int,default=12); ap.add_argument('--n-masked',type=int,default=24); ap.add_argument('--budget',type=int,default=4); ap.add_argument('--out',default='results/pareto/pareto.csv'); a=ap.parse_args(); rng=np.random.default_rng(20260920); rows=[]
 for fam in a.families.split(','):
  for t in range(a.trials):
   inst=generate_instance(f'{fam}-{t}',fam,a.n_committed,a.n_masked,rng)
   # synthetic benchmark's graph is encoded in dependency/violation edges; build a compatible CF graph
   from synapse_cut.graph import CounterfactualGraph
   L=a.n_committed+a.n_masked; inf=np.zeros((L,L),np.float32); edges=list(inst.violation_edges)
   for u,v,w in edges:inf[v,u]=w
   g=CounterfactualGraph(inf,edges,list(range(a.n_committed)),list(range(a.n_committed,L)))
   for sel,m in pareto_frontier(g,list(range(a.n_committed)),a.budget):
    rows.append({'family':fam,'trial':t,'selected':','.join(map(str,sel)),'repair_cost':m.repair_cost,'energy':m.energy,'blocker_depth':m.blocker_depth,'frontier':m.frontier})
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 print(f'Wrote {len(rows)} Pareto points to {a.out}')
if __name__=='__main__':main()
