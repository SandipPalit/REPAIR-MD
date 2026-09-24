#!/usr/bin/env python
"""Adversarial structural cases: high-confidence harmful commitments, weak-but-joint interactions, long chains."""
import argparse,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synapse_cut.repair_bench import greedy_repair,beam_repair,interaction_repair,exact_repair

def cases():
 return [
  ('high_conf_harmful',[(0,1,1.0),(0,2,1.0),(1,3,.05),(2,4,.05)],{0:.99,1:.98,2:.97,3:.96}),
  ('joint_synergy',[(0,2,.6),(1,2,.6),(0,3,.6),(1,3,.6)],{0:.9,1:.9,2:.2,3:.2}),
  ('long_chain',[(i,i+1,1.0) for i in range(15)],{i:.99 for i in range(16)}),
 ]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/adversarial/final.csv');a=ap.parse_args(); rows=[]
 for name,edges,scores in cases():
  nodes=sorted(scores); oracle=exact_repair(nodes,edges,4); 
  for method,fn in [('oracle',None),('greedy',greedy_repair),('beam',beam_repair),('interaction',interaction_repair)]:
   sol=oracle if method=='oracle' else (fn(nodes,edges,4,16,scores) if method=='beam' else (fn(nodes,edges,4,16,32) if method=='interaction' else fn(nodes,edges,4,scores)))
   rows.append({'case':name,'method':method,'selected':','.join(map(str,sol.selected)),'cost':len(sol.selected),'residual':sol.residual,'oracle_cost':len(oracle.selected),'regret':len(sol.selected)-len(oracle.selected) if sol.residual<=1e-8 else None})
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with open(a.out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(f'Wrote {a.out}')
if __name__=='__main__':main()
