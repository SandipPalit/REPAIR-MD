#!/usr/bin/env python
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from synapse_cut.repair_bench import greedy_repair,interaction_repair,residual_energy

def run_case(name,n=8):
    if name=='high_conf_wrong':
        # Confidence ranks 0 first, but structural violations are concentrated on 3.
        edges=[(3,i,1.0) for i in range(1,n)]+[(4,5,.2)]
        scores={0:.97,1:.94,2:.93,3:.60,4:.59,5:.58,6:.57,7:.56}
    elif name=='low_conf_harmless':
        # Token 0 is low-confidence but harmless; token 3 carries the structural load.
        edges=[(3,i,.8) for i in [1,2,4,5,6,7]]+[(0,1,.05)]
        scores={0:.55,1:.92,2:.91,3:.60,4:.90,5:.89,6:.88,7:.87}
    else:
        # Set-level overlap: neither singleton removes all violations, but the pair does.
        edges=[(0,2,.5),(1,2,.5),(0,3,.5),(1,3,.5)]
        scores={0:.50,1:.49,2:.90,3:.89,4:.88,5:.87,6:.86,7:.85}
    nodes=list(range(n)); initial=residual_energy(edges,())
    conf_top=sorted(nodes,key=lambda x:scores[x],reverse=True)[:2]
    greedy=greedy_repair(nodes,edges,2,scores); inter=interaction_repair(nodes,edges,2,16)
    return {'case':name,'initial_energy':initial,'confidence_top2':str(tuple(conf_top)),
            'confidence_residual':residual_energy(edges,conf_top),'greedy_selected':str(greedy.selected),
            'greedy_residual':greedy.residual,'interaction_selected':str(inter.selected),
            'interaction_residual':inter.residual}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results/adversarial'); a=p.parse_args(); Path(a.out).mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame([run_case(x) for x in ['high_conf_wrong','low_conf_harmless','pair_synergy']]); df.to_csv(Path(a.out)/'results.csv',index=False); print(df.to_string(index=False))
if __name__=='__main__': main()
