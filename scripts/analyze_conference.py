#!/usr/bin/env python
"""Conference-grade analysis: paired bootstrap CIs and compute/quality summaries."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synapse_cut.metrics import bootstrap_ci,paired_bootstrap

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='results/real_world/results.csv');ap.add_argument('--out',default='results/analysis');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);df=pd.read_csv(a.input)
 metrics=[m for m in ['rougeL','rouge1','rouge2','token_f1','wall_time_sec','total_model_forward_calls','nfe','peak_memory_mb','tokens_per_sec','remasked_tokens','counterfactual_time_sec'] if m in df.columns]
 summary=[]
 for keys,g in df.groupby(['dataset','method']):
  rec={'dataset':keys[0],'method':keys[1],'n':len(g)}
  for m in metrics:
   mean,lo,hi=bootstrap_ci(g[m].dropna().values); rec[f'{m}_mean']=mean;rec[f'{m}_ci_lo']=lo;rec[f'{m}_ci_hi']=hi
  summary.append(rec)
 pd.DataFrame(summary).to_csv(out/'summary_ci.csv',index=False)
 # paired comparisons on common example/seed rows
 methods=list(df.method.unique())
 if methods:
  base='confidence' if 'confidence' in methods else methods[0]; tests=[]
  for m in methods:
   if m==base:continue
   cols=['dataset','example_id','seed',m,base]
   piv=df[df.method.isin([base,m])].pivot_table(index=['dataset','example_id','seed'],columns='method',values='rougeL',aggfunc='mean').dropna()
   if not piv.empty:
    mean,lo,hi=paired_bootstrap(piv[m].values,piv[base].values); tests.append({'metric':'rougeL','comparison':f'{m}-vs-{base}','n':len(piv),'delta_mean':mean,'ci_lo':lo,'ci_hi':hi})
  pd.DataFrame(tests).to_csv(out/'paired_tests.csv',index=False)
 print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
