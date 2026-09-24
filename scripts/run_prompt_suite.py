#!/usr/bin/env python
"""Run a JSONL prompt suite with identical prompts across local decoders.

Input fields: id,prompt and optional reference. If reference is supplied,
exact normalized-match is reported; otherwise quality is intentionally left
unscored so no unsupported automatic quality claim is made.
"""
import argparse,json,sys,re,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from synapse_cut.model import load_model
from synapse_cut.llada_adapter import LLaDAAdapter
from synapse_cut.baselines import build_decoder
from synapse_cut.experiment import seed_everything,snapshot,save_json

def norm(s): return re.sub(r'\s+',' ',s.strip().lower())
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--model',default='GSAI-ML/iLLaDA-8B-Base'); p.add_argument('--methods',default='confidence,entropy,attention,synapse_cut'); p.add_argument('--mask-id',type=int,default=5); p.add_argument('--device',default='cuda'); p.add_argument('--dtype',default='bfloat16'); p.add_argument('--steps',type=int,default=64); p.add_argument('--max-new-tokens',type=int,default=256); p.add_argument('--cf-batch-size',type=int,default=1); p.add_argument('--seed',type=int,default=1234); p.add_argument('--out',default='results/prompt_suite'); a=p.parse_args(); seed_everything(a.seed)
 rows=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]; bundle=load_model(a.model,a.mask_id,a.device,a.dtype); adapter=LLaDAAdapter(bundle); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); save_json(out/'environment.json',snapshot()); allrows=[]
 for ex in rows:
  for method in [m.strip() for m in a.methods.split(',')]:
   inp=adapter.prepare(ex['prompt'],a.max_new_tokens); dec=build_decoder(method,bundle.model,bundle.tokenizer,bundle.mask_id,bundle.device); r=dec.generate(inp.ids,steps=a.steps,counterfactual_batch_size=a.cf_batch_size,seed=a.seed); text=adapter.decode(r.token_ids,inp.prompt_length); row={'id':ex.get('id',str(len(allrows))),'method':method,'output':text,'nfe':r.nfe,'wall_time_sec':r.wall_time,'remasked_tokens':r.remasked,'counterfactual_calls':r.counterfactual_calls,'repair_cost':r.repair_cost,'frontier_gain_sum':r.frontier_gain};
   if 'reference' in ex: row['exact_match']=float(norm(text)==norm(str(ex['reference'])))
   allrows.append(row)
 pd.DataFrame(allrows).to_csv(out/'results.csv',index=False); print(pd.DataFrame(allrows).groupby('method').agg({'nfe':'mean','wall_time_sec':'mean','remasked_tokens':'mean','exact_match':'mean'} if any('exact_match' in x for x in allrows) else {'nfe':'mean','wall_time_sec':'mean','remasked_tokens':'mean'}).to_string())
if __name__=='__main__': main()
