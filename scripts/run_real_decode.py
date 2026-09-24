#!/usr/bin/env python
import argparse,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,pandas as pd
from synapse_cut.model import load_model
from synapse_cut.llada_adapter import LLaDAAdapter
from synapse_cut.baselines import build_decoder
from synapse_cut.experiment import seed_everything,snapshot,save_json

def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',default='GSAI-ML/iLLaDA-8B-Base'); p.add_argument('--prompt',required=True); p.add_argument('--methods',default='confidence,entropy,attention,synapse_cut'); p.add_argument('--mask-id',type=int,default=5); p.add_argument('--device',default='cuda'); p.add_argument('--dtype',default='bfloat16'); p.add_argument('--steps',type=int,default=64); p.add_argument('--max-new-tokens',type=int,default=128); p.add_argument('--candidate-k',type=int,default=16); p.add_argument('--repair-budget',type=int,default=4); p.add_argument('--repair-every',type=int,default=4); p.add_argument('--cf-batch-size',type=int,default=1); p.add_argument('--top-k-edges',type=int,default=4); p.add_argument('--selection-fraction',type=float,default=.25); p.add_argument('--threshold',type=float,default=.10); p.add_argument('--solver',default='interaction'); p.add_argument('--seed',type=int,default=1234); p.add_argument('--out',default='results/real_decode'); a=p.parse_args(); seed_everything(a.seed)
 bundle=load_model(a.model,a.mask_id,a.device,a.dtype); adapter=LLaDAAdapter(bundle); data=[]; outdir=Path(a.out); outdir.mkdir(parents=True,exist_ok=True); save_json(outdir/'environment.json',snapshot())
 for method in [x.strip() for x in a.methods.split(',')]:
    inp=adapter.prepare(a.prompt,a.max_new_tokens); dec=build_decoder(method,bundle.model,bundle.tokenizer,bundle.mask_id,bundle.device); t=time.perf_counter(); result=dec.generate(inp.ids,steps=a.steps,max_new_tokens=a.max_new_tokens,candidate_k=a.candidate_k,repair_budget=a.repair_budget,repair_every=a.repair_every,counterfactual_batch_size=a.cf_batch_size,top_k_edges=a.top_k_edges,selection_fraction=a.selection_fraction,structural_threshold=a.threshold,repair_solver=a.solver,seed=a.seed); elapsed=time.perf_counter()-t
    row={'method':method,'model':a.model,'prompt':a.prompt,'output':adapter.decode(result.token_ids,inp.prompt_length),'nfe':result.nfe,'wall_time_sec':elapsed,'remasked_tokens':result.remasked,'counterfactual_calls':result.counterfactual_calls,'repair_events':result.repair_events,'repair_cost':result.repair_cost,'frontier_gain_sum':result.frontier_gain_gain if hasattr(result,'frontier_gain_gain') else result.frontier_gain,'graph_time_sec':result.graph_time,'counterfactual_time_sec':result.counterfactual_time,'solver_time_sec':result.solver_time,'seed':a.seed,'tokens_generated':a.max_new_tokens}
    data.append(row); (outdir/f'{method}_trace.json').write_text(json.dumps(result.trace,indent=2))
 pd.DataFrame(data).to_csv(outdir/'results.csv',index=False); print(pd.DataFrame(data)[['method','nfe','wall_time_sec','remasked_tokens','counterfactual_calls','repair_cost','tokens_generated']].to_string(index=False))
if __name__=='__main__': main()
