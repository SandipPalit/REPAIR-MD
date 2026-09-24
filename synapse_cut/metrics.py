from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Iterable,Optional
import numpy as np
from math import sqrt

@dataclass
class GenerationMetrics:
    method:str; example_id:str
    nfe:int=0; decode_forward_calls:int=0; counterfactual_calls:int=0; total_model_forward_calls:int=0
    diffusion_steps:int=0; generated_tokens:int=0; remasked_tokens:int=0; repair_events:int=0; repair_cost:int=0
    avg_tokens_per_step:float=0.; wall_time_sec:float=0.; decode_time_sec:float=0.; graph_time_sec:float=0.; counterfactual_time_sec:float=0.; solver_time_sec:float=0.
    peak_memory_mb:Optional[float]=None; tokens_per_sec:float=0.; frontier_gain:float=0.; blocker_depth_before:float=0.; blocker_depth_after:float=0.
    quality:Optional[float]=None; rouge1:Optional[float]=None; rouge2:Optional[float]=None; rougeL:Optional[float]=None; token_f1:Optional[float]=None; exact_match:Optional[float]=None
    seed:int=0
    def to_dict(self): return asdict(self)

def summarize(rows:Iterable[dict]):
    rows=list(rows)
    if not rows:return {}
    keys=['nfe','decode_forward_calls','counterfactual_calls','total_model_forward_calls','wall_time_sec','decode_time_sec','graph_time_sec','counterfactual_time_sec','solver_time_sec','remasked_tokens','repair_cost','frontier_gain','blocker_depth_before','blocker_depth_after','tokens_per_sec','rouge1','rouge2','rougeL','token_f1','exact_match','quality']
    out={'num_examples':len(rows)}
    for k in keys:
        vals=[float(r[k]) for r in rows if r.get(k) is not None]
        if vals: out[k+'_mean']=float(np.mean(vals)); out[k+'_median']=float(np.median(vals)); out[k+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.
    return out

def bootstrap_ci(values, statistic=np.mean, n_boot=2000, seed=1234, alpha=.05):
    x=np.asarray([v for v in values if v is not None],dtype=float)
    if len(x)==0:return (float('nan'),float('nan'),float('nan'))
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),size=(n_boot,len(x))); stats=np.asarray([statistic(x[i]) for i in idx]); lo,hi=np.quantile(stats,[alpha/2,1-alpha/2]); return float(statistic(x)),float(lo),float(hi)

def paired_bootstrap(a,b,n_boot=2000,seed=1234):
    a=np.asarray(a,float); b=np.asarray(b,float); d=a-b; return bootstrap_ci(d,n_boot=n_boot,seed=seed)
