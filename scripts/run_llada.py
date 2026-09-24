#!/usr/bin/env python
"""LLaDA/iLLaDA forward smoke test. This is not a paper-quality sampler."""
import argparse,time,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
import torch
from synapse_cut.model import load_model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='GSAI-ML/iLLaDA-8B-Base'); ap.add_argument('--prompt',required=True); ap.add_argument('--mask-id',type=int,default=None); ap.add_argument('--device',default='cuda'); ap.add_argument('--dtype',default='bfloat16'); a=ap.parse_args()
    bundle=load_model(a.model,a.mask_id,a.device,a.dtype); ids=bundle.tokenizer(a.prompt,return_tensors='pt').input_ids.to(a.device)
    with torch.no_grad():
        t=time.perf_counter(); out=bundle.model(ids); elapsed=time.perf_counter()-t
    logits=out.logits if hasattr(out,'logits') else out[0]
    print({'model':a.model,'mask_id':bundle.mask_id,'input_tokens':int(ids.shape[1]),'forward_seconds':elapsed,'logits_shape':list(logits.shape)})
if __name__=='__main__': main()
