from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import json, os, platform, subprocess

@dataclass
class DecodeConfig:
    steps:int=32
    max_new_tokens:int=128
    selection_fraction:float=0.25
    candidate_k:int=16
    candidate_low_k:int=16
    top_k_edges:int=4
    repair_budget:int=4
    repair_solver:str='interaction'
    beam_width:int=16
    interaction_pair_width:int=32
    structural_threshold:float=0.10
    counterfactual_batch_size:int=2
    repair_every:int=4
    temperature:float=0.0
    lambda_repair:float=1.0
    lambda_depth:float=0.25
    lambda_frontier:float=0.50
    lambda_energy:float=1.0
    schedule_mode:str='wavefront'
    seed:int=1234
    deterministic:bool=False
    use_amp:bool=True
    exact_repair_max_nodes:int=18

    def to_dict(self): return asdict(self)

@dataclass
class BenchmarkConfig:
    model:str='GSAI-ML/LLaDA-8B-Base'
    device:str='cuda'
    dtype:str='bfloat16'
    datasets:str='wikitext2,ag_news,cnn_dailymail,xsum,multi_news'
    samples_per_dataset:int=10
    seeds:str='1234,2026,2027'
    max_new_tokens:int=128
    steps:int=32
    methods:str='confidence,entropy,attention,synapse_cut'
    output_dir:str='results/real_world'


def environment_snapshot():
    out={'python':platform.python_version(),'platform':platform.platform(),'cwd':os.getcwd()}
    try:
        import torch
        out.update({'torch':torch.__version__,'cuda_available':bool(torch.cuda.is_available()),'cuda_version':torch.version.cuda})
        if torch.cuda.is_available():
            out['gpu_count']=torch.cuda.device_count()
            out['gpus']=[{'name':torch.cuda.get_device_name(i),'capability':torch.cuda.get_device_capability(i),'total_memory_mb':round(torch.cuda.get_device_properties(i).total_memory/2**20,2)} for i in range(torch.cuda.device_count())]
    except Exception as e: out['torch_error']=repr(e)
    for pkg in ('transformers','datasets','accelerate','numpy','pandas','scipy'):
        try:
            m=__import__(pkg); out[pkg]=getattr(m,'__version__','unknown')
        except Exception: out[pkg]=None
    try:
        out['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: out['git_commit']=None
    return out

# Backward-compatible public configuration name used by earlier scripts.
@dataclass
class SynapseConfig:
    seed:int=1234
    model_name:str='GSAI-ML/LLaDA-8B-Base'
    mask_id:Optional[int]=None
    dtype:str='bfloat16'
    steps:int=64
    max_new_tokens:int=256
    temperature:float=0.0
    candidate_k:int=16
    top_k_edges:int=4
    repair_budget:int=4
    structural_threshold:float=.10
    counterfactual_batch_size:int=1
