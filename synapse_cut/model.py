from __future__ import annotations
from dataclasses import dataclass
import warnings
import torch

@dataclass
class ModelBundle:
    model:object; tokenizer:object; mask_id:int; device:str

def model_placement(model):
    device_map=getattr(model,'hf_device_map',None) or {}
    counts={}
    for location in device_map.values():
        key=str(location)
        counts[key]=counts.get(key,0)+1
    return device_map,counts

def validate_model_placement(model,allow_disk_offload=False):
    device_map,counts=model_placement(model)
    disk=[name for name,location in device_map.items() if str(location)=='disk']
    if disk and not allow_disk_offload:
        summary=', '.join(f'{key}={value}' for key,value in sorted(counts.items()))
        raise RuntimeError(
            'Unsafe model placement: Transformers assigned '
            f'{len(disk)} module(s) to disk ({summary}). Disk-offloaded '
            'LLaDA-MoE forwards are unstable on this platform and may terminate '
            'with a native access violation. Increase RAM/VRAM or use a '
            'quantized/smaller checkpoint. To override this guard explicitly, '
            'pass --allow-disk-offload.'
        )
    return device_map,counts

def resolve_mask_id(tokenizer, model=None, explicit=None):
    if explicit is not None:return int(explicit)
    for obj in (tokenizer, getattr(model,'config',None)):
        if obj is None: continue
        for key in ('mask_token_id','mask_id'):
            v=getattr(obj,key,None)
            if v is not None:return int(v)
    # LLaDA family fallback only; callers should record this explicitly.
    name=str(getattr(model,'name_or_path','')).lower()
    if 'llada' in name:return 126336
    raise ValueError('Could not resolve mask token id. Pass --mask-id explicitly.')

def load_model(model_name,mask_id=None,device='cuda',dtype='bfloat16',trust_remote_code=True,attn_implementation=None,low_cpu_mem_usage=True,device_map=None,max_memory=None,offload_folder='offload'):
    from transformers import AutoModel,AutoTokenizer
    if device.startswith('cuda') and not torch.cuda.is_available():
        warnings.warn('CUDA requested but unavailable; falling back to CPU.', RuntimeWarning, stacklevel=2)
        device='cpu'
    torch_dtype=getattr(torch,dtype) if isinstance(dtype,str) and hasattr(torch,dtype) else dtype
    tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=trust_remote_code)
    kwargs={'trust_remote_code':trust_remote_code,'torch_dtype':torch_dtype,'low_cpu_mem_usage':low_cpu_mem_usage}
    if attn_implementation: kwargs['attn_implementation']=attn_implementation
    if device_map:
        kwargs.update({'device_map':device_map,'offload_folder':offload_folder,'offload_state_dict':True})
        if max_memory: kwargs['max_memory']=max_memory
        model=AutoModel.from_pretrained(model_name,**kwargs).eval()
    else:
        model=AutoModel.from_pretrained(model_name,**kwargs).to(device).eval()
    mid=resolve_mask_id(tok,model,mask_id)
    return ModelBundle(model,tok,mid,device)
