from __future__ import annotations
from dataclasses import dataclass
import torch
from .model import ModelBundle

@dataclass
class MaskedPrompt:
    ids:torch.Tensor; prompt_length:int; generation_positions:list[int]

class LLaDAAdapter:
    def __init__(self,bundle:ModelBundle): self.bundle=bundle
    @property
    def model(self): return self.bundle.model
    @property
    def tokenizer(self): return self.bundle.tokenizer
    @property
    def mask_id(self): return self.bundle.mask_id
    def prepare(self,prompt,max_new_tokens,device=None,max_input_tokens=None):
        device=device or self.bundle.device
        enc=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=True,truncation=max_input_tokens is not None,max_length=max_input_tokens)
        base=enc.input_ids.to(device)
        masks=torch.full((base.shape[0],max_new_tokens),self.mask_id,dtype=base.dtype,device=device)
        ids=torch.cat([base,masks],1)
        start=int(base.shape[1]); return MaskedPrompt(ids,start,list(range(start,start+max_new_tokens)))
    def decode(self,ids,prompt_length): return self.tokenizer.decode(ids[0,prompt_length:].tolist(),skip_special_tokens=True)
