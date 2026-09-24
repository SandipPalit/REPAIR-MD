"""Reference LLaDA sampler primitives, kept separate from SYNAPSE-CUT.

The implementation mirrors the public LLaDA transfer-count principle: distribute
initial masked positions approximately uniformly over diffusion steps. It is used
only as a reference condition; SYNAPSE-CUT remains responsible for structural repair.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

def get_num_transfer_tokens(mask_index,steps):
    mask_num=mask_index.sum(dim=1,keepdim=True); base=mask_num//steps; rem=mask_num%steps
    out=torch.zeros(mask_num.size(0),steps,device=mask_index.device,dtype=torch.int64)+base
    for i in range(mask_num.size(0)):out[i,:int(rem[i])]+=1
    return out

@torch.no_grad()
def generate(model,prompt,steps=128,gen_length=128,temperature=0.,remasking='low_confidence',mask_id=126336):
    x=torch.full((prompt.shape[0],prompt.shape[1]+gen_length),mask_id,dtype=torch.long,device=model.device); x[:,:prompt.shape[1]]=prompt
    prompt_index=(x!=mask_id); mask_index=x.eq(mask_id); transfer=get_num_transfer_tokens(mask_index[:,prompt.shape[1]:],steps)
    for i in range(steps):
        mask_index=x.eq(mask_id); logits=model(x).logits
        if temperature>0:
            noise=torch.rand_like(logits,dtype=torch.float64); logits=logits.float()-temperature*torch.log(-torch.log(noise.clamp_min(1e-12)))
        x0=logits.argmax(-1)
        if remasking=='low_confidence':
            p=F.softmax(logits.float(),dim=-1); conf=p.gather(-1,x0.unsqueeze(-1)).squeeze(-1)
        elif remasking=='random': conf=torch.rand_like(x,dtype=torch.float)
        else: raise ValueError(remasking)
        x0=torch.where(mask_index,x0,x); conf=torch.where(mask_index,conf,torch.tensor(float('-inf'),device=x.device))
        k=int(transfer[:,i].max().item()); k=min(k,int(mask_index[:,prompt.shape[1]:].sum().item()))
        if k>0:
            idx=torch.topk(conf,k=k,dim=-1).indices; rows=torch.arange(x.shape[0],device=x.device)[:,None]; x[rows,idx]=x0[rows,idx]
    return x
