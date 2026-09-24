from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, Optional
import numpy as np
import torch

@dataclass
class CounterfactualGraph:
    influence: np.ndarray
    edges:list[tuple[int,int,float]] # (target, source, influence)
    committed:list[int]
    masked:list[int]|None=None
    def node_strength(self):
        s={i:0.0 for i in self.committed}
        for t,src,w in self.edges:s[src]=s.get(src,0.)+float(w)
        return s
    def blocker_strength(self):
        s={i:0.0 for i in self.committed}
        ms=set(self.masked or [])
        for t,src,w in self.edges:
            if t in ms:s[src]=s.get(src,0.)+float(w)
        return s
    def subgraph(self,nodes):
        n=set(nodes); return [(t,s,w) for t,s,w in self.edges if s in n]

class InfluenceEstimator:
    def __init__(self,metric='js',top_k=4): self.metric=metric; self.top_k=top_k
    @staticmethod
    def _norm(x): return torch.softmax(x.float(),dim=-1)
    @staticmethod
    def _js(p,q,eps=1e-8):
        m=.5*(p+q); return .5*((p*((p+eps).log()-(m+eps).log())).sum(-1)+(q*((q+eps).log()-(m+eps).log())).sum(-1))
    def score(self,p,q):
        if self.metric=='l1':return (p-q).abs().sum(-1)
        if self.metric=='kl':return (p*(p.clamp_min(1e-8).log()-q.clamp_min(1e-8).log())).sum(-1)
        return self._js(p,q)
    def build(self,base_logits,cf_logits:Dict[int,torch.Tensor],committed:Sequence[int],masked:Sequence[int]):
        p=self._norm(base_logits); L=base_logits.shape[-2]; influence=np.zeros((L,L),np.float32); edges=[]
        targets=list(dict.fromkeys([*committed,*masked])); committed=list(committed); masked=list(masked)
        for src,cf in cf_logits.items():
            q=self._norm(cf[0]); scores=self.score(p[0],q).detach().cpu().numpy()
            for t in targets:
                if t!=src and t<L: influence[t,src]=float(scores[t])
        for src in committed:
            vals=sorted(((t,float(influence[t,src])) for t in targets if t!=src),key=lambda z:z[1],reverse=True)
            edges.extend((t,src,w) for t,w in vals[:self.top_k] if w>0)
        return CounterfactualGraph(influence,edges,committed,masked)

class BatchedCounterfactualEstimator:
    """Returns one baseline + ceil(K/B) counterfactual forwards and exact call accounting."""
    def __init__(self,forward,mask_id,metric='js',batch_size=8,top_k=4):
        self.forward=forward; self.mask_id=int(mask_id); self.estimator=InfluenceEstimator(metric,top_k); self.batch_size=max(1,int(batch_size))
    @torch.no_grad()
    def estimate(self,ids,candidates,target_positions):
        raw=self.forward(ids); base=raw.logits if hasattr(raw,'logits') else raw; calls=1; out={}
        cand=list(dict.fromkeys(candidates))
        for i in range(0,len(cand),self.batch_size):
            batch=cand[i:i+self.batch_size]; cf=ids.repeat(len(batch),1)
            for r,pos in enumerate(batch):cf[r,pos]=self.mask_id
            rawcf=self.forward(cf); logits=rawcf.logits if hasattr(rawcf,'logits') else rawcf; calls+=1
            for r,pos in enumerate(batch):out[int(pos)]=logits[r:r+1]
        return base,self.estimator.build(base,out,candidates,target_positions),calls
