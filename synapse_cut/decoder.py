from __future__ import annotations
from dataclasses import dataclass,field
import time, math
import numpy as np
import torch
from .graph import BatchedCounterfactualEstimator, CounterfactualGraph
from .repair_bench import greedy_repair,beam_repair,interaction_repair,local_search_repair,exact_repair,residual_energy
from .scheduler import build_dependency_wavefront
from .objective import evaluate_state,joint_objective

def _ratio(num,den):
    return float(num)/float(den) if den else 0.

@dataclass
class DecodeOutput:
    token_ids:torch.Tensor; nfe:int; steps:int; remasked:int; counterfactual_calls:int; repair_events:int; repair_cost:int
    frontier_gain:float; wall_time:float; decode_forward_calls:int=0; model_forward_calls:int=0
    graph_time:float=0.; counterfactual_time:float=0.; solver_time:float=0.; decode_time:float=0.; peak_memory_mb:float|None=None
    initial_frontier:int=0; final_frontier:int=0; blocker_depth_before:int=0; blocker_depth_after:int=0
    trace:list[dict]=field(default_factory=list)
    candidate_recall:float|None=None; oracle_repair_recall:float|None=None; repair_feasible:float|None=None
    repair_regret:float|None=None; residual_structural_violation:float|None=None; structural_violation_reduction:float|None=None

class BaseDecoder:
    def __init__(self,model,tokenizer,mask_id,device='cuda',generation_positions=None):
        self.model=model; self.tokenizer=tokenizer; self.mask_id=int(mask_id); self.device=torch.device(device); self.generation_positions=generation_positions
    @torch.inference_mode()
    def forward(self,ids,**kwargs):
        kwargs.setdefault('use_cache',False)
        out=self.model(ids,**kwargs); return out.logits if hasattr(out,'logits') else out[0]
    def _distribution(self,logits,temp=0.):
        p=torch.softmax(logits.float()/max(temp,1e-8),dim=-1) if temp>0 else torch.softmax(logits.float(),dim=-1); return p,p.max(-1).values,p.argmax(-1)
    def _masked(self,ids,allowed=None):
        pos=torch.where(ids[0].eq(self.mask_id))[0].tolist()
        if allowed is not None:
            allowed=set(allowed); pos=[p for p in pos if p in allowed]
        return pos
    def _mutable(self,ids):
        allpos=range(ids.shape[1]); allowed=set(self.generation_positions) if self.generation_positions is not None else set(allpos); return [i for i in allpos if i in allowed and ids[0,i].item()!=self.mask_id]
    def _unmask(self,ids,pos,tok):
        for p in pos:ids[0,p]=tok[0,p]
    def _remask(self,ids,pos):
        for p in pos:ids[0,p]=self.mask_id
    @staticmethod
    def _transfer_count(masked_count, step, total_steps, schedule='balanced', selection_fraction=.25):
        if masked_count<=0:return 0
        remaining=max(1,total_steps-step)
        if schedule=='fraction': return min(masked_count,max(1,int(round(masked_count*selection_fraction))))
        # Balanced schedule: distribute the currently masked tokens across the remaining
        # diffusion steps. This matches LLaDA's uniform transfer principle when no remasking occurs.
        return min(masked_count,max(1,(masked_count + remaining - 1)//remaining))
    def _peak(self):
        if self.device.type=='cuda':return float(torch.cuda.max_memory_allocated(self.device)/2**20)
        return None

class ConfidenceDecoder(BaseDecoder):
    def generate(self,input_ids,steps=64,selection_fraction=.25,temperature=0.,generation_positions=None,**kwargs):
        start=time.perf_counter(); ids=input_ids.clone().to(self.device); allowed=generation_positions or self.generation_positions; nfe=0; trace=[]; progress=bool(kwargs.get('progress',False))
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
        for step in range(steps):
            masked=self._masked(ids,allowed)
            if not masked:break
            if progress:print(f'  decode step {step + 1}/{steps}: {len(masked)} masked tokens; running model forward',flush=True)
            logits=self.forward(ids); nfe+=1; _,conf,tok=self._distribution(logits,temperature)
            k=self._transfer_count(len(masked),step,steps,'balanced',selection_fraction); pos=torch.tensor(masked,device=ids.device); chosen=pos[torch.topk(conf[0,pos],k).indices].tolist(); self._unmask(ids,chosen,tok)
            if progress:print(f'  decode step {step + 1}/{steps}: committed {len(chosen)} tokens; elapsed {time.perf_counter() - start:.1f}s',flush=True)
            trace.append({'step':step,'masked_before':len(masked),'unmasked':len(chosen)})
        elapsed=time.perf_counter()-start; steps_done=len(trace)
        return DecodeOutput(ids,nfe,steps_done,0,0,0,0,0.,elapsed,nfe,nfe,peak_memory_mb=self._peak(),trace=trace)

class EntropyDecoder(ConfidenceDecoder):
    def generate(self,input_ids,**kw):
        # Same mechanics, highest entropy first.
        start=time.perf_counter(); ids=input_ids.clone().to(self.device); allowed=kw.get('generation_positions',self.generation_positions); steps=kw.get('steps',64); frac=kw.get('selection_fraction',.25); nfe=0; trace=[]
        for step in range(steps):
            masked=self._masked(ids,allowed)
            if not masked:break
            logits=self.forward(ids); nfe+=1; p,_,tok=self._distribution(logits,kw.get('temperature',0.)); ent=-(p.clamp_min(1e-8)*p.clamp_min(1e-8).log()).sum(-1); k=self._transfer_count(len(masked),step,steps,kw.get('transfer_schedule','balanced'),frac); pos=torch.tensor(masked,device=ids.device); chosen=pos[torch.topk(ent[0,pos],k).indices].tolist(); self._unmask(ids,chosen,tok); trace.append({'step':step,'masked_before':len(masked),'unmasked':len(chosen)})
        return DecodeOutput(ids,nfe,len(trace),0,0,0,0,0.,time.perf_counter()-start,nfe,nfe,peak_memory_mb=self._peak(),trace=trace)

class RandomDecoder(ConfidenceDecoder):
    def generate(self,input_ids,**kw):
        g=torch.Generator(device=self.device).manual_seed(int(kw.get('seed',1234))); start=time.perf_counter(); ids=input_ids.clone().to(self.device); allowed=kw.get('generation_positions',self.generation_positions); steps=kw.get('steps',64); frac=kw.get('selection_fraction',.25); nfe=0; trace=[]
        for step in range(steps):
            masked=self._masked(ids,allowed)
            if not masked:break
            logits=self.forward(ids); nfe+=1; _,_,tok=self._distribution(logits,kw.get('temperature',0.)); k=self._transfer_count(len(masked),step,steps,kw.get('transfer_schedule','balanced'),frac); perm=torch.randperm(len(masked),generator=g,device=self.device)[:k]; chosen=[masked[int(i)] for i in perm]; self._unmask(ids,chosen,tok); trace.append({'step':step,'masked_before':len(masked),'unmasked':len(chosen)})
        return DecodeOutput(ids,nfe,len(trace),0,0,0,0,0.,time.perf_counter()-start,nfe,nfe,peak_memory_mb=self._peak(),trace=trace)

class AttentionCentralityDecoder(ConfidenceDecoder):
    def generate(self,input_ids,**kw):
        start=time.perf_counter(); ids=input_ids.clone().to(self.device); allowed=kw.get('generation_positions',self.generation_positions); steps=kw.get('steps',64); frac=kw.get('selection_fraction',.25); nfe=0; trace=[]
        for step in range(steps):
            masked=self._masked(ids,allowed)
            if not masked:break
            out=self.model(ids,output_attentions=True); nfe+=1; logits=out.logits; _,_,tok=self._distribution(logits,kw.get('temperature',0.)); att=getattr(out,'attentions',None)
            central=att[-1][0].float().mean(0).sum(0) if att else torch.softmax(logits.float(),-1).max(-1).values[0]
            k=self._transfer_count(len(masked),step,steps,kw.get('transfer_schedule','balanced'),frac); pos=torch.tensor(masked,device=self.device); chosen=pos[torch.topk(central[pos],k).indices].tolist(); self._unmask(ids,chosen,tok); trace.append({'step':step,'masked_before':len(masked),'unmasked':len(chosen)})
        return DecodeOutput(ids,nfe,len(trace),0,0,0,0,0.,time.perf_counter()-start,nfe,nfe,peak_memory_mb=self._peak(),trace=trace)

class SynapseCutDecoder(BaseDecoder):
    def generate(self,input_ids,steps=64,selection_fraction=.25,temperature=0.,candidate_k=16,candidate_low_k=16,top_k_edges=4,repair_budget=4,repair_solver='interaction',beam_width=16,structural_threshold=.1,counterfactual_batch_size=1,repair_every=4,generation_positions=None,lambda_repair=1.,lambda_depth=.25,lambda_frontier=.5,lambda_energy=1.,schedule_mode='wavefront',seed=1234,**kwargs):
        start=time.perf_counter(); ids=input_ids.clone().to(self.device); allowed=set(generation_positions or self.generation_positions or range(ids.shape[1])); mutable=allowed; nfe=0; cf_calls=0; remasked=events=repair_cost=0; trace=[]; graph_time=cf_time=solver_time=decode_time=0.; initial_frontier=final_frontier=0; depth_before=depth_after=0; prev_frontier=0; oracle_recalls=[]; repair_recalls=[]; feasibility=[]; regrets=[]; residuals=[]; reductions=[]
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
        estimator=BatchedCounterfactualEstimator(self.forward,self.mask_id,'js',counterfactual_batch_size,top_k_edges)
        for step in range(steps):
            masked=self._masked(ids,allowed)
            if not masked:break
            t0=time.perf_counter(); logits=self.forward(ids); nfe+=1; decode_time+=time.perf_counter()-t0; p,conf,tok=self._distribution(logits,temperature)
            committed=[i for i in mutable if ids[0,i].item()!=self.mask_id]
            graph=None; repair=[]; wave=None
            if committed and len(committed)>1 and step%max(1,repair_every)==0:
                # confidence-diverse candidate pool: high-confidence AND low-confidence commitments.
                high=sorted(committed,key=lambda x:float(conf[0,x]),reverse=True)[:max(0,candidate_k)]
                low=sorted(committed,key=lambda x:float(conf[0,x]))[:max(0,candidate_low_k)]
                candidates=list(dict.fromkeys(high+low))
                if candidates:
                    t=time.perf_counter(); cf0=time.perf_counter(); base,graph,calls=estimator.estimate(ids,candidates,masked); cf_time+=time.perf_counter()-cf0; graph_time+=time.perf_counter()-t; cf_calls+=calls
                    scores=graph.node_strength(); blocker=graph.blocker_strength()
                    nodes=sorted(candidates,key=lambda x:(blocker.get(x,0.)+scores.get(x,0.)),reverse=True)
                    edges=graph.edges
                    if edges:
                        oracle=exact_repair(committed,edges,repair_budget,structural_threshold)
                        ts=time.perf_counter()
                        if repair_solver=='greedy':sol=greedy_repair(nodes,edges,repair_budget,scores)
                        elif repair_solver=='beam':sol=beam_repair(nodes,edges,repair_budget,beam_width,scores)
                        elif repair_solver=='local':sol=local_search_repair(nodes,edges,repair_budget,scores)
                        else:sol=interaction_repair(nodes,edges,repair_budget,beam_width)
                        solver_time+=time.perf_counter()-ts
                        repair=list(sol.selected)
                        oracle_set=set(oracle.selected); repair_set=set(repair)
                        oracle_recalls.append(_ratio(len(set(candidates)&oracle_set),len(oracle_set)) if oracle_set else 1.)
                        repair_recalls.append(_ratio(len(repair_set&oracle_set),len(oracle_set)) if oracle_set else 1.)
                        feasibility.append(float(sol.residual<=structural_threshold))
                        regrets.append(_ratio(len(repair)-len(oracle.selected),len(oracle.selected)) if oracle.selected else float(len(repair)))
                        residuals.append(float(sol.residual)); reductions.append(float(residual_energy(edges,())-sol.residual))
                        before=evaluate_state(graph,[],structural_threshold); depth_before=before.blocker_depth; 
                        if initial_frontier==0:initial_frontier=before.frontier; prev_frontier=before.frontier
                        if repair and sol.residual<=1e-8:
                            self._remask(ids,repair); remasked+=len(repair); events+=1; repair_cost+=len(repair)
                            after=evaluate_state(graph,repair,structural_threshold); depth_after=after.blocker_depth; final_frontier=after.frontier; 
                            self._remask(ids,repair)
                            wave=build_dependency_wavefront(graph,self._masked(ids,allowed),structural_threshold,set(graph.committed)-set(repair))
                            # Keep repair reversible for this step; unmasking occurs below from updated logits.
                            self._remask(ids,repair)
                            # recompute logits after structural intervention; this is a real model forward and is counted.
                            t1=time.perf_counter(); logits=self.forward(ids); nfe+=1; decode_time+=time.perf_counter()-t1; p,conf,tok=self._distribution(logits,temperature)
            if graph is not None:
                wave=build_dependency_wavefront(graph,masked,structural_threshold,set(graph.committed)-set(repair))
                pool=[x for x in wave.frontier if x in masked] if schedule_mode=='wavefront' else []
            else:pool=[]
            if not pool:pool=masked
            k=self._transfer_count(len(pool),step,steps,'balanced',selection_fraction)
            pos=torch.tensor(pool,device=self.device); chosen=pos[torch.topk(conf[0,pos],k).indices].tolist(); self._unmask(ids,chosen,tok)
            trace.append({'step':step,'masked_before':len(masked),'unmasked':len(chosen),'remasked':len(repair),'candidate_count':len(committed),'candidate_screened':len(candidates) if committed else 0,'frontier':len(pool),'repair_cost':len(repair),'wave_depth':wave.depth if wave else 0})
        elapsed=time.perf_counter()-start; final_frontier=final_frontier or len(self._masked(ids,allowed)); depth_after=depth_after or depth_before
        return DecodeOutput(ids,nfe,len(trace),remasked,cf_calls,events,repair_cost,float(final_frontier-initial_frontier),elapsed,nfe+cf_calls,nfe,graph_time,cf_time,solver_time,decode_time,self._peak(),initial_frontier,final_frontier,depth_before,depth_after,trace,candidate_recall=float(np.mean(oracle_recalls)) if oracle_recalls else None,oracle_repair_recall=float(np.mean(repair_recalls)) if repair_recalls else None,repair_feasible=float(np.mean(feasibility)) if feasibility else None,repair_regret=float(np.mean(regrets)) if regrets else None,residual_structural_violation=float(np.mean(residuals)) if residuals else None,structural_violation_reduction=float(np.mean(reductions)) if reductions else None)
