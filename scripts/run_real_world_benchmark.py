#!/usr/bin/env python
from __future__ import annotations
import argparse,csv,json,random,time,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, pandas as pd
from synapse_cut.model import load_model,model_placement,validate_model_placement
from synapse_cut.llada_adapter import LLaDAAdapter
from synapse_cut.baselines import build_decoder
from synapse_cut.config import environment_snapshot

def norm(s):return ' '.join(str(s).lower().split())
def rouge_n(pred,ref,n):
    from collections import Counter
    def grams(x):return Counter(tuple(x[i:i+n]) for i in range(max(0,len(x)-n+1)))
    a,b=grams(norm(pred).split()),grams(norm(ref).split()); overlap=sum((a&b).values());
    if not a or not b:return 0.
    p=overlap/sum(a.values()); r=overlap/sum(b.values()); return 2*p*r/(p+r) if p+r else 0.
def rouge_l(pred,ref):
    a,b=norm(pred).split(),norm(ref).split(); dp=[0]*(len(b)+1)
    for x in a:
        old=0
        for j,y in enumerate(b,1):
            tmp=dp[j]; dp[j]=old+1 if x==y else max(dp[j],dp[j-1]); old=tmp
    l=dp[-1]; return 2*l/(len(a)+len(b)) if a and b else 0.
def token_f1(pred,ref):
    a=norm(pred).split(); b=norm(ref).split(); from collections import Counter
    ca,cb=Counter(a),Counter(b); overlap=sum((ca&cb).values()); p=overlap/len(a) if a else 0.; r=overlap/len(b) if b else 0.; return 2*p*r/(p+r) if p+r else 0.
def task_score(dataset,pred,ref,label=None):
    out={'rouge1':None,'rouge2':None,'rougeL':None,'token_f1':None,'exact_match':None,'accuracy':None,'predicted_label':None}
    if ref is not None:
        out.update({'rouge1':rouge_n(pred,ref,1),'rouge2':rouge_n(pred,ref,2),'rougeL':rouge_l(pred,ref),'token_f1':token_f1(pred,ref),'exact_match':float(norm(pred)==norm(ref))})
    if dataset=='ag_news' and label is not None:
        names=['World','Sports','Business','Sci/Tech']; text=norm(pred); predicted=next((name for name in names if norm(name) in text),None); out['predicted_label']=predicted; out['accuracy']=float(predicted==names[int(label)])
        out['quality']=out['accuracy']
    elif ref is not None: out['quality']=out['rougeL']
    else: out['quality']=None
    return out
def macro_f1(rows):
    labels=['World','Sports','Business','Sci/Tech']; scores=[]
    for name in labels:
        tp=sum(r['predicted_label']==name and int(r['label'])==labels.index(name) for r in rows)
        fp=sum(r['predicted_label']==name and int(r['label'])!=labels.index(name) for r in rows)
        fn=sum(r['predicted_label']!=name and int(r['label'])==labels.index(name) for r in rows)
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(scores)/len(scores)
def load_rows(name,split='test',n=10,seed=1234):
    from datasets import load_dataset
    specs={'wikitext2':('wikitext','wikitext-2-raw-v1','text',None),'ag_news':('ag_news',None,'text','label'),'cnn_dailymail':('cnn_dailymail','3.0.0','article','highlights'),'xsum':('EdinburghNLP/xsum',None,'document','summary'),'multi_news':('multi_news',None,'document','summary')}
    args=specs[name]; ds=load_dataset(args[0],args[1],split=split) if args[1] else load_dataset(args[0],split=split)
    rng=random.Random(seed); idx=list(range(len(ds))); rng.shuffle(idx); rows=[]
    for i in idx:
        r=ds[i]; text=str(r[args[2]]).strip(); ref=None if args[3] is None or args[3]=='label' else str(r[args[3]]).strip()
        if len(text)>40:rows.append({'id':f'{name}-{i}','text':text,'reference':ref,'label':r.get('label')});
        if len(rows)>=n:break
    return rows

def prompt_for(name,text):
    if name=='wikitext2':return 'Continue the following text naturally. Return only the continuation.\n\n'+text[:1800]
    if name=='ag_news':return 'Classify the following news article into one of: World, Sports, Business, Sci/Tech. Return only the class name.\n\n'+text[:1800]
    if name=='cnn_dailymail':return 'Summarize the following article concisely.\n\n'+text[:5000]
    if name=='xsum':return 'Write a one-sentence news summary of the following document.\n\n'+text[:5000]
    return 'Summarize the following collection of news articles.\n\n'+text[:6000]

def safe_ratio(num,den):
    return float(num)/float(den) if den else 0.

def trace_metrics(res,generated_tokens):
    trace=res.trace or []
    committed=[int(item.get('unmasked',0)) for item in trace]
    frontiers=[int(item['frontier']) for item in trace if 'frontier' in item]
    return {
        'generated_tokens_actual':int(generated_tokens),
        'avg_tokens_per_step':safe_ratio(sum(committed),len(committed)),
        'median_tokens_per_step':float(np.median(committed)) if committed else 0.,
        'p25_tokens_per_step':float(np.percentile(committed,25)) if committed else 0.,
        'p75_tokens_per_step':float(np.percentile(committed,75)) if committed else 0.,
        'max_frontier':max(frontiers) if frontiers else None,
        'avg_frontier':safe_ratio(sum(frontiers),len(frontiers)) if frontiers else None,
    }

def derived_metrics(res,wall,generated_tokens):
    timings={'decode_time_sec':res.decode_time,'graph_time_sec':res.graph_time,'counterfactual_time_sec':res.counterfactual_time,'solver_time_sec':res.solver_time}
    out=trace_metrics(res,generated_tokens)
    structural_trace=bool(res.trace and 'frontier' in res.trace[0])
    out.update({
        'nfe_total':res.model_forward_calls,
        'nfe_decode':res.decode_forward_calls,
        'nfe_counterfactual':res.counterfactual_calls,
        'forward_calls_per_token':safe_ratio(res.model_forward_calls,generated_tokens),
        'wall_time_per_token_ms':1000.*safe_ratio(wall,generated_tokens),
        'counterfactual_fraction':safe_ratio(res.counterfactual_calls,res.model_forward_calls),
        'decode_time_fraction':safe_ratio(res.decode_time,wall),
        'graph_time_fraction':safe_ratio(res.graph_time,wall),
        'counterfactual_time_fraction':safe_ratio(res.counterfactual_time,wall),
        'solver_time_fraction':safe_ratio(res.solver_time,wall),
        'remask_rate':safe_ratio(res.remasked,generated_tokens),
        'remasked_tokens_per_repair':safe_ratio(res.remasked,res.repair_events),
        'repair_events_per_step':safe_ratio(res.repair_events,res.steps),
        'repair_cost_per_token':safe_ratio(res.repair_cost,generated_tokens),
        'frontier_gain':res.frontier_gain if structural_trace else None,
        'blocker_depth_before':res.blocker_depth_before if structural_trace else None,
        'blocker_depth_after':res.blocker_depth_after if structural_trace else None,
        'blocker_depth_reduction':res.blocker_depth_before-res.blocker_depth_after if structural_trace else None,
        'timing_unaccounted_sec':wall-sum(timings.values()),
        'candidate_recall':res.candidate_recall,
        'oracle_repair_recall':res.oracle_repair_recall,
        'repair_feasible':res.repair_feasible,
        'repair_regret':res.repair_regret,
        'residual_structural_violation':res.residual_structural_violation,
        'structural_violation_reduction':res.structural_violation_reduction,
    })
    return out

def print_metric_table(title,metrics):
    items=[(str(k).replace('_',' ').title(), f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)) for k,v in metrics.items() if v is not None and v != 'N/A' and not (isinstance(v,(float,np.floating)) and np.isnan(v))]
    width=max(24,max((len(k) for k,_ in items),default=0)); value_width=max(14,max((len(v) for _,v in items),default=0))
    border='+'+'-'*(width+2)+'+'+'-'*(value_width+2)+'+'
    print(f'\n{title}',flush=True); print(border,flush=True); print(f'| {"Metric":<{width}} | {"Value":>{value_width}} |',flush=True); print(border,flush=True)
    for key,value in items: print(f'| {key:<{width}} | {value:>{value_width}} |',flush=True)
    print(border,flush=True)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',default='GSAI-ML/LLaDA-8B-Base'); ap.add_argument('--datasets',default='wikitext2,ag_news,cnn_dailymail,xsum,multi_news'); ap.add_argument('--samples-per-dataset',type=int,default=10); ap.add_argument('--seeds',default='1234'); ap.add_argument('--methods',default='confidence,entropy,attention,synapse_cut'); ap.add_argument('--device',default='cuda'); ap.add_argument('--dtype',default='bfloat16'); ap.add_argument('--steps',type=int,default=32); ap.add_argument('--max-new-tokens',type=int,default=128); ap.add_argument('--max-input-tokens',type=int,default=1024); ap.add_argument('--out',default='results/real_world');
 ap.add_argument('--candidate-k',type=int,default=16); ap.add_argument('--candidate-low-k',type=int,default=16); ap.add_argument('--repair-budget',type=int,default=4); ap.add_argument('--repair-every',type=int,default=4); ap.add_argument('--cf-batch-size',type=int,default=1); ap.add_argument('--top-k-edges',type=int,default=4); ap.add_argument('--solver',default='interaction'); ap.add_argument('--selection-fraction',type=float,default=.25); ap.add_argument('--threshold',type=float,default=.1); ap.add_argument('--device-map',default=None,choices=['auto','sequential']); ap.add_argument('--gpu-memory',default='3GiB'); ap.add_argument('--cpu-memory',default='4GiB'); ap.add_argument('--offload-folder',default='offload'); ap.add_argument('--allow-disk-offload',action='store_true'); a=ap.parse_args()
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/'environment.json').write_text(json.dumps(environment_snapshot(),indent=2)); print(f'Loading model: {a.model}',flush=True); max_memory={0:a.gpu_memory,'cpu':a.cpu_memory} if a.device_map else None
 try:
  bundle=load_model(a.model,None,a.device,a.dtype,device_map=a.device_map,max_memory=max_memory,offload_folder=a.offload_folder)
  _,placement=model_placement(bundle.model); print(f'Model placement: {placement}',flush=True)
  validate_model_placement(bundle.model,a.allow_disk_offload)
 except RuntimeError as exc:
  print(f'Benchmark stopped before decoding: {exc}',file=sys.stderr,flush=True); return 2
 adapter=LLaDAAdapter(bundle); print(f'Model ready on {bundle.device}',flush=True); rows=[]
 for seed in map(int,a.seeds.split(',')):
  torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
  for dataset in [x.strip() for x in a.datasets.split(',')]:
    print(f'Loading dataset={dataset} seed={seed}',flush=True); examples=load_rows(dataset,n=a.samples_per_dataset,seed=seed); print(f'Loaded {len(examples)} examples from {dataset}',flush=True)
    for ex in examples:
        prompt=prompt_for(dataset,ex['text']); inp=adapter.prepare(prompt,a.max_new_tokens,max_input_tokens=a.max_input_tokens); genpos=inp.generation_positions
        for method in [x.strip() for x in a.methods.split(',')]:
         print(f'Running dataset={dataset} example={ex["id"]} method={method}',flush=True)
         if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
         dec=build_decoder(method,bundle.model,bundle.tokenizer,bundle.mask_id,bundle.device,generation_positions=genpos)
         t=time.perf_counter()
         try:
          res=dec.generate(inp.ids,steps=a.steps,max_new_tokens=a.max_new_tokens,generation_positions=genpos,candidate_k=a.candidate_k,candidate_low_k=a.candidate_low_k,repair_budget=a.repair_budget,repair_every=a.repair_every,counterfactual_batch_size=a.cf_batch_size,top_k_edges=a.top_k_edges,selection_fraction=a.selection_fraction,structural_threshold=a.threshold,repair_solver=a.solver,seed=seed,progress=True)
         except RuntimeError as exc:
          if 'out of memory' not in str(exc).lower(): raise
          if torch.cuda.is_available(): torch.cuda.empty_cache()
          raise RuntimeError(f'Model forward ran out of memory. Reduce --max-input-tokens or --max-new-tokens (total input length: {inp.ids.shape[1]}).') from exc
         wall=time.perf_counter()-t
        pred=adapter.decode(res.token_ids,inp.prompt_length); metrics=task_score(dataset,pred,ex['reference'],ex.get('label'))
        actual_tokens=sum(int(res.token_ids[0,p].item()!=bundle.mask_id) for p in genpos)
        row={'dataset':dataset,'example_id':ex['id'],'method':method,'model':a.model,'seed':seed,'prediction':pred,'reference':ex['reference'],'label':ex.get('label'),'nfe':res.nfe,'decode_forward_calls':res.decode_forward_calls,'counterfactual_calls':res.counterfactual_calls,'total_model_forward_calls':res.model_forward_calls,'steps':res.steps,'generated_tokens':a.max_new_tokens,'wall_time_sec':wall,'decode_time_sec':res.decode_time,'graph_time_sec':res.graph_time,'counterfactual_time_sec':res.counterfactual_time,'solver_time_sec':res.solver_time,'remasked_tokens':res.remasked,'repair_events':res.repair_events,'repair_cost':res.repair_cost,'frontier_gain':res.frontier_gain,'blocker_depth_before':res.blocker_depth_before,'blocker_depth_after':res.blocker_depth_after,'tokens_per_sec':actual_tokens/max(wall,1e-9),'peak_memory_mb':res.peak_memory_mb}; row.update(metrics); row.update(derived_metrics(res,wall,actual_tokens)); rows.append(row); pd.DataFrame(rows).to_csv(out/'results.csv',index=False); print(f'Completed dataset={dataset} example={ex["id"]} method={method} in {wall:.1f}s: {pred!r}',flush=True); print_metric_table(f'Metrics: {dataset} / {ex["id"]} / {method}',{'Task quality':row.get('quality'),'Accuracy':row.get('accuracy'),'Exact match':row.get('exact_match'),'NFE total':row['nfe_total'],'NFE decode':row['nfe_decode'],'NFE counterfactual':row['nfe_counterfactual'],'Forward calls/token':row['forward_calls_per_token'],'Wall time (sec)':row['wall_time_sec'],'Milliseconds/token':row['wall_time_per_token_ms'],'Tokens/sec':row['tokens_per_sec'],'Decode time fraction':row['decode_time_fraction'],'Graph time fraction':row['graph_time_fraction'],'CF time fraction':row['counterfactual_time_fraction'],'Solver time fraction':row['solver_time_fraction'],'Unaccounted time (sec)':row['timing_unaccounted_sec'],'Steps':row['steps'],'Actual tokens':row['generated_tokens_actual'],'Avg tokens/step':row['avg_tokens_per_step'],'Median tokens/step':row['median_tokens_per_step'],'P25 tokens/step':row['p25_tokens_per_step'],'P75 tokens/step':row['p75_tokens_per_step'],'Max frontier':row['max_frontier'],'Avg frontier':row['avg_frontier'],'Frontier gain':row['frontier_gain'],'Remask rate':row['remask_rate'],'Remasked/repair':row['remasked_tokens_per_repair'],'Repair events/step':row['repair_events_per_step'],'Repair events':row['repair_events'],'Repair cost':row['repair_cost'],'Blocker depth reduction':row['blocker_depth_reduction'],'Peak GPU memory (MB)':row['peak_memory_mb'],'Candidate recall':row['candidate_recall'],'Repair regret':row['repair_regret'],'Residual structural violation':row['residual_structural_violation']})
 df=pd.DataFrame(rows); df.to_csv(out/'results.csv',index=False)
 summary=df.groupby(['dataset','method'])[['wall_time_sec','total_model_forward_calls','nfe','accuracy','rougeL','token_f1','quality','tokens_per_sec','nfe_decode','nfe_counterfactual','avg_tokens_per_step','median_tokens_per_step','max_frontier','avg_frontier','frontier_gain','remask_rate','remasked_tokens_per_repair','repair_events_per_step','repair_events','repair_cost','blocker_depth_reduction','peak_memory_mb','counterfactual_fraction','decode_time_fraction','graph_time_fraction','counterfactual_time_fraction','solver_time_fraction','forward_calls_per_token','wall_time_per_token_ms']].mean().reset_index()
 summary['speedup_vs_confidence']=None
 for dataset, group in summary.groupby('dataset'):
  baseline=group.loc[group.method=='confidence','wall_time_sec']
  if len(baseline): summary.loc[group.index,'speedup_vs_confidence']=float(baseline.iloc[0])/group['wall_time_sec']
 summary['macro_f1']=None
 for (dataset,method), group in df.groupby(['dataset','method']):
  if dataset=='ag_news': summary.loc[(summary.dataset==dataset)&(summary.method==method),'macro_f1']=macro_f1(group.to_dict('records'))
 for (dataset,method), group in df.groupby(['dataset','method']):
  print_metric_table(f'Exact repair-oracle metrics: {dataset} / {method}',{'Candidate recall':group['candidate_recall'].mean(),'Oracle repair recall':group['oracle_repair_recall'].mean(),'Repair feasible':group['repair_feasible'].mean(),'Repair regret':group['repair_regret'].mean(),'Residual structural violation':group['residual_structural_violation'].mean(),'Structural violation reduction':group['structural_violation_reduction'].mean()})
 summary.to_csv(out/'summary.csv',index=False); print(summary.round(4).to_string(index=False));
 for _, aggregate in summary.iterrows():
    print_metric_table(f'Aggregate metrics: {aggregate.dataset} / {aggregate.method}',{'Quality':aggregate.get('quality'),'Accuracy':aggregate.get('accuracy'),'Macro F1':aggregate.get('macro_f1'),'ROUGE-L':aggregate.get('rougeL'),'NFE total':aggregate.get('total_model_forward_calls'),'NFE decode':aggregate.get('nfe_decode'),'NFE counterfactual':aggregate.get('nfe_counterfactual'),'Forward calls/token':aggregate.get('forward_calls_per_token'),'Wall time (sec)':aggregate.get('wall_time_sec'),'Milliseconds/token':aggregate.get('wall_time_per_token_ms'),'Tokens/sec':aggregate.get('tokens_per_sec'),'Speedup vs confidence':aggregate.get('speedup_vs_confidence'),'Decode time fraction':aggregate.get('decode_time_fraction'),'Graph time fraction':aggregate.get('graph_time_fraction'),'CF time fraction':aggregate.get('counterfactual_time_fraction'),'Solver time fraction':aggregate.get('solver_time_fraction'),'Avg tokens/step':aggregate.get('avg_tokens_per_step'),'Median tokens/step':aggregate.get('median_tokens_per_step'),'Max frontier':aggregate.get('max_frontier'),'Avg frontier':aggregate.get('avg_frontier'),'Frontier gain':aggregate.get('frontier_gain'),'Remask rate':aggregate.get('remask_rate'),'Remasked/repair':aggregate.get('remasked_tokens_per_repair'),'Repair events/step':aggregate.get('repair_events_per_step'),'Repair events':aggregate.get('repair_events'),'Repair cost':aggregate.get('repair_cost'),'Blocker depth reduction':aggregate.get('blocker_depth_reduction'),'Counterfactual fraction':aggregate.get('counterfactual_fraction'),'Peak GPU memory (MB)':aggregate.get('peak_memory_mb'),'FLOPs':'N/A','Energy (J)':'N/A','GPU utilization':'N/A','Candidate recall':'N/A','Oracle repair recall':'N/A','Repair regret':'N/A','Residual structural violation':'N/A','ParallelBench score':'N/A','Pass@1':'N/A'})
 print(f'Results written to {out / "results.csv"}',flush=True); print(f'Summary written to {out / "summary.csv"}',flush=True)
if __name__=='__main__':raise SystemExit(main())
