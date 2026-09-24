#!/usr/bin/env python
"""Execute externally installed baseline implementations without fabricating results.

Each command is isolated, logged, and tagged with model/seed/budget metadata. Commands
are read from configs/sota_methods.csv and should be replaced by the official repo
commands appropriate to the exact checkpoint and environment.
"""
from __future__ import annotations
import argparse,csv,json,subprocess,time,os,sys,platform
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/sota_methods.csv'); ap.add_argument('--model',default='GSAI-ML/LLaDA-8B-Base'); ap.add_argument('--steps',type=int,default=32); ap.add_argument('--max-new-tokens',type=int,default=128); ap.add_argument('--seed',type=int,default=1234); ap.add_argument('--dataset-scope',default='ParallelBench'); ap.add_argument('--execute',action='store_true'); ap.add_argument('--out',default='results/matched_sota'); a=ap.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True); rows=[]
 with open(a.config,newline='') as f: methods=list(csv.DictReader(f))
 for m in methods:
  cmd=m.get('command','').strip(); rec={'method':m.get('method'),'command':cmd,'model':a.model,'steps':a.steps,'max_new_tokens':a.max_new_tokens,'seed':a.seed,'dataset_scope':a.dataset_scope,'status':'NOT_EXECUTED'}
  if a.execute and cmd:
   t=time.perf_counter(); p=subprocess.run(cmd,shell=True,capture_output=True,text=True); rec.update({'status':'PASS' if p.returncode==0 else 'FAIL','returncode':p.returncode,'wall_time_sec':time.perf_counter()-t,'stdout':p.stdout,'stderr':p.stderr})
  rows.append(rec)
 (out/'manifest.json').write_text(json.dumps({'python':platform.python_version(),'cwd':os.getcwd(),'rows':rows},indent=2))
 with open(out/'results.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
