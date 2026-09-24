#!/usr/bin/env python
"""Run the same real benchmark over multiple diffusion budgets for equal-NFE/Pareto analysis."""
import argparse,subprocess,sys
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='GSAI-ML/LLaDA-8B-Base');ap.add_argument('--steps',default='8,16,32,64,128');ap.add_argument('--samples-per-dataset',type=int,default=10);ap.add_argument('--seeds',default='1234');ap.add_argument('--out-root',default='results/compute_sweep');a=ap.parse_args();root=Path(a.out_root);root.mkdir(parents=True,exist_ok=True)
 for s in a.steps.split(','):
  out=root/f'steps_{s}'; cmd=[sys.executable,'scripts/run_real_world_benchmark.py','--model',a.model,'--steps',s,'--samples-per-dataset',str(a.samples_per_dataset),'--seeds',a.seeds,'--out',str(out)]
  print('>>>',' '.join(cmd)); subprocess.run(cmd,check=True)
if __name__=='__main__':main()
