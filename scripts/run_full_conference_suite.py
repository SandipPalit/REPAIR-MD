#!/usr/bin/env python
"""Single entry point for the reproducible conference experiment matrix."""
from __future__ import annotations
import argparse,subprocess,sys

def run(cmd):
 print('\n>>>',' '.join(cmd)); return subprocess.run(cmd,check=True).returncode

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['synthetic','real','all'],default='all');ap.add_argument('--real-samples',type=int,default=10);ap.add_argument('--real-seeds',default='1234');ap.add_argument('--model',default='GSAI-ML/LLaDA-8B-Base');a=ap.parse_args()
 py=sys.executable
 if a.mode in ('synthetic','all'):
  run([py,'scripts/run_ablation_suite.py','--trials','1000']); run([py,'scripts/run_adversarial_suite.py']); run([py,'scripts/run_pareto_benchmark.py','--trials','100']); run([py,'scripts/run_scaling.py','--trials','100'])
 if a.mode in ('real','all'):
  run([py,'scripts/run_real_world_benchmark.py','--model',a.model,'--samples-per-dataset',str(a.real_samples),'--seeds',a.real_seeds])
  run([py,'scripts/analyze_conference.py','--input','results/real_world/results.csv'])
if __name__=='__main__':main()
