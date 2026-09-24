#!/usr/bin/env python
"""Record or execute an external benchmark command without fabricating results."""
import argparse,json,subprocess,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--benchmark',required=True); ap.add_argument('--model',required=True); ap.add_argument('--method',default='synapse_cut'); ap.add_argument('--limit',type=int,default=None); ap.add_argument('--steps',type=int,default=64); ap.add_argument('--output',default='results/benchmark_command.json'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args()
    if a.benchmark.lower()=='parallelbench':
        cmd=['pb','eval','--model','parallelbench_llada','--model_args',f'model_path={a.model}','--gen_kwargs',f'steps={a.steps},block_length={a.steps},unmasking={a.method}','--tasks','parallelbench_all','--include_path','parallelbench/tasks','--batch_size','1']
    else:
        cmd=['echo','No built-in external adapter for',a.benchmark,'— use its official runner and record the command here.']
    record={'benchmark':a.benchmark,'model':a.model,'method':a.method,'command':cmd,'limit':a.limit}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(record,indent=2)); print(' '.join(cmd))
    if a.execute: subprocess.run(cmd,check=True)
if __name__=='__main__': main()
