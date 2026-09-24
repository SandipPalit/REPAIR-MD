
from __future__ import annotations
import json,platform,subprocess,time
from pathlib import Path
import numpy as np,torch

def seed_everything(seed):
    import random; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def snapshot():
    d={'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.cuda.is_available(),'cuda_version':torch.version.cuda}
    if torch.cuda.is_available(): d.update(gpu=torch.cuda.get_device_name(0),gpu_count=torch.cuda.device_count())
    try:d['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception:d['git_commit']=None
    return d

def save_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2,default=str))

def bootstrap_ci(values,seed=1234,n_boot=2000):
    a=np.asarray([x for x in values if x==x],dtype=float)
    if len(a)==0:return (float('nan'),float('nan'))
    rng=np.random.default_rng(seed); means=np.empty(n_boot)
    for i in range(n_boot): means[i]=rng.choice(a,size=len(a),replace=True).mean()
    return tuple(map(float,np.quantile(means,[.025,.975])))
