#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot(df,x,y,path,title):
    fig,ax=plt.subplots(figsize=(7,5))
    for method,g in df.groupby('method'):
        gg=g.sort_values(x); ax.plot(gg[x],gg[y],marker='o',label=method)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--out',default='results/figures'); a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(a.input)
    if {'wall_time_sec','nfe'}.issubset(df.columns): plot(df,'wall_time_sec','nfe',out/'nfe_vs_time.png','NFE vs wall-clock')
    if {'wall_time_sec','remasked_tokens'}.issubset(df.columns): plot(df,'wall_time_sec','remasked_tokens',out/'remasked_vs_time.png','Remasking vs wall-clock')
    if {'nfe','repair_cost'}.issubset(df.columns): plot(df,'nfe','repair_cost',out/'repair_vs_nfe.png','Repair cost vs NFE')
if __name__=='__main__': main()
