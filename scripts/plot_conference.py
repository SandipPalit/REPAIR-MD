#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='results/real_world/results.csv');ap.add_argument('--out',default='results/figures');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);df=pd.read_csv(a.input)
 if 'rougeL' in df and df.rougeL.notna().any():
  g=df.groupby('method').agg(quality=('rougeL','mean'),latency=('wall_time_sec','mean'),nfe=('total_model_forward_calls','mean')).reset_index();
  plt.figure();
  for _,r in g.iterrows():plt.scatter(r.latency,r.quality,label=r.method);plt.annotate(r.method,(r.latency,r.quality))
  plt.xlabel('Mean wall-clock (s)');plt.ylabel('Mean ROUGE-L');plt.title('Quality–latency Pareto view');plt.grid(alpha=.2);plt.tight_layout();plt.savefig(out/'quality_latency.png',dpi=220);plt.close()
  plt.figure();
  for _,r in g.iterrows():plt.scatter(r.nfe,r.quality,label=r.method);plt.annotate(r.method,(r.nfe,r.quality))
  plt.xlabel('Mean total model forward calls');plt.ylabel('Mean ROUGE-L');plt.title('Quality–compute view');plt.grid(alpha=.2);plt.tight_layout();plt.savefig(out/'quality_compute.png',dpi=220);plt.close()
 print('Wrote figures to',out)
if __name__=='__main__':main()
