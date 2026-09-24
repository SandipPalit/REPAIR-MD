#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT)), json
from pathlib import Path
import numpy as np
import pandas as pd
from synapse_cut.synthetic import evaluate_trial

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/synthetic")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--noise", type=float, default=0.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    families = ["chain", "star", "tree", "grid", "random_dag"]
    rows = []
    for family in families:
        for _ in range(args.trials):
            r = evaluate_trial(family, args.n, rng, args.noise)
            rows.append(r.__dict__)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_trial.csv", index=False)
    summary = df.groupby("graph_family").agg({
        "precision":"mean",
        "recall":"mean",
        "regret":"mean",
        "oracle_cost":"mean",
        "greedy_cost":"mean",
        "oracle_gain":"mean",
        "greedy_gain":"mean"
    }).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
