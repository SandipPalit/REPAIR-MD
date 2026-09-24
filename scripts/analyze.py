#!/usr/bin/env python
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    print(df.describe(include="all").transpose().to_string())

if __name__ == "__main__":
    main()
