#!/usr/bin/env bash
set -euo pipefail
pytest -q
python scripts/run_synthetic.py --out results/synthetic_smoke --n 12 --trials 50 --seed 1234
