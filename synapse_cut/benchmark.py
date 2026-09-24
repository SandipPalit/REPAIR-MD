from __future__ import annotations
import json, time, platform, subprocess
from pathlib import Path
from typing import Callable
import numpy as np
import torch
from .metrics import summarize

def environment_snapshot():
    snap = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        snap["gpu"] = torch.cuda.get_device_name(0)
        snap["cuda_version"] = torch.version.cuda
    try:
        snap["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        snap["git_commit"] = None
    return snap

def write_jsonl(path: str | Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def benchmark_summary(path):
    rows = read_jsonl(path)
    return summarize(rows)
