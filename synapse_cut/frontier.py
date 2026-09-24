
from __future__ import annotations
from typing import Iterable
from .graph import CounterfactualGraph

def safe_frontier(graph: CounterfactualGraph, masked: Iterable[int], threshold: float=0.10)->list[int]:
    masked=set(masked); vals={i:0.0 for i in masked}
    for target,source,w in graph.edges:
        if target in masked and source in graph.committed:
            vals[target]=max(vals[target],float(w))
    return [i for i in sorted(masked) if vals[i]<=threshold]

def frontier_gain(before,after): return float(len(after)-len(before))
