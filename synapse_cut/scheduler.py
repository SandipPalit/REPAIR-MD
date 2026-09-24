from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

@dataclass
class Wavefront:
    waves:list[list[int]]
    frontier:list[int]
    depth:int

def build_dependency_wavefront(graph, masked:Iterable[int], threshold=.10, active=None):
    masked=set(masked); active=set(graph.committed if active is None else active)
    blockers={m:set() for m in masked}
    succ={u:set() for u in active}
    for target,source,w in graph.edges:
        if source in active and w>threshold:
            if target in masked: blockers[target].add(source)
            if target in active: succ[source].add(target)
    frontier=sorted([m for m in masked if not blockers[m]])
    # dependency levels over committed nodes; masked token becomes available when all active blockers are resolved.
    waves=[]; remaining=set(masked); released=set()
    while remaining:
        wave=sorted([m for m in remaining if not (blockers[m]-released)])
        if not wave:
            wave=sorted(remaining)[:1]
        waves.append(wave); released.update(wave); remaining-=set(wave)
    return Wavefront(waves,frontier,len(waves))

def compatible_frontier(graph, masked, threshold=.10, active=None):
    return build_dependency_wavefront(graph,masked,threshold,active).frontier
