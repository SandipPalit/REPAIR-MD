
from __future__ import annotations
from .graph import CounterfactualGraph

def structural_energy(graph, active=None):
    active=set(graph.committed if active is None else active)
    return float(sum(w for target,source,w in graph.edges if source in active and target in active))

def blocker_energy(graph, active=None):
    active=set(graph.committed if active is None else active)
    masked=set(graph.masked)
    return float(sum(w for target,source,w in graph.edges if target in masked and source in active))

def marginal_energy_reduction(graph, active, node):
    a=set(active); return structural_energy(graph,a)-structural_energy(graph,a-{node})
