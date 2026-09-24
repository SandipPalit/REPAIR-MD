from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from .graph import CounterfactualGraph
from .energy import structural_energy, blocker_energy

@dataclass
class StateMetrics:
    energy:float
    blocker_energy:float
    blocker_depth:int
    frontier:int
    repair_cost:int

def blocker_depth(graph, active=None):
    active=set(graph.committed if active is None else active); masked=set(graph.masked or [])
    # longest path of active committed sources before a masked target; graph may be cyclic numerically,
    # so use the source->target dependency graph and a conservative iterative relaxation.
    import networkx as nx
    H=nx.DiGraph(); H.add_nodes_from(graph.committed)
    for t,s,w in graph.edges:
        if s in active and t in active and w>0: H.add_edge(s,t,weight=w)
    if not masked: return 0
    best=0
    try: topo=list(nx.topological_sort(H))
    except nx.NetworkXUnfeasible: topo=list(H.nodes)
    dp={v:(1 if v in active else 0) for v in H.nodes}
    for u in topo:
        for v in H.successors(u): dp[v]=max(dp.get(v,0),dp[u]+1)
    for t,s,w in graph.edges:
        if t in masked and s in active: best=max(best,dp.get(s,1))
    return int(best)

def frontier_size(graph, threshold=0.10, active=None):
    active=set(graph.committed if active is None else active); masked=set(graph.masked or [])
    blocked={t for t,s,w in graph.edges if t in masked and s in active and w>threshold}
    return len(masked-blocked)

def evaluate_state(graph, removed:Iterable[int], threshold=0.10):
    removed=set(removed); active=set(graph.committed)-removed
    return StateMetrics(structural_energy(graph,active), blocker_energy(graph,active), blocker_depth(graph,active), frontier_size(graph,threshold,active), len(removed))

def joint_objective(graph, removed, lambda_repair=1., lambda_depth=.25, lambda_frontier=.5, lambda_energy=1., threshold=.1):
    m=evaluate_state(graph,removed,threshold)
    return lambda_repair*m.repair_cost + lambda_depth*m.blocker_depth - lambda_frontier*m.frontier + lambda_energy*m.energy

def pareto_frontier(graph, candidate_nodes, budget, threshold=.1):
    from itertools import combinations
    points=[]
    for k in range(0,min(budget,len(candidate_nodes))+1):
        for c in combinations(candidate_nodes,k):
            m=evaluate_state(graph,c,threshold)
            points.append((tuple(c),m))
    out=[]
    for c,m in points:
        dominated=False
        for c2,m2 in points:
            if c==c2: continue
            no_worse=(m2.repair_cost<=m.repair_cost and m2.blocker_depth<=m.blocker_depth and m2.frontier>=m.frontier and m2.energy<=m.energy)
            strict=(m2.repair_cost<m.repair_cost or m2.blocker_depth<m.blocker_depth or m2.frontier>m.frontier or m2.energy<m.energy)
            if no_worse and strict: dominated=True; break
        if not dominated: out.append((c,m))
    return out
