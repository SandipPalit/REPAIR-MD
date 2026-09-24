from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Dict, List, Tuple, Set, Optional
import math, json
import numpy as np
import networkx as nx
from .repair_bench import exact_repair, greedy_repair, beam_repair, interaction_repair, local_search_repair, residual_energy

@dataclass
class BenchInstance:
    instance_id: str
    family: str
    n_committed: int
    n_masked: int
    dependency_edges: List[Tuple[int,int]]
    violation_edges: List[Tuple[int,int,float]]
    true_corruptions: List[int]
    observed_node_scores: Dict[int,float]
    base_dependency_depth: int
    repaired_dependency_depth_oracle: int
    frontier_before: int
    frontier_after_oracle: int
    oracle_repair: List[int]

@dataclass
class BenchResult:
    instance_id: str
    family: str
    method: str
    oracle_cost: int
    selected_cost: int
    repair_regret: float
    oracle_precision: float
    oracle_recall: float
    corruption_precision: float
    corruption_recall: float
    residual_violation_weight: float
    initial_violation_weight: float
    depth_before: int
    depth_after: int
    depth_reduction: int
    frontier_before: int
    frontier_after: int
    frontier_gain: int
    candidate_recall: float
    feasible: int
    conditional_cost_regret: float
    relative_depth_reduction: float
    relative_frontier_gain: float
    blocker_depth_before: int
    blocker_depth_after: int
    blocker_depth_reduction: int


def _family_graph(family: str, n: int, rng: np.random.Generator) -> nx.DiGraph:
    G=nx.DiGraph(); G.add_nodes_from(range(n))
    if family=='chain': G.add_edges_from((i,i+1) for i in range(n-1))
    elif family=='star': G.add_edges_from((0,i) for i in range(1,n))
    elif family=='tree': G.add_edges_from(((i-1)//2,i) for i in range(1,n))
    elif family=='grid':
        s=max(2,int(math.ceil(math.sqrt(n)))); nodes=[(r,c) for r in range(s) for c in range(s)][:n]; mp={x:i for i,x in enumerate(nodes)}
        for r,c in nodes:
            if (r,c+1) in mp: G.add_edge(mp[(r,c)],mp[(r,c+1)])
            if (r+1,c) in mp: G.add_edge(mp[(r,c)],mp[(r+1,c)])
    elif family=='random_dag':
        for i in range(n):
            for j in range(i+1,n):
                if rng.random()<min(0.20,3.5/n): G.add_edge(i,j)
    elif family=='clustered_dag':
        k=max(2,n//8); clusters=[i//k for i in range(n)]
        for i in range(n):
            for j in range(i+1,n):
                p=0.18 if clusters[i]==clusters[j] else 0.025
                if rng.random()<p: G.add_edge(i,j)
    elif family=='long_range_dag':
        for i in range(n-1): G.add_edge(i,i+1)
        for _ in range(max(1,n//2)):
            i=int(rng.integers(0,n-2)); j=int(rng.integers(i+2,n)); G.add_edge(i,j)
    else: raise ValueError(family)
    return G


def _masked_topology(G, committed, masked, removed):
    active_committed=committed-removed; H=nx.DiGraph(); H.add_nodes_from(masked); source=-1; H.add_node(source)
    for u,v in G.edges():
        if u in masked and v in masked: H.add_edge(u,v)
        elif u in active_committed and v in masked: H.add_edge(source,v)
    return H


def _depth_and_frontier(G, masked, committed=set(), removed=set()):
    H=_masked_topology(G,committed,masked,removed) if committed else G.subgraph(masked).copy()
    if not nx.is_directed_acyclic_graph(H): H=nx.DiGraph([(u,v) for u,v in H.edges if u<v])
    frontier=sum(1 for v in masked if H.in_degree(v)==0)
    source=-1; depth=0
    for v in masked:
        try: depth=max(depth,nx.shortest_path_length(H,source,v))
        except nx.NetworkXNoPath: pass
    return depth,frontier


def _critical_blocker_depth(G, committed, masked, removed):
    active=committed-removed; H=G.subgraph(active|masked).copy()
    if not nx.is_directed_acyclic_graph(H): return 0
    dp={v:(1 if v in active else 0) for v in H.nodes}
    for u in nx.topological_sort(H):
        for v in H.successors(u): dp[v]=max(dp.get(v,0),dp[u]+(1 if v in active else 0))
    return max((dp[v] for v in masked),default=0)


def exact_min_vertex_cover(nodes, edges, threshold=0.0, max_k=None):
    # Kept as a public compatibility function. Pairwise structural repair is a vertex-cover oracle.
    return list(exact_repair(nodes, [(u,v,w) for u,v,w in edges if w>threshold], max_k if max_k is not None else len(nodes), threshold=1e-8).selected)


def _repair_energy(edges, selected): return residual_energy(edges, selected)


def generate_instance(instance_id, family, n_committed, n_masked, rng, corruption_rate=0.15, edge_noise=0.05, blocker_rate=0.6):
    nc,nm=n_committed,n_masked; total=nc+nm; G=_family_graph(family,total,rng)
    committed=set(range(nc)); masked=set(range(nc,total)); dep_edges=list(G.edges())
    for c in committed:
        if rng.random()<blocker_rate:
            targets=rng.choice(list(masked),size=min(2,len(masked)),replace=False)
            for t in np.atleast_1d(targets): dep_edges.append((c,int(t)))
    dep_edges=list(dict.fromkeys(dep_edges))
    k=max(1,int(round(nc*corruption_rate))); true=list(map(int,rng.choice(sorted(committed),size=k,replace=False))); C=set(true)
    violations=[]
    for u,v in dep_edges:
        if u in C or v in C:
            w=max(0.05,float(1.0+rng.normal(0,edge_noise))); violations.append((u,v,w))
    clean=list(committed-C)
    for _ in range(max(0,int(0.08*len(dep_edges)))):
        if len(clean)>=2:
            u,v=map(int,rng.choice(clean,2,replace=False)); violations.append((u,v,float(rng.uniform(0.05,0.30))))
    scores={x:max(0.0,sum(w for u,v,w in violations if u==x or v==x)+rng.normal(0,edge_noise*max(1.0,sum(w for u,v,w in violations if u==x or v==x)))) for x in committed}
    oracle=exact_min_vertex_cover(sorted(committed),violations,max_k=nc)
    R=set(oracle)
    db,fb=_depth_and_frontier(G,masked,committed,set()); da,fa=_depth_and_frontier(G,masked,committed,R)
    return BenchInstance(instance_id,family,nc,nm,dep_edges,violations,true,scores,db,da,fb,fa,oracle)


def _select(inst, method, budget, candidate_k, rng=None, beam_width=16):
    nodes=list(range(inst.n_committed)); candidate=set(nodes)
    if candidate_k is not None and candidate_k<len(nodes): candidate=set(sorted(nodes,key=lambda x:inst.observed_node_scores[x],reverse=True)[:candidate_k])
    if method=='none': return []
    if method=='oracle': return list(inst.oracle_repair)
    if method=='random':
        rr=rng or np.random.default_rng(0); return list(map(int,rr.choice(sorted(candidate),size=min(budget,len(candidate)),replace=False))) if candidate else []
    if method in ('singleton','singleton_full'):
        pool=candidate if method=='singleton' else set(nodes); return sorted(pool,key=lambda x:inst.observed_node_scores[x],reverse=True)[:budget]
    pool=candidate if method in ('greedy','beam','interaction','local') else set(nodes)
    if method in ('greedy','greedy_full'): return list(greedy_repair(sorted(pool),inst.violation_edges,budget,inst.observed_node_scores).selected)
    if method in ('beam','beam_full'): return list(beam_repair(sorted(pool),inst.violation_edges,budget,beam_width,inst.observed_node_scores).selected)
    if method in ('interaction','interaction_full'): return list(interaction_repair(sorted(pool),inst.violation_edges,budget,beam_width=max(beam_width,8)).selected)
    if method in ('local','local_full'): return list(local_search_repair(sorted(pool),inst.violation_edges,budget,inst.observed_node_scores).selected)
    raise ValueError(method)


def evaluate_instance(inst, method, budget, candidate_k=None, rng=None, beam_width=16):
    nodes=set(range(inst.n_committed)); oracle=set(inst.oracle_repair); true=set(inst.true_corruptions); selected=set(_select(inst,method,budget,candidate_k,rng,beam_width))
    residual=_repair_energy(inst.violation_edges,selected); initial=sum(w for _,_,w in inst.violation_edges); feasible=int(residual<=1e-8)
    masked=set(range(inst.n_committed,inst.n_committed+inst.n_masked)); G=nx.DiGraph(); G.add_nodes_from(range(inst.n_committed+inst.n_masked)); G.add_edges_from(inst.dependency_edges)
    da,fa=_depth_and_frontier(G,masked,nodes,selected); bdb=_critical_blocker_depth(G,nodes,masked,set()); bda=_critical_blocker_depth(G,nodes,masked,selected)
    dr=inst.base_dependency_depth-da; fg=fa-inst.frontier_before
    # Repair regret is defined only for feasible solutions. Infeasible solutions are not negative-regret solutions.
    regret=float(len(selected)-len(oracle)) if feasible else float('nan')
    candidate=set(nodes) if candidate_k is None or candidate_k>=len(nodes) else set(sorted(nodes,key=lambda x:inst.observed_node_scores[x],reverse=True)[:candidate_k])
    cond=regret
    return BenchResult(inst.instance_id,inst.family,method,len(oracle),len(selected),regret,
        _pr(selected,oracle),_rc(selected,oracle),_pr(selected,true),_rc(selected,true),residual,initial,
        inst.base_dependency_depth,da,dr,inst.frontier_before,fa,fg,_rc(candidate,true),feasible,cond,
        dr/max(inst.base_dependency_depth,1),fg/max(inst.frontier_before,1),bdb,bda,bdb-bda)

def _pr(a,b): return len(set(a)&set(b))/len(a) if a else (1.0 if not b else 0.0)
def _rc(a,b): return len(set(a)&set(b))/len(b) if b else 1.0

def generate_benchmark(families,trials,n_committed,n_masked,seed,corruption_rate,edge_noise):
    rng=np.random.default_rng(seed); out=[]
    for fam in families:
        for t in range(trials): out.append(generate_instance(f'{fam}-{t:05d}',fam,n_committed,n_masked,rng,corruption_rate,edge_noise))
    return out
