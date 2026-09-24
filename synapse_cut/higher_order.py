from __future__ import annotations
from itertools import combinations
from dataclasses import dataclass
@dataclass
class HyperedgeSolution:
    selected: tuple[int,...]
    residual: float

def residual_hyperedges(hyperedges,selected):
    S=set(selected); return float(sum(w for nodes,w in hyperedges if not (S & set(nodes))))

def exact_hyperedge_repair(nodes,hyperedges,budget):
    nodes=tuple(nodes); best=(float('inf'),())
    for k in range(min(budget,len(nodes))+1):
        for c in combinations(nodes,k):
            r=residual_hyperedges(hyperedges,c)
            if r<best[0]-1e-12: best=(r,c)
    return HyperedgeSolution(tuple(best[1]),best[0])

def greedy_hyperedge_repair(nodes,hyperedges,budget):
    S=[]
    for _ in range(min(budget,len(nodes))):
        cur=residual_hyperedges(hyperedges,S)
        candidates=[n for n in nodes if n not in S]
        if not candidates: break
        n=max(candidates,key=lambda x:cur-residual_hyperedges(hyperedges,S+[x])); S.append(n)
        if residual_hyperedges(hyperedges,S)<=1e-12: break
    return HyperedgeSolution(tuple(S),residual_hyperedges(hyperedges,S))

def interaction_demo():
    # Two individually weak interventions jointly cover two higher-order constraints.
    h=[((0,2,4),1.0),((1,3,5),1.0),((0,1,6),.2)]
    return exact_hyperedge_repair(range(7),h,2),greedy_hyperedge_repair(range(7),h,2)
