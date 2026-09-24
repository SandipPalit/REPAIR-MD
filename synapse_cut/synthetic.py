from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import networkx as nx
from .graph import CounterfactualGraph
from .repair import MinimumRepairSolver
from .energy import structural_energy

@dataclass
class SyntheticResult:
    graph_family: str
    n: int
    oracle_cost: int
    greedy_cost: int
    precision: float
    recall: float
    regret: float
    oracle_gain: float
    greedy_gain: float
    initial_energy: float
    final_energy: float

def make_graph(family: str, n: int, rng: np.random.Generator):
    if family == "chain":
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from((i, i+1) for i in range(n-1))
    elif family == "star":
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from((0, i) for i in range(1, n))
    elif family == "tree":
        T = nx.balanced_tree(2, max(1, int(np.log2(max(2, n)))) )
        G = nx.DiGraph()
        G.add_nodes_from(range(min(n, len(T.nodes))))
        G.add_edges_from((u,v) for u,v in T.edges if u < n and v < n)
    elif family == "grid":
        side = int(np.sqrt(n))
        G = nx.grid_2d_graph(side, side, create_using=nx.DiGraph)
        mapping = {node:i for i,node in enumerate(G.nodes)}
        G = nx.relabel_nodes(G, mapping)
    elif family == "random_dag":
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < min(0.15, 3.0/n):
                    G.add_edge(i,j)
    else:
        raise ValueError(f"unknown family: {family}")
    return G

def graph_to_cf(G, rng, noise=0.0):
    n = len(G.nodes)
    influence = np.zeros((n,n), dtype=np.float32)
    edges = []
    for u,v in G.edges:
        # i=v changes when j=u is removed.
        w = 1.0 + float(rng.normal(0, noise))
        w = max(0.0, w)
        influence[v,u] = w
        edges.append((v,u,w))
    return CounterfactualGraph(influence, edges, list(range(n)))

def evaluate_trial(family, n, rng, noise=0.0, budget=4, threshold=1.5):
    G = make_graph(family, n, rng)
    graph = graph_to_cf(G, rng, noise)

    # Oracle is computed on the actual graph. To make the experiment nontrivial,
    # we designate a small corruption set by deleting nodes and asking the
    # solver to recover enough deletions to cross a structural threshold.
    oracle = MinimumRepairSolver(threshold=threshold, budget=budget).exact(graph)
    greedy = MinimumRepairSolver(threshold=threshold, budget=budget).greedy(graph)

    a, b = set(oracle.selected), set(greedy.selected)
    precision = len(a & b) / len(b) if b else (1.0 if not a else 0.0)
    recall = len(a & b) / len(a) if a else 1.0
    return SyntheticResult(
        family, n, len(a), len(b), precision, recall,
        len(b)-len(a), oracle.gain, greedy.gain,
        oracle.initial_energy, greedy.final_energy
    )
