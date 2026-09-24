from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional
from .energy import structural_energy, marginal_energy_reduction
from .graph import CounterfactualGraph

@dataclass
class RepairResult:
    selected: list[int]
    initial_energy: float
    final_energy: float
    cost: float
    gain: float
    optimal: bool = False

class MinimumRepairSolver:
    def __init__(self, threshold: float, budget: int):
        self.threshold = threshold
        self.budget = budget

    def exact(self, graph: CounterfactualGraph) -> RepairResult:
        nodes = list(graph.committed)
        initial = structural_energy(graph)
        if initial <= self.threshold:
            return RepairResult([], initial, initial, 0.0, 0.0, True)

        best = None
        for k in range(1, min(self.budget, len(nodes)) + 1):
            for combo in combinations(nodes, k):
                active = set(nodes) - set(combo)
                e = structural_energy(graph, active)
                if e <= self.threshold:
                    best = combo
                    break
            if best is not None:
                break

        if best is None:
            best = tuple(nodes[:self.budget])

        final = structural_energy(graph, set(nodes) - set(best))
        return RepairResult(list(best), initial, final, float(len(best)),
                            initial - final, True)

    def greedy(self, graph: CounterfactualGraph) -> RepairResult:
        nodes = set(graph.committed)
        initial = structural_energy(graph, nodes)
        selected = []

        while nodes and len(selected) < self.budget:
            current = structural_energy(graph, nodes)
            if current <= self.threshold:
                break

            best_node = max(nodes, key=lambda n: marginal_energy_reduction(graph, nodes, n))
            selected.append(best_node)
            nodes.remove(best_node)

        final = structural_energy(graph, nodes)
        return RepairResult(selected, initial, final,
                            float(len(selected)),
                            initial - final, False)
