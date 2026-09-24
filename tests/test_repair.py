import numpy as np
from synapse_cut.graph import CounterfactualGraph
from synapse_cut.energy import structural_energy
from synapse_cut.repair import MinimumRepairSolver

def make_graph():
    # 0 -> 1 and 0 -> 2. Removing 0 removes both edges.
    influence = np.zeros((3,3), dtype=np.float32)
    influence[1,0] = 1.0
    influence[2,0] = 1.0
    edges = [(1,0,1.0),(2,0,1.0)]
    return CounterfactualGraph(influence, edges, [0,1,2])

def test_energy():
    g = make_graph()
    assert structural_energy(g) == 2.0
    assert structural_energy(g, {1,2}) == 0.0

def test_exact_minimum_repair():
    g = make_graph()
    r = MinimumRepairSolver(0.5, 2).exact(g)
    assert r.selected == [0]
    assert r.final_energy == 0.0

def test_greedy():
    g = make_graph()
    r = MinimumRepairSolver(0.5, 2).greedy(g)
    assert r.selected == [0]
