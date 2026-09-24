import numpy as np
from synapse_cut.graph import CounterfactualGraph
from synapse_cut.frontier import safe_frontier

def test_frontier():
    g = CounterfactualGraph(
        np.array([[0,0],[0.05,0]], dtype=np.float32),
        [(1,0,0.05)],
        [0,1]
    )
    assert set(safe_frontier(g, [0,1], 0.1)) == {0,1}
