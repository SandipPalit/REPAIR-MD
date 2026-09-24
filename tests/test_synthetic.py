import numpy as np
from synapse_cut.synthetic import evaluate_trial

def test_synthetic_runs():
    r = evaluate_trial("chain", 12, np.random.default_rng(0))
    assert r.n == 12
    assert 0.0 <= r.precision <= 1.0
    assert 0.0 <= r.recall <= 1.0
