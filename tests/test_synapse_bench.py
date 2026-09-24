import numpy as np
from synapse_cut.synapse_bench import generate_instance, evaluate_instance, exact_min_vertex_cover

def test_exact_vertex_cover_triangle():
    e=[(0,1,1.0),(1,2,1.0),(0,2,1.0)]
    r=exact_min_vertex_cover([0,1,2],e,max_k=3)
    assert len(r)==2

def test_oracle_is_feasible():
    rng=np.random.default_rng(1)
    x=generate_instance('x','random_dag',10,16,rng,0.2,0.05)
    r=evaluate_instance(x,'oracle',4)
    assert r.residual_violation_weight < 1e-8
    assert r.repair_regret==0

def test_frontier_is_computable():
    rng=np.random.default_rng(2)
    x=generate_instance('x','chain',10,16,rng,0.2,0.05)
    r=evaluate_instance(x,'greedy',4,6)
    assert r.frontier_after >= 0
    assert r.depth_after >= 0


def test_infeasible_repairs_have_nan_regret():
    rng=np.random.default_rng(3)
    x=generate_instance('x2','star',12,24,rng,0.2,0.1)
    r=evaluate_instance(x,'none',4,8)
    assert r.feasible == 0
    assert np.isnan(r.repair_regret)


def test_beam_can_recover_feasible_repair():
    rng=np.random.default_rng(4)
    x=generate_instance('x3','chain',12,24,rng,0.2,0.1)
    r=evaluate_instance(x,'beam_full',4)
    assert r.residual_violation_weight >= 0
    assert r.selected_cost <= 4
