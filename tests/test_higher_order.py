from synapse_cut.higher_order import interaction_demo

def test_higher_order_oracle_and_greedy_are_valid():
    exact,greedy=interaction_demo()
    assert exact.residual >= 0 and greedy.residual >= 0
    assert len(exact.selected)<=2 and len(greedy.selected)<=2
