import numpy as np
from synapse_cut.graph import CounterfactualGraph
from synapse_cut.objective import evaluate_state,pareto_frontier
from synapse_cut.scheduler import build_dependency_wavefront
from synapse_cut.config import DecodeConfig

def test_masked_blocker_and_frontier():
 g=CounterfactualGraph(np.zeros((4,4),np.float32),[(3,0,.5),(2,1,.2)], [0,1],[2,3])
 m=evaluate_state(g,[],.1); assert m.blocker_energy>.6 and m.frontier==0
 m2=evaluate_state(g,[0],.1); assert m2.frontier>=1

def test_wavefront():
 g=CounterfactualGraph(np.zeros((4,4),np.float32),[(3,0,.5),(2,1,.2)], [0,1],[2,3])
 w=build_dependency_wavefront(g,[2,3],.1); assert len(w.frontier)==0; assert w.depth>=1

def test_pareto_and_config():
 g=CounterfactualGraph(np.zeros((3,3),np.float32),[(2,0,.9),(2,1,.2)], [0,1],[2])
 p=pareto_frontier(g,[0,1],2); assert p
 assert DecodeConfig().candidate_low_k>0
