import torch, numpy as np
from synapse_cut.graph import InfluenceEstimator
from synapse_cut.repair_bench import residual_energy,greedy_repair,interaction_repair
from synapse_cut.theory import diagnose

def test_graph_build():
 est=InfluenceEstimator('js',2); a=torch.zeros(1,4,5); b=a.clone(); b[0,1,0]=2
 g=est.build(a,{0:b},[0],[1,2,3]); assert g.influence.shape==(4,4); assert len(g.edges)>=0

def test_pairwise_energy_is_monotone():
 e=[(0,1,1.),(1,2,2.),(0,2,.5)]; assert residual_energy(e,{0})<=residual_energy(e,set())
 d=diagnose(range(3),e); assert d.monotone

def test_interaction_solver_respects_budget():
 e=[(0,2,1.),(1,2,1.),(0,3,1.),(1,3,1.)]; r=interaction_repair(range(4),e,2,8); assert len(r.selected)<=2 and r.residual<=1e-8

def test_batched_counterfactual_call_count():
 from synapse_cut.graph import BatchedCounterfactualEstimator
 class M:
  def __call__(self,x): return type('O',(),{'logits':torch.nn.functional.one_hot(x%5,5).float()*2})()
 ids=torch.tensor([[1,2,3,4]])
 e=BatchedCounterfactualEstimator(M(),0,batch_size=2)
 base,g,c=e.estimate(ids,[0,1,2],[0,1,2,3]); assert c==3
