#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synapse_cut.config import DecodeConfig,environment_snapshot
from synapse_cut.graph import CounterfactualGraph
from synapse_cut.objective import evaluate_state,joint_objective
from synapse_cut.scheduler import build_dependency_wavefront
import numpy as np

def main():
 g=CounterfactualGraph(np.zeros((5,5),np.float32),[(2,0,.5),(3,1,.7),(4,2,.9)], [0,1,2],[3,4])
 m=evaluate_state(g,[],.1); w=build_dependency_wavefront(g,[3,4],.1); assert m.blocker_depth>=1 and len(w.waves)>=1 and joint_objective(g,[0])>=0
 print('Conference smoke test: PASS'); print('DecodeConfig:',DecodeConfig()); print('Environment keys:',sorted(environment_snapshot().keys()))
if __name__=='__main__':main()
