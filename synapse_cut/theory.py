
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from .repair_bench import residual_energy
@dataclass
class SetFunctionDiagnostics:
    monotone: bool
    submodular: bool
    supermodular: bool
    max_monotonicity_violation: float
    max_submodularity_violation: float
    max_supermodularity_violation: float

def gain(edges,S,u): return residual_energy(edges,S)-residual_energy(edges,set(S)|{u})
def diagnose(nodes,edges,tol=1e-10):
    nodes=list(nodes); mono=True; sub=True; sup=True; mv=sv=vv=0.
    # Exhaustive small-set test; this is an empirical diagnostic, not a theorem.
    for r in range(len(nodes)+1):
        from itertools import combinations
        for A in combinations(nodes,r):
            A=set(A)
            for u in nodes:
                if u in A: continue
                g=gain(edges,A,u)
                if g < -tol: mono=False; mv=max(mv,-g)
                for Btuple in combinations([x for x in nodes if x not in A and x!=u], max(0,min(len(nodes)-len(A)-1,1))):
                    B=A|set(Btuple)
                    if gain(edges,A,u)+tol < gain(edges,B,u): sub=False; sv=max(sv,gain(edges,B,u)-gain(edges,A,u))
                    if gain(edges,A,u) > gain(edges,B,u)+tol: sup=False; vv=max(vv,gain(edges,A,u)-gain(edges,B,u))
    return SetFunctionDiagnostics(mono,sub,sup,mv,sv,vv)

def vertex_cover_special_case(nodes,edges):
    """For nonnegative pairwise violation edges, repair is weighted vertex cover.

    The oracle is exact only for the specified finite pairwise graph model;
    real MDLM structural consistency need not reduce to vertex cover.
    """
    return {'reduction':'minimum vertex cover on pairwise violation graph','guarantee':'exact under benchmark definition'}
