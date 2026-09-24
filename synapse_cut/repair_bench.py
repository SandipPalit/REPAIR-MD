from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RepairSolution:
    selected: tuple[int, ...]
    residual: float
    gain: float


def residual_energy(edges, selected: Iterable[int]) -> float:
    s = set(selected)
    return float(sum(w for u, v, w in edges if u not in s and v not in s))


def marginal_gain(edges, selected: Iterable[int], node: int) -> float:
    s = set(selected)
    before = residual_energy(edges, s)
    after = residual_energy(edges, s | {node})
    return before - after


def exact_repair(nodes: Sequence[int], edges, budget: int, threshold: float = 1e-8, milp_limit_nodes: int = 24) -> RepairSolution:
    """Exact minimum-cardinality vertex-cover repair when tractable.

    For small candidate sets, exhaustive search gives a transparent oracle. For larger
    sets, scipy.milp solves the same binary cover formulation without pretending that
    the problem is polynomial-time. If MILP is unavailable/fails, we fall back to a
    budget-bounded search and mark the returned solution only through its residual.
    """
    nodes=tuple(nodes); active_edges=[(u,v,w) for u,v,w in edges if w>threshold]
    initial=residual_energy(active_edges,())
    if initial<=threshold:return RepairSolution((),initial,0.)
    # Exact enumeration for the small oracle benchmark.
    if len(nodes)<=milp_limit_nodes:
        for k in range(1,min(budget,len(nodes))+1):
            for combo in combinations(nodes,k):
                r=residual_energy(active_edges,combo)
                if r<=threshold:return RepairSolution(tuple(combo),r,initial-r)
    # Exact binary minimum vertex cover via MILP. This is still exponential in the
    # worst case; it is an oracle, not an efficiency claim.
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        idx={n:i for i,n in enumerate(nodes)}; m=len(active_edges)
        if m:
            A=np.zeros((m,len(nodes)),float)
            for r,(u,v,w) in enumerate(active_edges):
                if u in idx:A[r,idx[u]]=1
                if v in idx:A[r,idx[v]]=1
            res=milp(c=np.ones(len(nodes)),integrality=np.ones(len(nodes)),bounds=Bounds(0,1),constraints=LinearConstraint(A,1,np.inf),options={'time_limit':120})
            if res.success and res.x is not None:
                sel=tuple(nodes[i] for i,x in enumerate(res.x) if x>.5)
                if len(sel)<=budget:
                    r=residual_energy(active_edges,sel); return RepairSolution(sel,r,initial-r)
        # If the unconstrained cover exceeds the repair budget, find the best budget set
        # only when the candidate space is still manageable.
    except Exception:
        pass
    k=min(budget,len(nodes))
    if k==0:return RepairSolution((),initial,0.)
    if len(nodes)<=28:
        best=min(combinations(nodes,k),key=lambda c:residual_energy(active_edges,c)); r=residual_energy(active_edges,best); return RepairSolution(tuple(best),r,initial-r)
    # Explicit heuristic fallback for large synthetic scaling. It is not labelled oracle.
    sol=greedy_repair(nodes,active_edges,budget)
    return sol


def greedy_repair(nodes: Sequence[int], edges, budget: int, scores=None) -> RepairSolution:
    selected: list[int] = []
    remaining = set(nodes)
    initial = residual_energy(edges, ())
    while remaining and len(selected) < budget:
        current = residual_energy(edges, selected)
        if current <= 1e-8:
            break
        best = max(remaining, key=lambda n: (marginal_gain(edges, selected, n),
                                             (scores or {}).get(n, 0.0), -n))
        selected.append(best)
        remaining.remove(best)
    final = residual_energy(edges, selected)
    return RepairSolution(tuple(selected), final, initial - final)


def beam_repair(nodes: Sequence[int], edges, budget: int, beam_width: int = 16, scores=None) -> RepairSolution:
    """Look-ahead set search. Unlike greedy, retains multiple partial repair hypotheses."""
    nodes = tuple(nodes)
    initial = residual_energy(edges, ())
    if initial <= 1e-8:
        return RepairSolution((), initial, 0.0)
    beam: list[tuple[int, ...]] = [()]
    for depth in range(min(budget, len(nodes))):
        candidates: dict[tuple[int, ...], float] = {}
        for state in beam:
            used = set(state)
            for n in nodes:
                if n in used:
                    continue
                nxt = tuple(sorted((*state, n)))
                candidates[nxt] = residual_energy(edges, nxt)
        ranked = sorted(candidates.items(), key=lambda kv: (kv[1], -sum((scores or {}).get(n, 0.0) for n in kv[0])))
        beam = [s for s, _ in ranked[:beam_width]]
        feasible = [s for s in beam if residual_energy(edges, s) <= 1e-8]
        if feasible:
            best = min(feasible, key=len)
            return RepairSolution(best, 0.0, initial)
    best = min(beam, key=lambda s: (residual_energy(edges, s), len(s)))
    r = residual_energy(edges, best)
    return RepairSolution(best, r, initial - r)


def interaction_repair(nodes: Sequence[int], edges, budget: int, beam_width: int = 8, pair_width: int = 32) -> RepairSolution:
    """Pair-aware beam search using exact set-level energy, not additive singleton scores."""
    nodes = tuple(nodes)
    initial = residual_energy(edges, ())
    if initial <= 1e-8:
        return RepairSolution((), initial, 0.0)
    # Seed with the strongest pairwise repairs, then expand with set-level marginal gain.
    pairs = []
    for a, b in combinations(nodes, 2):
        r = residual_energy(edges, (a, b))
        pairs.append(((a, b), r))
    pairs.sort(key=lambda x: x[1])
    beam = [s for s, _ in pairs[:pair_width]] if budget >= 2 else [()]
    if budget == 1:
        return greedy_repair(nodes, edges, 1)
    feasible = [s for s in beam if residual_energy(edges, s) <= 1e-8]
    if feasible:
        best = min(feasible, key=len)
        return RepairSolution(best, 0.0, initial)
    for _ in range(2, min(budget, len(nodes))):
        expanded = {}
        for state in beam:
            used = set(state)
            for n in nodes:
                if n in used:
                    continue
                nxt = tuple(sorted((*state, n)))
                expanded[nxt] = residual_energy(edges, nxt)
        beam = [s for s, _ in sorted(expanded.items(), key=lambda kv: (kv[1], len(kv[0])))[:beam_width]]
        feasible = [s for s in beam if residual_energy(edges, s) <= 1e-8]
        if feasible:
            best = min(feasible, key=len)
            return RepairSolution(best, 0.0, initial)
    best = min(beam, key=lambda s: (residual_energy(edges, s), len(s)))
    r = residual_energy(edges, best)
    return RepairSolution(best, r, initial - r)


def local_search_repair(nodes: Sequence[int], edges, budget: int, scores=None, restarts: int = 8) -> RepairSolution:
    """Greedy initialization followed by 1-for-1 swaps that reduce residual energy."""
    seed = greedy_repair(nodes, edges, budget, scores)
    best = seed
    node_set = set(nodes)
    for start in range(restarts):
        if start == 0:
            current = list(seed.selected)
        else:
            # deterministic diversified starts: score order with cyclic offset
            ordered = sorted(nodes, key=lambda n: ((scores or {}).get(n, 0.0), -n), reverse=True)
            off = start % max(1, len(ordered))
            rotated = ordered[off:] + ordered[:off]
            current = rotated[:min(budget, len(rotated))]
        current = list(dict.fromkeys(current))
        cur_r = residual_energy(edges, current)
        improved = True
        while improved:
            improved = False
            for i, old in enumerate(list(current)):
                for new in sorted(node_set - set(current)):
                    trial = current.copy(); trial[i] = new
                    r = residual_energy(edges, trial)
                    if r + 1e-12 < cur_r:
                        current, cur_r = trial, r
                        improved = True
                        break
                if improved:
                    break
        sol = RepairSolution(tuple(sorted(current)), cur_r, residual_energy(edges, ()) - cur_r)
        if (sol.residual, len(sol.selected)) < (best.residual, len(best.selected)):
            best = sol
        if best.residual <= 1e-8:
            break
    return best
