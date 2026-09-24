# Theory and optimization notes

For a graph-induced structural energy

`S(R) = sum_{(u,v) not hit by R} w_uv`,

with nonnegative pairwise violation weights, selecting a repair set that drives residual energy to zero is a minimum vertex-cover problem on the violation graph. The exact oracle is therefore exponential in the worst case and is used only for small synthetic instances.

The gain function `g(R)=S(empty)-S(R)` is monotone for this pairwise nonnegative model. The marginal gain of a node cannot increase after additional nodes have been selected, so the gain is submodular for the pure edge-cover formulation. This property should **not** be automatically transferred to the real MDLM energy: counterfactual influence estimation, higher-order interactions, thresholds, and graph reconstruction can violate the assumptions.

The repository therefore includes `run_theory_diagnostics.py` as an empirical sanity check and keeps the theorem boundary explicit.
