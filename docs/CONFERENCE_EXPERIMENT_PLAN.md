# Conference-grade experiment plan

SYNAPSE-CUT is not considered validated by synthetic results alone. The release therefore separates **structural validation** from **real MDLM validation** and refuses to manufacture external benchmark results.

## Claims
1. Structural: committed tokens can create dependency bottlenecks.
2. Algorithmic: sparse counterfactual influence supports near-minimum set repair under an explicit estimated structural energy.
3. Inference: structural repair can improve the quality/latency/parallelism trade-off of masked diffusion decoding.

## Required real-model comparisons
- Standard confidence MDLM
- entropy remasking
- attention-centrality remasking
- random remasking
- SYNAPSE-CUT greedy/beam/interaction/local
- external DAPD/CoRe/DEMASK/DOS/Fast-dLLM++ implementations when compatible with the exact model and protocol

External methods are not reimplemented by this repository unless their licenses and interfaces permit it. Use their official code and record the exact commit/command.

## Compute-matched views
Report quality vs wall time, quality vs NFE, quality vs FLOPs when available, and frontier size vs remasking. Never report only NFE.

## Counterfactual accounting
`BatchedCounterfactualEstimator` evaluates multiple counterfactual masks in one model batch. Count the original forward, every counterfactual batch, every post-repair forward, and all solver/graph time. A batched forward is still a model forward and must not be hidden.

## Theoretical honesty
The exact oracle in SYNAPSE-Bench is exact only for the finite pairwise violation-graph formulation. `run_theory_diagnostics.py` empirically checks monotonicity/submodularity/supermodularity on small instances; it does not turn empirical behavior into a theorem.
