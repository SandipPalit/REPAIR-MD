# SYNAPSE-Bench v1.0 — Primary Structural Benchmark Report

## Run

- Date: 2026-09-20
- Families: chain, tree, star, grid, random_dag, clustered_dag, long_range_dag
- Generated trials per family: 1,000
- Committed positions: 12
- Masked positions: 24
- Corruption rate: 0.15
- Influence noise σ: 0.10
- Primary repair budget: 4
- Candidate K: 8
- Primary inclusion rule: exact oracle repair cost ≤ 4
- Seed: 20260920
- Retained instances: 3,841
- Exact oracle: exhaustive minimum vertex cover over the violation graph

## Core structural result

The oracle intervention changed the critical committed-blocker topology substantially:

- blocker depth reduction: **2.751** levels on average;
- frontier gain: **+2.674** masked positions on average;
- residual violation: **0** by construction.

The no-repair condition had:

- blocker-depth reduction: 0;
- frontier change: **−1.203** positions on average.

This establishes, on the synthetic structural world, the first required causal link:

`minimum structural repair → removal of critical committed blockers → larger safe frontier`.

It is not evidence that an MDLM benefits yet; that requires the model experiment.

## Solver results

| Method | Feasible repair rate | Mean residual violation | Blocker-depth reduction | Frontier gain | Conditional cost regret |
|---|---:|---:|---:|---:|---:|
| Oracle | 100.0% | 0.000 | 2.751 | 2.674 | 0.000 |
| Greedy, K=8 | 19.8% | 1.646 | 2.552 | 2.568 | 0.037 |
| Greedy, full candidate set | 44.0% | 1.597 | 2.752 | 2.739 | 0.019 |
| Singleton, K=8 | 20.9% | 0.237 | 2.413 | 3.204 | 0.718 |
| Random | 0.9% | 4.717 | 2.279 | −0.022 | 1.088 |
| No repair | 0.0% | 7.704 | 0.000 | −1.203 | — |

## Interpretation

The current results are **not sufficient to claim SYNAPSE-CUT superiority**.

The oracle validates the structural mechanism, but the present greedy approximation is not yet a reliable oracle surrogate: with K=8 candidates it produces a fully feasible repair on only 19.8% of retained instances; using all committed candidates raises feasibility to 44.0%.

The important positive result is that, when the greedy method does find a feasible repair, its conditional cost regret is small (0.037 for K=8 and 0.019 with the full candidate set). This suggests that the principal failure mode is candidate/repair feasibility rather than consistently choosing unnecessarily large feasible repair sets.

Singleton selection achieves larger raw frontier gains but at the fixed budget of four it has much worse conditional cost regret and low feasibility. Therefore frontier gain must always be reported jointly with repair cost and feasibility; otherwise over-remasking can look artificially attractive.

## Required next scientific step

The repair objective needs a stronger estimator/solver before model-scale evaluation:

1. use pairwise counterfactual influence rather than only noisy singleton incident-weight sums;
2. score **set interactions** explicitly;
3. add a feasibility-aware candidate expansion step;
4. compare greedy, beam search, local search, and exact oracle on the same instances;
5. evaluate the frontier at matched repair cost, not merely at a fixed maximum budget;
6. retain the exact oracle as the scientific reference.

Only after this structural gap is reduced should the model-level LLaDA/iLLaDA experiment be treated as the main paper result.

## Model benchmark status

No LLaDA/iLLaDA CUDA benchmark is reported in this run. The available execution environment is CPU-only (`torch 2.10.0+cpu`, CUDA unavailable). Reporting model numbers without actually loading the weights and running the benchmark would be scientifically invalid.

The external benchmark protocol is nevertheless specified for direct execution on a CUDA machine. ParallelBench currently supports model wrappers and saves evaluation results locally, with PB90/PB80/PB70/PB60 analysis. The official DAPD repository also provides ParallelBench and lm-eval evaluation runners. See the repository's SOTA integration documentation.
