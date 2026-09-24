# SYNAPSE-Bench v1.0 — Final Scientific Definition

## 1. Purpose

SYNAPSE-Bench is an oracle-grounded benchmark for testing the central SYNAPSE-CUT claim independently of any language model:

> A partially committed sequence contains a sparse structural dependency topology and a small set of corrupted commitments. The goal is to identify the minimum-cost set of commitments whose removal eliminates structural inconsistency and, when those commitments act as dependency blockers, increases the safe parallel decoding frontier.

The benchmark intentionally separates **repair correctness** from **language-model quality**. LLaDA/iLLaDA/other dLLMs are evaluated only after this structural claim has been validated.

## 2. Ground-truth generative process

Each instance contains:

- `C`: committed positions, |C| = Nc.
- `M`: masked positions, |M| = Nm.
- `G_dep`: a DAG describing dependency structure.
- `B ⊂ C`: ground-truth corrupted commitments.
- `V`: structural-violation edges induced by B, with positive weights.
- `\hat I`: noisy counterfactual influence observations used by SYNAPSE-CUT.

Corrupted commitments induce violations on dependency edges incident to at least one corrupted commitment. A small controlled fraction of clean-clean spurious violations prevents trivial recovery.

The observed node-level singleton repair signal is:

`gain(j) = sum_{e incident to j} w_e + noise`.

The benchmark therefore tests whether **set-level repair** can outperform ranking by individual singleton influence when violations overlap.

## 3. Exact oracle

The oracle is NOT the ground-truth corruption set. It is the true minimum structural repair:

`R* = argmin_R |R| subject to every violation edge being covered by R.`

For the current pairwise benchmark this is the exact minimum vertex-cover problem on the violation graph. Exhaustive enumeration is used for the committed-set sizes shipped in the benchmark. The ground-truth corruption set B is retained separately for diagnostic precision/recall.

This distinction is essential: multiple different repair sets can be valid, so evaluating only exact recovery of B would incorrectly penalize alternative minimum repairs.

## 4. Graph families

Seven families are mandatory:

1. chain — maximal serial dependency depth;
2. balanced tree — hierarchical dependencies;
3. star — high fan-out, low depth;
4. grid — distributed local dependencies;
5. random DAG — heterogeneous sparse topology;
6. clustered DAG — local communities plus sparse long-range links;
7. long-range DAG — serial backbone plus skip dependencies.

## 5. Default scale

Structural-oracle validation:

- Nc = 12 committed positions
- Nm = 24 masked positions
- 200 trials/family for smoke/reproducibility
- 5,000 trials/family for final paper statistics
- corruption rate = 20%
- edge-noise σ = 0.10
- repair budget = 4
- candidate K = 8

Scaling study:

- Nc ∈ {8, 12, 16, 20}
- Nm ∈ {16, 24, 32, 64}
- corruption rate ∈ {0.10, 0.20, 0.30}
- noise σ ∈ {0, 0.05, 0.10, 0.20}
- candidate K ∈ {4, 8, 16, Nc}

Exact oracle is required for all primary reported structural results. Larger instances may use a separately labelled exact/ILP oracle implementation; heuristic solutions must never be called ground truth.

## 6. Methods

### Oracle
Exact minimum repair.

### Random
Random repair under the same budget.

### Singleton
Rank candidates by noisy singleton counterfactual gain and select top-K.

### Greedy SYNAPSE-CUT
At each iteration, recompute marginal structural repair gain on the residual violation graph and select the candidate with maximum gain until the exact structural constraint is satisfied or the repair budget is exhausted.

The language-model implementation replaces the synthetic violation oracle with batched counterfactual forward passes, but the optimization interface is identical.

## 7. Primary metrics

### Repair regret

`Regret = |R_hat| - |R*|`.

### Oracle precision / recall

Measure overlap with *any selected oracle R*** for the instance. Do not use corruption-set recall as the primary metric because alternative minimum repairs may exist.

### Corruption precision / recall

Secondary diagnostic against the planted corruption set B.

### Residual violation

`S(X^{-R_hat}) / S(X)` and the absolute residual.

### Candidate recall

Fraction of oracle repair nodes retained by the candidate-screening stage.

## 8. Topology metrics

The primary structural-depth metric is **critical committed-blocker depth**, not the raw longest path of the remaining masked-token DAG. This distinction is important: removing a committed blocker can expose a pre-existing masked-to-masked chain and therefore increase the raw masked-path length even while eliminating the very commitments that prevented parallel decoding.

Define `D_B(G)` as the maximum number of active committed blocker nodes on any dependency path ending at a masked token. A repair removes a subset of committed blockers and produces:

`BDR = D_B(before) - D_B(after)`

The parallel frontier is `F(G)`: masked positions with no remaining active dependency blocker under the benchmark's dependency criterion.

`PG = F_after - F_before`

Also report normalized versions:

`relative_BDR = BDR / max(D_B(before),1)`

`relative_PG = PG / max(F_before,1)`

Raw masked-DAG depth is retained only as a secondary diagnostic.

## 9. Causal hypothesis tests

The benchmark must explicitly test the chain:

`repair success → dependency-depth reduction → parallelism gain`.

Required analyses:

1. Compare oracle repair vs no repair.
2. Compare SYNAPSE repair vs singleton and random at equal repair budget.
3. Report paired Δdepth and Δfrontier per instance.
4. Report Spearman correlation between repair regret and topology outcomes.
5. Stratify by graph family.
6. Stratify by corruption rate and influence noise.

A stronger result is one where SYNAPSE approaches the oracle repair cost and its topology outcomes approach the oracle outcomes.

## 10. Primary paired analysis

For the primary table, restrict to instances whose exact oracle cost is at most the selected repair budget. This prevents a fixed-budget method from being compared against an infeasible oracle. The default primary setting is `Nc=12, Nm=24, oracle_cost<=4, budget=4`.

Also run a budget-sweep secondary analysis with budgets `{1,2,4,6}` and report feasible-repair rate and conditional cost regret.

## 11. Falsification criteria

SYNAPSE-CUT should NOT be promoted as a successful structural method if any of the following holds across the primary test suite:

- greedy repair has consistently high repair regret;
- candidate screening frequently excludes oracle repair nodes;
- repair does not materially reduce dependency depth on blocker-containing instances;
- frontier gain is absent after successful repair;
- topology gain comes only from over-remasking many tokens;
- singleton ranking performs equivalently to minimum-set repair.

## 12. Required figures

1. Repair regret CDF.
2. Oracle-vs-SYNAPSE repair cost.
3. Repair success vs influence noise.
4. Dependency depth before/after repair.
5. Frontier size before/after repair.
6. Frontier gain vs repair cost.
7. Candidate-K recall curve.
8. Family-wise heatmap of regret.
9. Noise × corruption-rate heatmap.
10. Scatter: repair regret vs depth reduction.

## 13. Transition to real MDLMs

Only after SYNAPSE-Bench validates the structural mechanism should the exact same interface be connected to model-generated counterfactual influence:

`I_ij = JS(p_i(.|X), p_i(.|X^{-j}))`.

The model benchmark must preserve the same metrics and add:

- NFE;
- counterfactual forward calls;
- wall-clock latency;
- peak GPU memory;
- generated quality;
- tokens/step;
- ParallelBench PB90/PB80/PB70/PB60.

Published SOTA methods should be evaluated from their official implementations whenever possible.
