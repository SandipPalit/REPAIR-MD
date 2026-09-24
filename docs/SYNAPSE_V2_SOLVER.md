# SYNAPSE-CUT v2: Interaction-Aware Minimum Repair

## Scientific correction

`repair_regret` is now defined **only for feasible repairs**:

\[
R= C(\hat R)-C(R^*)
\]

when the selected set satisfies the structural repair constraint. Infeasible selections have `repair_regret = NaN` and are reported through `infeasible_rate`.

## Solvers

- `oracle`: exact minimum repair for the pairwise benchmark (minimum vertex cover over violation edges).
- `singleton`: score-only top-K baseline.
- `greedy`: recomputed set-level marginal coverage.
- `beam`: look-ahead beam search over repair sets.
- `interaction`: pair-seeded set-level search; explicitly evaluates joint residual energy, so it does not assume singleton additivity.
- `local`: greedy initialization followed by 1-for-1 swap local search.
- `random`, `none`: controls.

## Primary gate

The next research gate is the budget × candidate-K ablation:

`B={1,2,4,6,8}` and `K={2,4,8,16,all}`.

The desired evidence is a reduction in oracle gap/infeasibility without obtaining frontier gains merely by over-remasking.

## Interpretation

The synthetic benchmark validates the structural mechanism independently of a language model. It does **not** establish that the same dependency/violation process is faithful to a real MDLM. Real LLaDA/iLLaDA experiments remain a separate stage.
