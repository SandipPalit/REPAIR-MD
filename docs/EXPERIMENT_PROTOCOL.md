# Paper Experiment Protocol

## 1. Core claim

The paper should test:

> Minimal structural intervention can restore consistency and enlarge the safe parallel decoding frontier with less revision than token-wise instability remasking.

## 2. Primary hypotheses

H1. Counterfactual influence predicts structural dependency better than raw confidence alone.

H2. Set-level minimum repair has lower repair regret than independent token-wise remasking under the same repair budget.

H3. Successful repair increases safe parallel frontier size.

H4. SYNAPSE-CUT improves the quality/parallelism frontier under equal compute.

H5. The additional counterfactual overhead does not erase the wall-clock benefit in the target regime.

## 3. Synthetic benchmark

Graph families:
- chain
- balanced tree
- star
- grid
- random DAG
- clustered DAG
- long-range DAG

Sizes:
- 16, 32, 64, 128, 256

Corruption:
- 1, 2, 4, 8, 16 structural interventions

Noise:
- 0, 0.01, 0.05, 0.10

Metrics:
- repair precision
- repair recall
- repair regret
- oracle gap
- structural energy reduction
- frontier gain
- runtime

For n <= 16, compute an exact minimum repair by exhaustive search. For larger n, use exact optimization only where tractable and otherwise use a documented oracle construction.

## 4. Model baselines

At minimum:
- random
- confidence/top-k
- top-2 margin
- official CoRe
- official Fast-dLLM where compatible
- official ParallelBench confidence methods
- SYNAPSE-CUT

Do not reimplement published methods if the authors provide official code unless there is a specific scientific reason.

## 5. Compute matching

Every comparison should report:

### Equal NFE
Same total model forward evaluations.

### Equal counterfactual budget
Same number of additional stress-test forwards.

### Equal wall clock
Same GPU and implementation environment.

### Equal output length
Same prompt and generation length.

## 6. Primary external benchmark

ParallelBench should be the primary parallel-decoding benchmark because it reports PBx metrics and TPS.

Recommended settings:
- complete benchmark
- batch size 1
- fixed seeds
- multiple TPS values
- three independent seeds when practical

Report:
- PB90
- PB80
- PB70
- average score
- tokens/step
- NFE
- wall time

## 7. Reasoning/code

Use:
- GSM8K
- MATH
- HumanEval
- MBPP

Report exact-match/pass@k using the benchmark's official evaluator.

## 8. Statistical reporting

For stochastic methods:
- at least 3 seeds
- mean ± standard deviation
- paired bootstrap confidence intervals for accuracy
- paired latency measurements
- Wilcoxon signed-rank test when assumptions for a parametric test are not justified

Do not report a single lucky seed as the final result.

## 9. Critical ablations

A. confidence only

B. confidence + graph

C. counterfactual singleton repair

D. greedy set repair

E. exact minimum repair on small synthetic graphs

F. repaired graph + frontier

G. full SYNAPSE-CUT

Also ablate:
- candidate K
- graph top-k
- repair budget
- structural threshold
- counterfactual metric
- revision frequency

## 10. Failure analysis

Categorize:
- under-repair
- over-repair
- false dependency
- missed dependency
- harmful remasking
- counterfactual overhead
- frontier inflation without quality preservation

## 11. Paper-standard result table

Never hard-code expected numbers. Generate tables directly from raw JSONL/CSV experiment logs.

## 12. Reproducibility checklist

Archive:
- exact commit
- model revision
- tokenizer revision
- environment lockfile
- GPU type
- CUDA version
- benchmark revision
- prompt templates
- random seeds
- generation parameters
- raw per-example outputs
- aggregate metrics
