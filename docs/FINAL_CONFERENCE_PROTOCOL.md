# SYNAPSE-CUT++ Final Conference Protocol

This repository is designed to support a reproducible empirical paper. It does **not** encode fabricated SOTA numbers and does not claim theoretical guarantees that have not been proved.

## Scientific claims the implementation can test

1. Counterfactual candidate screening can recover structurally relevant repair candidates.
2. Set-level intervention selection is harder than candidate discovery.
3. Minimum-cost structural repair and maximum parallel frontier are distinct objectives.
4. A joint repair/scheduling objective can trade small repair cost against structural depth/frontier.
5. Structural repair overhead must be counted as model compute and wall-clock time.
6. SYNAPSE-CUT does not make a sub-quadratic Transformer inference claim; the Transformer forward pass remains model-determined.

## Required experiments

### Synthetic oracle benchmark
- 7 graph families: chain, star, tree, grid, random DAG, clustered DAG, long-range DAG.
- At least 5 seeds for the final paper.
- Candidate counts: 8, 12, 16, 20, 32, 48, 64.
- Masked counts: 16, 32, 64, 128, 256.
- Corruption: 0.05, 0.10, 0.20, 0.30, 0.40.
- Noise: 0, 0.05, 0.10, 0.20, 0.40.
- Candidate K: 4, 8, 16, 32, 64.
- Report feasibility, residual violation, repair regret, candidate recall, blocker-depth reduction, frontier gain, and runtime.

### Adversarial suite
- High-confidence structurally harmful commitments.
- Low-confidence structurally harmless commitments.
- Long dependency chains.
- Weak individually useful but strongly interacting repair sets.

### Pareto repair
For small instances enumerate feasible repair sets and report the Pareto frontier over:
- repair cost,
- residual structural energy,
- blocker depth,
- frontier size.

### Real MDLM benchmark
Use the same checkpoint, tokenizer, GPU, dtype, prompt set, generation length, and seed across methods. The LLaDA reference transfer-count principle is implemented separately in `llada_official.py` and is not conflated with SYNAPSE-CUT.

Primary real-world tasks should be task-appropriate:
- WikiText-2: conditional continuation / likelihood-oriented evaluation.
- AG News: classification accuracy from constrained class outputs.
- CNN/DailyMail: ROUGE-1/2/L.
- XSum: ROUGE-1/2/L.
- Multi-News: ROUGE-1/2/L.

### Matched-compute SOTA
Run official external implementations in isolated environments when possible. Record:
- repository commit,
- checkpoint,
- CUDA/PyTorch/Transformers versions,
- GPU model,
- dtype,
- generation length,
- diffusion steps,
- seed,
- wall-clock,
- total model forwards/NFE where available,
- peak memory.

Do not mix paper-reported numbers with newly measured numbers in one table without clearly labelling the source.

## Compute accounting

For SYNAPSE-CUT++:

`total_model_forward_calls = decode_forward_calls + counterfactual_calls`.

Counterfactual batches are genuine model forwards and therefore count toward NFE/compute. Report wall-clock separately because batching and GPU utilization can make equal-NFE and equal-time conclusions differ.

## Statistical reporting

Final tables should include mean ± standard deviation and 95% bootstrap confidence intervals over examples/seeds. For paired method comparisons, align the same `(dataset, example_id, seed)` and report paired bootstrap confidence intervals for quality and latency deltas.

## Reproducibility checklist

- Fixed seeds recorded.
- Environment snapshot saved.
- Exact model identifier and implementation commit recorded.
- Exact dataset/config recorded.
- Prompt templates version-controlled.
- Raw JSONL predictions preserved.
- CSV metrics preserved.
- External baseline stdout/stderr preserved.
- No result is generated from a missing external implementation.
- CUDA peak memory reset before each method.
- Prompt positions are immutable; only generation positions may be remasked.
