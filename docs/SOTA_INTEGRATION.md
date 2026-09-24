# SOTA Integration

This repository deliberately separates the scientific implementation from
published external repositories.

## CoRe

Use the authors' official implementation for the comparison rather than
copying a partial reimplementation.

The current CoRe repository provides:
- low-confidence baseline
- top-k margin
- random
- CoRe
- compute-matched remasking controls
- GSM8K, HumanEval, Minerva-MATH, BBH and MBPP evaluation.

Run the official implementation with the same:
- model
- seed
- output length
- diffusion steps
- candidate budget
- GPU.

Record its raw output into `external/core/`.

## Fast-dLLM

Use the official repository for its supported models and acceleration
configuration. Report its actual wall-clock behavior separately from a
scheduler-only comparison.

## ParallelBench

ParallelBench is the primary structured parallel-decoding evaluation suite.
Register SYNAPSE-CUT as an unmasking policy in the model wrapper, then run the
official `pb eval` and `pb analyze` commands.

## Why external official code?

Published SOTA systems often contain implementation-specific optimizations
such as KV-cache reuse. A paper comparison must not accidentally compare:
"our optimized implementation" against "their algorithmic description."

For each baseline, distinguish:
1. algorithmic scheduler comparison,
2. end-to-end implementation comparison.
