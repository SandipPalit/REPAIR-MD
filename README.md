# SYNAPSE-CUT — conference-ready research codebase

**Counterfactual Minimum Structural Intervention for Parallel Masked Diffusion Decoding**

This release implements the complete research workflow requested for a top-conference study:

- SYNAPSE-Bench structural oracle validation
- robustness sweeps over corruption/noise/budget/candidate-K/seeds
- adversarial cases for confidence-vs-structure and interaction effects
- empirical theory diagnostics with explicit theorem boundaries
- batched counterfactual evaluation
- graph/repair/frontier timing accounting
- real LLaDA/iLLaDA masked-suffix smoke/evaluation runner
- confidence, entropy, attention-centrality and random baselines
- SYNAPSE-CUT greedy/beam/interaction/local solvers
- sequence-length scaling harness
- reproducibility/environment snapshots
- no fabricated external benchmark/SOTA results

## Install

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Tests

```bash
pytest -q
```

## Synthetic robustness

```bash
python scripts/run_synthetic_robustness.py --trials 100 --seeds 1,2,3
python scripts/run_adversarial_bench.py
python scripts/run_theory_diagnostics.py --trials 50
python scripts/run_scaling.py
```

## Real LLaDA/iLLaDA

Requires a CUDA environment and access to the model weights:

```bash
python scripts/run_real_decode.py \
  --model GSAI-ML/iLLaDA-8B-Base \
  --prompt "Explain why dependency-aware parallel decoding can reduce diffusion decoding latency." \
  --methods confidence,entropy,attention,synapse_cut \
  --steps 32 --max-new-tokens 128 --device cuda
```

This runner is a **generic research adapter**, not a claim that its masking schedule exactly reproduces every official LLaDA/iLLaDA sampler revision. For final paper numbers, use the exact official sampler for the model revision and expose its pre-unmask logits through the adapter.

## Quantized GGUF LLaDA

For machines with approximately 8 GB system RAM and 4 GB VRAM, use a compatible
LLaDA GGUF checkpoint with an external `diffuse-cpp` or `llama.cpp` executable.
The GGUF runner does not load Transformers weights and therefore avoids the
disk-offload path used by the full-precision runner:

```powershell
python scripts/run_gguf_benchmark.py `
  --executable C:\path\to\diffuse-cli.exe `
  --model C:\models\llada-8b-instruct-q3_k_s.gguf `
  --gpu-layers 0 --context-size 2048 --steps 16 `
  --datasets ag_news --samples-per-dataset 1
```

The current diffuse-cpp integration is CPU-only. The existing
`run_real_world_benchmark.py` remains the token-logit research path for
Transformers models; text-only GGUF engines cannot run its confidence,
attention, or SYNAPSE-CUT decoders without an engine API that exposes logits.

To build diffuse-cpp on Windows, install CMake and the Visual Studio C++
build tools, open a Developer PowerShell, then run:

```powershell
.\scripts\check_gguf_runtime.ps1
```

## ParallelBench / external SOTA

The repository intentionally does not re-create or fabricate third-party benchmark results. Use official implementations for DAPD, CoRe, DEMASK, DOS, Fast-dLLM++ and other applicable methods. Record their commit and commands. `scripts/run_benchmark.py` can record an external command without pretending that it executed successfully.

## Important accounting rule

Counterfactual batches are model forwards. Total inference cost must include:

`model_forward_time + counterfactual_time + graph_time + solver_time`.

The code does not claim sub-quadratic Transformer inference. Sparse graph construction can be `O(nk)` for fixed top-k sparsity; the Transformer forward remains model-dependent.

## Scientific boundary

The synthetic exact oracle is exact for the benchmark's pairwise nonnegative violation graph. It is not the ground-truth causal structure of an MDLM. Real-model validation is mandatory before making inference/SOTA claims.
