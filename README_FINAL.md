# SYNAPSE-CUT++ — Final Conference-Ready Research Codebase

SYNAPSE-CUT++ treats masked-diffusion decoding as **counterfactual structural repair followed by dependency-aware scheduling**.

The implementation contains:

- sparse counterfactual dependency graphs;
- confidence-diverse candidate screening;
- minimum-cost set repair;
- greedy, beam, interaction, and local solvers;
- adaptive/wavefront scheduling;
- repair/depth/frontier joint objective;
- Pareto repair analysis;
- adversarial structural cases;
- synthetic scaling and robustness experiments;
- real LLaDA five-dataset evaluation;
- task-aware metrics;
- equalized balanced transfer scheduling;
- LLaDA reference sampler primitives;
- exact model-forward/NFE accounting including counterfactual forwards;
- GPU peak-memory and timing decomposition;
- matched-compute external SOTA bridge;
- paired bootstrap confidence intervals;
- environment/reproducibility manifests;
- conference experiment orchestration and smoke tests.

## First commands

```powershell
pip install -r requirements.txt
python scripts/smoke_conference.py
```

Synthetic validation:

```powershell
python scripts/run_ablation_suite.py --trials 1000
python scripts/run_adversarial_suite.py
python scripts/run_pareto_benchmark.py --trials 100
python scripts/run_scaling.py --trials 100
```

CUDA pilot:

```powershell
python scripts/run_real_world_benchmark.py `
  --model GSAI-ML/LLaDA-8B-Base `
  --device cuda `
  --dtype bfloat16 `
  --steps 32 `
  --max-new-tokens 128 `
  --samples-per-dataset 10 `
  --seeds 1234
```

Analysis:

```powershell
python scripts/analyze_conference.py --input results/real_world/results.csv
```

External baselines:

```powershell
python scripts/run_matched_sota.py --execute
```

The external commands in `configs/sota_methods.csv` are placeholders and must be replaced by the exact commands/checkpoints of the official implementations in the target environment. This is deliberate: the package never fabricates SOTA results.
