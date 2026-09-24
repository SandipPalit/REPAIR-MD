# Final upgrade changelog

## v5 — Conference-ready consolidation

- Added a formal `DecodeConfig` and `BenchmarkConfig`.
- Added environment/GPU/package/git manifest capture.
- Added confidence-diverse high/low candidate screening.
- Added explicit masked-target blocker energy.
- Added blocker depth and dependency wavefront scheduling.
- Added repair/depth/frontier joint objective and Pareto enumeration.
- Added balanced transfer scheduling so baseline and SYNAPSE use the same initial transfer principle instead of an arbitrary fraction.
- Added separate LLaDA reference sampler primitives based on the public transfer-count formulation.
- Corrected mask-ID resolution; no longer defaults to `5` for LLaDA.
- Prompt positions are immutable; only generation positions can be remasked/unmasked.
- Counterfactual model forwards are included in total model-forward/NFE accounting.
- Added timing decomposition and GPU peak-memory reporting.
- Added task-aware real-world metrics and seed-aware raw result preservation.
- Added compute sweeps for quality-vs-NFE and quality-vs-wall-clock analysis.
- Added synthetic scaling, Pareto, adversarial, and ablation runners.
- Added paired bootstrap confidence intervals.
- Added matched-SOTA external execution bridge with stdout/stderr/manifest capture.
- Added final experiment matrix and reproducibility protocol.
- Added 18 automated tests and a conference smoke test.

## Important scientific guardrails

- No fabricated SOTA numbers.
- No claim that candidate screening is theoretically optimal.
- No claim that greedy/beam has an approximation guarantee unless separately proved.
- No sub-quadratic Transformer inference claim.
- No claim of real-model superiority until CUDA measurements are generated.
- External baselines must be executed from their exact compatible implementations/checkpoints.
