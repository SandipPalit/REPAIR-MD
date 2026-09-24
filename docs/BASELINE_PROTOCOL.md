# Baseline protocol

All local baselines receive the same model, tokenizer, prompt set, seed, output length, hardware, and maximum decoding steps.

- `confidence`: lowest-information-risk confidence frontier.
- `entropy`: highest predictive entropy among currently masked positions.
- `attention`: attention-centrality selection if model attentions are available; otherwise explicitly falls back to confidence.
- `random`: fixed-seed random control.
- `synapse_cut`: counterfactual graph + repair + safe frontier.

External baselines must be run from their official repositories. Record commit, command, model revision, and any model-specific generation semantics in `results/metadata.json`.
