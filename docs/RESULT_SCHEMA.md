# Result schema

Each generation row should contain:

```json
{
  "method": "synapse_cut",
  "model": "GSAI-ML/iLLaDA-8B-Base",
  "benchmark": "parallelbench",
  "task": "waiting_line_copy",
  "example_id": "0",
  "seed": 1234,
  "correct": 1,
  "nfe": 64,
  "counterfactual_calls": 8,
  "remasked_tokens": 3,
  "repair_events": 2,
  "repair_cost": 3,
  "avg_tokens_per_step": 8.1,
  "wall_time_sec": 4.21,
  "peak_memory_mb": 15432,
  "frontier_gain_sum": 12
}
```

Raw generated text should be stored separately so that benchmark evaluators
can be rerun without regenerating.
