#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synapse_cut.config import environment_snapshot
from synapse_cut.gguf_runner import run_gguf


def load_rows(name: str, split: str, count: int, seed: int):
    from scripts.run_real_world_benchmark import load_rows as load_dataset_rows
    return load_dataset_rows(name, split=split, n=count, seed=seed)


def prompt_for(name: str, text: str) -> str:
    if name == 'ag_news':
        return 'Classify the following news article into one of: World, Sports, Business, Sci/Tech. Return only the class name.\n\n' + text
    if name == 'wikitext2':
        return 'Continue the following text naturally. Return only the continuation.\n\n' + text
    return 'Summarize the following text concisely.\n\n' + text


def main() -> int:
    parser = argparse.ArgumentParser(description='Run quantized LLaDA GGUF inference through an external engine.')
    parser.add_argument('--executable', required=True, help='Path to diffuse-cpp diffuse-cli executable')
    parser.add_argument('--tokenizer', default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--model', required=True, help='Local .gguf model path')
    parser.add_argument('--datasets', default='ag_news')
    parser.add_argument('--samples-per-dataset', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--context-size', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--gpu-layers', type=int, default=0, help='Reserved for future GPU support; currently must be 0.')
    parser.add_argument('--entropy-exit', type=float, default=None)
    parser.add_argument('--timeout', type=float, default=None)
    parser.add_argument('--out', default='results/gguf')
    args = parser.parse_args()

    executable = Path(args.executable)
    if not executable.is_file():
        parser.error(
            f'diffuse-cli executable does not exist: {executable}. '
            'Build diffuse-cpp first with .\\scripts\\check_gguf_runtime.ps1.'
        )
    model = Path(args.model)
    if not model.is_file():
        parser.error(f'GGUF model does not exist: {model}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'environment.json').write_text(json.dumps(environment_snapshot(), indent=2), encoding='utf-8')
    rows = []
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        for dataset in [item.strip() for item in args.datasets.split(',')]:
            for example in load_rows(dataset, 'test', args.samples_per_dataset, args.seed):
                result = run_gguf(
                    str(executable),
                    str(model),
                    prompt_for(dataset, example['text']),
                    max_new_tokens=args.max_new_tokens,
                    context_size=args.context_size,
                    steps=args.steps,
                    gpu_layers=args.gpu_layers,
                    entropy_exit=args.entropy_exit,
                    timeout=args.timeout,
                    tokenizer=tokenizer,
                )
                rows.append({
                    'dataset': dataset,
                    'example_id': example['id'],
                    'model': str(model),
                    'prediction': result.text,
                    'reference': example['reference'],
                    'label': example.get('label'),
                    'elapsed_sec': result.elapsed_sec,
                    'steps': args.steps,
                    'max_new_tokens': args.max_new_tokens,
                    'context_size': args.context_size,
                    'gpu_layers': args.gpu_layers,
                    'command': json.dumps(result.command),
                })
                print(f'Completed {dataset}/{example["id"]} in {result.elapsed_sec:.1f}s: {result.text!r}', flush=True)
    except RuntimeError as exc:
        print(f'GGUF benchmark stopped: {exc}', file=sys.stderr, flush=True)
        return 2
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ['dataset'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Results written to {out / "results.csv"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())