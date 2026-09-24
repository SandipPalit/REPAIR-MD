from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time


@dataclass
class GGUFResult:
    text: str
    elapsed_sec: float
    command: list[str]
    returncode: int


def run_gguf(
    executable: str,
    model: str,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    context_size: int = 2048,
    steps: int = 16,
    gpu_layers: int = 0,
    entropy_exit: float | None = None,
    timeout: float | None = None,
    tokenizer=None,
) -> GGUFResult:
    """Run a LLaDA GGUF through diffuse-cpp's diffuse-cli executable."""
    name=Path(executable).name.lower()
    if 'llama' in name and 'diffuse' not in name:
        raise RuntimeError('llama.cpp does not support LLaDA diffusion sampling; pass diffuse-cli instead.')
    if tokenizer is None:
        raise RuntimeError('A LLaDA tokenizer is required for diffuse-cpp input tokenization.')
    input_tokens=tokenizer.encode(prompt,add_special_tokens=True)
    if len(input_tokens)+max_new_tokens>context_size:
        raise RuntimeError(f'Prompt plus generation exceeds --context-size {context_size}.')
    if gpu_layers:
        raise RuntimeError('The current diffuse-cpp integration supports CPU execution only; use --gpu-layers 0.')
    command=[executable,'-m',model,'--tokens',','.join(map(str,input_tokens)),
             '-n',str(max_new_tokens),'-s',str(steps),'-t','1',
             '--remasking','entropy_exit','--entropy-threshold',str(entropy_exit or 1.5)]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f'GGUF engine not found: {executable!r}. Install diffuse-cpp or llama.cpp, '
            'or pass the full path to its executable.'
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'GGUF inference timed out after {timeout}s.') from exc
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(
            f'GGUF engine failed with exit code {completed.returncode}: {diagnostic}'
        )
    lines=completed.stdout.strip().splitlines()
    token_line=next((line.strip() for line in reversed(lines) if line.strip() and all(part.strip().lstrip('-').isdigit() for part in line.split(','))),None)
    if token_line is None:
        raise RuntimeError('diffuse-cli completed without generated token IDs.')
    token_ids=[int(part) for part in token_line.split(',')]
    text=tokenizer.decode(token_ids[len(input_tokens):],skip_special_tokens=True)
    return GGUFResult(text, elapsed, command, completed.returncode)