"""Small, optional helpers for the public NIAH notebook.

The model and trainer calls in the notebook use the public TokenGate API. This
module keeps dataset generation, baseline measurement, and display code out of
the walkthrough. It is not part of the training or inference implementation.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import time
from itertools import groupby
from pathlib import Path


DEMO_CITIES = ("Amman", "Hanoi", "Regina")
DEMO_DEPTHS = (0.1, 0.5, 0.9)


def niah_examples(repo_root: str | Path, doc_tokens: int = 16384,
                  cache_dir: str | Path | None = None) -> list[dict]:
    """Three distinct held-out facts at early, middle, and late depths.

    At the 16k tier, use the repository's frozen 150-row test partition. The
    generator produces the same 200 evaluation rows as the released recipe;
    no train examples are generated for this demo.
    """
    root = Path(repo_root)
    cache = Path(cache_dir) if cache_dir is not None else (
        Path("/content") if Path("/content").is_dir() else Path.cwd())
    cache.mkdir(parents=True, exist_ok=True)
    paper_tier = doc_tokens == 16384
    path = cache / f"tokengate_niah_test_{doc_tokens}_seed0.jsonl"
    if not path.exists():
        cmd = [sys.executable,
               str(root / "datasets/niah_kamradt/generate_niah_kamradt.py"),
               "--output-file", str(path), "--n-train", "0",
               "--n-test-cities", "20", "--seed", "0"]
        if not paper_tier:
            cmd += ["--test-target", str(doc_tokens), "--test-jit", "0"]
        subprocess.run(cmd, check=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if paper_tier:
        split = json.loads((root / "tokengate/datasets/eval_splits/longbench_eval_split_niah_iso_v1.json").read_text())["niah_kamradt"]
        test_ids = set(split["test"])
        assert len(split["val"]) == 50 and len(test_ids) == 150
        rows = [row for row in rows if row["id"] in test_ids]
        assert len(rows) == 150
    chosen = []
    for city, target in zip(DEMO_CITIES, DEMO_DEPTHS):
        candidates = [row for row in rows if row["city"] == city]
        if not candidates:
            raise ValueError(f"no held-out NIAH rows for {city}")
        row = min(candidates, key=lambda r: (abs(r["depth_actual"] - target), r["id"]))
        assert row["answer"] in row["body"]
        chosen.append(dict(row, target_depth=target))
    assert len({row["id"] for row in chosen}) == 3
    return chosen


def niah_prompt(tokenizer, row: dict) -> tuple[list[int], list[int]]:
    """Return (TokenGate sentinel prompt, ordinary full-context prompt)."""
    from tokengate import apply_chat_template, split_state_span
    from tokengate.datasets.tasks import get_task

    task = get_task("niah_kamradt")
    prompt = task.format_prompt("{state}\n\nQuestion: " + row["question"])
    gated = apply_chat_template(tokenizer, prompt, state=row["body"])
    plain, _, _ = split_state_span(gated)
    return gated, plain


def _decode_answer(tokenizer, ids: list[int]) -> str:
    from tokengate.generators import _strip_thinking
    return _strip_thinking(tokenizer.decode(ids, skip_special_tokens=True))


def plain_generate(model, tokenizer, prompt_ids: list[int], *,
                   max_new_tokens: int = 64, chunk_tokens: int | None = None
                   ) -> tuple[str, float]:
    """Greedy full-context generation; return (answer, prefill-to-first-token s).

    ``chunk_tokens=None`` is the unmodified model's *single-pass* prefill, the
    same reference definition as ``benchmarks/run_benchmark.py``. A chunk size
    is a memory-safe full-context T4 path, but its timing is not paper-style.
    """
    import torch
    from transformers.cache_utils import DynamicCache

    if max_new_tokens < 1 or (chunk_tokens is not None and chunk_tokens < 1):
        raise ValueError("max_new_tokens and chunk_tokens must be positive")
    if chunk_tokens and "qwen3_5" in model.config.model_type:
        # Transformers 5.5 DeltaNet reuses its recurrent state only for
        # one-token decode; multi-token incremental chunks restart that state.
        raise ValueError("Qwen3.5 needs single-pass prefill in this demo; select an A100 baseline")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    length = ids.shape[1]
    cache = DynamicCache(config=model.config) if chunk_tokens else None
    steps = chunk_tokens or length
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, length, steps):
            end = min(start + steps, length)
            kw = dict(input_ids=ids[:, start:end], past_key_values=cache,
                      use_cache=True, logits_to_keep=1)
            if chunk_tokens:
                kw.update(attention_mask=torch.ones((1, end), device=model.device, dtype=torch.long),
                          cache_position=torch.arange(start, end, device=model.device))
            out = model(**kw)
            cache = out.past_key_values
            if end == length:
                next_id = out.logits[:, -1].argmax(-1, keepdim=True)
            del out
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - start_time
        generated = []
        for step in range(max_new_tokens):
            token = next_id.item()
            if token == tokenizer.eos_token_id:
                break
            generated.append(token)
            if step == max_new_tokens - 1:
                break
            kw = dict(input_ids=next_id, past_key_values=cache,
                      use_cache=True, logits_to_keep=1)
            if chunk_tokens:
                kw.update(attention_mask=torch.ones((1, length + step + 1), device=model.device, dtype=torch.long),
                          cache_position=torch.tensor([length + step], device=model.device))
            out = model(**kw)
            cache = out.past_key_values
            next_id = out.logits[:, -1].argmax(-1, keepdim=True)
            del out
    return _decode_answer(tokenizer, generated), prefill_s


def plain_results(model, tokenizer, prompts: list[list[int]], rows: list[dict],
                  *, chunk_tokens: int | None = None) -> list[dict]:
    """Warm each shape, then score three ordinary-model calls."""
    import torch
    from tokengate.datasets.tasks import get_task

    task = get_task("niah_kamradt")
    results = []
    for prompt, row in zip(prompts, rows):
        plain_generate(model, tokenizer, prompt, chunk_tokens=chunk_tokens)
        torch.cuda.reset_peak_memory_stats()
        answer, seconds = plain_generate(model, tokenizer, prompt, chunk_tokens=chunk_tokens)
        results.append(dict(city=row["city"], depth=row["depth_actual"],
                            budget="full", answer=answer,
                            correct=task.evaluate(answer, row["answer"]),
                            prefill_s=seconds,
                            peak_gib=torch.cuda.max_memory_allocated() / 2**30))
    return results


def gate_results(model, tokenizer, prompts: list[list[int]], rows: list[dict],
                 budgets: tuple[int, ...]) -> list[dict]:
    """Call the public TokenGate generate API at full and each requested K."""
    import torch
    from tokengate import split_state_span
    from tokengate.datasets.tasks import get_task

    task = get_task("niah_kamradt")
    results = []
    for k in (None,) + budgets:
        for prompt, row in zip(prompts, rows):
            model.generate([prompt], tokenizer, max_kept_prefill_tokens=k, max_new_tokens=64)
            torch.cuda.reset_peak_memory_stats()
            out = model.generate([prompt], tokenizer, max_kept_prefill_tokens=k,
                                 max_new_tokens=64)
            _, state_start, state_end = split_state_span(prompt)
            kept_state = sum(state_start <= p < state_end for p in out.kept_pos[0])
            results.append(dict(city=row["city"], depth=row["depth_actual"],
                                budget="full" if k is None else k,
                                answer=out.answers[0],
                                correct=task.evaluate(out.answers[0], row["answer"]),
                                prefill_s=out.prefill_s,
                                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                                retained=kept_state / (state_end - state_start)))
    return results


def result_figure(plain: list[dict], gated: list[dict], *,
                  baseline_name: str = "single-pass base"):
    """Show accuracy and mean TTFT speedup versus the chosen plain baseline."""
    import matplotlib.pyplot as plt
    from statistics import mean

    base_s = mean(r["prefill_s"] for r in plain)
    budgets = list(dict.fromkeys(r["budget"] for r in gated))
    means = [mean(r["prefill_s"] for r in gated if r["budget"] == k) for k in budgets]
    accuracies = [mean(r["correct"] for r in gated if r["budget"] == k) for k in budgets]
    names = ["All" if k == "full" else f"K={k}" for k in budgets]
    fig, ax = plt.subplots(figsize=(7, 3.3))
    bars = ax.bar(names, [base_s / s for s in means], color="#296d5c")
    ax.axhline(1, color="#777", linestyle="--", linewidth=1)
    ax.set(ylabel="Time-to-first-token speedup",
           title=f"Three held-out NIAH documents · vs {baseline_name}")
    ax.bar_label(bars, labels=[f"{base_s/s:.1f}× · {a:.0%}" for s, a in zip(means, accuracies)], padding=3)
    ax.set_ylim(0, max(base_s / s for s in means) * 1.22)
    fig.tight_layout()
    return fig


def show_kept_tokens(model, tokenizer, prompt: list[int], row: dict,
                     budget: int = 100):
    """Highlight retained document spans in a scrollable notebook display."""
    from IPython.display import HTML, display
    from tokengate import split_state_span

    out = model.generate([prompt], tokenizer, max_kept_prefill_tokens=budget,
                         max_new_tokens=64)
    clean, start, end = split_state_span(prompt)
    selected = set(out.kept_pos[0])
    spans = []
    for keep, positions in groupby(range(start, end), key=lambda p: p in selected):
        piece = html.escape(tokenizer.decode([clean[p] for p in positions]))
        style = "background:#bbf7d0;color:#14532d" if keep else "color:#777"
        spans.append(f'<span style="{style}">{piece}</span>')
    print(f"{row['city']} · depth {row['depth_actual']:.0%} · "
          f"kept {sum(start <= p < end for p in selected):,}/{end-start:,} document tokens")
    print("Answer:", out.answers[0])
    display(HTML('<div style="max-height:320px;overflow:auto;white-space:pre-wrap;line-height:1.55">'
                 + "".join(spans) + "</div>"))


def training_rows(repo_root: str | Path, output_dir: str | Path,
                  n_train: int = 128, n_val: int = 8,
                  seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Build a small training pool disjoint from the selected test cities."""
    import random

    path = Path(output_dir) / f"niah_training_pool_{n_train}_{n_val}_seed{seed}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run([sys.executable,
                        str(Path(repo_root) / "datasets/niah_kamradt/generate_niah_kamradt.py"),
                        "--output-file", str(path), "--n-train", str(n_train + n_val),
                        "--n-test-cities", "1", "--test-target", "2048",
                        "--test-jit", "0", "--seed", str(seed)], check=True)
    pool = [r for r in map(json.loads, path.read_text().splitlines())
            if r["split"] == "train"]
    if len(pool) != n_train + n_val:
        raise ValueError("cached training pool has the wrong row count")
    random.Random(seed).shuffle(pool)
    def convert(rows):
        return [dict(id=r["id"], article=r["body"] + "\n\nQuestion: " + r["question"],
                     answer=r["answer"]) for r in rows]
    assert not {r["city"] for r in pool} & set(DEMO_CITIES)
    return convert(pool[n_val:]), convert(pool[:n_val])
