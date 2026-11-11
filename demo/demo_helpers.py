"""Small, optional helpers for the public NIAH notebook.

The model and trainer calls in the notebook use the public TokenGate API. This
module keeps dataset generation, baseline measurement, and display code out of
the walkthrough. It is not part of the training or inference implementation.
"""

from __future__ import annotations

import base64
import html
import io
import json
import subprocess
import sys
import time
from itertools import groupby
from pathlib import Path


DEMO_CITIES = ("Amman", "Hanoi", "Regina")
DEMO_DEPTHS = (0.1, 0.5, 0.9)
DEMO_CACHE_VERSION = 2


def prepare_runtime(source: str | Path, gpu_runtime: str, model_name: str) -> None:
    """Install notebook dependencies and check its selected GPU."""
    import logging
    import os
    import warnings

    import torch

    assert torch.cuda.is_available(), "Choose a GPU runtime first."
    assert gpu_runtime in torch.cuda.get_device_name(0), (
        f"Select a {gpu_runtime} GPU in Colab: Runtime → Change runtime type.")
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="You are sending unauthenticated requests to the HF Hub.*")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                    str(source), "transformers==5.5.0", "matplotlib",
                    "ipywidgets>=8"], check=True)
    if model_name == "Qwen3.5-2B":
        from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
        if not is_fast_path_available:
            logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").setLevel(logging.ERROR)
            print("Qwen3.5: PyTorch DeltaNet fallback is active; timings include it.")


def load_sft_baseline(base_model: str, checkpoint_id: str,
                      checkpoint_file: str, *, revision: str, dtype):
    """Load and fold the released full-prefill LoRA baseline for this demo."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM
    from tokengate.e2e_gate.gate_module.lora import (
        apply_lora_to_layers, load_lora_state_dict,
        merge_adapters_into_layers, resolve_all_adapter_layers,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype, attn_implementation="sdpa").cuda().eval()
    layers = resolve_all_adapter_layers(model)
    apply_lora_to_layers(model, layers, rank=16, alpha=32,
                         targets=["q_proj", "v_proj"])
    adapter = torch.load(hf_hub_download(checkpoint_id, checkpoint_file,
                                         revision=revision),
                         map_location="cpu", weights_only=True)
    load_lora_state_dict(model, adapter)
    merge_adapters_into_layers(model, layers)
    return model


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


def niah_prompt_text(row: dict) -> str:
    """Format a NIAH question with the public API's {state} placeholder."""
    from tokengate.datasets.tasks import get_task

    task = get_task("niah_kamradt")
    return task.format_prompt("{state}\n\nQuestion: " + row["question"])


def niah_needle(row: dict) -> str:
    """Return the exact sentence planted in this NIAH document."""
    needle = f"The best thing to do in {row['city']} is {row['answer']}."
    if needle not in row["body"]:
        raise ValueError("The expected NIAH needle is missing from this document")
    return needle


def niah_prompt(tokenizer, row: dict) -> tuple[list[int], list[int]]:
    """Return (TokenGate sentinel prompt, ordinary full-context prompt)."""
    from tokengate import apply_chat_template, split_state_span

    prompt = niah_prompt_text(row)
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
    """Warm each shape, then score the selected ordinary-model prompts."""
    import torch
    from tokengate.datasets.tasks import get_task

    task = get_task("niah_kamradt")
    results = []
    for prompt, row in zip(prompts, rows):
        plain_generate(model, tokenizer, prompt, chunk_tokens=chunk_tokens)
        torch.cuda.reset_peak_memory_stats()
        answer, seconds = plain_generate(model, tokenizer, prompt, chunk_tokens=chunk_tokens)
        results.append(dict(city=row["city"], depth=row["depth_actual"],
                            budget="full", answer=answer, prompt_tokens=len(prompt),
                            correct=task.evaluate(answer, row["answer"]),
                            prefill_s=seconds,
                            peak_gib=torch.cuda.max_memory_allocated() / 2**30))
    return results


def gate_results(model, tokenizer, prompts: list[list[int]], rows: list[dict],
                 budgets: tuple[int, ...]) -> list[dict]:
    """Call the public TokenGate generate API at each requested K."""
    import torch
    from tokengate import split_state_span
    from tokengate.datasets.tasks import get_task

    task = get_task("niah_kamradt")
    results = []
    for k in budgets:
        for prompt, row in zip(prompts, rows):
            model.generate([prompt], tokenizer, max_kept_prefill_tokens=k, max_new_tokens=64)
            torch.cuda.reset_peak_memory_stats()
            out = model.generate([prompt], tokenizer, max_kept_prefill_tokens=k,
                                 max_new_tokens=64)
            _, state_start, state_end = split_state_span(prompt)
            kept_state = sum(state_start <= p < state_end for p in out.kept_pos[0])
            results.append(dict(city=row["city"], depth=row["depth_actual"],
                                budget=k,
                                answer=out.answers[0],
                                prompt_tokens=out.n_kept[0],
                                correct=task.evaluate(out.answers[0], row["answer"]),
                                prefill_s=out.prefill_s,
                                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                                kept_pos=list(out.kept_pos[0]),
                                kept_state=kept_state,
                                retained=kept_state / (state_end - state_start)))
    return results


def show_inference_results(plain: list[dict], gated: list[dict], *,
                           baseline_name: str = "Fine-tuned baseline") -> None:
    """Show the chart, answers, and a measured takeaway in one result panel."""
    from IPython.display import HTML, display

    def cells(values):
        return "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>"

    rows = []
    for result in plain:
        rows.append(("FT baseline", "Full", result["answer"],
                     "Yes" if result["correct"] else "No",
                     f"{result['prefill_s'] * 1000:.0f}",
                     f"{result['peak_gib']:.2f}", "All",
                     f"{result['prompt_tokens']:,}" if "prompt_tokens" in result else "—"))
    for result in gated:
        kept = (f"{result['kept_state']:,} ({result['retained']:.1%})"
                if "kept_state" in result else f"{result['retained']:.1%}")
        rows.append(("TokenGate", f"{result['budget']:,}", result["answer"],
                     "Yes" if result["correct"] else "No",
                     f"{result['prefill_s'] * 1000:.0f}",
                     f"{result['peak_gib']:.2f}", kept,
                     f"{result['prompt_tokens']:,}" if "prompt_tokens" in result else "—"))
    headers = ("Method", "Token budget", "Answer", "Correct?", "TTFT (ms)",
               "Peak PyTorch (GiB)", "Document kept", "Total prompt kept")
    heading = "<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>"
    figure = result_figures(plain, gated, baseline_name=baseline_name)[0]
    png = io.BytesIO()
    figure.savefig(png, format="png", dpi=130, bbox_inches="tight")
    image = base64.b64encode(png.getvalue()).decode("ascii")
    style = ("<style>.tg-results{border:1px solid #d6dce2;border-radius:12px;"
             "padding:18px;max-width:1100px}.tg-results img{max-width:100%;"
             "height:auto;display:block;margin:auto}.tg-results table{border-collapse:collapse;"
             "width:100%;font-size:13px}.tg-results th,.tg-results td{border-bottom:"
             "1px solid #e5e7eb;padding:7px;text-align:left;vertical-align:top}"
             ".tg-results th{font-weight:600}.tg-results tr:nth-child(2) td:first-child"
             "{color:#64748b;font-weight:600}.tg-results tr:not(:nth-child(2))"
             " td:first-child{color:#296d5c;font-weight:600}"
             ".tg-takeaway{background:#e8f5f0;border-left:4px solid #296d5c;"
             "padding:10px 12px;margin-top:14px;border-radius:4px;color:#173f33}</style>")
    takeaway = "Compare a correct answer with TTFT before interpreting a speedup."
    if len(plain) == 1:
        passing = [result for result in gated if result["correct"]]
        if plain[0]["correct"] and passing:
            chosen = next((r for r in passing if r["budget"] == 100),
                          min(passing, key=lambda r: r["prefill_s"]))
            time_ratio = plain[0]["prefill_s"] / chosen["prefill_s"]
            memory_ratio = plain[0]["peak_gib"] / chosen["peak_gib"]
            pace = (f"{time_ratio:.1f}× faster" if time_ratio > 1.05
                    else f"{1 / time_ratio:.1f}× slower" if time_ratio < 0.95
                    else "about the same")
            memory = (f"{memory_ratio:.1f}× lower" if memory_ratio > 1.05
                      else f"{1 / memory_ratio:.1f}× higher" if memory_ratio < 0.95
                      else "about the same")
            takeaway = (f"At K={chosen['budget']:,}, TokenGate still found the needle "
                        f"while keeping {chosen['retained']:.1%} of the document. "
                        f"TTFT was {pace} and peak PyTorch memory was {memory} "
                        f"compared with the fine-tuned baseline.")
        elif not passing:
            takeaway = "None of the tested TokenGate budgets found this needle; try a larger budget."
        else:
            takeaway = ("TokenGate found this needle, but the fine-tuned baseline "
                        "missed it. Compare answers before treating their timing ratio as a matched-accuracy speedup.")
    display(HTML(style + '<div class="tg-results"><img alt="Measured prefill time with '
                 'speedup labels" src="data:image/png;base64,' + image + '">'
                 '<div style="overflow-x:auto"><table>' + heading
                 + "".join(cells(row) for row in rows) + "</table></div>"
                 + '<div class="tg-takeaway"><strong>This run:</strong> '
                 + html.escape(takeaway) + '</div></div>'))


def result_figures(plain: list[dict], gated: list[dict], *,
                   baseline_name: str = "Fine-tuned baseline"):
    """Plot prefill time and label each correct budget's speedup."""
    import matplotlib.pyplot as plt
    from statistics import mean

    base_ms = 1000 * mean(r["prefill_s"] for r in plain)
    budgets = [k for k in dict.fromkeys(r["budget"] for r in gated)
               if all(r["correct"] for r in gated if r["budget"] == k)]
    gate_ms = [1000 * mean(r["prefill_s"] for r in gated if r["budget"] == k)
               for k in budgets]
    names = [f"K={k}" for k in budgets]
    scope = ("One NIAH document" if len(plain) == 1
             else f"Mean over {len(plain)} NIAH documents")
    time_fig, time_ax = plt.subplots(figsize=(7, 3.3))
    time_bars = time_ax.bar(["Baseline", *names], [base_ms, *gate_ms],
                            color=["#64748b", *["#296d5c"] * len(names)])
    time_ax.bar_label(time_bars,
                      labels=[f"{base_ms:.0f} ms",
                              *[f"{ms:.0f} ms · {base_ms / ms:.1f}×"
                                for ms in gate_ms]], padding=3)
    time_ax.set(ylabel="TTFT (ms)",
                title=f"{scope} · {baseline_name}")
    time_ax.set_ylim(0, max([base_ms, *gate_ms]) * 1.25)
    if not gate_ms:
        time_ax.text(0.5, 0.8, "No correct TokenGate budgets",
                     transform=time_ax.transAxes, ha="center")
    time_fig.tight_layout()

    # Colab auto-renders open Matplotlib figures after the cell. The notebook
    # displays these explicitly, so close them here to prevent duplicates.
    plt.close(time_fig)
    return (time_fig,)


def show_kept_tokens(tokenizer, prompt: list[int], row: dict, result: dict):
    """Highlight kept document tokens after the notebook names the target."""
    from IPython.display import HTML, display
    from tokengate import split_state_span

    clean, start, end = split_state_span(prompt)
    selected = set(result["kept_pos"])
    spans = []
    for keep, positions in groupby(range(start, end), key=lambda p: p in selected):
        piece = html.escape(tokenizer.decode([clean[p] for p in positions]))
        style = "background:#bbf7d0;color:#14532d" if keep else "color:#777"
        spans.append(f'<span style="{style}">{piece}</span>')
    print(f"Depth {row['depth_actual']:.0%} · K={result['budget']} · "
          f"kept {sum(start <= p < end for p in selected):,}/{end-start:,} document tokens")
    display(HTML('<div style="max-height:320px;overflow:auto;white-space:pre-wrap;line-height:1.55">'
                 + "".join(spans) + "</div>"))


def run_niah_exploration(repo_root: str | Path, *, base_model: str,
                         checkpoint_id: str, checkpoint_revision: str,
                         baseline_checkpoint: tuple[str, str, str],
                         gpu_runtime: str, doc_tokens: int, example_index: int,
                         budgets: tuple[int, ...]) -> dict:
    """Run one selectable example end to end for the notebook's exploration cell."""
    import gc

    import torch
    from huggingface_hub import snapshot_download
    from IPython.utils.io import capture_output
    from transformers import AutoTokenizer
    from tokengate import load_model, split_state_span

    if not budgets or any(k < 1 for k in budgets):
        raise ValueError("Provide at least one positive token budget")
    if gpu_runtime == "T4" and "Qwen3.5" in base_model and doc_tokens > 8192:
        raise ValueError("Qwen3.5 on T4 supports up to 8k document tokens")
    if gpu_runtime == "A100" and torch.cuda.get_device_properties(0).total_memory < 35 * 2**30:
        raise ValueError("Use an A100-class GPU for single-pass prefill")

    row = niah_examples(repo_root, doc_tokens)[example_index]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    gate_prompt, plain_prompt = niah_prompt(tokenizer, row)
    _, state_start, state_end = split_state_span(gate_prompt)
    chunk_tokens = (1024 if gpu_runtime == "T4" and "Qwen3-1.7B" in base_model
                    and doc_tokens > 4096 else None)
    baseline_name = ("chunked FT baseline (T4)" if chunk_tokens
                     else f"single-pass FT baseline ({gpu_runtime})")
    dtype = torch.float16 if gpu_runtime == "T4" else torch.bfloat16
    print(f"Question: {row['question']}")
    print(f"Expected answer: {row['answer']}")
    print(f"Needle depth: {row['depth_actual']:.0%}; document: {state_end - state_start:,} tokens")
    print(f"Ordinary-model prefill: {baseline_name}")

    with capture_output():
        plain_model = load_sft_baseline(
            base_model, baseline_checkpoint[0], baseline_checkpoint[2],
            revision=baseline_checkpoint[1], dtype=dtype)
    try:
        plain = plain_results(plain_model, tokenizer, [plain_prompt], [row],
                              chunk_tokens=chunk_tokens)
    finally:
        del plain_model
        gc.collect()
        torch.cuda.empty_cache()
    with capture_output():
        checkpoint = snapshot_download(checkpoint_id, revision=checkpoint_revision)
        gate = load_model(checkpoint, gate_attn="sdpa" if gpu_runtime == "T4" else "flex",
                          dtype=dtype)
    try:
        gated = gate_results(gate, tokenizer, [gate_prompt], [row], budgets)
        show_inference_results(plain, gated, baseline_name=baseline_name)
    finally:
        del gate
        gc.collect()
        torch.cuda.empty_cache()
    return {"cache_version": DEMO_CACHE_VERSION,
            "sample": row, "plain": plain, "gated": gated,
            "tokenizer": tokenizer, "gate_prompt": gate_prompt}


def release_niah_rows(repo_root: str | Path, output_dir: str | Path,
                      n_train: int = 12000) -> tuple[list[dict], list[dict]]:
    """Generate the stated NIAH-only release recipe and frozen eval pool.

    The 200 eval rows contain 50 validation and 150 reserved test IDs. The
    trainer receives the committed split manifest and sees only the former.
    """
    root = Path(repo_root)
    path = Path(output_dir) / f"niah_release_{n_train}_seed0.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".incomplete.jsonl")
        subprocess.run([sys.executable,
                        str(root / "datasets/niah_kamradt/generate_niah_kamradt.py"),
                        "--output-file", str(tmp), "--n-train", str(n_train),
                        "--n-test-cities", "20", "--seed", "0"], check=True)
        tmp.replace(path)

    split_path = root / "tokengate/datasets/eval_splits/longbench_eval_split_niah_iso_v1.json"
    split = json.loads(split_path.read_text())["niah_kamradt"]
    val_ids, test_ids = set(split["val"]), set(split["test"])
    assert len(val_ids) == 50 and len(test_ids) == 150 and not val_ids & test_ids
    train, eval_rows = [], []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            converted = dict(id=row["id"],
                             article=row["body"] + "\n\nQuestion: " + row["question"],
                             answer=row["answer"])
            (train if row["split"] == "train" else eval_rows).append(converted)
    assert len(train) == n_train and len({r["id"] for r in train}) == n_train
    assert len(eval_rows) == 200
    assert {r["id"] for r in eval_rows} == val_ids | test_ids
    assert not {r["id"] for r in train} & (val_ids | test_ids)
    return train, eval_rows


def release_niah_config_kwargs(repo_root: str | Path, base_model: str,
                               output_dir: str | Path, data_root: str | Path,
                               num_gpus: int) -> dict:
    """Public TokenGateConfig inputs for the stated NIAH-only release recipe."""
    if num_gpus not in (1, 2, 4):
        raise ValueError("use 1, 2, or 4 visible A100s for effective batch size 4")
    root = Path(repo_root)
    if base_model not in ("Qwen/Qwen3-1.7B", "Qwen/Qwen3.5-2B"):
        raise ValueError(f"no released NIAH recipe for {base_model}")
    return dict(
        base_model=base_model, gate_layer=1, output_dir=str(output_dir),
        learning_rate=1e-4, epochs=20, max_steps=(3000 if base_model.endswith("1.7B") else 18000),
        k_eval=[2000], lora_rank=16, lora_alpha=32, lora_lr=1e-4,
        lora_targets=["q_proj", "v_proj"],
        num_gpus=num_gpus, batch_size=1, grad_accum=4 // num_gpus,
        warmup_steps=1000, lr_schedule="constant", optim="muon",
        gate_optimizer="muon", weight_decay=0.1, no_decay_scalars=True,
        phase2_loguniform=True, shuffle_epochs=True, seed=0,
        min_article_tokens=2048, max_article_tokens=16384,
        eval_min_article_tokens=0, eval_max_article_tokens=39000,
        eval_freq=600, checkpoint_freq=600, eval_gen_batch_size=1,
        gate_full_eval_freq=5, swa_freq=1, swa_start=0, swa_eval_freq=1,
        eval_split=str(root / "tokengate/datasets/eval_splits/longbench_eval_split_niah_iso_v1.json"),
        eval_group_override="niah:niah_kamradt", eval_test_full_tasks=None,
        target_balanced_map=str(root / "tokengate/datasets/maps/eval_target_train_map_v17_niah_only.yaml"),
        data_root=str(data_root), packed=True, grad_checkpoint=False,
        wandb_project=None,
    )


def train_demo_sft_baseline(base_model: str, data_root: str | Path,
                            eval_split: str | Path, output_dir: str | Path,
                            num_gpus: int) -> Path:
    """Run the notebook's separate full-prefill LoRA control on the same rows."""
    output_dir = Path(output_dir)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft>=0.15"],
                   check=True)
    if (output_dir / "adapter_model.safetensors").exists():
        return output_dir
    steps = 10200 if base_model == "Qwen/Qwen3-1.7B" else 600
    subprocess.run([
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}", "--master_port=29501",
        str(Path(__file__).with_name("train_lora_baseline.py")), "--base-model", base_model,
        "--data-root", str(data_root), "--eval-split", str(eval_split),
        "--output-dir", str(output_dir), "--max-steps", str(steps),
        "--grad-accum", str(4 // num_gpus),
    ], check=True)
    return output_dir
