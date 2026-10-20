#!/usr/bin/env python3
"""Benchmark E2E gate training: step time and max batch size before OOM.

Sweeps batch_size × K combinations and reports:
  - Peak GPU memory (forward + backward)
  - Step wall-clock time
  - Recommended max batch size for your GPU

Uses either real tokenized samples from a cached dataset (preferred, gives
realistic seq lengths) or synthetic sequences of a specified length.

Usage
-----
# With real data (reads from a tokenized cache or builds one):
python -m tokengate.e2e_gate.tests.benchmark_training \\
    --base-model Qwen/Qwen3-0.6B \\
    --gate-layer 8 --k-low 50 --k-high 5000 \\
    --dataset biodex --task biodex_reactions

# Synthetic sequences (no dataset needed):
python -m tokengate.e2e_gate.tests.benchmark_training \\
    --base-model Qwen/Qwen3-0.6B \\
    --gate-layer 8 --k-low 50 --k-high 5000 \\
    --synthetic --seq-len 8192 --article-len 6000

# Sweep specific batch sizes only:
python -m tokengate.e2e_gate.tests.benchmark_training \\
    --base-model Qwen/Qwen3-0.6B --gate-layer 8 \\
    --batch-sizes 1 2 4 8 16 --k-values 500 2000 5000
"""

import argparse
import gc
import os
import random
import sys
import time
from typing import List, Optional

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
from tokengate.e2e_gate.attn.flex_attention_utils import warmup_flex_attention
from tokengate.e2e_gate.gate_module.gate_forward import build_gate_branch
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    get_full_attn_layer_indices,
    patch_model_with_flex_attention,
    set_gate_bias_for_layers,
)
from tokengate.e2e_gate.train.train_e2e_gate import forward_with_gate_batched


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gb(bytes_: int) -> str:
    return f"{bytes_ / 1e9:.2f} GB"


def _reset_peak():
    torch.cuda.reset_peak_memory_stats()


def _peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


def _free_gb(device) -> float:
    free, total = torch.cuda.mem_get_info(device)
    return free / 1e9


def _total_gb(device) -> float:
    _, total = torch.cuda.mem_get_info(device)
    return total / 1e9


def _model_param_gb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9


def _make_synthetic_batch(
    batch_size: int,
    seq_len: int,
    article_len: int,
    n_ans: int,
    K: int,
    vocab_size: int,
    rng: random.Random,
) -> List[dict]:
    """Build a batch of random token sequences with realistic article layout."""
    batch = []
    for _ in range(batch_size):
        # Layout: [pre_article | article | post_article | answer]
        pre_len = rng.randint(64, 256)
        post_len = rng.randint(32, 128)
        actual_art = article_len
        actual_seq = pre_len + actual_art + post_len + n_ans

        full_ids = [rng.randint(1, vocab_size - 1) for _ in range(actual_seq)]
        ctx_len = pre_len + actual_art + post_len
        batch.append({
            "input_ids": full_ids,
            "article_start": pre_len,
            "article_end": pre_len + actual_art,
            "ctx_len": ctx_len,
            "n_ans": n_ans,
            "K": K,
        })
    return batch


def _make_real_batch(
    tokenized_samples: List[dict],
    batch_size: int,
    K: int,
    rng: random.Random,
) -> List[dict]:
    """Sample a batch from real tokenized data."""
    chosen = [rng.choice(tokenized_samples) for _ in range(batch_size)]
    return [
        {
            "input_ids": s["full_ids"],
            "article_start": s["article_start"],
            "article_end": s["article_end"],
            "ctx_len": s["ctx_len"],
            "n_ans": s["n_ans"],
            "K": K,
        }
        for s in chosen
    ]


def _load_real_data(args, tokenizer):
    """Load tokenized samples from cache or build them."""
    from tokengate.e2e_gate.train.sft_dataset import build_sft_samples
    from tokengate.e2e_gate.train.train_e2e_gate import (
        _apply_chat_template, _find_article_boundaries,
    )
    from tokengate.datasets.tasks import get_task
    import pickle

    task = get_task(args.task)
    cache_key = f"{args.dataset}_{args.task}_500_50_{args.seed}_fa"
    if args.max_article_tokens:
        cache_key += f"_maxart{args.max_article_tokens}"
    cache_path = os.path.join("/tmp", f"bench_cache_{cache_key}.pkl")

    if os.path.exists(cache_path):
        print(f"[bench] Loading tokenized cache from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[bench] Building tokenized samples for {args.dataset}/{args.task}...")
    raw = build_sft_samples(
        task=task, dataset_name=args.dataset, n=500, seed=args.seed,
        max_article_tokens=args.max_article_tokens, tokenizer=tokenizer,
    )
    tokenized = []
    for s in raw:
        prompt_ids = _apply_chat_template(tokenizer, s["input"])
        answer_ids = tokenizer.encode(
            "FINAL ANSWER: " + s["target"] + tokenizer.eos_token, add_special_tokens=False,
        )
        full_ids = prompt_ids + answer_ids
        non_comp = task.get_non_compressible_context(s["article"])
        empty_prompt = task.format_prompt(non_comp)
        empty_ids = _apply_chat_template(tokenizer, empty_prompt)
        pre_len, post_len = _find_article_boundaries(prompt_ids, empty_ids)
        art_start = pre_len
        art_end = len(prompt_ids) - post_len
        n_art = art_end - art_start
        if n_art <= 0 or len(answer_ids) <= 0:
            continue
        if args.max_article_tokens and n_art > args.max_article_tokens:
            art_end = art_start + args.max_article_tokens
            prompt_ids = prompt_ids[:art_end] + prompt_ids[art_end:]
            full_ids = prompt_ids + answer_ids
        tokenized.append({
            "full_ids": full_ids,
            "ctx_len": len(prompt_ids),
            "article_start": art_start,
            "article_end": art_end,
            "n_ans": len(answer_ids),
        })
    with open(cache_path, "wb") as f:
        pickle.dump(tokenized, f)
    print(f"[bench] Built {len(tokenized)} samples, cached to {cache_path}")
    return tokenized


# ── Single-step benchmark ─────────────────────────────────────────────────────

def benchmark_step(
    model,
    gate,
    batch_data: List[dict],
    gate_layer: int,
    n_warmup: int = 1,
    n_timed: int = 3,
) -> dict:
    """
    Run forward + backward for the given batch, return timing + memory stats.

    Returns dict with keys:
        ok: bool — completed without OOM
        fwd_ms: float — average forward pass ms
        bwd_ms: float — average backward pass ms
        step_ms: float — average total step ms (fwd + bwd)
        peak_fwd_gb: float — peak memory after forward
        peak_bwd_gb: float — peak memory after backward (includes grad buffers)
        oom_phase: str or None — "forward" / "backward" if OOM occurred
    """
    device = next(model.parameters()).device

    def _run_once(timed: bool):
        optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-4)
        optimizer.zero_grad()

        _reset_peak()
        t0 = time.perf_counter()
        try:
            logits, loss_mask, _ = forward_with_gate_batched(
                model, gate, batch_data, gate_layer,
                frozen_backbone=True, prune_phase1_kv=True,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"ok": False, "oom_phase": "forward"}
        fwd_t = time.perf_counter() - t0
        peak_fwd = _peak_gb()

        # Compute a real loss so backward has gradients to flow
        batch_loss = 0.0
        n_valid = 0
        for b_idx, item in enumerate(batch_data):
            mask = loss_mask[b_idx].to(logits.device)
            if not mask.any():
                continue
            sample_logits = logits[b_idx, mask, :]
            answer_ids = item["input_ids"][item["ctx_len"]: item["ctx_len"] + item["n_ans"]]
            labels = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
            if sample_logits.shape[0] != labels.shape[0]:
                continue
            batch_loss = batch_loss + F.cross_entropy(sample_logits, labels)
            n_valid += 1

        if n_valid == 0:
            return {"ok": False, "oom_phase": "no_valid_loss"}

        loss = batch_loss / n_valid
        _reset_peak()
        t1 = time.perf_counter()
        try:
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"ok": False, "oom_phase": "backward"}
        bwd_t = time.perf_counter() - t1
        peak_bwd = _peak_gb()

        optimizer.step()
        optimizer.zero_grad()
        del logits, loss_mask, loss
        torch.cuda.empty_cache()

        if not timed:
            return {"ok": True}
        return {
            "ok": True,
            "fwd_ms": fwd_t * 1000,
            "bwd_ms": bwd_t * 1000,
            "step_ms": (fwd_t + bwd_t) * 1000,
            "peak_fwd_gb": peak_fwd,
            "peak_bwd_gb": peak_bwd,
        }

    # Warmup runs
    for _ in range(n_warmup):
        result = _run_once(timed=False)
        if not result["ok"]:
            return result

    # Timed runs
    fwd_times, bwd_times, peak_fwds, peak_bwds = [], [], [], []
    for _ in range(n_timed):
        result = _run_once(timed=True)
        if not result["ok"]:
            return result
        fwd_times.append(result["fwd_ms"])
        bwd_times.append(result["bwd_ms"])
        peak_fwds.append(result["peak_fwd_gb"])
        peak_bwds.append(result["peak_bwd_gb"])

    return {
        "ok": True,
        "fwd_ms": sum(fwd_times) / len(fwd_times),
        "bwd_ms": sum(bwd_times) / len(bwd_times),
        "step_ms": sum(f + b for f, b in zip(fwd_times, bwd_times)) / len(fwd_times),
        "peak_fwd_gb": max(peak_fwds),
        "peak_bwd_gb": max(peak_bwds),
        "oom_phase": None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark E2E gate training step.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gate-layer", type=int, required=True)

    # K sweep
    parser.add_argument("--k-low",  type=int, default=50)
    parser.add_argument("--k-high", type=int, default=5000)
    parser.add_argument("--k-values", type=int, nargs="+", default=None,
                        help="Explicit K values to test. Default: [k_low, (k_low+k_high)//2, k_high]")

    # Batch size sweep
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="Batch sizes to sweep.")
    parser.add_argument("--stop-on-oom", action="store_true", default=True,
                        help="Stop sweeping batch sizes after first OOM (default: True).")
    parser.add_argument("--no-stop-on-oom", dest="stop_on_oom", action="store_false")

    # Data source
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic random sequences instead of real data.")
    parser.add_argument("--seq-len", type=int, default=8192,
                        help="Sequence length for synthetic data.")
    parser.add_argument("--article-len", type=int, default=6000,
                        help="Article length for synthetic data (must be < seq-len).")
    parser.add_argument("--n-ans", type=int, default=8,
                        help="Answer tokens for synthetic data.")

    # Real data args (ignored with --synthetic)
    parser.add_argument("--dataset", type=str, default="biodex")
    parser.add_argument("--task", type=str, default="biodex_reactions")
    parser.add_argument("--max-article-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)

    # Benchmark config
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup steps (not timed) per configuration.")
    parser.add_argument("--timed-steps", type=int, default=3,
                        help="Timed steps to average over per configuration.")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index.")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    rng = random.Random(args.seed)

    k_values = args.k_values or [args.k_low, (args.k_low + args.k_high) // 2, args.k_high]
    k_values = sorted(set(k_values))

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[bench] Loading {args.base_model} on cuda:{args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    model = patch_model_with_flex_attention(model)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.lm_head.parameters():
        p.requires_grad_(True)
    for p in model.model.norm.parameters():
        p.requires_grad_(True)

    d_model = model.config.hidden_size
    vocab_size = model.config.vocab_size
    num_layers = len(model.model.layers)
    model_gb = _model_param_gb(model)

    print(f"[bench] Model: {num_layers} layers, d_model={d_model}, vocab={vocab_size}")
    print(f"[bench] Model weights: {model_gb:.2f} GB")
    print(f"[bench] GPU total: {_total_gb(device):.1f} GB  |  free after model load: {_free_gb(device):.1f} GB")

    warmup_flex_attention(device)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.synthetic:
        print(f"[bench] Using synthetic data: seq_len={args.seq_len}, article_len={args.article_len}")
        tokenized_samples = None
    else:
        tokenizer_for_data = tokenizer
        tokenized_samples = _load_real_data(args, tokenizer_for_data)
        seq_lens = [len(s["full_ids"]) for s in tokenized_samples]
        art_lens = [s["article_end"] - s["article_start"] for s in tokenized_samples]
        print(f"[bench] Real data: {len(tokenized_samples)} samples")
        print(f"[bench]   seq_len  — median={sorted(seq_lens)[len(seq_lens)//2]:,}  "
              f"p90={sorted(seq_lens)[int(len(seq_lens)*0.9)]:,}  max={max(seq_lens):,}")
        print(f"[bench]   art_len  — median={sorted(art_lens)[len(art_lens)//2]:,}  "
              f"p90={sorted(art_lens)[int(len(art_lens)*0.9)]:,}  max={max(art_lens):,}")

    # ── Gate ──────────────────────────────────────────────────────────────────
    nlb = len([i for i in get_full_attn_layer_indices(model) if i > args.gate_layer])
    gate = E2EGate(d_model=d_model, n_heads=model.config.num_attention_heads,
                   n_layers_bias=nlb).to(device).to(torch.float32)
    gate.set_qk_norm_scales(model, args.gate_layer)
    build_gate_branch(model, gate, args.gate_layer)
    # non-trivial scores (the folded readout row is zero-init)
    torch.nn.init.normal_(gate.gate_branch.down_proj.weight, std=0.02)

    # ── Print header ──────────────────────────────────────────────────────────
    print(f"\n[bench] gate_layer={args.gate_layer}  K values={k_values}  "
          f"batch_sizes={args.batch_sizes}")
    print(f"\n{'batch':>6}  {'K':>6}  {'seq_len':>8}  "
          f"{'fwd(ms)':>8}  {'bwd(ms)':>8}  {'step(ms)':>9}  "
          f"{'peak_fwd':>9}  {'peak_bwd':>9}  {'status':>10}")
    print("-" * 90)

    results = []
    last_ok_batch = {}  # K → last successful batch size

    for K in k_values:
        hit_oom = False
        for bs in sorted(args.batch_sizes):
            if hit_oom and args.stop_on_oom:
                print(f"{'':>6}  {K:>6}  {'':>8}  {'':>8}  {'':>8}  {'':>9}  "
                      f"{'':>9}  {'':>9}  {'skipped':>10}")
                continue

            # Build batch
            if args.synthetic:
                batch = _make_synthetic_batch(
                    bs, args.seq_len, args.article_len, args.n_ans, K, vocab_size, rng,
                )
                seq_len_shown = args.seq_len
            else:
                batch = _make_real_batch(tokenized_samples, bs, K, rng)
                seq_len_shown = max(len(item["input_ids"]) for item in batch)

            result = benchmark_step(
                model, gate, batch, args.gate_layer,
                n_warmup=args.warmup, n_timed=args.timed_steps,
            )
            result["batch_size"] = bs
            result["K"] = K
            result["seq_len"] = seq_len_shown
            results.append(result)

            if result["ok"]:
                last_ok_batch[K] = bs
                print(f"{bs:>6}  {K:>6}  {seq_len_shown:>8,}  "
                      f"{result['fwd_ms']:>8.1f}  {result['bwd_ms']:>8.1f}  "
                      f"{result['step_ms']:>9.1f}  "
                      f"{result['peak_fwd_gb']:>8.2f}G  "
                      f"{result['peak_bwd_gb']:>8.2f}G  "
                      f"{'OK':>10}")
            else:
                hit_oom = True
                print(f"{bs:>6}  {K:>6}  {seq_len_shown:>8,}  "
                      f"{'':>8}  {'':>8}  {'':>9}  "
                      f"{'':>9}  {'':>9}  "
                      f"{'OOM(' + result['oom_phase'] + ')':>10}")
                gc.collect()
                torch.cuda.empty_cache()

        print()  # blank line between K values

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 90)
    print("[bench] SUMMARY")
    print(f"  Model               : {args.base_model}")
    print(f"  Gate layer          : {args.gate_layer}")
    print(f"  GPU                 : cuda:{args.device}  ({_total_gb(device):.0f} GB total)")
    print(f"  Model weight memory : {model_gb:.2f} GB")
    print()

    for K in k_values:
        ok_rows = [r for r in results if r["K"] == K and r["ok"]]
        oom_rows = [r for r in results if r["K"] == K and not r["ok"]]
        if ok_rows:
            max_bs = max(r["batch_size"] for r in ok_rows)
            best = max(ok_rows, key=lambda r: r["batch_size"])
            steps_per_hour = 3600 / (best["step_ms"] / 1000) if best["step_ms"] > 0 else 0
            print(f"  K={K:<6}  max_batch_size={max_bs:<4}  "
                  f"step_ms={best['step_ms']:.1f}  "
                  f"peak_bwd={best['peak_bwd_gb']:.2f} GB  "
                  f"({steps_per_hour:.0f} steps/hr at bs={max_bs})")
        else:
            print(f"  K={K:<6}  OOM at batch_size=1")

    print()
    # Overall recommendation: use the minimum max_batch_size across K values
    # (K_high is the worst case for memory)
    all_max_bs = [max((r["batch_size"] for r in results if r["K"] == K and r["ok"]), default=0)
                  for K in k_values]
    safe_bs = min(b for b in all_max_bs if b > 0) if any(b > 0 for b in all_max_bs) else 0

    print(f"  Recommended --batch-size : {safe_bs}  (safe across all K values tested)")
    print(f"  With 8-GPU DDP, effective batch = {safe_bs * 8}")
    print("=" * 90)


if __name__ == "__main__":
    main()
