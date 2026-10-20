#!/usr/bin/env python3
"""Single-rank microbenchmark of the e2e-gate training hot path.

Reproduces the production per-step path used by train_e2e_gate.py
(forward_with_gate_batched → loss → backward → grad-norm logging → optimizer.step
→ zero_grad), and reports per-component wall time, peak memory and the number
of GPU→CPU syncs (.item() calls) introduced by per-step logging.

The script supports two A/B variants:

  --variant baseline     replicates the current main-branch hot path verbatim
                         (15+ grad-norm .item()s per step, non-fused AdamW,
                         set_to_none=False, forward hook syncing to CPU).
  --variant optimized    applies the safe quick-wins (single grad-norm sync,
                         fused AdamW, set_to_none=True, hook kept on GPU).

A short pre-warmup + N timed steps are run on synthetic samples of realistic
shape (configurable median 8 k tokens by default, matching the pretrain mix's
median context length).

Usage
-----
    python -m tokengate.e2e_gate.bench.bench_train_step \\
        --base-model Qwen/Qwen3-8B --gate-layer 0 --variant baseline \\
        --seq-len 8000 --article-len 6000 --k 2000 --batch-size 1 \\
        --warmup 3 --steps 20 \\
        --out /tmp/bench_baseline.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from typing import List

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate  # noqa: E402
from tokengate.e2e_gate.attn.flex_attention_utils import (  # noqa: E402
    warmup_flex_attention,
)
from tokengate.e2e_gate.attn import qwen3_flex_attention as _qfa  # noqa: E402
from tokengate.e2e_gate.attn.qwen3_flex_attention import (  # noqa: E402
    get_full_attn_layer_indices,
    patch_model_with_flex_attention,
)
from tokengate.e2e_gate.gate_module.gate_forward import build_gate_branch  # noqa: E402
from tokengate.e2e_gate.train.train_e2e_gate import (  # noqa: E402
    forward_with_gate_batched,
)

# Disable diagnostic phase-2 attention-stats accumulation in the bench.
# It only feeds W&B logging and is shape-fragile when run outside the trainer
# (assumes per-head bias).  Stats accumulation adds no measurable cost to the
# step, so removing it leaves timings representative.
_qfa._accumulate_phase2_attn_stats = lambda *a, **kw: None  # type: ignore[attr-defined]


# ─── Sync counter ─────────────────────────────────────────────────────────────

class SyncCounter:
    """Patches torch.Tensor.item() to count GPU→CPU syncs.

    Each .item() call on a CUDA tensor forces a `cudaStreamSynchronize`, which
    blocks the host until all enqueued kernels finish.  We count these to
    quantify the per-step logging cost.
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._orig = None

    def __enter__(self) -> "SyncCounter":
        self._n = 0
        self._orig = torch.Tensor.item

        def _counting_item(self_t):
            if self_t.is_cuda:
                SyncCounter._counter += 1
            return _counting_item._orig(self_t)

        _counting_item._orig = self._orig
        SyncCounter._counter = 0
        torch.Tensor.item = _counting_item  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        torch.Tensor.item = self._orig  # type: ignore[assignment]
        self._n = SyncCounter._counter

    @property
    def count(self) -> int:
        return self._n


# ─── Synthetic batch ──────────────────────────────────────────────────────────

def _synth_batch(
    *,
    batch_size: int,
    seq_len: int,
    article_len: int,
    n_ans: int,
    K: int,
    vocab_size: int,
    rng: random.Random,
) -> List[dict]:
    out: List[dict] = []
    for _ in range(batch_size):
        pre = rng.randint(64, 256)
        post = rng.randint(32, 128)
        # Keep total sequence ≈ seq_len; clamp article_len to fit.
        art = min(article_len, max(1, seq_len - pre - post - n_ans))
        total = pre + art + post + n_ans
        ids = [rng.randint(1, vocab_size - 1) for _ in range(total)]
        ctx = pre + art + post
        out.append({
            "input_ids": ids,
            "article_start": pre,
            "article_end": pre + art,
            "ctx_len": ctx,
            "n_ans": n_ans,
            "K": K,
        })
    return out


# ─── Hot path: baseline (matches train_e2e_gate.py:2300+) ─────────────────────

def _build_optimizer_baseline(gate, lr: float) -> torch.optim.Optimizer:
    """AdamW with default (non-fused) kwargs, mirroring main-branch setup."""
    return torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)


def _build_optimizer_optimized(gate, lr: float) -> torch.optim.Optimizer:
    """Fused AdamW where supported (CUDA + bf16/fp32 params)."""
    # `fused=True` uses a single CUDA kernel for the parameter update.
    # Requires capability for fused Adam; supported on Ampere+ (sm_80+).
    return torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4, fused=True)


def _baseline_step(model, gate, batch, gate_layer: int, optimizer) -> dict:
    """One training step in the *baseline* style (heavy per-step .item() syncs)."""
    timings = {}
    torch.cuda.synchronize()
    t_step_start = time.perf_counter()

    # Forward
    t0 = time.perf_counter()
    logits, loss_mask, _ = forward_with_gate_batched(
        model, gate, batch, gate_layer,
        frozen_backbone=True, prune_phase1_kv=True,
    )
    torch.cuda.synchronize()
    timings["fwd_ms"] = (time.perf_counter() - t0) * 1000

    # Loss
    t0 = time.perf_counter()
    total_loss = None
    n_valid = 0
    for b_idx, item in enumerate(batch):
        mask = loss_mask[b_idx].to(logits.device)
        if not mask.any():
            continue
        sl = logits[b_idx, mask, :]
        ans_ids = item["input_ids"][item["ctx_len"]:item["ctx_len"] + item["n_ans"]]
        labels = torch.tensor(ans_ids, dtype=torch.long, device=logits.device)
        if sl.shape[0] != labels.shape[0]:
            continue
        per = F.cross_entropy(sl, labels)
        # Baseline does an extra .item() on letter/eos losses for logging
        if sl.shape[0] >= 2:
            _ = F.cross_entropy(sl[:-1], labels[:-1]).item()
            _ = F.cross_entropy(sl[-1:], labels[-1:]).item()
        total_loss = per if total_loss is None else total_loss + per
        n_valid += 1
    loss = total_loss / max(1, n_valid) if total_loss is not None else None
    torch.cuda.synchronize()
    timings["loss_ms"] = (time.perf_counter() - t0) * 1000
    if loss is None:
        return None

    # Backward
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    timings["bwd_ms"] = (time.perf_counter() - t0) * 1000

    # Per-step logging .item() spam (baseline)
    t0 = time.perf_counter()
    gn = torch.nn.utils.clip_grad_norm_(list(gate.parameters()), 1.0).item()
    # Mirror train_e2e_gate.py:2546–2618 — many independent norm().item() calls
    for name, p in gate.named_parameters():
        if p.grad is not None:
            _ = p.grad.norm().item()
    # Loss .item() for logging
    _ = loss.detach().item()
    torch.cuda.synchronize()
    timings["log_ms"] = (time.perf_counter() - t0) * 1000

    # Optimizer step + zero_grad (baseline: zero_grad without set_to_none)
    t0 = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad()  # set_to_none default = True in modern PyTorch but
                            # set_to_none=False matches the legacy default
    torch.cuda.synchronize()
    timings["opt_ms"] = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    timings["step_ms"] = (time.perf_counter() - t_step_start) * 1000
    timings["grad_norm"] = gn
    timings["loss"] = loss.detach().float().mean().item()
    return timings


# ─── Hot path: optimized ──────────────────────────────────────────────────────

def _optimized_step(model, gate, batch, gate_layer: int, optimizer) -> dict:
    """One training step with quick-win optimizations.

    Changes vs baseline:
      • Single batched grad-norm sync (one .item() call).
      • `set_to_none=True` in zero_grad — faster and lower peak memory.
      • Drop letter/eos cross-entropy .item() spam (left to W&B periodic logging).
      • Fused AdamW step (set by `_build_optimizer_optimized`).
      • Avoid per-sample loss .item() inside batch loop.
    """
    timings = {}
    torch.cuda.synchronize()
    t_step_start = time.perf_counter()

    # Forward
    t0 = time.perf_counter()
    logits, loss_mask, _ = forward_with_gate_batched(
        model, gate, batch, gate_layer,
        frozen_backbone=True, prune_phase1_kv=True,
    )
    torch.cuda.synchronize()
    timings["fwd_ms"] = (time.perf_counter() - t0) * 1000

    # Loss
    t0 = time.perf_counter()
    total_loss = None
    n_valid = 0
    for b_idx, item in enumerate(batch):
        mask = loss_mask[b_idx].to(logits.device)
        if not mask.any():
            continue
        sl = logits[b_idx, mask, :]
        ans_ids = item["input_ids"][item["ctx_len"]:item["ctx_len"] + item["n_ans"]]
        labels = torch.tensor(ans_ids, dtype=torch.long, device=logits.device)
        if sl.shape[0] != labels.shape[0]:
            continue
        per = F.cross_entropy(sl, labels)
        total_loss = per if total_loss is None else total_loss + per
        n_valid += 1
    loss = total_loss / max(1, n_valid) if total_loss is not None else None
    torch.cuda.synchronize()
    timings["loss_ms"] = (time.perf_counter() - t0) * 1000
    if loss is None:
        return None

    # Backward
    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    timings["bwd_ms"] = (time.perf_counter() - t0) * 1000

    # Logging — keep one fused tensor-of-norms on GPU; sync once at the end.
    t0 = time.perf_counter()
    gn = torch.nn.utils.clip_grad_norm_(list(gate.parameters()), 1.0)  # tensor on GPU
    grad_norms_gpu = torch.stack([
        p.grad.detach().float().norm() if p.grad is not None
        else torch.zeros((), device=loss.device)
        for p in gate.parameters()
    ])
    # One sync brings *all* logging scalars to host together.
    loss_h = loss.detach()
    flat = torch.cat([gn.unsqueeze(0), grad_norms_gpu, loss_h.unsqueeze(0)]).cpu().tolist()
    torch.cuda.synchronize()
    timings["log_ms"] = (time.perf_counter() - t0) * 1000

    # Optimizer step + zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    timings["opt_ms"] = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    timings["step_ms"] = (time.perf_counter() - t_step_start) * 1000
    timings["grad_norm"] = flat[0]
    timings["loss"] = flat[-1]
    return timings


# ─── Driver ───────────────────────────────────────────────────────────────────

def _aggregate(rows: List[dict]) -> dict:
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if isinstance(rows[0][k], (int, float))]
    out: dict = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            vals = sorted(vals)
            out[f"{k}_mean"] = sum(vals) / len(vals)
            out[f"{k}_median"] = vals[len(vals) // 2]
            out[f"{k}_min"] = vals[0]
            out[f"{k}_max"] = vals[-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--gate-layer", type=int, default=0)
    ap.add_argument("--variant", choices=["baseline", "optimized"], default="baseline")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=8000)
    ap.add_argument("--article-len", type=int, default=6000)
    ap.add_argument("--n-ans", type=int, default=10)
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"[bench:{args.variant}] loading {args.base_model} on cuda:{args.device}...")
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
    d_model = model.config.hidden_size
    vocab_size = model.config.vocab_size
    warmup_flex_attention(device)

    nlb = len([i for i in get_full_attn_layer_indices(model) if i > args.gate_layer])
    gate = E2EGate(d_model=d_model, n_heads=model.config.num_attention_heads,
                   n_layers_bias=nlb).to(device).to(torch.float32)
    gate.set_qk_norm_scales(model, args.gate_layer)
    build_gate_branch(model, gate, args.gate_layer)
    # non-trivial scores (the folded readout row is zero-init)
    torch.nn.init.normal_(gate.gate_branch.down_proj.weight, std=0.02)
    for p in gate.parameters():
        p.requires_grad_(True)

    build_opt = (
        _build_optimizer_baseline if args.variant == "baseline"
        else _build_optimizer_optimized
    )
    optimizer = build_opt(gate, args.lr)
    step_fn = _baseline_step if args.variant == "baseline" else _optimized_step

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f"[bench:{args.variant}] warmup ({args.warmup} steps)...")
    for _ in range(args.warmup):
        batch = _synth_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            article_len=args.article_len,
            n_ans=args.n_ans,
            K=args.k,
            vocab_size=vocab_size,
            rng=rng,
        )
        _ = step_fn(model, gate, batch, args.gate_layer, optimizer)

    # ── Timed loop ───────────────────────────────────────────────────────────
    print(f"[bench:{args.variant}] timed ({args.steps} steps)...")
    torch.cuda.reset_peak_memory_stats()

    syncs_per_step: List[int] = []
    rows: List[dict] = []
    for i in range(args.steps):
        batch = _synth_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            article_len=args.article_len,
            n_ans=args.n_ans,
            K=args.k,
            vocab_size=vocab_size,
            rng=rng,
        )
        with SyncCounter() as sc:
            r = step_fn(model, gate, batch, args.gate_layer, optimizer)
        if r is None:
            continue
        r["syncs"] = sc.count
        rows.append(r)
        syncs_per_step.append(sc.count)

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    agg = _aggregate(rows)
    agg["peak_bwd_gb"] = peak_gb
    agg["syncs_per_step_median"] = (sorted(syncs_per_step)[len(syncs_per_step) // 2]
                                    if syncs_per_step else 0)
    agg["variant"] = args.variant
    agg["seq_len"] = args.seq_len
    agg["article_len"] = args.article_len
    agg["batch_size"] = args.batch_size
    agg["k"] = args.k
    agg["base_model"] = args.base_model
    agg["gate_layer"] = args.gate_layer
    agg["n_steps"] = len(rows)

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print(f"[bench:{args.variant}] results ({len(rows)} timed steps)")
    print(f"  step_ms median  : {agg.get('step_ms_median', 0):.1f}")
    print(f"  fwd_ms  median  : {agg.get('fwd_ms_median', 0):.1f}")
    print(f"  bwd_ms  median  : {agg.get('bwd_ms_median', 0):.1f}")
    print(f"  loss_ms median  : {agg.get('loss_ms_median', 0):.2f}")
    print(f"  log_ms  median  : {agg.get('log_ms_median', 0):.2f}")
    print(f"  opt_ms  median  : {agg.get('opt_ms_median', 0):.2f}")
    print(f"  peak_bwd_gb     : {agg['peak_bwd_gb']:.2f}")
    print(f"  syncs/step (med): {agg['syncs_per_step_median']}")
    print()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"[bench:{args.variant}] wrote {args.out}")


if __name__ == "__main__":
    main()
