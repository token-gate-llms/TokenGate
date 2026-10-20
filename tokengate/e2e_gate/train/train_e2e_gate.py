#!/usr/bin/env python3
"""Train the E2E Gate for layer-based KV token pruning.

The gate learns a per-token bias b_j = W_B @ x_j^L that:
  1. Selects which top-K article tokens to keep at layer L.
  2. Biases attention toward kept tokens at layers > L.

Training uses SFT (cross-entropy on ground-truth answers); per-step K
targets come from the configured phase-2 target source (--phase2-loguniform
/ --phase2-len-list / --phase2-bucket-lists).  Gradients flow through the
attention bias for the selected top-K tokens (MoE-style).

Usage
-----
torchrun -m tokengate.e2e_gate.train.train_e2e_gate \\
    --base-model Qwen/Qwen3-1.7B \\
    --dataset mfqa_2wiki_wiki --task mfqa_2wiki_wiki \\
    --gate-layer 1 \\
    --phase2-loguniform --k-eval 2000 \\
    --eval-split-manifest tokengate/datasets/eval_splits/longbench_eval_split_2wikimqa_iso_v1.json \\
    --output-dir e2e_gates/example/L1 \\
    --epochs 20
"""

import argparse
import asyncio
import copy
import csv
import datetime
import json
import math
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import wandb
    _HAVE_WANDB = True
except ImportError:
    _HAVE_WANDB = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


def _make_param_heatmap(tensor: "torch.Tensor", title: str) -> "wandb.Image":
    """Render a (n_layers, n_heads) parameter tensor as a W&B heatmap image."""
    data = tensor.detach().float().cpu().numpy()  # (L, H)
    n_layers, n_heads = data.shape
    fig, ax = plt.subplots(figsize=(max(4, n_heads // 4), max(3, n_layers // 2)))
    vabs = max(abs(data.min()), abs(data.max()), 1e-6)
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    ax.set_xlabel("Head")
    ax.set_ylabel("Phase-2 layer index")
    ax.set_title(title)
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L+{i+1}" for i in range(n_layers)])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _validate_cache_prompt_version(cache_dict: dict, task, cache_path: str) -> None:
    """Raise loudly if a cached tokenization was built against a different prompt.

    The cache_dict comes from pickle.load on a per-pair tokenization cache.
    Mismatch causes are: (a) the Task class bumped its prompt_version since the
    cache was built, or (b) the cache predates prompt_version tracking entirely.
    Either way, the cached tokens are stale and silently training on them would
    waste compute — so we fail loudly and tell the user to delete the file.
    """
    expected = task.get_prompt_version()
    cached = cache_dict.get("prompt_version")
    if cached != expected:
        raise RuntimeError(
            f"Cached tokenization at {cache_path} was built with "
            f"prompt_version={cached!r} but Task {type(task).__name__} now "
            f"declares prompt_version={expected!r}.  The cache is stale "
            f"(prompt changed, or cache predates prompt versioning).  "
            f"Delete the cache and rerun:\n    rm {cache_path}"
        )


import numpy as np
import torch
import torch._dynamo  # noqa: F401 (eagerly bind so torch._dynamo.reset() works inside train())
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# CPU-backed (gloo) group for python-object collectives. On the NCCL backend,
# gather_object/all_gather_object stage the pickled objects through fresh GPU
# byte tensors — allocations that can fail exactly when eval has the device at
# its ceiling (a step-10800 eval died with 'NCCL Error 1: unhandled cuda error'
# inside gather_object). None until train() creates it; None → default group,
# preserving behavior for importers that never initialize it (e.g. the RL trainer).
_OBJ_PG = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokengate.e2e_gate.gate_module.e2e_gate_module import set_gate_diag_stats  # noqa: E402
from tokengate.e2e_gate.gate_module.e2e_gate_module import (
    E2EGate, save_gate, load_gate, assert_fp32_params, swa_inplace_update,
)
from tokengate.e2e_gate.attn.flex_attention_utils import (
    flex_attention_causal,
    flex_attention_gate_bias,
    warmup_flex_attention,
    get_block_mask_cache_stats,
)
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    GATE_ATTN,
    set_attention_backends_for_layers,
    patch_model_with_flex_attention,
    set_gate_bias_for_layers,
    get_full_attn_layer_indices,
    set_phase2_stats_enabled,
    _phase2_attn_stats_buf as _flex_attn_stats_buf,
)
from tokengate.e2e_gate.gate_module.gate_forward import (
    apply_chat_template,
    find_article_boundaries,
    compute_prunable_span,
    forward_lora_baseline_batched,
    forward_with_gate_batched,
)
from tokengate.e2e_gate.decode.gate_decode import generate_batch
# Module alias so the profile path can read the opt-in PHASE_TIMING global
# (populated per step by forward_backward_with_gate_multi_K_detach when
# PROFILE_PHASES=1).  See training_bench/bench_homogenization.sh.
from tokengate.e2e_gate.gate_module import gate_forward as _gate_forward_mod
from tokengate.datasets.tasks import get_task, get_task_category, PRETRAIN_VAL_CATEGORIES

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False

# Module-level seeds removed — seeding is done inside train() with args.seed
# so that --seed actually controls all RNG sources.


# ── b_j distribution tracking ──────────────────────
def _sarle_bimodality(b_np) -> Optional[float]:
    """Sarle's bimodality coefficient.

    BC = (skew² + 1) / (kurt + 3·(n-1)²/((n-2)(n-3)))
    where ``kurt`` is excess kurtosis.  Heuristic: BC > 5/9 ≈ 0.555 ⇒ bimodal
    or uniform; BC < 0.555 ⇒ unimodal.

    Returns None on degenerate inputs (n < 4, zero variance, or non-positive
    denominator).
    """
    import numpy as np
    n = b_np.size
    if n < 4:
        return None
    m = float(b_np.mean())
    d = b_np - m
    m2 = float((d ** 2).mean())
    if m2 <= 0:
        return None
    m3 = float((d ** 3).mean())
    m4 = float((d ** 4).mean())
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2) - 3.0
    denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom <= 0:
        return None
    return (skew ** 2 + 1.0) / denom


def _log_bj_distribution(
    wandb_log: dict,
    bj_buf,
    n_K_eval: int,
    k_eval_values,
    n_tracked: int = 4,
) -> None:
    """Add b_j-distribution metrics to ``wandb_log`` (mutates in place).

    ``bj_buf`` is the post-eval drain of ``_gate_output_eval_buf`` — one
    ``(n_art,)`` (or ``(n_art, n_heads)`` for legacy per-head gate) tensor per
    gate forward call.  Each eval cycle runs the gate ``n_K_eval`` times per
    sample (once per K in K_EVAL_VALUES), so the first ``len(buf) // n_K_eval``
    entries cover the rank-local eval set exactly once.

    Logs:
      - For the first ``n_tracked`` rank-local samples (deterministic — rank 0
        always sees the same shard in the same order):
        ``bj_dist/article{i}/hist`` (W&B Histogram, 50 bins),
        ``mean/std/min/max/skew/kurt/bc``, and a ``threshold_K{k}`` scalar per
        K in ``k_eval_values`` (the b_j value at the K-th sorted position —
        moves of this scalar over training show where the kept/dropped
        boundary is migrating).
      - Aggregated across all rank-local samples this eval cycle:
        ``bj_dist/bc_median``, ``bj_dist/bc_p10``, ``bj_dist/bc_p90``.

    Cost: ~milliseconds per eval cycle, ~4 histograms + ~30 scalars to W&B.
    Caller must be rank-0 (the hook only populates ``_gate_output_eval_buf``
    on rank 0).
    """
    if not bj_buf or n_K_eval <= 0:
        return
    import numpy as np
    try:
        import wandb
    except ImportError:
        return  # no-op when wandb isn't installed

    n_per_K = len(bj_buf) // max(1, n_K_eval)
    if n_per_K == 0:
        return  # nothing to log (shouldn't happen but be defensive)
    samples = bj_buf[:n_per_K]   # first K's pass — one b_j per rank-local sample

    # Per-article tracked subset.
    for i in range(min(n_tracked, len(samples))):
        b = samples[i]
        if b.dim() == 3:
            # per_layer_gate: (n_art, n_layers_bias, n_heads) -> per-token scalar,
            # matching E2EGate.aggregate()'s pooled reduction (tracking only).
            b = b.mean(dim=(-2, -1))
        elif b.dim() == 2:
            # Legacy per-head gate: aggregate to scalar for tracking only.
            b = b.mean(dim=-1)
        b_np = b.detach().cpu().numpy().astype(np.float32)
        # Guard: if the gate output is non-finite (NaN/inf — e.g. a diverged
        # run), wandb.Histogram raises "autodetected range [nan, nan] is not
        # finite" and HARD-CRASHES the whole training process via the eval
        # path.  Logging must never kill the run; skip this article's b_j
        # metrics when non-finite and let the trainer's own NaN handling (or
        # the user) deal with the divergence.
        if not np.isfinite(b_np).all():
            continue
        prefix = f"bj_dist/article{i}"
        wandb_log[f"{prefix}/hist"] = wandb.Histogram(b_np, num_bins=50)
        wandb_log[f"{prefix}/mean"] = float(b_np.mean())
        wandb_log[f"{prefix}/std"] = float(b_np.std())
        wandb_log[f"{prefix}/min"] = float(b_np.min())
        wandb_log[f"{prefix}/max"] = float(b_np.max())
        bc = _sarle_bimodality(b_np)
        if bc is not None:
            wandb_log[f"{prefix}/bc"] = float(bc)
        # Sorted threshold-at-K scalars.  Per-sample sort is cheap (≤30k tokens).
        b_sorted, _ = torch.sort(b.detach().cpu(), descending=True)
        for k in k_eval_values:
            if 1 <= k <= b_sorted.numel():
                wandb_log[f"{prefix}/threshold_K{k}"] = float(b_sorted[k - 1].item())

    # Aggregate bimodality across rank-local samples this cycle.
    bcs = []
    for b in samples:
        if b.dim() == 3:
            b = b.mean(dim=(-2, -1))
        elif b.dim() == 2:
            b = b.mean(dim=-1)
        b_np = b.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(b_np).all():
            continue
        bc = _sarle_bimodality(b_np)
        if bc is not None:
            bcs.append(bc)
    if bcs:
        bcs_arr = np.array(bcs, dtype=np.float32)
        wandb_log["bj_dist/bc_median"] = float(np.median(bcs_arr))
        wandb_log["bj_dist/bc_p10"] = float(np.percentile(bcs_arr, 10))
        wandb_log["bj_dist/bc_p90"] = float(np.percentile(bcs_arr, 90))


# ── Attention helpers ─────────────────────────────────────────────────────────

# Aliases for backward compatibility
_apply_chat_template = apply_chat_template
_find_article_boundaries = find_article_boundaries
_compute_prunable_span = compute_prunable_span

# forward_with_gate, forward_with_gate_batched, and prefill_for_decode are
# imported from gate_forward.py — that module is the single source of truth
# for all forward-pass logic.

def _mem_census_dump(tag, rank):
    """E2E_EVAL_MEM_LOG diagnostics: dump live CUDA tensors by shape + allocator summary."""
    import gc as _g
    from collections import Counter as _C
    _c = _C()
    for _o in _g.get_objects():
        try:
            if torch.is_tensor(_o) and _o.is_cuda:
                _c[(tuple(_o.shape), str(_o.dtype))] += _o.numel() * _o.element_size()
        except Exception:
            pass
    print(f"[eval-mem] CENSUS {tag} rank={rank} allocated="
          f"{torch.cuda.memory_allocated()/2**30:.2f}GB python-visible={sum(_c.values())/2**30:.2f}GB",
          flush=True)
    for (_sh, _dt), _b in _c.most_common(25):
        print(f"  [eval-mem] {_b/2**30:8.3f}GB {_dt:14s} {_sh}", flush=True)
    print(torch.cuda.memory_summary(abbreviated=True), flush=True)



def evaluate(
    model,
    gate: E2EGate,
    tokenizer,
    eval_samples: List[dict],
    gate_layer: int,
    K_eval: int,
    task,
    max_gen_tokens: int = 10,
    device: str = "cuda",
    log_examples: int = 0,
    baseline_predictions: Optional[Dict[int, str]] = None,
    rank: int = 0,
    world_size: int = 1,
    eval_gen_batch_size: int = 1,
    tasks_by_name: Optional[Dict[str, "Task"]] = None,
    sparsity_lambda: float = 0.0,
) -> Tuple[float, float, List[dict]]:
    """Run inference on eval_samples.

    When `tasks_by_name` is provided, `eval_samples` may mix datasets/tasks:
    the task for templating, boundary detection, and scoring is resolved
    per-sample from `sample["task"]`.  This is what lets a single pooled
    metric-group eval set (e.g. all QA datasets) be scored in one call.  When
    None, the single `task` arg is used for every sample (legacy per-dataset
    path).  A per-dataset breakdown (`per_ds`) is returned as the final element
    so callers can report group micro-averages AND per-dataset numbers from the
    same generations.

    Returns
    -------
    accuracy : float
        Mean primary task accuracy (LLM judge when active, token-F1 otherwise).
    f1_accuracy : float
        Mean token-level F1 accuracy (always computed regardless of judge).
    eval_loss : float
        Mean teacher-forced cross-entropy loss across eval_samples
        (same objective as training, computed in mini-batches via
        forward_with_gate_batched).
    eval_loss_letter, eval_loss_eos : float
        Sub-components of eval_loss.
    logged_responses : List[dict]
        Up to log_examples dicts with keys: sample_idx, n_art, K, target,
        prediction, correct.  Empty when log_examples=0.
    rougeL_vs_baseline : float or None
        Mean ROUGE-L of compressed predictions against no-compression baseline
        predictions.  Only computed when baseline_predictions is provided and
        the task has a ROUGE scorer.  None otherwise.
    mean_n_tokens_full : float
        Mean full prompt length (system + chat template + article + question),
        averaged over eval samples.  Same denominator as the original metric.
    mean_n_tokens_pruned : float
        Mean total tokens sent to the model after pruning — includes the
        always-kept non-prunable prefix/suffix plus the gate-selected article
        tokens.  Use mean_n_tokens_pruned / mean_n_tokens_full to read the
        KV-cache footprint shrinkage.
    mean_n_tokens_full_art : float
        Mean number of *article* (prunable) tokens per sample.  Excludes
        non-prunable prefix/suffix.
    mean_n_tokens_pruned_art : float
        Mean number of *article* tokens the gate kept.  Use
        mean_n_tokens_pruned_art / mean_n_tokens_full_art to read the gate's
        article-level kept fraction (matches train/mean_kept_frac semantics).
    """
    from tokengate.generators import _strip_thinking
    gate.eval()
    model.eval()

    def _task_of(_s):
        """Task for a sample — per-sample when tasks_by_name is given (pooled
        mixed-task group set), else the single `task` arg (legacy path)."""
        if tasks_by_name is not None:
            return tasks_by_name[_s["task"]]
        return task

    correct = 0
    f1_correct = 0
    n_logged = 0
    rouge_vs_baseline_sum = 0.0
    n_rouge_baseline = 0
    n_tokens_full_sum = 0          # sum of full prompt length (incl. non-prunable prefix/suffix)
    n_tokens_pruned_sum = 0        # sum of total kept after pruning (incl. non-prunable)
    n_tokens_full_art_sum = 0      # sum of n_art (prunable article length only)
    n_tokens_pruned_art_sum = 0    # sum of n_kept_art (gate-kept article tokens only)
    n_tokens_valid = 0         # count of samples contributing to the two sums above
    # Per-dataset breakdown (the "cheap extra"): same generations, bucketed by
    # source dataset.  acc/f1/n/token sums; averaged + gathered at the end.
    local_per_ds: Dict[str, dict] = {}
    loss_batch_items: List[dict] = []  # accumulate for teacher-forced loss
    logged_responses: List[dict] = []

    if os.environ.get("E2E_EVAL_MEM_LOG") == "1":
        print(f"[eval-mem] rank={rank} evaluate() ENTRY allocated="
              f"{torch.cuda.memory_allocated()/2**30:.2f}GB", flush=True)
    my_indices = list(range(rank, len(eval_samples), world_size))
    _eff_batch = max(1, eval_gen_batch_size)

    for _batch_start in range(0, len(my_indices), _eff_batch):
        _batch_gidxs = my_indices[_batch_start: _batch_start + _eff_batch]

        # ── Compute article boundaries for each sample in the mini-batch ─────
        _batch_meta: List[dict] = []
        for _gi in _batch_gidxs:
            _s = eval_samples[_gi]
            _ts = _task_of(_s)
            _pids = _apply_chat_template(tokenizer, _s["input"], use_chat_template=_ts.use_chat_template)
            _nc   = _ts.get_non_compressible_context(_s["article"])
            _eids = _apply_chat_template(tokenizer, _ts.format_prompt(_nc), use_chat_template=_ts.use_chat_template)
            _art_start, _art_end = _compute_prunable_span(_pids, _eids)
            _n_art     = _art_end - _art_start
            _batch_meta.append({
                "global_idx": _gi,
                "sample": _s,
                "prompt_ids": _pids,
                "article_start": _art_start,
                "article_end": _art_end,
                "n_art": _n_art,
                "valid": _n_art > 0 and _art_start < _art_end,
            })

        # ── Accumulate loss items for all valid samples ───────────────────────
        for _m in _batch_meta:
            if not _m["valid"]:
                continue
            _aid = tokenizer.encode(
                _m["sample"]["target"] + tokenizer.eos_token, add_special_tokens=False,
            )
            loss_batch_items.append({
                "input_ids": list(_m["prompt_ids"]) + _aid,
                "article_start": _m["article_start"],
                "article_end":   _m["article_end"],
                "ctx_len":  len(_m["prompt_ids"]),
                "n_ans":    len(_aid),
                "K": max(1, min(K_eval, _m["n_art"])),
            })

        _valid_meta = [_m for _m in _batch_meta if _m["valid"]]
        if not _valid_meta:
            continue

        input_device = next(model.parameters()).device
        inner = model.model

        # ── Batched prefill (or fall back to sequential on OOM) ───────────────
        _answers: List[str] = []
        _kept_pos_list: List[torch.Tensor] = []   # per sample, (1, n_kept)
        _K_clamped_list: List[int] = []
        _n_art_list: List[int] = [_m["n_art"] for _m in _valid_meta]
        _art_start_list: List[int] = [_m["article_start"] for _m in _valid_meta]
        _art_end_list:   List[int] = [_m["article_end"]   for _m in _valid_meta]
        _pids_list: List[List[int]] = [_m["prompt_ids"] for _m in _valid_meta]

        _use_batch = len(_valid_meta) > 1
        _oom_fallback = False

        # Chunk the batched prefill. The old code passed the ENTIRE rank shard as one
        # batch, right-padded to the shard's longest prompt — on long-prompt mixes
        # (v16a: ~28k tokens, ~13 samples/rank single-node) the padded full-model KV
        # cache alone is ~50GB+, so the batched path OOM'd and SILENTLY fell back to
        # the slow sequential loop on every eval. (This was NOT the cause of the v16a
        # step-0 crashes — that was grad-mode decode retention, fixed separately with
        # no_grad — but unchunked batches waste the batched path exactly on the runs
        # that need it most.) Chunks preserve sample order (results append
        # positionally), are capped at eval_gen_batch_size samples, and at
        # _EVAL_PAD_BUDGET padded token slots (chunk_len × longest prompt in chunk) —
        # 48k = the proven-safe qasper operating point (4 × ~12k). A single
        # over-budget prompt still forms its own chunk.
        _EVAL_PAD_BUDGET = 48_000
        _chunks: List[List[int]] = []
        _cur: List[int] = []
        for _ci, _m in enumerate(_valid_meta):
            _trial = _cur + [_ci]
            _pad = len(_trial) * max(len(_valid_meta[_j]["prompt_ids"]) for _j in _trial)
            if _cur and (len(_trial) > eval_gen_batch_size or _pad > _EVAL_PAD_BUDGET):
                _chunks.append(_cur)
                _cur = [_ci]
            else:
                _cur = _trial
        if _cur:
            _chunks.append(_cur)

        if _use_batch:
            try:
                for _chunk in _chunks:
                    # Shared decode utility (gate_decode.generate_batch) —
                    # the SAME implementation the benchmark harness uses, so
                    # in-training eval and bench eval cannot drift.  no_grad
                    # (load-bearing: grad-mode decode retained every step's
                    # KV cache, the v16a step-0 OOM) and the pad-slot /
                    # first-EOS / form-decode / below-gate-routing fixes all
                    # live inside it.
                    _res = generate_batch(
                        model, gate, tokenizer,
                        [_pids_list[_j] for _j in _chunk],
                        [_art_start_list[_j] for _j in _chunk],
                        [_art_end_list[_j] for _j in _chunk],
                        gate_layer, K_eval, max_gen_tokens,
                        decode_phase2_form=True,
                    )
                    _answers.extend(_res.answers)
                    for _b in range(len(_chunk)):
                        _kept_pos_list.append(
                            _res.kept_pos[_b:_b + 1, :_res.n_kepts[_b]].cpu())
                        _K_clamped_list.append(_res.K_clampeds[_b])
                    del _res
                    torch.cuda.empty_cache()
                    if os.environ.get("E2E_EVAL_MEM_LOG") == "1":
                        _al = torch.cuda.memory_allocated() / 2**30
                        print(f"[eval-mem] rank={rank} chunk done ({len(_chunk)} samples, "
                              f"maxS={max(len(_pids_list[_j]) for _j in _chunk)}) "
                              f"allocated={_al:.2f}GB", flush=True)
                        if _al > 50:
                            _mem_census_dump("chunk-high-water", rank)
                            raise SystemExit("[eval-mem] high-water stop")

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                _oom_fallback = True
                if os.environ.get("E2E_EVAL_MEM_LOG") == "1":
                    print(f"[eval-mem] rank={rank} BATCHED OOM -> fallback "
                          f"(allocated={torch.cuda.memory_allocated()/2**30:.2f}GB)", flush=True)

        if not _use_batch or _oom_fallback:
            if _oom_fallback:
                # Hygiene before retrying on a memory-starved GPU: collect any cyclic
                # garbage from the failed attempt and return cached blocks. NOTE this
                # was originally added under the (disproven) theory that pinned
                # exception frames caused the v16a step-0 OOMs — the census showed the
                # real culprit was grad-mode decode retaining every KV-cache step
                # (fixed with no_grad on the decode loops). Kept because it is cheap,
                # runs only on the rare OOM-fallback path, and can only help.
                import gc as _gc_oom
                _gc_oom.collect()
                torch.cuda.empty_cache()
            # Sequential fallback: the SAME shared utility at B=1 (a batch of
            # one is pad-free, so this is exactly the old single-sample path).
            # Skips samples the chunked batched path already completed
            # (appends are positional).
            _n_done = len(_answers)
            for _mi, _m in enumerate(_valid_meta[_n_done:], start=_n_done):
                if os.environ.get("E2E_EVAL_MEM_LOG") == "1":
                    _al = torch.cuda.memory_allocated() / 2**30
                    print(f"[eval-mem] rank={rank} seq sample {_mi+1}/{len(_valid_meta)} START "
                          f"S={len(_m['prompt_ids'])} allocated={_al:.2f}GB", flush=True)
                    if _al > 50:
                        _mem_census_dump("seq-high-water", rank)
                        raise SystemExit("[eval-mem] high-water stop")
                _res = generate_batch(
                    model, gate, tokenizer,
                    [_m["prompt_ids"]], [_m["article_start"]], [_m["article_end"]],
                    gate_layer, K_eval, max_gen_tokens,
                    decode_phase2_form=True,
                )
                _answers.append(_res.answers[0])
                _kept_pos_list.append(_res.kept_pos[:1, :_res.n_kepts[0]].cpu())
                _K_clamped_list.append(_res.K_clampeds[0])
                del _res
                if os.environ.get("E2E_EVAL_MEM_LOG") == "1":
                    _al = torch.cuda.memory_allocated() / 2**30
                    print(f"[eval-mem] rank={rank} seq sample {_mi+1}/{len(_valid_meta)} "
                          f"S={len(_m['prompt_ids'])} allocated={_al:.2f}GB", flush=True)
                    if _al > 55:
                        _mem_census_dump("seq-high-water-end", rank)
                        raise SystemExit("[eval-mem] high-water stop")

        # ── Score and accumulate metrics ──────────────────────────────────────
        for _bi, (_m, _answer, _kpos_cpu, _K_clamped) in enumerate(
            zip(_valid_meta, _answers, _kept_pos_list, _K_clamped_list)
        ):
            _global_idx   = _m["global_idx"]
            _sample       = _m["sample"]
            _article_start = _m["article_start"]
            _article_end   = _m["article_end"]
            _n_art         = _m["n_art"]
            _pids          = _m["prompt_ids"]

            _t = _task_of(_sample)
            acc_i = _t.evaluate(_answer, _sample["target"], sample=_sample)
            f1_i  = _t.evaluate_f1(_answer, _sample["target"], sample=_sample)
            correct    += acc_i
            f1_correct += f1_i

            # Two parallel token accounting pairs.  Both are useful and W&B
            # logs both:
            #   * (n_tokens_full, n_tokens_pruned) — TOTAL, including the
            #     non-prunable system+chat-template prefix and post-article
            #     question that the gate cannot drop.  This is the KV-cache
            #     footprint shrinkage.
            #   * (n_tokens_full_art, n_tokens_pruned_art) — ARTICLE only,
            #     matching the gate's actual decision domain.  Use this for
            #     the article-level kept fraction; parity with the train-time
            #     `train/mean_kept_frac` (which is also article-only).
            _n_tokens_full = len(_pids)
            _n_tokens_pruned = int(_kpos_cpu.shape[1])
            _n_tokens_full_art = _n_art
            _n_tokens_pruned_art = int(_K_clamped)
            n_tokens_full_sum += _n_tokens_full
            n_tokens_pruned_sum += _n_tokens_pruned
            n_tokens_full_art_sum += _n_tokens_full_art
            n_tokens_pruned_art_sum += _n_tokens_pruned_art
            n_tokens_valid += 1

            # Per-dataset breakdown (cheap extra) from the same generation.
            _ds_b = local_per_ds.setdefault(
                _sample.get("dataset", "?"),
                {"acc": 0.0, "f1": 0.0, "n": 0,
                 "ntf": 0, "ntp": 0, "ntfa": 0, "ntpa": 0},
            )
            _ds_b["acc"] += acc_i
            _ds_b["f1"]  += f1_i
            _ds_b["n"]   += 1
            _ds_b["ntf"] += _n_tokens_full
            _ds_b["ntp"] += _n_tokens_pruned
            _ds_b["ntfa"] += _n_tokens_full_art
            _ds_b["ntpa"] += _n_tokens_pruned_art

            _rouge_scorer = getattr(_t, '_get_scorer', None)
            if baseline_predictions is not None and _rouge_scorer is not None:
                _bp = baseline_predictions.get(_global_idx)
                if _bp is not None:
                    _sc = _rouge_scorer().score(_bp, _answer)
                    rouge_vs_baseline_sum += _sc["rougeL"].fmeasure
                    n_rouge_baseline += 1

            if n_logged < log_examples:
                _kp_sq = _kpos_cpu.squeeze(0)
                _is_art = (_kp_sq >= _article_start) & (_kp_sq < _article_end)
                _art_kept_local = sorted(_kp_sq[_is_art].sub(_article_start).tolist())
                _kept_art_ids = [_pids[_article_start + i] for i in _art_kept_local]
                _kept_tokens_text = tokenizer.decode(_kept_art_ids, skip_special_tokens=True)
                _kept_pids = [_pids[i] for i in _kp_sq.tolist()]
                _pruned_ctx = tokenizer.decode(_kept_pids, skip_special_tokens=False)
                _full_prompt = tokenizer.decode(_pids, skip_special_tokens=False)
                _pre_article_prompt = tokenizer.decode(_pids[:_article_start], skip_special_tokens=False)
                _prunable_article = tokenizer.decode(_pids[_article_start:_article_end], skip_special_tokens=False)
                _prompt_tail = tokenizer.decode(_pids[_article_end:], skip_special_tokens=False)
                _judge_expl  = getattr(_t, "last_judge_explanation", None)
                _judge_prompt = getattr(_t, "last_judge_prompt", None)
                _acc_tag = _t.acc_metric_name
                _f1_tag  = _t.secondary_metric_name
                if _judge_expl:
                    _score_str = f"judge={acc_i:.1f}  {_f1_tag}={f1_i:.2f}"
                elif _acc_tag == _f1_tag:
                    _score_str = f"{_acc_tag}={acc_i:.2f}"
                else:
                    _score_str = f"{_acc_tag}={acc_i:.2f}  {_f1_tag}={f1_i:.2f}"
                if rank == 0:
                    print(f"\n  [eval example {n_logged + 1}]"
                          f"  n_art={_n_art}  K={_K_clamped}  {_score_str}")
                    print(f"    prompt tail : {_prompt_tail[-200:]!r}")
                    print(f"    generated   : {_answer!r}")
                    print(f"    target      : {_sample['target']!r}")
                    if _judge_expl:
                        print(f"    judge       : {_judge_expl!r}")
                logged_responses.append({
                    "sample_idx":     n_logged,
                    "n_art":          _n_art,
                    "K":              _K_clamped,
                    "n_tokens_full":   _n_tokens_full,
                    "n_tokens_pruned": _n_tokens_pruned,
                    "n_tokens_full_art":   _n_tokens_full_art,
                    "n_tokens_pruned_art": _n_tokens_pruned_art,
                    "target":         _sample["target"],
                    "prediction":     _answer,
                    "correct":        acc_i,
                    "correct_f1":     f1_i,
                    "full_prompt":         _full_prompt,
                    "pre_article_prompt":  _pre_article_prompt,
                    "prunable_article":    _prunable_article,
                    "post_article_prompt": _prompt_tail,
                    "kept_tokens":    _kept_tokens_text,
                    "pruned_context": _pruned_ctx,
                    "judge_prompt":   _judge_prompt or "",
                    "judge_explanation": _judge_expl or "",
                })
                n_logged += 1

    # ── Teacher-forced eval loss ──────────────────────────────────────────────
    # Run forward_with_gate_batched in mini-batches on the accumulated items.
    # This mirrors the training loss exactly (same K, same phase 1/2 split).
    total_loss = 0.0
    total_loss_letter = 0.0
    total_loss_eos = 0.0
    n_loss_valid = 0
    # Enable rank-local sparsity-term accumulation through the eval loss path
    # so we can report `eval/sparsity_loss` as the matching component of the
    # eval loss decomposition (ce + sparsity, same as training).
    _gate_inner_eval = gate.module if hasattr(gate, "module") else gate
    _gate_inner_eval._sparsity_term = None
    _gate_inner_eval._sparsity_eval_capture = True
    # Pad-budget chunking (mirrors the prefill chunker): fixed size-4 batches of
    # ~24k-token rows OOM'd every mini-batch under the 39k eval cap, and the
    # except below SKIPPED them silently — the whole group's loss printed 0.0
    # (observed on narrqa-iso-v3, 2026-08-20). Cap a chunk at 4 rows AND 48k
    # padded slots (chunk_len × longest row), singleton chunks for over-budget rows.
    # 24k, NOT the prefill's 48k: this path materializes full-vocab logits over
    # the whole padded sequence (B x seq x 151k x 2B — 2x23k rows ≈ 14GB of
    # logits alone), so its budget must be ~half the KV-bound prefill's.
    # Observed on narrqa-iso-v3r32 (2026-08-20): n=2 @ 23k and n=3 @ 13k chunks
    # OOM'd under the 48k budget.
    _LOSS_PAD_BUDGET = 24_000
    _loss_chunks: List[List[int]] = []
    _cur: List[int] = []
    for _ci, _item in enumerate(loss_batch_items):
        _trial = _cur + [_ci]
        _pad = len(_trial) * max(len(loss_batch_items[_j]["input_ids"]) for _j in _trial)
        if _cur and (len(_trial) > 4 or _pad > _LOSS_PAD_BUDGET):
            _loss_chunks.append(_cur)
            _cur = [_ci]
        else:
            _cur = _trial
    if _cur:
        _loss_chunks.append(_cur)
    for _chunk in _loss_chunks:
        _mini = [loss_batch_items[_j] for _j in _chunk]
        try:
            # no_grad is essential: this reuses the TRAINING forward, and with any
            # trainable adapter attached, grad mode retains the full-sequence
            # activations of all 36 layers for a backward that never runs —
            # tens of GB at 30k+ tokens. (Same bug family as the 2026-08-16
            # decode-loop fix; exposed by the 39k eval cap.)
            with torch.no_grad():
                _logits, _loss_mask, _ = forward_with_gate_batched(
                    model, gate, _mini, gate_layer,
                    frozen_backbone=True, prune_phase1_kv=False,
                )
            for _b, _item in enumerate(_mini):
                _mask = _loss_mask[_b].to(_logits.device)
                if not _mask.any():
                    continue
                _sl = _logits[_b, _mask, :]
                _lbl = torch.tensor(
                    _item["input_ids"][_item["ctx_len"]: _item["ctx_len"] + _item["n_ans"]],
                    dtype=torch.long, device=_logits.device,
                )
                if _sl.shape[0] != _lbl.shape[0]:
                    continue
                total_loss += F.cross_entropy(_sl, _lbl).item()
                if _sl.shape[0] >= 2:
                    total_loss_letter += F.cross_entropy(_sl[:-1], _lbl[:-1]).item()
                    total_loss_eos += F.cross_entropy(_sl[-1:], _lbl[-1:]).item()
                n_loss_valid += 1
        except torch.cuda.OutOfMemoryError:
            # Loud skip: silent drops here zeroed a whole group's loss series.
            print(f"[eval-loss] rank={rank} OOM on loss chunk (n={len(_mini)}, "
                  f"max_len={max(len(x['input_ids']) for x in _mini)}), skipped",
                  flush=True)
            torch.cuda.empty_cache()

    # Drain the rank-local sparsity-term accumulation produced by the loss loop.
    # _spars_term is a tensor sum-of-soft over the rank's eval samples;
    # multiply by lambda → rank-local sparsity-loss SUM (per-sample-mean is
    # taken after the cross-rank all_reduce, paralleling CE).
    _gate_inner_eval._sparsity_eval_capture = False
    _spars_term = getattr(_gate_inner_eval, "_sparsity_term", None)
    total_sparsity_loss = 0.0
    if _spars_term is not None and sparsity_lambda != 0.0:
        total_sparsity_loss = float((sparsity_lambda * _spars_term).detach().item())
    _gate_inner_eval._sparsity_term = None

    # ── Aggregate across ranks ────────────────────────────────────────────────
    _stats = torch.tensor(
        [correct, f1_correct, rouge_vs_baseline_sum, float(n_rouge_baseline),
         total_loss, total_loss_letter, total_loss_eos, float(n_loss_valid),
         float(n_tokens_full_sum), float(n_tokens_pruned_sum), float(n_tokens_valid),
         float(n_tokens_full_art_sum), float(n_tokens_pruned_art_sum),
         total_sparsity_loss],
        dtype=torch.float64, device=device,
    )
    if world_size > 1:
        dist.all_reduce(_stats, op=dist.ReduceOp.SUM)
    (agg_correct, agg_f1, agg_rouge_sum, agg_rouge_n,
     agg_loss, agg_loss_letter, agg_loss_eos, agg_loss_n,
     agg_n_tokens_full, agg_n_tokens_pruned, agg_n_tokens_valid,
     agg_n_tokens_full_art, agg_n_tokens_pruned_art,
     agg_sparsity_loss) = [s.item() for s in _stats]

    accuracy = agg_correct / max(len(eval_samples), 1)
    f1_accuracy = agg_f1 / max(len(eval_samples), 1)
    eval_loss = agg_loss / max(agg_loss_n, 1)
    eval_loss_letter = agg_loss_letter / max(agg_loss_n, 1)
    eval_loss_eos = agg_loss_eos / max(agg_loss_n, 1)
    # CE component (alias of eval_loss; explicit so the eval decomposition reads
    # ce + sparsity == total in W&B panels) + sparsity component, both per-sample
    # means like CE.
    eval_ce_loss = eval_loss
    eval_sparsity_loss = agg_sparsity_loss / max(agg_loss_n, 1)
    rougeL_vs_baseline = agg_rouge_sum / agg_rouge_n if agg_rouge_n > 0 else None
    mean_n_tokens_full = agg_n_tokens_full / max(agg_n_tokens_valid, 1)
    mean_n_tokens_pruned = agg_n_tokens_pruned / max(agg_n_tokens_valid, 1)
    mean_n_tokens_full_art = agg_n_tokens_full_art / max(agg_n_tokens_valid, 1)
    mean_n_tokens_pruned_art = agg_n_tokens_pruned_art / max(agg_n_tokens_valid, 1)

    if world_size > 1:
        _gathered = [None] * world_size
        dist.gather_object(logged_responses, _gathered if rank == 0 else None, dst=0,
                           group=_OBJ_PG)
        if rank == 0:
            logged_responses = [r for shard in _gathered for r in shard]
        else:
            logged_responses = []

    # Reduce the per-dataset breakdown across ranks (sum the per-rank shards),
    # then convert to averaged acc/f1.  Available on every rank.
    if world_size > 1:
        _pd_gathered = [None] * world_size
        dist.all_gather_object(_pd_gathered, local_per_ds, group=_OBJ_PG)
    else:
        _pd_gathered = [local_per_ds]
    _pd_sum: Dict[str, dict] = {}
    for _shard in _pd_gathered:
        for _ds, _e in _shard.items():
            _acc = _pd_sum.setdefault(
                _ds,
                {"acc": 0.0, "f1": 0.0, "n": 0,
                 "ntf": 0, "ntp": 0, "ntfa": 0, "ntpa": 0},
            )
            for _k in ("acc", "f1", "n", "ntf", "ntp", "ntfa", "ntpa"):
                # Older shards (pre-fix) may lack the article-only counts; default to 0.
                _acc[_k] += _e.get(_k, 0)
    per_ds: Dict[str, dict] = {
        _ds: {
            "accuracy": _e["acc"] / max(_e["n"], 1),
            "f1":       _e["f1"]  / max(_e["n"], 1),
            "n_samples": _e["n"],
            "n_tokens_full":       _e["ntf"]  / max(_e["n"], 1),
            "n_tokens_pruned":     _e["ntp"]  / max(_e["n"], 1),
            "n_tokens_full_art":   _e["ntfa"] / max(_e["n"], 1),
            "n_tokens_pruned_art": _e["ntpa"] / max(_e["n"], 1),
        }
        for _ds, _e in _pd_sum.items()
    }

    return (accuracy, f1_accuracy, eval_loss, eval_loss_letter, eval_loss_eos,
            logged_responses, rougeL_vs_baseline,
            mean_n_tokens_full, mean_n_tokens_pruned,
            mean_n_tokens_full_art, mean_n_tokens_pruned_art, per_ds,
            eval_ce_loss, eval_sparsity_loss)


# ── No-compression baseline eval (distributed across all ranks) ───────────────

_SCAFFOLD_DROP_WARNINGS = 0


def _warn_scaffold_infeasible(L_list, plen, n_art, where, is_main):
    """Audit fix (2026-08-28): a phase-2 target L is dropped SILENTLY when the
    non-prunable scaffold (prompt_len) >= L — a config/dataset mismatch that
    would null a loss lane for the whole run (distinct from the documented
    per-sample drop when n_art is too small for high L). Warn loudly on the
    first few occurrences instead of staying silent."""
    global _SCAFFOLD_DROP_WARNINGS
    if not is_main or _SCAFFOLD_DROP_WARNINGS >= 3:
        return
    bad = [int(L) for L in L_list if int(L) != -1 and int(L) - plen <= 0]
    if bad:
        _SCAFFOLD_DROP_WARNINGS += 1
        print(f"[train][WARN] {where}: phase-2 target(s) {bad} dropped because "
              f"the non-prunable scaffold (prompt_len={plen}) >= L — these loss "
              f"lanes get NO gradient from such samples (n_art={n_art}). If this "
              f"repeats, the bucket list is incompatible with this dataset's "
              f"scaffold. (warning {_SCAFFOLD_DROP_WARNINGS}/3)", flush=True)


@torch.no_grad()
def evaluate_no_compression(
    model,
    tokenizer,
    eval_samples: List[dict],
    task,
    max_gen_tokens: int = 10,
    device: str = "cuda",
    log_examples: int = 3,
    world_size: int = 1,
    rank: int = 0,
    wandb_run=None,
    output_dir: str = "",
    source: str = "no_compress",
    label: str = "no-compress eval",
    tasks_by_name: Optional[Dict[str, "Task"]] = None,
    return_per_ds: bool = False,
) -> Tuple[float, float, float]:
    """Evaluate model accuracy on full (uncompressed) context, distributed across ranks.

    Each rank processes eval_samples[rank::world_size].  After generation,
    accuracy and loss sums are all_reduced so every rank has the global result.
    Printing and W&B logging are done by rank 0 only.

    Returns (accuracy, f1_accuracy, loss, predictions_by_idx) aggregated over all
    ranks.  loss is teacher-forced cross-entropy on answer tokens (same objective
    as training, but with full uncompressed context).

    tasks_by_name : when provided, the task for each sample is resolved per-sample
        via tasks_by_name[sample["task"]] (so a mixed-task pool — e.g. a pooled
        per-metric-group eval set — is scored with each sample's own metric).
        When None, `task` is applied to every sample (single-task back-compat).
    return_per_ds : when True, append a 5th return value: a per-dataset breakdown
        {dataset: {accuracy, f1, loss(None), n_samples, acc_metric_tag,
        f1_metric_tag}} bucketed from the SAME generations (rank 0 only; other
        ranks return {}).  Mirrors the gated eval's cheap-extra per-ds breakdown.
    """
    from tokengate.generators import _strip_thinking
    from tokengate.e2e_gate.attn.qwen3_flex_attention import (
        set_attn_scale_for_layers, set_ssmax_ratio_for_layers)
    model.eval()
    inner = model.model
    # Audit fix (2026-08-28): clear any stale gate bias / attn_scale / ssmax
    # left by a preceding gated-eval arm. Previously safe only by call
    # ordering — an eval-arm reordering would have silently biased the
    # no-compress baseline (compiled kernel, no error).
    set_gate_bias_for_layers(model, 0, len(inner.layers), None)
    set_attn_scale_for_layers(model, 0, len(inner.layers), None)
    set_ssmax_ratio_for_layers(model, 0, len(inner.layers), None)

    my_samples = eval_samples[rank::world_size]
    correct = 0.0
    f1_correct = 0.0
    n_ok = 0
    console_logged = 0  # how many examples printed to console (capped at log_examples)
    all_responses: List[dict] = []  # ALL scored samples on this rank, for W&B
    # Parallel per-sample list kept OUT of all_responses (whose dict schema is
    # locked to _BASELINE_RESPONSE_COLS by the CSV/W&B writers).  Carries the
    # source dataset + metric tags so rank 0 can bucket a per-dataset breakdown
    # after gather when return_per_ds=True.
    _per_sample_ds: List[dict] = []
    _loss_items: List[tuple] = []  # (prompt_ids, answer_ids) for teacher-forced loss

    # local_batch_size controls how many consecutive samples we attempt before
    # an OOM triggers a retry.  It is purely local — it never propagates to
    # args.batch_size or any other training state.  We start at 1 because
    # full-context prefills are memory-heavy; raise it here if your GPU has
    # headroom and you want faster eval.
    local_batch_size = 1

    idx = 0
    while idx < len(my_samples):
        chunk = my_samples[idx : idx + local_batch_size]
        n_done_in_chunk = 0
        oom_hit = False

        for sample in chunk:
            _t = tasks_by_name[sample["task"]] if tasks_by_name is not None else task
            prompt_ids = _apply_chat_template(tokenizer, sample["input"], use_chat_template=_t.use_chat_template)
            _answer_ids = tokenizer.encode(
                sample["target"] + tokenizer.eos_token, add_special_tokens=False,
            )
            _loss_items.append((prompt_ids, _answer_ids))
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            seq_len = input_ids.shape[1]

            try:
                # ── Full-context prefill (all layers, no pruning) ─────────────
                hidden_states = inner.embed_tokens(input_ids)
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                pe = inner.rotary_emb(hidden_states, position_ids)
                cache = DynamicCache(config=model.config)
                cache_pos = torch.arange(seq_len, device=device)

                for layer_i in range(len(inner.layers)):
                    hidden_states = inner.layers[layer_i](
                        hidden_states,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=cache_pos,
                        position_embeddings=pe,
                    )

                hidden_states = inner.norm(hidden_states)
                first_token = model.lm_head(hidden_states)[:, -1].argmax(-1, keepdim=True)

                # ── Greedy decode ─────────────────────────────────────────────
                generated = [first_token]
                next_pos = seq_len
                cache_next = seq_len

                for _ in range(max_gen_tokens - 1):
                    if generated[-1].item() == tokenizer.eos_token_id:
                        break
                    cur_tok = generated[-1]
                    hs_dec = inner.embed_tokens(cur_tok)
                    pos = torch.tensor([[next_pos]], device=device)
                    pe_tok = inner.rotary_emb(hs_dec, pos)
                    cp = torch.tensor([cache_next], device=device)

                    for layer_i in range(len(inner.layers)):
                        hs_dec = inner.layers[layer_i](
                            hs_dec,
                            attention_mask=None,
                            position_ids=pos,
                            past_key_values=cache,
                            use_cache=True,
                            cache_position=cp,
                            position_embeddings=pe_tok,
                        )

                    hs_dec = inner.norm(hs_dec)
                    tok = model.lm_head(hs_dec)[:, -1].argmax(-1, keepdim=True)
                    generated.append(tok)
                    next_pos += 1
                    cache_next += 1

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom_hit = True
                break

            if generated and generated[-1].item() == tokenizer.eos_token_id:
                generated = generated[:-1]

            answer = (
                _strip_thinking(tokenizer.decode(
                    torch.cat(generated, dim=1)[0].cpu(), skip_special_tokens=True
                ))
                if generated else ""
            )

            acc_i = _t.evaluate(answer, sample["target"], sample=sample)
            f1_i = _t.evaluate_f1(answer, sample["target"], sample=sample)
            correct += acc_i
            f1_correct += f1_i
            # global index of this sample in eval_samples
            global_idx = rank + n_ok * world_size
            n_ok += 1
            n_done_in_chunk += 1

            # Collect for W&B/CSV logging (all ranks, all samples)
            judge_prompt_nc = getattr(_t, "last_judge_prompt", None)
            judge_expl_nc = getattr(_t, "last_judge_explanation", None)
            all_responses.append({
                "sample_idx": global_idx,
                "prompt": sample["input"],
                "target": sample["target"],
                "prediction": answer,
                "correct": acc_i,
                "correct_f1": f1_i,
                "judge_prompt": judge_prompt_nc or "",
                "judge_explanation": judge_expl_nc or "",
            })
            # Parallel per-sample record for the per-dataset breakdown (kept out
            # of all_responses so the locked CSV/W&B schema isn't disturbed).
            _per_sample_ds.append({
                "dataset": sample.get("dataset"),
                "acc": acc_i,
                "f1": f1_i,
                "acc_metric_tag": _t.acc_metric_name,
                "f1_metric_tag": _t.secondary_metric_name,
            })

            # Console print (rank 0 only, capped at log_examples)
            if rank == 0 and console_logged < log_examples:
                judge_expl = getattr(_t, "last_judge_explanation", None)
                _acc_tag = _t.acc_metric_name
                _f1_tag  = _t.secondary_metric_name
                if judge_expl:
                    _score_str = f"judge={acc_i:.1f}  {_f1_tag}={f1_i:.2f}"
                elif _acc_tag == _f1_tag:
                    _score_str = f"{_acc_tag}={acc_i:.2f}"
                else:
                    _score_str = f"{_acc_tag}={acc_i:.2f}  {_f1_tag}={f1_i:.2f}"
                print(f"\n  [{label} example {console_logged + 1}]"
                      f"  n_tok={seq_len}  {_score_str}")
                print(f"    generated : {answer!r}")
                print(f"    target    : {sample['target']!r}")
                if judge_expl:
                    print(f"    judge     : {judge_expl!r}")
                console_logged += 1

        # Advance past the samples we finished; retry the failed one with a
        # smaller batch_size (which gives it a clean GPU memory slate).
        idx += n_done_in_chunk
        if oom_hit:
            if local_batch_size > 1:
                local_batch_size //= 2
                if rank == 0:
                    print(f"\n  [{label}] OOM — reducing local_batch_size to {local_batch_size}, retrying")
            else:
                raise RuntimeError(
                    f"[{label}] OOM on rank {rank} at local_batch_size=1 "
                    f"(seq_len={seq_len}).  Cannot evaluate this sample without "
                    f"truncating the context, which is not permitted."
                )

    # ── Teacher-forced loss (no compression) ─────────────────────────────────
    # Separate forward pass over prompt+answer with use_cache=False, one sample
    # at a time.  Done after generation so GPU memory from KV caches is freed.
    total_loss = 0.0
    n_loss_valid = 0
    for _prompt_ids, _answer_ids in _loss_items:
        try:
            _full = list(_prompt_ids) + _answer_ids
            _full_t = torch.tensor([_full], dtype=torch.long, device=device)
            _p = len(_prompt_ids)
            _a = len(_answer_ids)
            _logits = model(_full_t, use_cache=False).logits[0]  # [full_len, vocab]
            _shift = _logits[_p - 1 : _p + _a - 1]              # predicts answer tokens
            _lbl = torch.tensor(_answer_ids, dtype=torch.long, device=device)
            if _shift.shape[0] == _lbl.shape[0] and _a > 0:
                total_loss += F.cross_entropy(_shift, _lbl).item()
                n_loss_valid += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    # ── Aggregate across ranks ────────────────────────────────────────────────
    stats = torch.tensor(
        [correct, f1_correct, float(n_ok), total_loss, float(n_loss_valid)],
        dtype=torch.float64, device=device,
    )
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_correct, total_f1, total_n, agg_loss, agg_loss_n = (
        stats[0].item(), stats[1].item(), stats[2].item(),
        stats[3].item(), stats[4].item(),
    )

    accuracy = total_correct / max(total_n, 1)
    f1_accuracy = total_f1 / max(total_n, 1)
    nc_loss = agg_loss / max(agg_loss_n, 1)

    # ── Gather per-sample responses to rank 0 for logging ────────────────────
    if world_size > 1:
        gathered = [None] * world_size
        dist.gather_object(all_responses, gathered if rank == 0 else None, dst=0,
                           group=_OBJ_PG)
        if rank == 0:
            flat_responses = [r for shard in gathered for r in shard]
        else:
            flat_responses = []
    else:
        flat_responses = all_responses

    predictions_by_idx: dict = {}
    if rank == 0:
        _acc_tag = task.acc_metric_name
        _f1_tag  = task.secondary_metric_name
        _nc_metrics = (f"  {_acc_tag}={accuracy:.3f}  {_f1_tag}={f1_accuracy:.3f}"
                       if _acc_tag != _f1_tag else f"  {_acc_tag}={accuracy:.3f}")
        print(f"\n  [{label}]  n={int(total_n)}{_nc_metrics}"
              f"  loss={nc_loss:.4f}  (full context, no gate)")
        if output_dir:
            _log_baseline_responses(flat_responses, source, output_dir, wandb_run)
        predictions_by_idx = {r["sample_idx"]: r["prediction"] for r in flat_responses}

    if not return_per_ds:
        return accuracy, f1_accuracy, nc_loss, predictions_by_idx

    # ── Per-dataset breakdown (rank 0 only) ──────────────────────────────────
    # Gather the parallel per-sample records and bucket by source dataset.
    if world_size > 1:
        _gathered_ds = [None] * world_size
        dist.gather_object(_per_sample_ds, _gathered_ds if rank == 0 else None, dst=0,
                           group=_OBJ_PG)
        _flat_ds = [r for shard in _gathered_ds for r in shard] if rank == 0 else []
    else:
        _flat_ds = _per_sample_ds
    per_ds_metrics: Dict[str, dict] = {}
    if rank == 0:
        _by_ds: Dict[str, List[dict]] = {}
        for _r in _flat_ds:
            _by_ds.setdefault(_r["dataset"], []).append(_r)
        for _ds, _rs in _by_ds.items():
            _n = len(_rs)
            per_ds_metrics[_ds] = {
                "accuracy": sum(r["acc"] for r in _rs) / max(_n, 1),
                "f1": sum(r["f1"] for r in _rs) / max(_n, 1),
                "loss": None,  # per-ds loss not computed (matches gated per-ds)
                "n_samples": _n,
                "acc_metric_tag": _rs[0]["acc_metric_tag"],
                "f1_metric_tag": _rs[0]["f1_metric_tag"],
            }
    return accuracy, f1_accuracy, nc_loss, predictions_by_idx, per_ds_metrics




# ── Response logging helper ───────────────────────────────────────────────────

_RESPONSE_CSV_COLS = ["step", "dataset", "sample_idx", "n_art", "K",
                      "n_tokens_full", "n_tokens_pruned",
                      "n_tokens_full_art", "n_tokens_pruned_art",
                      "target", "prediction", "correct",
                      "correct_f1", "full_prompt", "pre_article_prompt", "prunable_article",
                      "post_article_prompt", "kept_tokens", "pruned_context",
                      "judge_prompt", "judge_explanation"]

_BASELINE_RESPONSE_COLS = ["source", "sample_idx", "prompt", "target", "prediction", "correct", "correct_f1",
                           "judge_prompt", "judge_explanation"]

# Max characters of any single text cell sent to a W&B Table.  The full,
# untruncated text is always written to the CSV on disk; W&B only needs enough
# to eyeball predictions vs. targets.  Without this cap, logging 20 examples ×
# (full_prompt + prunable_article + pruned_context + …) per eval cycle pushed
# multi-hundred-KB strings into W&B's in-memory/on-disk run buffer every cycle
# (the "Serializing object of type str that is N bytes" warnings) — a steady
# memory/IO drain over a long run.
_WANDB_CELL_MAX_CHARS = 2000


def _clip_cell(v, n: int = _WANDB_CELL_MAX_CHARS):
    """Truncate long strings for W&B tables; pass non-strings through."""
    if isinstance(v, str) and len(v) > n:
        return v[:n] + f"…[+{len(v) - n} chars]"
    return v


class PackedTrainStore:
    """List-of-dicts–compatible view over memory-mapped packed token data.

    Replaces the in-RAM ``List[dict]`` of training rows used by the
    pretraining-mixture loader.  Token ids live in per-dataset int32 ``.npy``
    files opened with ``mmap_mode='r'`` — they are paged in on demand by the OS
    and shared (one physical copy per node) across all ranks via the page
    cache, instead of being unpickled into ~9x-larger Python ``int`` lists and
    replicated per rank.  Only the small int64 ``meta`` table (one row of
    indices/boundaries per sample) is resident in process RAM.

    Indexing returns the exact dict shape a training step consumes
    (``full_ids`` + boundaries + ``dataset``/``task``).  The raw article/target
    *text* is intentionally absent — training never reads it (only the capped
    eval set, loaded separately, keeps text).  Supports ``len()``,
    ``store[i]`` and iteration, so ``random.choice`` / the shuffle-epoch
    sampler work unchanged.  ``meta`` is a single concatenated array across
    datasets, already sharded to this rank's slice.
    """

    __slots__ = ("_tok", "_ds_names", "_ds_tasks", "_meta")

    def __init__(self, token_arrays, ds_names, ds_tasks, meta):
        # token_arrays[k] is the mmap for dataset k; meta[:, DSI] indexes it.
        self._tok = token_arrays
        self._ds_names = ds_names
        self._ds_tasks = ds_tasks
        self._meta = meta  # (N, 8) int64: off,len,ctx,as,ae,nans,hash,dataset_idx

    # meta column for dataset index (appended after pack.HASH).
    _DSI = 7

    def __len__(self):
        return int(self._meta.shape[0])

    def __getitem__(self, i):
        from tokengate.datasets import pack
        m = self._meta[i]
        dsi = int(m[self._DSI])
        off = int(m[pack.OFF]); ln = int(m[pack.LEN])
        # .tolist() copies just this one row's tokens out of the mmap.
        ids = self._tok[dsi][off:off + ln].tolist()
        return {
            "full_ids": ids,
            "ctx_len": int(m[pack.CTX]),
            "article_start": int(m[pack.AS]),
            "article_end": int(m[pack.AE]),
            "n_ans": int(m[pack.NANS]),
            "dataset": self._ds_names[dsi],
            "task": self._ds_tasks[dsi],
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ── Vectorized helpers (avoid materializing dicts for bulk ops) ──────────
    def article_lengths(self) -> "np.ndarray":
        from tokengate.datasets import pack
        return self._meta[:, pack.AE] - self._meta[:, pack.AS]

    def dataset_idx(self) -> "np.ndarray":
        return self._meta[:, self._DSI]

    def dataset_name(self, i: int) -> str:
        return self._ds_names[int(self._meta[i, self._DSI])]


def _target_balanced_index_weights(map_path, ds_names, ds_tasks, row_counts, is_main):
    """Per-(dataset,task)-INDEX multiplicity for TARGET-BALANCED sampling.

    Draw semantics: pick an eval TARGET uniformly (1/N) → pick a ROW uniformly from
    that target's pooled training rows (the union of its pool members' rows).
    Realised over a shuffle-epoch as a per-index multiplicity: each of the N targets
    gets equal expected rows per epoch (R = included_rows / N), spread uniformly over
    its n_t pooled rows, so every row of every member of target t has the same weight
    w = R / n_t — members contribute in proportion to their size; a small member is
    NOT upweighted. (Before 2026-08-16 the middle draw was uniform over pool MEMBERS,
    giving each member 1/|pool| of the target's mass regardless of its row count.)
    Members in no target pool (excluded) get w = 0 (never sampled)."""
    import yaml
    m = yaml.safe_load(open(map_path))
    pool_of = {}                       # (dataset, task) -> (target, pool_size)
    for tgt, pool in m["targets"].items():
        for p in pool:
            pool_of[(p["dataset"], p["task"])] = (tgt, len(pool))
    n_targets = len(m["targets"])
    keys = [(ds_names[i], ds_tasks[i]) for i in range(len(ds_names))]
    included = sum(int(row_counts[i]) for i, k in enumerate(keys)
                   if k in pool_of and row_counts[i] > 0)
    # Total pooled rows per target (denominator of the uniform-over-rows draw).
    tgt_rows: Dict[str, int] = {}
    for i, k in enumerate(keys):
        if k in pool_of and row_counts[i] > 0:
            _tgt = pool_of[k][0]
            tgt_rows[_tgt] = tgt_rows.get(_tgt, 0) + int(row_counts[i])
    R = included / max(n_targets, 1)   # rows per target per epoch
    w = np.zeros(len(ds_names), dtype=np.float64)
    for i, k in enumerate(keys):
        n_i = int(row_counts[i])
        if k in pool_of and n_i > 0:
            _tgt = pool_of[k][0]
            w[i] = R / tgt_rows[_tgt]
    if is_main:
        print(f"[train] TARGET-BALANCED sampling (uniform over pooled rows) ← {map_path}: "
              f"{n_targets} targets, R≈{R:.0f} rows/target/epoch, epoch≈{included} rows",
              flush=True)
        excl = [k for i, k in enumerate(keys) if k not in pool_of]
        if excl:
            print(f"[train]   EXCLUDED (weight 0, never sampled): {excl}", flush=True)
        for i, k in enumerate(keys):
            if k in pool_of:
                _tgt, P = pool_of[k]
                n_i = int(row_counts[i])
                _mass = 100.0 / n_targets * n_i / max(tgt_rows.get(_tgt, 1), 1)
                print(f"[train]   {k[0]}/{k[1]:26s} → {_tgt:30s} pool={P} "
                      f"n={n_i:6d} w={w[i]:.3f} (→{_mass:.2f}% mass)", flush=True)
    return w


def _log_baseline_responses(
    responses: List[dict],
    source: str,
    output_dir: str,
    wandb_run=None,
) -> None:
    """Write no-compress baseline responses to CSV and a W&B Table."""
    if not responses:
        return

    csv_path = os.path.join(output_dir, f"baseline_responses_{source}.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        # quoting=QUOTE_ALL (+ escapechar) so model outputs containing commas,
        # quotes, newlines, CRs etc. (common in v3 summarization predictions/
        # targets) are always quoted — never "need to escape, but no escapechar
        # set".  Explicit fmtparams override the writer's inherited dialect.
        writer = csv.DictWriter(f, fieldnames=_BASELINE_RESPONSE_COLS,
                                quoting=csv.QUOTE_ALL, escapechar="\\")
        if write_header:
            writer.writeheader()
        for r in responses:
            writer.writerow({"source": source, **r})

    if wandb_run is not None:
        tbl = wandb.Table(columns=_BASELINE_RESPONSE_COLS)
        for r in responses:
            tbl.add_data(
                source, r["sample_idx"], _clip_cell(r["prompt"]),
                _clip_cell(r["target"]), _clip_cell(r["prediction"]),
                r["correct"], r["correct_f1"],
                _clip_cell(r.get("judge_prompt", "")),
                _clip_cell(r.get("judge_explanation", "")),
            )
        wandb_run.log({f"eval/baseline_responses_{source}": tbl}, step=0)

def _log_responses(
    responses: List[dict],
    step: int,
    output_dir: str,
    wandb_run=None,
) -> None:
    """Append eval responses to a single CSV and log per-dataset W&B tables.

    Each row dict should carry a "dataset" key (set by _evaluate_per_group;
    in grouped eval this is the metric-group name).
    Rows without one are bucketed under "default" so the function stays
    backward-compatible with callers that don't tag responses.

    Output:
      - CSV (single, concatenated): eval_responses.csv  (includes a `dataset` col).
      - W&B: one table per dataset at  eval/responses/{dataset}  so each eval
        set is inspectable independently in the W&B UI.
    """
    if not responses:
        return

    csv_path = os.path.join(output_dir, "eval_responses.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RESPONSE_CSV_COLS,
                                quoting=csv.QUOTE_ALL, escapechar="\\")
        if write_header:
            writer.writeheader()
        for r in responses:
            writer.writerow({"step": step, "dataset": r.get("dataset", "default"), **r})

    if wandb_run is not None:
        # Group by dataset → one W&B table per dataset.
        from collections import defaultdict
        by_ds: Dict[str, List[dict]] = defaultdict(list)
        for r in responses:
            by_ds[r.get("dataset", "default")].append(r)
        _log_payload: Dict[str, "wandb.Table"] = {}
        for _ds, _rows in by_ds.items():
            tbl = wandb.Table(columns=_RESPONSE_CSV_COLS)
            for r in _rows:
                tbl.add_data(step, _ds, r["sample_idx"], r["n_art"], r["K"],
                             r.get("n_tokens_full", ""), r.get("n_tokens_pruned", ""),
                             r.get("n_tokens_full_art", ""), r.get("n_tokens_pruned_art", ""),
                             _clip_cell(r["target"]), _clip_cell(r["prediction"]),
                             r["correct"],
                             r.get("correct_f1", ""),
                             _clip_cell(r.get("full_prompt", "")),
                             _clip_cell(r.get("pre_article_prompt", "")),
                             _clip_cell(r.get("prunable_article", "")),
                             _clip_cell(r.get("post_article_prompt", "")),
                             _clip_cell(r.get("kept_tokens", "")),
                             _clip_cell(r.get("pruned_context", "")),
                             _clip_cell(r.get("judge_prompt", "")),
                             _clip_cell(r.get("judge_explanation", "")))
            _log_payload[f"eval/responses/{_ds}"] = tbl
        wandb_run.log(_log_payload, step=step)


# ── Per-dataset eval helpers (multitask) ──────────────────────────────────────

def _evaluate_per_group(
    model,
    gate,
    tokenizer,
    eval_samples_by_group: Dict[str, List[dict]],
    tasks_by_name: Dict[str, "Task"],
    ds_task: Dict[str, str],
    K_eval: int,
    args,
    *,
    rank: int,
    world_size: int,
    device: str,
    log_examples: int = 0,
) -> Tuple[Dict[str, dict], List[dict], Dict[str, dict]]:
    """Call evaluate() once per metric group on that group's pooled (mixed-task)
    eval set, resolving each sample's task per-sample.  Returns:
      per_group_metrics: {group: {accuracy (micro over the pooled set), f1,
                                  loss, loss_letter, loss_eos,
                                  rougeL_vs_baseline, n_tokens_full,
                                  n_tokens_pruned, n_tokens_full_art,
                                  n_tokens_pruned_art, n_samples,
                                  acc_metric_tag, f1_metric_tag}}
      logged_responses: concatenated across groups (rank 0 only after gather).
      per_ds_metrics: {dataset: {accuracy, f1, n_samples, n_tokens_full,
                                  n_tokens_pruned, n_tokens_full_art,
                                  n_tokens_pruned_art, acc_metric_tag,
                                  f1_metric_tag}}
        — the "cheap extra" breakdown, derived from the SAME generations by
        bucketing per-sample scores by source dataset (no extra inference).
    """
    per_group_metrics: Dict[str, dict] = {}
    per_ds_metrics: Dict[str, dict] = {}
    all_responses: List[dict] = []
    for grp, _samps in eval_samples_by_group.items():
        if not _samps:
            continue
        (_acc, _f1, _loss, _loss_letter, _loss_eos, _resp, _rougeL,
         _ntok_full, _ntok_pruned,
         _ntok_full_art, _ntok_pruned_art, _per_ds,
         _ce_loss, _sparsity_loss) = evaluate(
            model, gate, tokenizer, _samps,
            args.gate_layer, K_eval, None,
            log_examples=log_examples,
            device=device,
            max_gen_tokens=args.max_gen_tokens,
            baseline_predictions=None,
            rank=rank, world_size=world_size,
            eval_gen_batch_size=args.eval_gen_batch_size,
            tasks_by_name=tasks_by_name,
            sparsity_lambda=getattr(args, "sparsity_lambda", 0.0),
        )
        # Metric tags are homogeneous within a group; read from any member task.
        _grp_task = tasks_by_name[_samps[0]["task"]]
        # Tag each logged row with the group so _log_responses splits per group.
        for _r in _resp:
            _r["dataset"] = grp
        per_group_metrics[grp] = {
            "accuracy": _acc, "f1": _f1,
            "loss": _loss, "loss_letter": _loss_letter, "loss_eos": _loss_eos,
            "ce_loss": _ce_loss, "sparsity_loss": _sparsity_loss,
            "rougeL_vs_baseline": _rougeL,
            "n_tokens_full": _ntok_full, "n_tokens_pruned": _ntok_pruned,
            "n_tokens_full_art": _ntok_full_art,
            "n_tokens_pruned_art": _ntok_pruned_art,
            "n_samples": len(_samps),
            "acc_metric_tag": _grp_task.acc_metric_name,
            "f1_metric_tag":  _grp_task.secondary_metric_name,
        }
        # Cheap-extra per-dataset breakdown (acc/f1/tokens; no per-ds loss).
        for _ds, _m in _per_ds.items():
            _dt = tasks_by_name[ds_task[_ds]]
            per_ds_metrics[_ds] = {
                "accuracy": _m["accuracy"], "f1": _m["f1"],
                "loss": None, "loss_letter": None, "loss_eos": None,
                "rougeL_vs_baseline": None,
                "n_tokens_full": _m["n_tokens_full"],
                "n_tokens_pruned": _m["n_tokens_pruned"],
                "n_tokens_full_art": _m["n_tokens_full_art"],
                "n_tokens_pruned_art": _m["n_tokens_pruned_art"],
                "n_samples": _m["n_samples"],
                "acc_metric_tag": _dt.acc_metric_name,
                "f1_metric_tag":  _dt.secondary_metric_name,
            }
        all_responses.extend(_resp)
    return per_group_metrics, all_responses, per_ds_metrics


_EXTRACT_METRIC_TAGS = {"F1", "EM", "ES", "acc"}
_ROUGE_METRIC_TAGS = {"rougeL", "rouge1", "rouge2", "rougeLsum"}


def _emit_eval_wandb_keys(
    wandb_log: dict,
    per_group: Dict[str, dict],
    per_ds: Dict[str, dict],
    K: int,
    legacy_acc_metric_tag: str,
    judge_active: bool,
    swa: bool = False,
    source: str = "",
) -> None:
    """Write the headline per-metric-group eval keys (`eval/cat_<group>/...`,
    one micro-averaged val set per group) plus the cheap per-dataset breakdown
    (`eval/<dataset>/...`) and legacy flat keys (from the primary group, for
    W&B history continuity) into `wandb_log`.

    per_group : {group: metrics}     — the headline (one pooled set per group)
    per_ds    : {dataset: metrics}    — the cheap-extra breakdown (acc/f1/tokens)
    source    : optional infix after 'eval/' (e.g. 'randomk_', 'gatefull_') to give a
                checkpoint-baseline overlay its own key namespace; ignored when swa=True.
    """
    _prefix = "eval/swa_" if swa else f"eval/{source}"

    # ── Headline: one micro-averaged curve per metric group ──────────────────
    for grp, m in per_group.items():
        tag = m["acc_metric_tag"]
        wandb_log[f"{_prefix}cat_{grp}/accuracy_{tag}_K{K}"] = m["accuracy"]
        if m["f1"] is not None and m["f1_metric_tag"] != tag:
            wandb_log[f"{_prefix}cat_{grp}/accuracy_{m['f1_metric_tag']}_K{K}"] = m["f1"]
        wandb_log[f"{_prefix}cat_{grp}/loss_K{K}"] = m["loss"]
        wandb_log[f"{_prefix}cat_{grp}/loss_letter_K{K}"] = m["loss_letter"]
        wandb_log[f"{_prefix}cat_{grp}/loss_eos_K{K}"] = m["loss_eos"]
        # CE + sparsity loss decomposition.  ce_loss is identical to loss above
        # (kept as an explicit alias for clarity in W&B panels); ce + sparsity
        # equals the total training-objective loss at this K.
        if "ce_loss" in m:
            wandb_log[f"{_prefix}cat_{grp}/ce_loss_K{K}"] = m["ce_loss"]
        if "sparsity_loss" in m:
            wandb_log[f"{_prefix}cat_{grp}/sparsity_loss_K{K}"] = m["sparsity_loss"]
            wandb_log[f"{_prefix}cat_{grp}/total_loss_K{K}"] = m["ce_loss"] + m["sparsity_loss"]
        # Total counts (incl. always-kept non-prunable prefix+suffix) — KV-cache footprint.
        wandb_log[f"{_prefix}cat_{grp}/n_tokens_full_K{K}"] = m["n_tokens_full"]
        wandb_log[f"{_prefix}cat_{grp}/n_tokens_pruned_K{K}"] = m["n_tokens_pruned"]
        # Article-only counts (gate-prunable region) — gate's actual decision footprint.
        wandb_log[f"{_prefix}cat_{grp}/n_tokens_full_art_K{K}"] = m["n_tokens_full_art"]
        wandb_log[f"{_prefix}cat_{grp}/n_tokens_pruned_art_K{K}"] = m["n_tokens_pruned_art"]
        wandb_log[f"{_prefix}cat_{grp}/n_samples_K{K}"] = m["n_samples"]

    # ── Cheap extra: per-dataset accuracy/f1 (same generations) ──────────────
    for ds, m in per_ds.items():
        tag = m["acc_metric_tag"]
        wandb_log[f"{_prefix}{ds}/accuracy_{tag}_K{K}"] = m["accuracy"]
        if m["f1"] is not None and m["f1_metric_tag"] != tag:
            wandb_log[f"{_prefix}{ds}/accuracy_{m['f1_metric_tag']}_K{K}"] = m["f1"]
        wandb_log[f"{_prefix}{ds}/n_tokens_full_K{K}"] = m["n_tokens_full"]
        wandb_log[f"{_prefix}{ds}/n_tokens_pruned_K{K}"] = m["n_tokens_pruned"]
        wandb_log[f"{_prefix}{ds}/n_tokens_full_art_K{K}"] = m["n_tokens_full_art"]
        wandb_log[f"{_prefix}{ds}/n_tokens_pruned_art_K{K}"] = m["n_tokens_pruned_art"]
        wandb_log[f"{_prefix}{ds}/n_samples_K{K}"] = m["n_samples"]

    # ── Legacy flat keys (primary/first group) for single-curve history ──────
    if per_group:
        _primary = next(iter(per_group))
        _pm = per_group[_primary]
        wandb_log[f"{_prefix}accuracy_{legacy_acc_metric_tag}_K{K}"] = _pm["accuracy"]
        if judge_active and _pm["f1"] is not None:
            wandb_log[f"{_prefix}accuracy_F1_K{K}"] = _pm["f1"]
        wandb_log[f"{_prefix}loss_K{K}"] = _pm["loss"]
        wandb_log[f"{_prefix}loss_letter_K{K}"] = _pm["loss_letter"]
        wandb_log[f"{_prefix}loss_eos_K{K}"] = _pm["loss_eos"]
        # CE + sparsity decomposition of eval loss at this K (legacy flat).
        if "ce_loss" in _pm:
            wandb_log[f"{_prefix}ce_loss_K{K}"] = _pm["ce_loss"]
        if "sparsity_loss" in _pm:
            wandb_log[f"{_prefix}sparsity_loss_K{K}"] = _pm["sparsity_loss"]
            wandb_log[f"{_prefix}total_loss_K{K}"] = _pm["ce_loss"] + _pm["sparsity_loss"]
        wandb_log[f"{_prefix}n_tokens_full_K{K}"] = _pm["n_tokens_full"]
        wandb_log[f"{_prefix}n_tokens_pruned_K{K}"] = _pm["n_tokens_pruned"]
        wandb_log[f"{_prefix}n_tokens_full_art_K{K}"] = _pm["n_tokens_full_art"]
        wandb_log[f"{_prefix}n_tokens_pruned_art_K{K}"] = _pm["n_tokens_pruned_art"]


# ── Training loop ─────────────────────────────────────────────────────────────

# Step-checkpoint family.  `train_state` files carry optimizer/scheduler/
# SWA/RNG/wandb state for mid-run resume; the other three are weights-only.
# All four are bounded by the same last-N + best-N retention helper.
_CKPT_STEP_RE = re.compile(
    r"^(?:gate|lora|train_state|swa_gate|swa_lora)_step(\d+)\.pt$"
)

# train_state-only regex for --resume-from path resolution.
_TRAIN_STATE_RE = re.compile(r"^train_state_step(\d+)\.pt$")

def _require_k_eval(args) -> list:
    """Eval K list; --k-eval is mandatory (the legacy k_set /
    (k_low+k_high)//2 fallbacks were removed 2026-08-25)."""
    if not getattr(args, "k_eval", None):
        raise ValueError(
            "--k-eval is required: the legacy fallbacks (k_set, "
            "(k_low+k_high)//2) were removed with the [k_low, k_high] "
            "sampling machinery.")
    return sorted(set(args.k_eval))


# Critical args that must match between the saved train_state and the
# resuming run.  A mismatch raises rather than silently mis-resuming (per
# the "fail loudly" working agreement in CLAUDE.md).
_RESUME_CRITICAL_ARGS = (
    "gate_layer",
    "phase2_len_list", "phase2_bucket_lists", "phase2_loguniform",
    "length_buckets",
    "base_model", "task", "dataset",
    "adapter_type", "lora_rank", "lora_alpha", "lora_targets", "lora_layers",
)


def _prune_step_checkpoints(
    output_dir: str,
    saved_step_accs: Dict[int, float],
    keep_last: int,
    keep_best: int,
    keep_all_after_step: Optional[int] = None,
) -> Tuple[int, List[int]]:
    """Delete periodic step checkpoints not in the retention set.

    Considers files matching ``(gate|lora|train_state)_step{N}.pt``
    in ``output_dir``.  ``*_final.pt`` and ``swa_gate_final.pt`` never match
    and are always kept.

    Retention set is the union of:
      - the ``keep_last`` most-recent saved steps (by numeric step), and
      - the ``keep_best`` highest-accuracy saved steps (by ``saved_step_accs``).

    Returns ``(num_files_removed, sorted_kept_steps)``.
    """
    keep_steps: set = set()
    if saved_step_accs:
        by_step = sorted(saved_step_accs.keys())
        keep_steps.update(by_step[-max(0, keep_last):])
        if keep_best > 0:
            by_acc = sorted(
                saved_step_accs.items(), key=lambda x: x[1], reverse=True
            )
            keep_steps.update(s for s, _ in by_acc[:keep_best])

    n_removed = 0
    for fname in os.listdir(output_dir):
        m = _CKPT_STEP_RE.match(fname)
        if m is None:
            continue
        step = int(m.group(1))
        if step in keep_steps:
            continue
        # --checkpoint-keep-all-after-warmup: gate_/lora_ step
        # files past the threshold are always kept; train_state stays on
        # keep-last (resume needs only recent state).
        if (keep_all_after_step is not None and step >= keep_all_after_step
                and not fname.startswith("train_state")):
            continue
        os.remove(os.path.join(output_dir, fname))
        n_removed += 1
    return n_removed, sorted(keep_steps)


def _resolve_resume_path(resume_arg: str) -> str:
    """Resolve --resume-from value to a specific train_state_step*.pt file.

    Accepts either a file path (used directly, must match the
    train_state_step{N}.pt pattern) or a directory (returns the highest-step
    train_state file inside).  Raises FileNotFoundError or ValueError on
    invalid input — never returns a stale or ambiguous path.
    """
    if os.path.isfile(resume_arg):
        if not _TRAIN_STATE_RE.match(os.path.basename(resume_arg)):
            raise ValueError(
                f"--resume-from file must be named train_state_step{{N}}.pt, "
                f"got {resume_arg!r}"
            )
        return resume_arg
    if os.path.isdir(resume_arg):
        candidates = []
        for f in os.listdir(resume_arg):
            m = _TRAIN_STATE_RE.match(f)
            if m:
                candidates.append((int(m.group(1)), f))
        if not candidates:
            raise FileNotFoundError(
                f"No train_state_step*.pt files found in {resume_arg}"
            )
        candidates.sort()
        return os.path.join(resume_arg, candidates[-1][1])
    raise FileNotFoundError(f"--resume-from path does not exist: {resume_arg}")


def _verify_resume_args(saved: dict, args) -> None:
    """Fail loudly if any critical arg differs between saved checkpoint and
    the current run.  Re-using a checkpoint with different gate_layer,
    K-source lists, dataset, or LoRA config produces a silent mis-resume that
    is very hard to debug; refuse instead.
    """
    mismatches = []
    for k in _RESUME_CRITICAL_ARGS:
        cur = getattr(args, k, None)
        if k not in saved:
            continue  # checkpoint predates this arg; allow
        if cur != saved[k]:
            mismatches.append(f"  {k}: current={cur!r}  saved={saved[k]!r}")
    if mismatches:
        raise ValueError(
            "--resume-from refused: critical args do not match the saved "
            "checkpoint:\n" + "\n".join(mismatches) +
            "\nRe-use with matching hyperparameters or start a fresh run."
        )


def _save_train_state(
    path: str, *,
    step: int,
    optimizer,
    scheduler,
    swa_gate,
    swa_n: int,
    rng: random.Random,
    wandb_run_id: Optional[str],
    args,
    swa_adapter_sd: Optional[dict] = None,
    no_compress_per_group: Optional[dict] = None,
    no_compress_per_ds: Optional[dict] = None,
) -> None:
    """Persist mid-run resume state next to the periodic gate checkpoint.

    Called inside the periodic-checkpoint `is_main` block, so the saved RNG
    state is rank 0's only.  Non-zero ranks re-seed deterministically from
    ``args.seed + rank`` on resume — their batch sequence diverges from the
    original run, but each rank's stream is still reproducible.  The model's
    randomness is dominated by rank-0 sampling (rng) plus warmup-only torch
    init, so skipping torch.{cuda}_rng_state is fine in practice.

    Also stores the step-0 no-compression baseline
    metrics so a resumed run can re-emit them as flat W&B reference lines
    without re-running the (~90-min) baseline eval at the start of training.
    """
    state = {
        "step": step,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "swa_n": swa_n,
        "swa_gate_state": swa_gate.state_dict() if swa_gate is not None else None,
        "swa_adapter_state": ({k: v.detach().cpu() for k, v in swa_adapter_sd.items()}
                              if swa_adapter_sd is not None else None),
        "rng_state": rng.getstate(),
        "wandb_run_id": wandb_run_id,
        "args_critical": {k: getattr(args, k, None) for k in _RESUME_CRITICAL_ARGS},
        "no_compress_per_group": no_compress_per_group or {},
        "no_compress_per_ds": no_compress_per_ds or {},
    }
    torch.save(state, path)


def train(args):
    # ── Distributed setup ─────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # ── Make the first-step torch.compile survivable under DDP ───────────────
    # The first training step compiles the FlexAttention kernels for the run's
    # sequence shapes.  On summarization-heavy mixes (30k-token articles) that
    # codegen takes many minutes, and two things then go wrong on multi-GPU:
    #   (a) all ranks share one Inductor/Triton cache dir and serialize on its
    #       file lock (~world_size× slower); and
    #   (b) while a rank holds the GIL in codegen, the NCCL watchdog can't
    #       heartbeat, so at its 480 s default it declares a hang and SIGABRTs
    #       the whole run — observed: an 8-GPU summarization run died at step 0.
    # Set startup-only env here (must precede init_process_group / first compile):
    #   1) TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC — THE crash fix: a slow first compile
    #      is no longer mistaken for a hang (the 2 h *collective* timeout below
    #      still catches real deadlocks, e.g. a genuine cross-rank desync).
    #   2) per-rank compile caches — hygiene: avoids shared-cache file-lock
    #      contention.  NB: this did NOT speed up the observed 8-GPU summarization
    #      compile stall (root cause is deeper; see the report) — a 1-GPU run on
    #      the same data compiles + trains in ~0.5 min, so the config is sound.
    if world_size > 1:
        _cache_root = os.environ.get("E2E_COMPILE_CACHE_ROOT", "/tmp/e2e_compile_cache")
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{_cache_root}/inductor_rank{local_rank}")
        os.environ.setdefault("TRITON_CACHE_DIR", f"{_cache_root}/triton_rank{local_rank}")
        os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
        # Bound Inductor's compile-worker pool ONLY when the default
        # (min(32, cores)) per rank would oversubscribe the node across ranks,
        # i.e. when 32·world_size would exceed the core count.  On a big-core
        # node (e.g. 255 cores / 8 ranks) the default already fits, and capping
        # lower just under-utilizes cores and slows the compile — so leave it
        # alone there.  (Capping did NOT fix the observed 8-GPU summarization
        # first-step compile stall, whose root cause is a deeper multi-rank
        # torch.compile/FlexAttention interaction — see the report; the heartbeat
        # timeout above is what prevents the resulting crash.)
        _cores = os.cpu_count() or (32 * world_size)
        if 32 * world_size > _cores:
            os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS",
                                  str(max(1, _cores // world_size)))

    if world_size > 1:
        # 2-hour timeout: the default 10 min is too short when rank 0 runs eval
        # (regular + SWA init) on a large model while ranks 1-N wait at a barrier.
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=2),
        )
        # Object collectives (see _OBJ_PG at module top) go over gloo so they
        # never allocate GPU staging buffers.
        global _OBJ_PG
        _OBJ_PG = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=2))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    # Phase-2 attention stats (W&B b_normalized / qk_normalized / softmax_input)
    # are consumed only by rank 0.  Disable accumulation on every other rank so
    # the global buffer never grows there (it is cleared only in the rank-0
    # logging block) and the extra per-layer QK matmuls are skipped.
    set_phase2_stats_enabled(is_main)
    # The gate's own per-call scalars (train/rms_bj, train/tanh_bj) likewise.
    set_gate_diag_stats(is_main)

    # ── Deterministic seeding ─────────────────────────────────────────────────
    # Must happen AFTER set_device so torch.cuda.manual_seed seeds the right
    # device.  Use args.seed + rank so each rank has a distinct but reproducible
    # RNG stream.  Rank 0 gets args.seed exactly so the no-compress eval (rank 0
    # only) is reproducible across runs with the same --seed.
    _seed = args.seed + rank
    random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)          # covers all devices on this node
    torch.backends.cudnn.deterministic = True  # cuDNN picks deterministic algos
    torch.backends.cudnn.benchmark = False     # disables auto-tuner (non-det)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Resume from train_state (mid-run resume) ──────────────────────────────
    # Pre-load on every rank: critical-args verification has to happen
    # consistently; only rank 0 will read the wandb_run_id from it.  The
    # heavy optimizer / SWA / RNG fields are applied lazily at their
    # respective construction sites later in train().
    if args.resume_from and args.resume_gate:
        raise ValueError(
            "--resume-from and --resume-gate are mutually exclusive: the "
            "former restores full training state including gate weights; the "
            "latter loads weights only."
        )
    _resume_state: Optional[dict] = None
    _resume_step_file_step: Optional[int] = None
    _resume_dir: Optional[str] = None
    if args.resume_from:
        _resume_path = _resolve_resume_path(args.resume_from)
        if is_main:
            print(f"[train] Resuming from {_resume_path}")
        _resume_state = torch.load(_resume_path, map_location="cpu")
        _verify_resume_args(_resume_state.get("args_critical", {}), args)
        _m = _TRAIN_STATE_RE.match(os.path.basename(_resume_path))
        _resume_step_file_step = int(_m.group(1))
        if _resume_state["step"] != _resume_step_file_step:
            raise ValueError(
                f"--resume-from train_state file step in filename "
                f"({_resume_step_file_step}) disagrees with state['step'] "
                f"({_resume_state['step']}) in {_resume_path}"
            )
        _resume_dir = os.path.dirname(_resume_path)

    if is_main:
        # Save params
        params = vars(args)
        with open(os.path.join(args.output_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)
        print(f"[train] Params saved to {args.output_dir}/params.json")

        # ── W&B init ─────────────────────────────────────────────────────────
        _wandb_run = None
        if _HAVE_WANDB and args.wandb_project:
            run_name = args.wandb_run_name or os.path.basename(args.output_dir)
            _wandb_kwargs = dict(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                dir=args.output_dir,
            )
            _resume_wandb_id = (
                _resume_state.get("wandb_run_id") if _resume_state else None
            )
            if _resume_wandb_id:
                _wandb_kwargs["id"] = _resume_wandb_id
                _wandb_kwargs["resume"] = "allow"
            _wandb_run = wandb.init(**_wandb_kwargs)
            print(f"[train] W&B run: {_wandb_run.url}"
                  + (" (resumed)" if _resume_wandb_id else ""))
        else:
            if not _HAVE_WANDB and getattr(args, "wandb_project", None):
                print("[train] Warning: --wandb-project set but wandb not installed.")
    else:
        _wandb_run = None

    # ── Load model + tokenizer ────────────────────────────────────────────────
    if is_main:
        print(f"[train] Loading model {args.base_model} on rank {rank} / {world_size}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # YaRN RoPE scaling for long-context training and inference.
    #  At factor=1.0
    # (default) no scaling is applied — base model behaves as before.
    # At factor F>1, max_position_embeddings extends to F × the model's base
    # max_position_embeddings (Qwen3-8B base = 40960, so F=3.2 → 131072 = 128k).
    #
    # transformers 5.x stores the RoPE config in ``config.rope_parameters``
    # (aliased to the legacy ``rope_scaling``) with the key ``rope_type`` (NOT
    # ``type``).  Setting the legacy ``{"type": "yarn"}`` schema raises
    # KeyError: 'rope_type' inside Qwen3RotaryEmbedding.  The base rope_theta
    # must be preserved or RoPE frequencies change.
    _model_kwargs = dict(
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation="sdpa",
    )
    if getattr(args, "long_ctx_factor", 1.0) > 1.0:
        from transformers import AutoConfig
        _cfg = AutoConfig.from_pretrained(args.base_model)
        _orig_max = int(getattr(_cfg, "max_position_embeddings", 40960))
        _base_rope = dict(getattr(_cfg, "rope_parameters", None)
                          or getattr(_cfg, "rope_scaling", None) or {})
        _rope_theta = _base_rope.get("rope_theta", 1000000)
        _cfg.rope_parameters = {
            "rope_type": "yarn",
            "factor": float(args.long_ctx_factor),
            "original_max_position_embeddings": _orig_max,
            "rope_theta": _rope_theta,
        }
        _cfg.max_position_embeddings = int(_orig_max * args.long_ctx_factor)
        if is_main:
            print(f"[train] YaRN RoPE scaling enabled: factor={args.long_ctx_factor}, "
                  f"original_max_position_embeddings={_orig_max} → effective "
                  f"max ctx = {_cfg.max_position_embeddings}")
        _model_kwargs["config"] = _cfg

    # Each rank loads a full model replica on its own GPU.
    # device_map={"": local_rank} pins all layers to this rank's GPU.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **_model_kwargs,
    ).eval()
    num_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(f"[train] Model: {num_layers} layers, d_model={d_model}, vocab={vocab_size}")
    
    if args.gate_layer >= num_layers:
        raise ValueError(f"--gate-layer {args.gate_layer} >= num_layers {num_layers}")

    # ── Freeze backbone (always — the trainer has no full-FT mode) ────────────
    for p in model.parameters():
        p.requires_grad_(False)
    # Gradient flows to the gate via Phase 2 attention score_mod:
    # gate_bias (requires_grad=True) → attn_output → residual → norm → lm_head → loss.
    # lm_head and norm don't need requires_grad=True for this chain — they act as
    # frozen matrices that the gradient passes through, not parameters being trained.
    # Keeping them at requires_grad=False saves ~1.24 GB VRAM/GPU (lm_head gradient).
    print("[train] Backbone frozen. Gradient flows to gate via Phase 2 attention score_mod.")

    # ── Adapter at the gate layer (factored LoRA or additive delta) ──────────
    # lora: enabled when --lora-rank > 0 (attn q/k/v/o).  delta: --adapter-type
    # delta (rank/alpha-free direct ΔW on attn q/k/v/o + MLP gate/up/down;
    # Muon-native).  The lora_* variable names and lora_*.pt
    # checkpoint filenames are the legacy adapter API covering both types
    # (see lora.py docstring); gate meta "adapter_type" disambiguates.
    _adapter_type = args.adapter_type
    _lora_enabled = args.lora_rank > 0 or _adapter_type == "delta"
    lora_start_layer = -1
    lora_layer_ids: List[int] = []
    lora_params: List = []
    # Named adapter param groups for W&B norm logging (train/{name}_norm,
    # train/{name}_grad_norm): lora_A/lora_B for lora, delta/delta_bias for delta.
    _adapter_norm_groups: Dict[str, List] = {}
    if _lora_enabled:
        from tokengate.e2e_gate.gate_module.lora import (
            _DELTA_DEFAULT_TARGETS,
            apply_delta_to_layers,
            apply_lora_to_layers,
            collect_lora_params,
            collect_lora_named_params,
            collect_lora_state_dict,
            load_lora_state_dict,
            resolve_all_adapter_layers,
        )
        lora_layer_ids = [args.gate_layer]
        if getattr(args, "lora_layers", None):
            # Explicit layer selection overrides the gate-layer default
            # (co-adapt default: LoRA on ALL layers, q_proj+v_proj).
            if _adapter_type == "delta":
                raise ValueError("--lora-layers is a LoRA-only override; "
                                 "ADAPTER_TYPE=delta keeps the legacy derivation")
            if args.lora_layers.strip().lower() == "all":
                # Full-attn layers only on hybrid models (Qwen3.5); the full
                # range on homogeneous Qwen3 — see resolve_all_adapter_layers.
                lora_layer_ids = resolve_all_adapter_layers(model)
            else:
                lora_layer_ids = sorted({int(x) for x in args.lora_layers.split(",")
                                         if x.strip()})
        # Audit fix (2026-08-28): derive from the ACTUAL adapter layers, not
        # the gate layer. `= args.gate_layer` was equivalent at gate_layer=0
        # (min(all)=0) but any future L>0 run would silently run layers
        # < gate_layer under no_grad in Phase 1, freezing their adapters.
        lora_start_layer = min(lora_layer_ids)
        if _adapter_type == "delta":
            apply_delta_to_layers(model, lora_layer_ids)
        else:
            apply_lora_to_layers(model, lora_layer_ids, args.lora_rank, args.lora_alpha, args.lora_targets)
        lora_params = collect_lora_params(model)
        if _adapter_type == "delta":
            _adapter_norm_groups = {"delta": [p for p in lora_params if p.ndim == 2]}
            _delta_biases = [p for p in lora_params if p.ndim == 1]
            if _delta_biases:
                _adapter_norm_groups["delta_bias"] = _delta_biases
        else:
            _adapter_norm_groups = {"lora_A": lora_params[::2],
                                    "lora_B": lora_params[1::2]}
        if args.resume_lora:
            _lora_sd = torch.load(args.resume_lora, map_location=device)
            load_lora_state_dict(model, _lora_sd)
            if is_main:
                print(f"[train] Resumed adapter from {args.resume_lora}")
        if is_main:
            n_lora = sum(p.numel() for p in lora_params)
            if _adapter_type == "delta":
                print(f"[train] Adapter: delta at layers={lora_layer_ids}, "
                      f"targets={_DELTA_DEFAULT_TARGETS}, params={n_lora/1e6:.2f}M")
            else:
                _lora_tgts = args.lora_targets or ["q_proj", "k_proj", "v_proj", "o_proj"]
                print(f"[train] LoRA: layers={lora_layer_ids}, rank={args.lora_rank}, "
                      f"alpha={args.lora_alpha}, targets={_lora_tgts}, params={n_lora/1e6:.2f}M")

    # ── Optional: gradient checkpointing on the gated path ──────────────────
    # The LoRA-only loop enables this unconditionally (line ~4949); on the gated
    # path it's opt-in (--grad-checkpoint) because gate-only training at K=2000
    # doesn't need it and grad-ckpt costs ~10-20% wallclock. Required when LoRA
    # trains on a frozen backbone at high K (e.g. K=30000 / 8000-tok sequences),
    # where every layer's activations would otherwise exceed 80 GB. Recomputes
    # layer activations on backward instead of storing them (~30x memory drop).
    if args.grad_checkpoint:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if is_main:
            print("[train] Gradient checkpointing ENABLED on backbone (use_reentrant=False).")

    # ── Gate ──────────────────────────────────────────────────────────────────
    n_heads = model.config.num_attention_heads
    # The gate layer may be ANY layer (run-17: gate at a DeltaNet layer is a
    # deliberate variant — the a_j split dispatches to linear_attn in
    # _attn_only_forward_eager, and the gate branch clones the layer's MLP,
    # which every hybrid layer has). Bias still lands only on the FULL-attn
    # layers above L; a DeltaNet gate layer simply contributes scoring, not a
    # biased mixer. Print the layout so the config is visible in the log.
    _full_attn_indices = get_full_attn_layer_indices(model)
    if args.gate_layer >= len(model.model.layers) - 1:
        raise ValueError(f"gate_layer={args.gate_layer} leaves no Phase-2 layers")
    if is_main and args.gate_layer not in _full_attn_indices:
        print(f"[train] gate_layer={args.gate_layer} is a LINEAR-ATTENTION "
              f"(DeltaNet) layer — scoring reads its hidden state; bias applies "
              f"to full-attn layers {[i for i in _full_attn_indices if i > args.gate_layer]}")
    # n_layers_bias: number of Phase-2 layers that receive the gate bias.
    # For Qwen3.5 hybrid only the FULL-attention layers strictly above the gate
    # layer are biased (linear-attn/DeltaNet layers carry no per-token KV bias in
    # the current baseline). For homogeneous Qwen3 this equals num_layers-L-1.
    _full_attn_above_L = [i for i in _full_attn_indices if i > args.gate_layer]
    n_layers_bias = len(_full_attn_above_L)
    gate = E2EGate(
        d_model=d_model,
        n_heads=n_heads,
        n_layers_bias=n_layers_bias,
        selection_mode=args.gate_selection,
        tau=args.tau,
    ).to(device).to(torch.float32)
    # Dedicated SCORING branch (--gate-branch): unties the token-ranking readout from
    # the hidden state that gets FORWARDED to layer gate_layer+1. Hypothesis: the layer-0
    # delta adapter is weight-tied across two incompatible jobs. The forward path is
    # UNCHANGED (delta-adapted layer 0, Phase-2 biases, learnable gate_scale/attn_scale);
    # only the gate projection's input changes. Built from the CURRENT layer weights, so
    # at init h_gate == h_fwd and the model is byte-identical to the tied architecture.
    # Attached to `gate` so gate.parameters() (optimizer + _sync_params + SWA) covers it.
    # gate_scale init: beta = softplus^-1(s_l) from the pretrained qk-norm gains
    # (set_qk_norm_scales) — the bias starts at ~one native logit-sigma per layer.
    _qk_s_l_vals = gate.set_qk_norm_scales(model, args.gate_layer)
    if rank == 0:
        print(f"[train] gate_scale init: s_l = ||g_q (.) g_k||_2/sqrt(d) per phase-2 layer "
              f"(min={min(_qk_s_l_vals):.3f} mean={sum(_qk_s_l_vals)/len(_qk_s_l_vals):.3f} "
              f"max={max(_qk_s_l_vals):.3f}); gate_scale LEARNABLE softplus(beta), "
              f"init softplus^-1(s_l); temperature: s_eff = r * softplus(sigma), init 1; "
              f"b_norm gain FROZEN at {float(gate.b_norm.weight.item()):.2f}",
              flush=True)
    if rank == 0 and args.wandb_project:
        import wandb as _wandb_slog
        _sl_log = {f"train/s_l/layer{args.gate_layer + 1 + _i}": _v
                   for _i, _v in enumerate(_qk_s_l_vals)}
        _sl_log["train/s_l_mean"] = sum(_qk_s_l_vals) / len(_qk_s_l_vals)
        _sl_log["train/s_l_min"] = min(_qk_s_l_vals)
        _sl_log["train/s_l_max"] = max(_qk_s_l_vals)
        _wandb_slog.log(_sl_log, step=0)


    from tokengate.e2e_gate.gate_module.gate_forward import build_gate_branch
    _br = build_gate_branch(model, gate, args.gate_layer).to(device)
    if is_main:
        print(f"[train] GATE BRANCH: dedicated scoring branch "
              f"init from layer {args.gate_layer} "
              f"({sum(p.numel() for p in _br.parameters()) / 1e6:.2f}M trainable, "
              f"fp32 storage, compute in the backbone dtype). "
              "Split active: Phase 1 ends at a_j; the layer's own MLP runs "
              "on kept tokens in Phase 2.")

    # Plumb the sparsity-penalty config onto the gate object so
    # gate_forward._accumulate_gate_sparsity can read it on each forward.
    # (E2EGate's __init__ doesn't accept these — they're train-time-only.)
    gate.sparsity_mode = args.sparsity_mode
    gate.sparsity_sharpness = float(args.sparsity_sharpness)
    # --random-topk: replace gate top-K with a uniform random K-subset of article
    # tokens (see gate_forward._select_topk_or_random).  Freeze all gate params so
    # the gate stays at zero-init throughout (b=0 → no Phase 2 attention bias);
    # the gate runs forward but its output is functionally a no-op besides
    # selecting K positions at random.
    if getattr(args, "random_topk", False):
        gate.random_topk = True
        for p in gate.parameters():
            p.requires_grad_(False)
        if is_main:
            print(f"[train] --random-topk: gate FROZEN ({sum(p.numel() for p in gate.parameters())/1e6:.2f}M "
                  f"params, requires_grad=False), top-K replaced by uniform random K-subset.")
    if is_main:
        print(f"[train] Gate: n_heads={n_heads}, "
              f"n_layers_bias={n_layers_bias}, "
              f"selection={args.gate_selection}, tau={args.tau}, sparsity_lambda={args.sparsity_lambda}, "
              f"sparsity_mode={args.sparsity_mode}, sparsity_sharpness={args.sparsity_sharpness}")
    if args.resume_gate:
        gate, meta = load_gate(args.resume_gate)
        gate = gate.to(device).to(torch.float32)
        if is_main:
            print(f"[train] Resumed gate from {args.resume_gate}")
    if _resume_state is not None:
        # Restore gate + LoRA + full-FT backbone weights from sibling
        # *_step{N}.pt files written by the same periodic checkpoint.
        _gate_ckpt_path = os.path.join(
            _resume_dir, f"gate_step{_resume_step_file_step}.pt"
        )
        # Strict-load the RAW saved state dict into the live gate (which already
        # has the scoring branch attached). load_state_dict copies values into
        # the existing fp32 params (a pre-fp32-storage checkpoint's bf16 branch
        # weights upcast exactly); the branch computes in the backbone dtype
        # regardless of its param dtype (CastLinear).
        _gate_obj = torch.load(_gate_ckpt_path, map_location="cpu", weights_only=False)
        gate.load_state_dict(_gate_obj["state_dict"])
        del _gate_obj
        gate = gate.to(device)
        if is_main:
            print(f"[train] Restored gate weights from {_gate_ckpt_path}")
        if _lora_enabled:
            _lora_ckpt_path = os.path.join(
                _resume_dir, f"lora_step{_resume_step_file_step}.pt"
            )
            if not os.path.isfile(_lora_ckpt_path):
                raise FileNotFoundError(
                    f"--resume-from expects LoRA sibling at {_lora_ckpt_path} "
                    "(LoRA is enabled but the checkpoint is missing)."
                )
            _lora_sd = torch.load(_lora_ckpt_path, map_location="cpu")
            # collect_lora_state_dict uses full module paths; load_state_dict
            # on the whole model with strict=False targets just those keys.
            model.load_state_dict(_lora_sd, strict=False)
            del _lora_sd
            if is_main:
                print(f"[train] Restored LoRA weights from {_lora_ckpt_path}")

    if is_main:
        # requires_grad filter on BOTH sides: a frozen gate (--random-topk,
        # frozen b_norm gain) must not be reported as trainable. The sftbase
        # banner printed "trainable=12.26M (gate=4.59M)" for a FROZEN gate
        # before this fix.
        n_gate = sum(p.numel() for p in gate.parameters() if p.requires_grad)
        n_gate_frozen = sum(p.numel() for p in gate.parameters() if not p.requires_grad)
        n_backbone_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_backbone_total = sum(p.numel() for p in model.parameters())
        n_trainable = n_gate + n_backbone_trainable
        n_total = n_gate + n_gate_frozen + n_backbone_total
        print(f"[train] Params: trainable={n_trainable/1e6:.2f}M "
              f"(gate={n_gate/1e6:.2f}M, backbone={n_backbone_trainable/1e6:.2f}M), "
              f"total={n_total/1e6:.2f}M")
        print(f"[mem] after model+gate+LoRA setup: allocated={torch.cuda.memory_allocated()/1e9:.1f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.1f}GB", flush=True)

    # Gate is NOT wrapped in DDP.  attn_scale and gate_scale receive their
    # gradients from Phase 2 attention layers (outside gate.forward()), so DDP's
    # bucket mechanism would never see their grad_fns and would either deadlock or
    # silently skip their gradients.  Instead, all gate parameter gradients are
    # manually all-reduced in the training loop, mirroring how the unfrozen
    # backbone is handled.
    if world_size > 1:
        # Synchronise initial gate weights so all ranks start from the same point.
        for p in gate.parameters():
            dist.broadcast(p.data, src=0)
        # Synchronise LoRA weights (rank 0 may have loaded a resume checkpoint).
        for p in lora_params:
            dist.broadcast(p.data, src=0)

    # Split gate parameters into scalar params (attn_scale, v_attn) and the rest.
    # Scalar params need a higher LR because AdamW's adaptive normalisation gives a
    # per-step update of ~±lr regardless of gradient magnitude, so with a small base lr
    # (e.g. 1e-6) these parameters barely move.  A separate param group with
    # lr * scalar_lr_mult (default 100×) lets them train at a reasonable rate.
    #
    # v_attn (attnpool) is included here: it controls which instruction tokens drive
    # W_k/W_q gradients via the attention pooling.  It starts at zero (uniform attention),
    # so the effective gradient to W_k is diluted by 1/sqrt(n_inst) relative to a linear
    # gate.  Giving v_attn the same higher LR as attn_scale lets it sharpen early.
    #
    # gate_scale (normed_gate) is deliberately NOT in this group: it controls the magnitude
    # of the gate bias added to Phase 2 attention logits.  At 100× LR, gate_scale grows
    # too quickly and makes the gate bias dominate attention before the gate has learned
    # meaningful token selection, causing accuracy to decrease.  Base LR gives the right
    # growth rate for gate_scale to track what the model actually needs.
    # log_temp_ca (crossattn) controls internal softmax peakiness — same role as
    # v_attn in attnpool.  Needs scalar_lr_mult so it can sharpen as W_Qa/W_Kc learn.
    # Every trainable gate param is fp32 storage (the scoring branch computes in its
    # compute dtype through CastLinear).  The optimizer update and the SWA running
    # mean both rely on this; fail here rather than train a silently-quantised param.
    assert_fp32_params(gate, "gate (incl. scoring branch)")

    _scalar_param_names = {"attn_scale", "v_attn", "log_temp_ca"}
    # --scalar-lr-include-gate-scale: explicit opt-in (run-4 experiment). The
    # default exclusion above is the documented 100x-era safety choice; at
    # modest mults under tanh (bounded bias + measured depth-structured G
    # signal) amplifying G is the point. Monitor tanh_bj/frac_sat.
    if getattr(args, "scalar_lr_include_gate_scale", False):
        _scalar_param_names = _scalar_param_names | {"gate_scale"}
    _scalar_params = [
        p for n, p in gate.named_parameters()
        if n in _scalar_param_names and p is not None
    ]
    # requires_grad filter: frozen gate params (the b_norm gain, the whole gate
    # under --random-topk) must not enter the optimizer -- AdamW would treat
    # them as live and waste optimizer state on them, and param-count prints
    # would overcount.
    _main_gate_params = [
        p for n, p in gate.named_parameters()
        if n not in _scalar_param_names and p.requires_grad
    ]
    _scalar_params = [p for p in _scalar_params if p.requires_grad]
    _scalar_lr = args.lr * args.scalar_lr_mult
    _lora_lr = args.lora_lr if (getattr(args, "lora_lr", None) is not None) else args.lr

    # Fused AdamW: single CUDA kernel for the param update, ~2-5x faster opt step
    # for the gate (small params) and modestly faster for LoRA.  Disabled when
    # CUDA is unavailable; falls back to default multi-tensor implementation.
    # foreach=False kept on LoRA/backbone groups because their large param count
    # makes the multi-tensor path's peak-memory spike (gather all params into one
    # buffer) trigger OOMs on the 80GB GPUs we run on.
    # --weight-decay applies to every param group.  Unset (None) keeps the legacy
    # defaults: 1e-4 on the AdamW path, 0.0 on the Muon path.
    _adamw_wd = args.weight_decay if args.weight_decay is not None else 1e-4
    _muon_wd = args.weight_decay if args.weight_decay is not None else 0.0
    _opt_kwargs = {"weight_decay": _adamw_wd}
    if torch.cuda.is_available():
        try:
            torch.optim.AdamW([torch.zeros(1, device="cuda")], fused=True)
            _opt_kwargs["fused"] = True
        except (RuntimeError, ValueError, TypeError):
            pass  # fused unsupported on this PyTorch / GPU pair

    # --no-decay-scalars: learned scales / temperatures / vectors (gate_scale,
    # attn_scale, v_attn, log_temp_ca, biases, any <2D tensor) carry no weight
    # decay.  Decay on a temperature pulls it toward 0 (= no attention) and on a
    # magnitude toward 0 (= no gate bias); neither is a prior anyone wants.
    _no_decay_names = {"gate_scale", "b_norm.weight"} | _scalar_param_names
    _nd_ids = set()
    if getattr(args, "no_decay_scalars", False):
        # min(shape)==1 mirrors the Muon exclusion: the (1, d) absorbed
        # readout row is a vector in matrix clothing. It was silently
        # DECAYED before this fix (ndim==2 passed the old test) despite the
        # run scripts documenting it as wd-exempt -- a shrink prior on the
        # exact parameter whose growth bottlenecks the bias.
        _nd_ids = {id(p) for n, p in gate.named_parameters()
                   if n in _no_decay_names or p.ndim < 2 or min(p.shape) == 1}
        _main_decay = [p for p in _main_gate_params if id(p) not in _nd_ids]
        _main_nodecay = [p for p in _main_gate_params if id(p) in _nd_ids]
    else:
        _main_decay, _main_nodecay = _main_gate_params, []

    _opt_groups = []
    if _main_decay:
        _opt_groups.append({"params": _main_decay, "lr": args.lr})
    if _main_nodecay:
        _opt_groups.append({"params": _main_nodecay, "lr": args.lr, "weight_decay": 0.0})
    if _scalar_params:
        _sg = {"params": _scalar_params, "lr": _scalar_lr}
        if getattr(args, "no_decay_scalars", False):
            _sg["weight_decay"] = 0.0
        _opt_groups.append(_sg)
        if is_main:
            print(f"[train] Scalar params ({[n for n, p in gate.named_parameters() if n in _scalar_param_names]}) "
                  f"using lr={_scalar_lr:.2e} ({args.scalar_lr_mult}× base lr={args.lr:.2e})")
    if getattr(args, "no_decay_scalars", False) and is_main:
        print(f"[train] No weight decay on "
              f"{[n for n, p in gate.named_parameters() if id(p) in _nd_ids]}")
    if lora_params:
        # foreach=False on LoRA group: see opt_kwargs comment above.
        # Cannot use fused with mixed-precision LoRA params unless they're fp32;
        # the foreach kwarg overrides fused for this group implicitly.
        _opt_groups.append({"params": lora_params, "lr": _lora_lr, "foreach": False})
        if is_main:
            print(f"[train] Adapter ({_adapter_type}) params: lr={_lora_lr:.2e}")
    if is_main:
        _wd_eff = (_muon_wd if getattr(args, "optimizer", "adamw") == "muon" else _adamw_wd)
        _br_ids = {id(p) for p in gate.gate_branch.parameters()}
        if _wd_eff > 0 and any(id(p) in _br_ids for p in _main_decay):
            print(f"[train] WARNING: weight_decay={_wd_eff} applies to the scoring branch "
                  f"(fp32 storage; initialised from the pretrained layer-{args.gate_layer} "
                  f"MLP). Decoupled decay pulls it toward 0, not toward its init; under the "
                  f"previous bf16 storage the decay was a silent no-op. Set WEIGHT_DECAY=0 "
                  f"to keep the previous effective behaviour.", flush=True)
    if not _opt_groups:
        raise RuntimeError(
            "Optimizer has no parameters.  --random-topk freezes the gate, so a "
            "trainable param source (--lora-rank > 0) must be set."
        )
    if getattr(args, "optimizer", "adamw") == "muon":
        from tokengate.e2e_gate.train.muon import (
            build_torch_muon_optimizer,
        )
        # PyTorch-native torch.optim.Muon (2D weight matrices: LoRA A/B) + aux
        # torch.optim.AdamW (everything else). adjust_lr_fn="match_rms_adamw"
        # (always) makes the Muon lr AdamW-comparable, so every group — Muon and
        # aux-AdamW — keeps its per-group lr (lora_lr, scalar_lr, ...); there is
        # no separate Muon LR.
        #
        # Force to AdamW (excluded from Muon):
        #   - scalar params (gate_scale/attn_scale/v_attn/log_temp_ca): 2D but
        #     learned SCALES, not linear transforms — orthogonalization is meaningless.
        #   - the gate's own linear proj (_main_gate_params): kept on AdamW because
        #     Muon lagged AdamW on the gate at rank-16 (project_muon_lags_adamw_gate).
        #     We test Muon ONLY on the big (full-rank) LoRA matrices.
        # --muon-gate: the gate's genuine linear maps (2D, both dims > 1 — the
        # mlp branch's gate_proj/up_proj, a per-head proj) go to Muon too.  The
        # rank-16 result that motivated the exclusion predates the 100M-param
        # scoring branch.  Scale grids (gate_scale/attn_scale), vectors and the
        # (1, d) readout stay on AdamW — orthogonalising them is meaningless.
        _adamw_ids = {id(p) for p in _scalar_params}
        if getattr(args, "muon_gate", False):
            _adamw_ids |= {id(p) for n, p in gate.named_parameters()
                           if n in _no_decay_names or p.ndim != 2 or min(p.shape) == 1}
        else:
            _adamw_ids |= {id(p) for p in _main_gate_params}
        # Factored (low-rank) LoRA A/B are SKINNY matrices — orthogonalizing a
        # (r, d) factor is not the full-rank map Muon is designed for, and the
        # repo's own rank-16 result (project_muon_lags_adamw_gate) found Muon
        # lagging AdamW on factored adapters. Route them to AdamW unless
        # --lora-muon explicitly opts in (full-rank LoRA, e.g. r=4096, should).
        # The delta adapter (full-rank additive ΔW) keeps riding Muon.
        if (_adapter_type == "lora" and lora_params
                and not getattr(args, "lora_muon", False)):
            _adamw_ids |= {id(p) for p in lora_params}
        optimizer, _n_muon, _n_aux = build_torch_muon_optimizer(
            _opt_groups,
            adamw_param_ids=_adamw_ids,
            weight_decay=_muon_wd,
        )
        if is_main:
            print(f"[train] Optimizer: torch.optim.Muon "
                  f"(per-group lrs, adjust_lr_fn=match_rms_adamw, "
                  f"weight_decay={_muon_wd}, muon_gate={getattr(args, 'muon_gate', False)}) "
                  f"on {_n_muon/1e6:.2f}M matrix params "
                  f"+ aux-AdamW on {_n_aux/1e6:.2f}M params")
    else:
        optimizer = torch.optim.AdamW(_opt_groups, **_opt_kwargs)

    if _resume_state is not None:
        optimizer.load_state_dict(_resume_state["optimizer_state"])
        if is_main:
            print("[train] Restored optimizer state (AdamW moments + steps)")

    # Scheduler is created after the dataset so --epochs can set args.steps first.

    # Forward hook on the gate module to capture the norm of its output b_j
    # (per-token gate scores, shape (n_article,)).  Fires once per sample;
    # with batch_size > 1 it fires multiple times per step — accumulated and averaged.
    _gate_raw_for_hook = gate
    # One raw 1-D float tensor per gate() call (one per sample in the batch).
    # Use non_blocking=True copy so the host transfer can overlap with the
    # backward graph; pin the destination buffer via .pin_memory() so the copy
    # uses DMA and stays out of the critical path.  The buffer is consumed in
    # the logging block where the host transfer is implicitly synchronised by
    # the subsequent .item() reads.
    # Only rank 0 ever consumes this buffer (it feeds the W&B gate-output stats
    # in the rank-0 logging block).  On ranks 1..N the hook would otherwise
    # append a CPU tensor on every gate() call — train AND eval — and the buffer
    # is cleared only inside `if is_main`, so those ranks leaked CPU memory for
    # the whole run.  Gate the append on is_main so non-main ranks never grow it
    # (and skip the wasted .cpu() transfer).
    _gate_output_raw_buf: List[torch.Tensor] = []
    # Parallel buffer used during eval to compute the eval-time
    # `gate_pct_above_tau` metric (fraction of article tokens whose
    # sigmoid(b) >= training tau).  Drained at the end of each eval round.
    _gate_output_eval_buf: List[torch.Tensor] = []
    def _gate_output_hook(module, input, output):
        if not is_main:
            return
        if _gate_capture[0]:
            _gate_output_raw_buf.append(
                output.detach().float().cpu()
            )
        elif _gate_eval_capture[0]:
            _gate_output_eval_buf.append(
                output.detach().float().cpu()
            )
    # _gate_capture gates appends to the training-step forward only, so eval-time
    # gate() calls (which also fire this hook) don't pollute the next step's
    # stats.  Set True right before the train forward, False afterwards.
    # _gate_eval_capture is the symmetric flag for eval — set True around the
    # eval forward calls so we can compute pct_above_tau without polluting train.
    _gate_capture = [False]
    _gate_eval_capture = [False]
    _gate_hook_handle = _gate_raw_for_hook.register_forward_hook(_gate_output_hook)

    # ── Build SFT dataset ─────────────────────────────────────────────────────
    # Parse --dataset / --task into aligned lists.  Single-dataset case: list of len 1.
    _datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    _task_names = [t.strip() for t in args.task.split(",") if t.strip()]
    if len(_datasets) != len(_task_names):
        raise ValueError(
            f"--dataset has {len(_datasets)} entries but --task has {len(_task_names)}; "
            f"they must be aligned 1:1."
        )
    if not _datasets:
        raise ValueError("At least one --dataset/--task pair is required.")
    _pairs: List[Tuple[str, str]] = list(zip(_datasets, _task_names))

    # Eval-only (held-out) pairs: subsets we evaluate but NEVER train on.  Parsed
    # from --pretrain-eval-only-pairs ("ds:tk,ds:tk").  They are deliberately NOT
    # added to _pairs, so the train loop and the packed store never see them; only
    # their val.pkl is loaded into the eval pool (below) and reported in separate
    # "_heldout" groups.
    _eval_only_pairs: List[Tuple[str, str]] = []
    for _tok in (getattr(args, "pretrain_eval_only_pairs", "") or "").split(","):
        _tok = _tok.strip()
        if not _tok:
            continue
        if ":" not in _tok:
            raise ValueError(
                f"--pretrain-eval-only-pairs entries must be 'dataset:task', got {_tok!r}")
        _eds, _etk = _tok.split(":", 1)
        _eval_only_pairs.append((_eds.strip(), _etk.strip()))
    _eval_only_tasks: set = {tk for _, tk in _eval_only_pairs}

    # Build one Task object per unique task name.  Held-out eval-only tasks
    # need a Task object too (for per-sample scoring), so include them
    # alongside the trained task names.  (The LLM-judge / oracle-baseline
    # wiring was removed — every surviving task scores with its own metric.)
    tasks_by_name: Dict[str, "Task"] = {}
    for _tname in list(_task_names) + [tk for _, tk in _eval_only_pairs]:
        if _tname in tasks_by_name:
            continue
        tasks_by_name[_tname] = get_task(_tname)

    # Single-task convenience handle.  Multitask runs still rely on this for
    # the *global* W&B metric tag and judge-active flag; per-dataset keys
    # produced by evaluate() use the per-sample task lookup.
    task = tasks_by_name[_task_names[0]]
    _judge_active = False  # judge wiring removed; tasks score with their own metric
    _acc_metric_tag = task.acc_metric_name
    _f1_metric_tag  = task.secondary_metric_name

    import pickle
    model_name = os.path.basename(args.base_model)  # e.g. "Qwen3-8B"

    # ── Pre-prepped tokenized data (the only data path) ──────────────────────
    # Each (dataset, task) pair reads the files written by
    # datasets/<ds>/<task>/prepare.py; the trainer never builds or
    # tokenizes data itself (the legacy in-trainer cache build was removed).
    _ptr_root = args.pretrain_data_root
    _pair_caches = []
    for ds, tk in _pairs:
        _td = os.path.join(_ptr_root, ds, tk, "tokenized", model_name)
        _tr = os.path.join(_td, "train.pkl")
        _vl = os.path.join(_td, "val.pkl")  # optional; absent → empty eval
        if not os.path.exists(_tr):
            raise FileNotFoundError(
                f"Expected tokenized train file at {_tr} but it was not "
                f"found.  Run `python datasets/{ds}/{tk}/prepare.py` "
                f"first to generate it."
            )
        # Stash the val path in the cache-dir slot so the loader can read it.
        _pair_caches.append((ds, tk, _td, _vl, _tr))
    print(f"[train] Loading {len(_pair_caches)} pre-prepped tokenized "
          f"dataset(s) from {_ptr_root}/")

    # ── All ranks: load every per-pair cache and concatenate ─────────────────
    tokenized_train: List[dict] = []
    eval_samples: List[dict] = []
    _pretrain_packed = getattr(args, "pretrain_packed", False)
    # Shard the pool across ranks AT LOAD (per-dataset, below) rather than after
    # building the full pool on every rank: with --shuffle-epochs each rank only
    # ever trains on its [rank::world_size] slice, and replicating the whole
    # pool (v3: ~314GB unpickled) on 8 ranks peaks at ~2.5TB during the load →
    # OOM.  _n_train_global accumulates the pre-shard global count so the step
    # estimate stays "one epoch = one pass over the full pool across ranks".
    _shard_at_load = (getattr(args, "shuffle_epochs", False) and world_size > 1)
    _n_train_global = 0
    # Per project policy we NEVER truncate a sample: a row whose prunable article
    # is outside [MIN_ARTICLE_TOKENS, MAX_ARTICLE_TOKENS] is DROPPED, not
    # shortened.  Truncating a long doc loses the content its target depends on,
    # and running the full (e.g. 137K-token) context uncompressed in the
    # no-compress baseline eval OOMs the GPU.
    _MAX_ART = args.max_article_tokens
    _MIN_ART = args.min_article_tokens
    # Drop (never truncate — same policy as the article cap) rows whose target
    # answer exceeds --max-answer-tokens.  Summarization datasets have long,
    # highly-variable answers (arxiv n_ans up to ~40k, p95 ~650, thousands of
    # distinct lengths); since the compiled Phase-2 sequence is ~K_kept + n_ans,
    # that variability is what floods torch.compile/FlexAttention with distinct
    # shapes and stalls the first-step compile on multi-GPU.  Capping n_ans
    # collapses the shape spread.  None = no cap (legacy behaviour).
    _MAX_ANS = getattr(args, "max_answer_tokens", None)
    def _in_article_range(_row):
        _n = _row["article_end"] - _row["article_start"]
        if _MIN_ART is not None and _n < _MIN_ART:
            return False
        if _MAX_ART is not None and _n > _MAX_ART:
            return False
        if _MAX_ANS is not None and _row.get("n_ans", 0) > _MAX_ANS:
            return False
        return True

    # Eval (val) rows can use a SEPARATE, wider length window than training so the
    # in-training eval set stays frozen while the training window is tightened for
    # speed (decoupled comparability).  Falls back to the train caps when unset.
    _EVAL_MIN_ART = (args.eval_min_article_tokens
                     if args.eval_min_article_tokens is not None else _MIN_ART)
    _EVAL_MAX_ART = (args.eval_max_article_tokens
                     if args.eval_max_article_tokens is not None else _MAX_ART)
    def _eval_in_article_range(_row):
        _n = _row["article_end"] - _row["article_start"]
        if _EVAL_MIN_ART is not None and _n < _EVAL_MIN_ART:
            return False
        if _EVAL_MAX_ART is not None and _n > _EVAL_MAX_ART:
            return False
        if _MAX_ANS is not None and _row.get("n_ans", 0) > _MAX_ANS:
            return False
        return True
    if is_main and (_EVAL_MIN_ART != _MIN_ART or _EVAL_MAX_ART != _MAX_ART):
        print(f"[train] eval length window DECOUPLED from training: "
              f"train=[{_MIN_ART},{_MAX_ART}]  eval=[{_EVAL_MIN_ART},{_EVAL_MAX_ART}]",
              flush=True)

    # ── Packed-store fast path ───────────────────────────────────────────────
    # When --pretrain-packed: memory-map each dataset's token array, build the
    # rank-local meta table, and assemble a PackedTrainStore.  Training tokens
    # are never unpickled into Python lists (10x less RAM, ms-scale load, shared
    # across ranks via the page cache).  Eval still comes from val.pkl (small,
    # and the no-compress baseline needs the raw article text).
    if _pretrain_packed:
        from tokengate.datasets import pack as _pack
        _pk_token_arrays: List = []
        _pk_meta_blocks: List = []
        _pk_ds_names: List[str] = []
        _pk_ds_tasks: List[str] = []
        for ds, tk, cd, ck, cp in _pair_caches:
            _t_ptr = tasks_by_name[tk]
            _tokens, _meta, _info = _pack.load_packed(cd)
            # Same loud prompt-version check as the pkl path.
            _validate_cache_prompt_version(
                {"prompt_version": _info.get("prompt_version")}, _t_ptr, cd)
            # Vectorized drop-not-truncate filter on article length + answer length.
            _art_len = _meta[:, _pack.AE] - _meta[:, _pack.AS]
            _keep = np.ones(_meta.shape[0], dtype=bool)
            if _MIN_ART is not None:
                _keep &= _art_len >= _MIN_ART
            if _MAX_ART is not None:
                _keep &= _art_len <= _MAX_ART
            if _MAX_ANS is not None:
                _keep &= _meta[:, _pack.NANS] <= _MAX_ANS
            _n_tr_before = int(_meta.shape[0])
            _meta = _meta[_keep]
            _n_tr_dropped = _n_tr_before - int(_meta.shape[0])
            _n_train_global += int(_meta.shape[0])  # pre-shard global count
            if _shard_at_load:
                _meta = _meta[rank::world_size]
            # Attach a dataset-index column (points into _pk_token_arrays).
            _ds_idx = len(_pk_token_arrays)
            _block = np.empty((_meta.shape[0], _pack.N_META_COLS + 1), dtype=np.int64)
            _block[:, :_pack.N_META_COLS] = _meta
            _block[:, PackedTrainStore._DSI] = _ds_idx
            _pk_token_arrays.append(_tokens)
            _pk_meta_blocks.append(_block)
            _pk_ds_names.append(ds)
            _pk_ds_tasks.append(tk)

            # ── Eval (val.pkl), identical semantics to the pkl path ──────────
            _vrows: List[dict] = []
            _n_ev_before = 0
            if ck and os.path.exists(ck):
                with open(ck, "rb") as _vf:
                    _vraw = pickle.load(_vf)
                if not isinstance(_vraw, list):
                    _vraw = _vraw.get("eval_samples", [])
                _n_ev_before = len(_vraw)
                _vrows = [r for r in _vraw if _eval_in_article_range(r)]
                for _r in _vrows:
                    _r.setdefault("input", _t_ptr.format_prompt(_r["article"]))
                    _r.setdefault("dataset", ds)
                    _r.setdefault("task", tk)
            _n_ev_dropped = _n_ev_before - len(_vrows)
            eval_samples.extend(_vrows)
            if (_n_tr_dropped or _n_ev_dropped) and is_main:
                print(f"[train]   {ds}/{tk}: dropped {_n_tr_dropped} train + "
                      f"{_n_ev_dropped} eval rows outside [{_MIN_ART}, {_MAX_ART}] "
                      f"article tokens (kept {int(_block.shape[0])} local train, "
                      f"{len(_vrows)} eval)")

        _pk_meta = (np.concatenate(_pk_meta_blocks, axis=0) if _pk_meta_blocks
                    else np.empty((0, _pack.N_META_COLS + 1), dtype=np.int64))

        # ── Train/eval dedup (token-hash safety net) ─────────────────────────
        # The pkl path hashed article *text*; here we hash the article *token*
        # sub-array (stored per train row in the meta HASH column, computed the
        # same way over each eval row at load) so the two are comparable without
        # keeping any train-row text in RAM.  Identical eval set on every rank →
        # identical drop, so DDP shards stay consistent.
        if eval_samples:
            _eval_hashes = np.fromiter(
                (_pack.art_token_hash(
                    _s["full_ids"][_s["article_start"]:_s["article_end"]])
                 for _s in eval_samples),
                dtype=np.int64, count=len(eval_samples),
            )
            _eval_hash_set = np.unique(_eval_hashes)
            _dup_mask = np.isin(_pk_meta[:, _pack.HASH], _eval_hash_set)
            _n_dup = int(_dup_mask.sum())
            if _n_dup:
                _pk_meta = _pk_meta[~_dup_mask]
                if is_main:
                    print(f"[train] Train/eval dedup: dropped {_n_dup} train rows "
                          f"whose article tokens also appear in eval.")
            elif is_main:
                print("[train] Train/eval dedup: no overlapping articles between "
                      "train and eval (clean splits).")

        tokenized_train = PackedTrainStore(
            _pk_token_arrays, _pk_ds_names, _pk_ds_tasks, _pk_meta)
        print(f"[rank {rank}] Loaded {len(tokenized_train)} train (packed mmap) / "
              f"{len(eval_samples)} eval samples from {len(_pair_caches)} dataset(s).")

    for ds, tk, cd, ck, cp in (_pair_caches if not _pretrain_packed else []):
        if not os.path.exists(cp):
            raise RuntimeError(
                f"[rank {rank}] Expected tokenized train file at {cp} but it "
                f"was not found."
            )
        with open(cp, "rb") as f:
            _data = pickle.load(f)
        # Pretraining-mode .pkl files are List[Dict] (prepare.py's write_pkl).
        # Synthesize the dict shape the rest of this code expects.
        # Per-row prompt_version (written by prepare.py) is validated against
        # the current Task class — same loud-failure semantics as multitask.
        _tok_train = _data if isinstance(_data, list) else _data["tokenized_train"]
        _t_ptr = tasks_by_name[tk]
        # Drop out-of-range rows (NEVER truncate), then count globally
        # (post-filter, pre-shard) for step accounting, then shard AT LOAD so
        # each rank holds ~1/world_size of the pool (the v3 pool unpickles to
        # ~314GB → ~2.5TB across 8 ranks under full replication → OOM).
        # Re-binding _tok_train below drops the full list's last reference at
        # the `_data = {...}` build, so refcounting frees the discarded rows.
        _n_tr_before = len(_tok_train)
        _tok_train = [r for r in _tok_train if _in_article_range(r)]
        _n_tr_dropped = _n_tr_before - len(_tok_train)
        _n_train_global += len(_tok_train)
        if _shard_at_load:
            _tok_train = _tok_train[rank::world_size]
        # Synthesize the eval-time "input" (= format_prompt(article)); over-cap
        # rows are gone so every article here is <= MAX_ART tokens (the
        # no-compress baseline eval re-tokenizes sample["input"]).
        for _r in _tok_train:
            _r.setdefault("input", _t_ptr.format_prompt(_r["article"]))
        # eval (val.pkl): per-dataset cap, then the same drop-not-truncate filter.
        _vrows: List[dict] = []
        _n_ev_before = 0
        _val_path = ck
        if _val_path and os.path.exists(_val_path):
            with open(_val_path, "rb") as _vf:
                _vraw = pickle.load(_vf)
            if not isinstance(_vraw, list):
                # Fall back to dict shape if a multitask-style val.pkl was placed there.
                _vraw = _vraw.get("eval_samples", [])
            _n_ev_before = len(_vraw)
            _vrows = [r for r in _vraw if _eval_in_article_range(r)]
            for _r in _vrows:
                _r.setdefault("input", _t_ptr.format_prompt(_r["article"]))
        _n_ev_dropped = _n_ev_before - len(_vrows)
        # prompt_version from whichever split survived (same loud-failure check).
        _pv_rows = _tok_train if _tok_train else _vrows
        _per_row_pv = _pv_rows[0].get("prompt_version") if _pv_rows else None
        _data = {
            "tokenized_train": _tok_train,
            "eval_samples": _vrows,
            "prompt_version": _per_row_pv,
        }
        if (_n_tr_dropped or _n_ev_dropped) and is_main:
            print(f"[train]   {ds}/{tk}: dropped {_n_tr_dropped} train + "
                  f"{_n_ev_dropped} eval rows outside [{_MIN_ART}, {_MAX_ART}] "
                  f"article tokens (kept {len(_tok_train)} local train, "
                  f"{len(_vrows)} eval)")
        # If removal left this dataset with nothing on this rank, skip it —
        # nothing to validate/extend.  (A dataset whose articles ALL exceed
        # MAX_ART should be dropped from the manifest entirely.)
        if not _tok_train and not _vrows:
            if is_main:
                print(f"[train]   {ds}/{tk}: SKIPPED — no rows within the article cap.")
            continue
        _validate_cache_prompt_version(_data, tasks_by_name[tk], cp)
        # Older caches were built before build_sft_samples emitted "dataset"/"task";
        # stamp them on every loaded sample so the multitask lookups below work.
        for _row in _data["tokenized_train"]:
            _row.setdefault("dataset", ds)
            _row.setdefault("task", tk)
        for _row in _data["eval_samples"]:
            _row.setdefault("dataset", ds)
            _row.setdefault("task", tk)
        tokenized_train.extend(_data["tokenized_train"])
        eval_samples.extend(_data["eval_samples"])
    if not _pretrain_packed:
        print(f"[rank {rank}] Loaded {len(tokenized_train)} train / "
              f"{len(eval_samples)} eval samples from {len(_pair_caches)} cache(s).")

    # ── Train/eval content dedup (safety net) ────────────────────────────────
    # train.pkl / val.pkl already come from disjoint source splits, but this
    # guards against the same document landing in BOTH splits (which would let
    # the gate memorise token selection for an eval article).  Drop any train
    # row whose article matches an eval row's article (hash on the article =
    # prunable context).  Deterministic, so every rank drops the same rows and
    # tokenized_train stays consistent across ranks for DDP.  pretrain-mix only.
    # (Packed mode runs the equivalent token-hash dedup in its own branch above.)
    if eval_samples and not _pretrain_packed:
        import hashlib

        def _art_hash(_s: dict) -> str:
            return hashlib.sha1(_s["article"].encode("utf-8")).hexdigest()

        _eval_art_hashes = {_art_hash(_s) for _s in eval_samples}
        _n_before = len(tokenized_train)
        _dropped_by_ds: Dict[str, int] = {}
        _kept_train: List[dict] = []
        for _s in tokenized_train:
            if _art_hash(_s) in _eval_art_hashes:
                _dropped_by_ds[_s["dataset"]] = _dropped_by_ds.get(_s["dataset"], 0) + 1
            else:
                _kept_train.append(_s)
        tokenized_train = _kept_train
        _n_dropped = _n_before - len(tokenized_train)
        if is_main:
            if _n_dropped:
                print(f"[train] Train/eval dedup: dropped {_n_dropped} train rows whose "
                      f"article also appears in eval (per dataset: {_dropped_by_ds}); "
                      f"train {_n_before} → {len(tokenized_train)}")
            else:
                print("[train] Train/eval dedup: no overlapping articles between "
                      "train and eval (clean splits).")

    # ── DDP data sharding (memory) ───────────────────────────────────────────
    # The pool was sharded AT LOAD (per-dataset, in the loop above) for
    # pretrain-mix + --shuffle-epochs + multi-GPU, so tokenized_train already
    # holds only this rank's [rank::world_size] slice.  Doing it at load — rather
    # than building the full pool on every rank and slicing afterwards — keeps the
    # load-time peak to ~one dataset's transient + the accumulated shard.  The v3
    # 28-dataset pool unpickles to ~314GB, which would peak at ~2.5TB across 8
    # ranks (→ OOM) under full replication; per-dataset load-time sharding keeps
    # it to ~700GB.  _n_train_global was summed pre-shard in the loop so the step
    # count stays one epoch = one pass over the full pool across all ranks.
    _data_presharded = _shard_at_load
    if not _data_presharded:
        # Single-GPU / non-shuffle: no load-time shard happened, so this rank
        # holds the full pool; the global count is just its (post-dedup) length.
        _n_train_global = len(tokenized_train)
    else:
        print(f"[rank {rank}] DDP load-time shard: ~{_n_train_global} global train "
              f"rows → {len(tokenized_train)} local ([{rank}::{world_size}])")

    # ── Eval-only (held-out) val sets ────────────────────────────────────────
    # Load val.pkl for subsets we evaluate but never train on (e.g. the LongBench
    # subsets with no upstream original, or the few-shot ones deferred from
    # training).  These rows join the eval pool only — train/pack never see them.
    # Loaded once on every rank (eval is the full set per rank; evaluate() shards
    # it internally), for both the packed and pkl paths (eval is always val.pkl).
    if _eval_only_pairs:
        _n_eo = 0
        for _eds, _etk in _eval_only_pairs:
            _vl = os.path.join(args.pretrain_data_root, _eds, _etk,
                               "tokenized", model_name, "val.pkl")
            if not os.path.exists(_vl):
                raise FileNotFoundError(
                    f"--pretrain-eval-only-pairs expected a tokenized val file at "
                    f"{_vl} but it was not found.  Run the subset's prepare.py first.")
            with open(_vl, "rb") as _f:
                _vraw = pickle.load(_f)
            if not isinstance(_vraw, list):
                _vraw = _vraw.get("eval_samples", [])
            _vrows = [r for r in _vraw if _eval_in_article_range(r)]
            _eo_task = tasks_by_name[_etk]
            for _r in _vrows:
                _r.setdefault("dataset", _eds)
                _r.setdefault("task", _etk)
                # evaluate() reads sample["input"] (the rendered prompt); set it
                # the same way the trained val path does.
                _r.setdefault("input", _eo_task.format_prompt(_r["article"]))
            eval_samples.extend(_vrows)
            _n_eo += len(_vrows)
            if is_main:
                print(f"[train]   eval-only {_eds}/{_etk}: {len(_vrows)} held-out val rows")
        if is_main:
            print(f"[train] Loaded {_n_eo} held-out eval rows from "
                  f"{len(_eval_only_pairs)} eval-only subset(s).")

    # ── Eval val/test split: keep ONLY manifest `val` rows ────────────────────
    # The split manifest is the EXPLICIT per-dataset eval/val size mechanism:
    # each subset's `val` ids form the in-training val set and the reserved
    # `test` ids the held-out test set (the implicit eval-samples-per-group
    # budget was removed).  Drops (a) every subset not in the manifest and
    # (b) the reserved `test` rows.  Applied after train/eval dedup so train is
    # deduped against the FULL eval set (test rows included) — no train/test
    # article leakage.
    if eval_samples and not getattr(args, "eval_split_manifest", ""):
        raise ValueError(
            "--eval-split-manifest is required when eval rows are loaded: the "
            "val/test set per dataset must be pinned explicitly by a split "
            "manifest (tokengate/datasets/make_longbench_eval_split.py).")
    if getattr(args, "eval_split_manifest", ""):
        with open(args.eval_split_manifest) as _sf:
            _split = json.load(_sf)
        _val_ids = {tk: set(v["val"]) for tk, v in _split.items()}
        # --eval-test-full-tasks: capture the listed subsets' full `test` partition
        # BEFORE the val-only filter drops it; appended later as uncapped group(s).
        # Each entry is `task` or `task:wandb_group` (default group `<task>_test`).
        _tf_pairs: List[Tuple[str, str]] = []
        for _e in (getattr(args, "eval_test_full_tasks", None) or []):
            _tk, _, _grp = _e.partition(":")
            _tf_pairs.append((_tk, _grp or f"{_tk}_test"))
        _test_full_tasks = [tk for tk, _ in _tf_pairs]
        _bad_tf = [tk for tk in _test_full_tasks if tk not in _split]
        if _bad_tf:
            raise ValueError(
                f"--eval-test-full-tasks: {_bad_tf} not in eval-split manifest "
                f"{args.eval_split_manifest} (tracked: {sorted(_split)})")
        _tf_ids = {tk: set(_split[tk]["test"]) for tk in _test_full_tasks}
        _test_full_rows: Dict[str, List[dict]] = {tk: [] for tk in _test_full_tasks}
        for _r in eval_samples:
            if _r.get("task") in _tf_ids and _r.get("id") in _tf_ids[_r["task"]]:
                # COPY the row: when the split json's `test` list overlaps `val`
                # (e.g. the narrativeqa iso split where test = val+test), the
                # captured rows are the same dicts kept in eval_samples for the
                # val group — mutating `dataset` on a shared dict would relabel
                # the val group's rows too.
                _test_full_rows[_r["task"]].append(dict(_r))
        _before = len(eval_samples)
        _keep_untracked = getattr(args, "eval_split_keep_untracked", False)
        # Tracked (in-manifest, LongBench) subsets: keep ONLY the `val` partition.
        # Untracked (trained subsets absent from the manifest, e.g. the few-shot
        # classification packs): keep ALL their val rows iff --eval-split-keep-untracked,
        # else drop (legacy LongBench-only behavior).
        eval_samples = [r for r in eval_samples
                        if (r.get("task") in _val_ids
                            and r.get("id") in _val_ids[r["task"]])
                        or (_keep_untracked and r.get("task") not in _val_ids)]
        if is_main:
            from collections import Counter
            _kept = Counter(r["task"] for r in eval_samples)
            print(f"[train] eval-split-manifest ({os.path.basename(args.eval_split_manifest)}): "
                  f"kept {len(eval_samples)}/{_before} eval rows over {len(_kept)} "
                  f"LongBench subset(s) (val partition only; test rows reserved).")
            # A small shortfall is expected+fine: the article-token range filter
            # (MAX_ARTICLE_TOKENS) legitimately drops the occasional long val row, so
            # a group ends up with 49 instead of 50. Only warn (the filter is
            # deterministic across ranks, so every rank keeps the identical set).
            _short = [(tk, _kept.get(tk, 0), len(_val_ids[tk]))
                      for tk in _val_ids if _kept.get(tk, 0) < len(_val_ids[tk])]
            if _short:
                print("[train]   note: subsets below manifest val count (article-range "
                      "filter dropped long rows): "
                      + ", ".join(f"{tk} {k}/{n}" for tk, k, n in _short))

    # ── Multitask lookups ────────────────────────────────────────────────────
    # Per-subset eval identity = the TASK.  The LongBench-matched mix gives every
    # subset the SAME manifest dataset name (distinguished only by task), so
    # keying eval grouping/breakdown by "dataset" would collapse all subsets into
    # one bucket.  Re-key the eval pool by task, and rewrite each eval row's
    # "dataset" to its task so evaluate()'s internal per-dataset breakdown also
    # buckets per subset.  For 1:1 dataset/task mixes (v3) this is a no-op relabel
    # (dataset and task are already in bijection there).
    ds_task: Dict[str, str] = {ds: tk for ds, tk in _pairs}
    for _s in eval_samples:
        _s["dataset"] = _s["task"]
    eval_samples_by_dataset: Dict[str, List[dict]] = {}
    for _s in eval_samples:
        eval_samples_by_dataset.setdefault(_s["task"], []).append(_s)
    # subset id == task → ds_task resolves each eval subset to its own task.
    for _subset in eval_samples_by_dataset:
        ds_task[_subset] = _subset
    # ── Task-category grouping for eval ──────────────────────────────────────
    # The pretraining eval is reported as category val sets — QA, code completion,
    # summarization, … — each pooling every subset whose task maps to that
    # category.  Category is resolved PER SUBSET (= per task), so a shared-dataset-
    # name mix (LongBench-matched) still separates into its true metric families.
    # Held-out (eval-only) subsets are split into a separate "<category>_heldout"
    # group when that metric family also has a trained member, so trained vs
    # held-out generalization are reported side-by-side but never pooled together.
    # Families with no trained member (e.g. classification=trec, synthetic=passage_*)
    # keep their bare category name (they are held-out by nature).
    _trained_cats = {get_task_category(tk) for _, tk in _pairs}
    _trained_tasks = {tk for _, tk in _pairs}

    # Eval subsets whose SAME dataset is trained via a disjoint-split twin task
    # (a DIFFERENT task name) are IN-DISTRIBUTION, not held-out — even though the
    # eval-subset task itself is never trained.  When that twin is present in the
    # mix, report the subset under its bare category (pooled with the trained
    # members) instead of "<category>_heldout".  triviaqa/multifieldqa have only
    # DIFFERENT-dataset proxies (squad / natural_questions), so they are NOT listed
    # here and stay held-out.  (These are also relabeled in the split manifest.)
    _SAME_DATASET_TWIN = {
        "longbench_samsum": "samsum_style_fewshot",   # knkarthick/samsum train, deduped
        # longbench_trec was listed here while trec_style_fewshot mined real
        # TREC-train demos (same dataset → in-distribution).  The decontam-policy
        # rebuild replaced that pool with LLM-generated questions + TREC-test
        # survivors, all excluded from every eval prompt — so trec is HELD-OUT
        # again and must report as classification_heldout (pooling it into the
        # capped classification group also diluted its 50-row val partition to a
        # ~11-row subsample, breaking cross-run no-compress comparability).
    }

    # Optional explicit eval-group override: "group:task1,task2;group2:task3" pins a
    # subset's reported group REGARDLESS of what is trained.  Lets a harness reproduce
    # v10's exact groups (qa = the 5 multi-hop subsets, qa_heldout = multifieldqa+
    # triviaqa) while those subsets are held OUT of training to measure transfer —
    # the trained-category heuristic below would otherwise dump every eval-only qa
    # subset into qa_heldout.  Keyed by task name (== subset id here).
    _grp_override: Dict[str, str] = {}
    _ovr = getattr(args, "eval_group_override", "") or ""
    for _seg in _ovr.split(";"):
        _seg = _seg.strip()
        if not _seg or ":" not in _seg:
            continue
        _grp, _tks = _seg.split(":", 1)
        for _tk in _tks.split(","):
            if _tk.strip():
                _grp_override[_tk.strip()] = _grp.strip()

    def _eval_group_label(_tk: str) -> str:
        if _tk in _grp_override:
            return _grp_override[_tk]
        _c = get_task_category(_tk)
        # Same-dataset twin in the mix → in-distribution (bare category), not held-out.
        _twin = _SAME_DATASET_TWIN.get(_tk)
        if _twin is not None and _twin in _trained_tasks:
            return _c
        if _tk in _eval_only_tasks and _c in _trained_cats:
            return f"{_c}_heldout"
        return _c

    # cat_of_ds is keyed by subset id (== task) to match eval_samples_by_dataset.
    cat_of_ds: Dict[str, str] = {
        _subset: _eval_group_label(_subset) for _subset in eval_samples_by_dataset
    }
    if is_main:
        _ds_by_cat: Dict[str, List[str]] = {}
        for _ds, _c in cat_of_ds.items():
            _ds_by_cat.setdefault(_c, []).append(_ds)
        print("[train] Eval val sets by task category:")
        for _c in PRETRAIN_VAL_CATEGORIES:
            _members = _ds_by_cat.get(_c, [])
            print(f"[train]   {_c:16s}: {_members if _members else '(none in this mix)'}")
        _other_cats = [c for c in _ds_by_cat if c not in PRETRAIN_VAL_CATEGORIES]
        if _other_cats:
            _other_pairs = [
                f"{_ds} ({cat_of_ds[_ds]})"
                for _c in sorted(_other_cats) for _ds in _ds_by_cat[_c]
            ]
            print("[train]   NOTE: these datasets match none of the three "
                  "QA / code_completion / summarization val sets and are "
                  f"reported under their own category: {_other_pairs}")

    # ── Pooled per-metric-group eval sets ────────────────────────────────────
    # One val set per metric group (qa / code_completion / summarization / ...):
    # pool every member dataset's val rows.  The set SIZE is explicit — the
    # split manifest's per-subset `val` id lists pin it exactly (the implicit
    # per-group sampling budget was removed) — so each group's set is simply
    # its full pool.  Deterministic load order on every rank keeps the DDP
    # shard (range(rank, n, world_size)) consistent.
    _grp_pool: Dict[str, List[dict]] = {}
    for _ds, _samps in eval_samples_by_dataset.items():
        _grp_pool.setdefault(cat_of_ds[_ds], []).extend(_samps)
    eval_samples_by_group: Dict[str, List[dict]] = {
        _g: list(_pool) for _g, _pool in _grp_pool.items()
    }
    # --eval-test-full-tasks: append the captured full test partitions as extra
    # groups, with subset identity `<task>_test` so their rows never pool into
    # the standard groups.
    if getattr(args, "eval_test_full_tasks", None):
        if not getattr(args, "eval_split_manifest", ""):
            raise ValueError("--eval-test-full-tasks requires --eval-split-manifest")
        for _tk, _tf_grp in _tf_pairs:
            _rows = _test_full_rows[_tk]
            if not _rows:
                raise ValueError(
                    f"--eval-test-full-tasks: no test rows for {_tk} survived loading "
                    "(dropped by the article-token filters, or the subset is absent "
                    "from this mix's eval pool)")
            if _tf_grp in eval_samples_by_group:
                raise ValueError(
                    f"--eval-test-full-tasks: group {_tf_grp!r} already exists (members: "
                    f"{sorted({r['dataset'] for r in eval_samples_by_group[_tf_grp]})}); "
                    "move its members to another group with --eval-group-override "
                    "before claiming the name")
            for _r in _rows:
                _r["dataset"] = f"{_tk}_test"
            ds_task[f"{_tk}_test"] = _tk
            eval_samples_by_group[_tf_grp] = _rows
            if is_main:
                print(f"[train]   {_tf_grp}: FULL reserved test partition of {_tk}, "
                      f"n={len(_rows)} (uncapped; --eval-test-full-tasks)", flush=True)
    # --eval-ind-train-groups: uncapped group(s) of TRAINING rows (in-distribution
    # eval; rows remain in training — see the argparse help).
    for _ig in (getattr(args, "eval_ind_train_groups", None) or []):
        _ig_grp, _ig_n, _ig_pairs = _ig.split(":", 2)
        _ig_n = int(_ig_n)
        if _ig_grp in eval_samples_by_group:
            raise ValueError(f"--eval-ind-train-groups: group {_ig_grp!r} already exists")
        _ig_pool: List[dict] = []
        for _pair in _ig_pairs.split(","):
            _ids, _itk = _pair.strip().split("/", 1)
            _tp = os.path.join(args.pretrain_data_root, _ids, _itk,
                               "tokenized", model_name, "train.pkl")
            if not os.path.exists(_tp):
                raise FileNotFoundError(f"--eval-ind-train-groups: no tokenized train "
                                        f"file at {_tp}")
            with open(_tp, "rb") as _f:
                _trows = pickle.load(_f)
            _it = tasks_by_name[_itk]
            for _r in _trows:
                # TRAIN window, not the (wider) eval range: the group must sample
                # only rows the training loop can actually draw.
                _al = _r["article_end"] - _r["article_start"]
                if args.min_article_tokens is not None and _al < args.min_article_tokens:
                    continue
                if args.max_article_tokens is not None and _al > args.max_article_tokens:
                    continue
                _r = dict(_r)
                _r["dataset"] = f"{_ids}_ind"
                _r["task"] = _itk
                _r.setdefault("input", _it.format_prompt(_r["article"]))
                _ig_pool.append(_r)
            ds_task[f"{_ids}_ind"] = _itk
        if len(_ig_pool) < _ig_n:
            raise ValueError(f"--eval-ind-train-groups: pool for {_ig_grp!r} has only "
                             f"{len(_ig_pool)} in-range rows (< N={_ig_n})")
        _ig_rng = random.Random(args.seed + 7919)
        eval_samples_by_group[_ig_grp] = _ig_rng.sample(_ig_pool, _ig_n)
        if is_main:
            from collections import Counter as _C
            _mix = dict(_C(r["dataset"] for r in eval_samples_by_group[_ig_grp]))
            print(f"[train]   {_ig_grp}: {_ig_n} TRAINING rows sampled uniformly from "
                  f"{len(_ig_pool)} pooled ({_mix}); rows remain in training "
                  f"(--eval-ind-train-groups)", flush=True)
    if is_main:
        print("[train] Pooled eval set per metric group:")
        for _g, _s in eval_samples_by_group.items():
            _src = {}
            for _r in _s:
                _src[_r["dataset"]] = _src.get(_r["dataset"], 0) + 1
            print(f"[train]   {_g:16s}: n={len(_s)}  from {_src}")

    # ── One-time dataset stats ────────────────────────────────────────────────
    if is_main:
        if _pretrain_packed:
            _art_lens = tokenized_train.article_lengths()
            avg_train_n_art = float(_art_lens.mean()) if _art_lens.size else 0.0
        else:
            avg_train_n_art = (
                sum(s["article_end"] - s["article_start"] for s in tokenized_train)
                / max(len(tokenized_train), 1)
            )
        # Eval samples are raw dicts without precomputed boundaries; tokenize article field.
        eval_art_tokens = [
            len(tokenizer.encode(s["article"], add_special_tokens=False))
            for s in eval_samples
        ]
        avg_eval_n_art = sum(eval_art_tokens) / max(len(eval_art_tokens), 1)
        print(f"[train] Avg article tokens — train: {avg_train_n_art:.0f}, eval: {avg_eval_n_art:.0f}")
        if _wandb_run is not None:
            _wandb_run.summary["data/avg_train_article_tokens"] = avg_train_n_art
            _wandb_run.summary["data/avg_eval_article_tokens"] = avg_eval_n_art
            _wandb_run.summary["data/n_train_samples"] = _n_train_global
            _wandb_run.summary["data/n_eval_samples"] = len(eval_samples)
            # Per-dataset sample counts for multitask runs.
            if _pretrain_packed:
                # Vectorized: count rows per dataset-array index, map back to the
                # array's TASK (eval keys are per-subset = per-task; a held-out
                # subset has no array → correctly resolves to 0 train rows).
                _dsi = tokenized_train.dataset_idx()
                _counts = np.bincount(_dsi, minlength=len(_pk_ds_names)) if len(_dsi) else \
                    np.zeros(len(_pk_ds_names), dtype=np.int64)
                _train_n_by_ds = {_pk_ds_tasks[_k]: int(_counts[_k])
                                  for _k in range(len(_pk_ds_names))}
            else:
                _train_n_by_ds = None
            for _ds, _samps in eval_samples_by_dataset.items():
                if _train_n_by_ds is not None:
                    _ds_train_n = _train_n_by_ds.get(_ds, 0)
                else:
                    _ds_train_n = sum(1 for s in tokenized_train if s["task"] == _ds)
                _wandb_run.summary[f"data/{_ds}/n_train_samples"] = _ds_train_n
                _wandb_run.summary[f"data/{_ds}/n_eval_samples"] = len(_samps)

    # ── Resolve steps from epochs (if --epochs was given) ────────────────────
    import math
    if args.epochs is not None:
        effective_batch = args.batch_size * world_size
        # Use the GLOBAL (pre-shard) train size: under DDP data sharding each
        # rank holds 1/world_size of the data, but an "epoch" is still one pass
        # over the full pool across all ranks.
        args.steps = math.ceil(args.epochs * _n_train_global / effective_batch)
        if is_main:
            print(f"[train] --epochs {args.epochs} → {args.steps} steps "
                  f"({_n_train_global} global samples, eff. batch {effective_batch})")

    warmup_steps = min(args.warmup_steps, args.steps)
    use_constant_lr = args.lr_schedule == "constant"
    if warmup_steps > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps,
        )
        if use_constant_lr:
            # ConstantLR with factor=1.0 is a no-op — holds LR at its current value.
            post_warmup_sched = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=max(args.steps - warmup_steps, 1),
            )
        else:
            post_warmup_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.steps - warmup_steps, 1), eta_min=args.lr * 0.1,
            )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, post_warmup_sched], milestones=[warmup_steps],
        )
    else:
        if use_constant_lr:
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=args.steps,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.steps, eta_min=args.lr * 0.1,
            )

    if _resume_state is not None:
        # Fast-forward a FRESH scheduler to the saved position instead of
        # load_state_dict.  load_state_dict restores the scheduler's counters but
        # not the optimizer's live lrs: constructing LinearLR above already
        # multiplied every group's lr by start_factor (1e-8), and the post-warmup
        # ConstantLR(factor=1.0) is multiplicative on whatever the group holds,
        # so a resumed run trained at lr*1e-8 forever (seen: a 600-step cycle
        # with byte-identical eval and train/lr=1e-12) -- and then saved THAT lr
        # into its own train_state, so trusting the checkpoint's _last_lr would
        # propagate the damage.  Replaying the schedule from the base lrs is
        # exact for every schedule here and costs nothing.
        _saved_epoch = int(_resume_state["scheduler_state"]["last_epoch"])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "scheduler.step() before optimizer.step()"
            for _ in range(_saved_epoch):
                scheduler.step()
        if scheduler.last_epoch != _saved_epoch:
            raise RuntimeError(f"resume: scheduler replay reached epoch {scheduler.last_epoch}, "
                               f"checkpoint says {_saved_epoch}")
        _last_lrs = [g["lr"] for g in optimizer.param_groups]
        if any(x < args.lr * 1e-3 for x in _last_lrs):
            raise RuntimeError(f"resume: restored group lrs {_last_lrs} are far below --lr {args.lr}")
        if is_main:
            print(f"[train] Restored LR scheduler position {_saved_epoch} (group lrs: "
                  f"{', '.join(f'{x:.2e}' for x in _last_lrs)})")

    # ── SWA setup ─────────────────────────────────────────────────────────────
    # NOTE: intentionally NOT gated on is_main so that all ranks enter the SWA
    # block and hit the dist.barrier() calls inside it.  Rank 0's eval at SWA
    # init would desync DDP if other ranks skipped the block entirely.
    # Profiling cap: --max-steps hard-bounds the loop for a cheap throughput
    # run (applied AFTER the epochs→steps conversion above).  Shrinking
    # args.steps also auto-disables SWA below, so a short profile never waits
    # for swa_start.
    if args.max_steps is not None:
        args.steps = min(args.steps, args.max_steps)
        if is_main:
            print(f"[profile] --max-steps={args.max_steps} → capping loop at "
                  f"args.steps={args.steps}", flush=True)

    # swa_start_step: absolute step at which SWA averaging begins.
    # Defaults to warmup_steps (current behaviour) when --swa-start is 0.
    # If --swa-start > 0, SWA starts at max(warmup_steps, swa_start) so it
    # never fires before warmup ends.
    swa_start_step = max(warmup_steps, args.swa_start) if args.swa_start > 0 else warmup_steps
    swa_enabled = args.swa_freq > 0 and swa_start_step < args.steps
    if swa_enabled and args.lr_schedule == "cosine":
        print(
            "[train] WARNING: SWA with lr_schedule=cosine averages over a decaying LR "
            "trajectory — the average converges to the final weights and gains little. "
            "Consider lr_schedule=constant, or disable SWA with --swa-freq -1."
        )
    swa_gate: Optional[E2EGate] = None  # averaged gate (all ranks when --swa-eval-ddp, else rank 0)
    swa_n = 0                            # number of SWA updates so far
    # SWA must average ALL trainable params — the gate proj AND the adapter (delta ΔW /
    # LoRA).  For delta runs the adapter is the bulk of what's learned and it lives inside
    # `model` (not in the gate), so we track a running-averaged adapter state dict alongside
    # swa_gate and swap it into `model` for SWA eval (see the SWA-eval block).  Maintained on
    # ALL ranks when --swa-eval-ddp (default; enables the sharded SWA eval), else rank 0 only.
    # GPU-resident fp32 like swa_gate (one extra fp32 copy of the adapter): the per-step
    # update is then a ~1 ms in-place op on the live params.  The previous CPU-resident
    # master cost a full D2H copy of the adapter plus a CPU running-mean pass every step
    # (~0.1-0.2 s/step for a 109M-param delta, 3-6% of the step) on every rank.
    swa_adapter_sd: Optional[Dict[str, torch.Tensor]] = None
    if _resume_state is not None and _resume_state.get("swa_n", 0) > 0:
        # Materialize swa_gate as a deep copy of the (just-restored) gate, then
        # overlay the saved SWA-averaged weights.  fp32 storage like the live gate
        # (asserted: the running mean is exact only in fp32; a pre-fp32-storage
        # train_state's bf16 average loads upcast and averaging continues from
        # whatever it holds).  Materialized on every rank, like the cold-start path.
        swa_gate = copy.deepcopy(gate)
        assert_fp32_params(swa_gate, "swa_gate (resume)")
        swa_gate.load_state_dict(_resume_state["swa_gate_state"])
        swa_gate.eval()
        swa_n = int(_resume_state["swa_n"])
        if _lora_enabled and _resume_state.get("swa_adapter_state") is not None:
            # train_state stores CPU copies; the running average lives on the device
            # next to the live adapter params it is updated from.
            swa_adapter_sd = {k: v.to(device=device, dtype=torch.float32)
                              for k, v in _resume_state["swa_adapter_state"].items()}
        if is_main:
            print(f"[train] Restored SWA state (swa_n={swa_n}, "
                  f"adapter={'yes' if swa_adapter_sd is not None else 'no'})")
    # In threshold mode the SWA eval sweeps the same tau values as the live gate.
    # See the k_eval_list construction below for the naming convention.
    if args.gate_selection == "threshold":
        if args.tau_eval is not None:
            _swa_k_list = sorted(set(float(t) for t in args.tau_eval))
        else:
            _swa_k_list = [float(args.tau)]
    else:
        _swa_k_list = _require_k_eval(args)
    swa_eval_steps:        list = []
    swa_eval_per_k_accs:   dict = {k: [] for k in _swa_k_list}
    swa_eval_per_k_losses: dict = {k: [] for k in _swa_k_list}
    swa_train_losses:      list = []  # spot-check loss on current batch with SWA weights
    # Counter for --swa-eval-freq: increments each time we enter the in-loop
    # SWA-eval block.  When swa_eval_freq=1 every cycle runs (current behavior);
    # >1 skips (swa_eval_freq-1) cycles between SWA evals.  Does NOT gate the
    # final post-training SWA eval.
    _swa_eval_cycle = 0

    def _swa_update(gate_raw: E2EGate) -> None:
        nonlocal swa_n
        swa_inplace_update(swa_gate.named_parameters(), gate_raw.named_parameters(), swa_n)
        swa_n += 1

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses = []
    k_log = []          # (step, K_sampled, K_clamped) per step
    bias_norm_log = []   # (step, gate_output_norm) — avg ‖b_j‖ per step
    attn_scale_log = []  # (step, attn_scale) — value of learned attn_scale parameter
    eval_steps = []
    # Eval-sweep value list.  In top-K mode this is integer K values; in
    # threshold mode it is float tau values that the eval loop applies via
    # gate.tau before each call (K passed to the gate is ignored).  All
    # downstream W&B keys, curve dicts, and *_per_k_* state keep their "K"
    # naming and just index off these values — for threshold runs you'll see
    # keys like `eval/acc_K0.5` which mean "acc at tau=0.5".
    if args.gate_selection == "threshold":
        if args.tau_eval is not None:
            k_eval_list = sorted(set(float(t) for t in args.tau_eval))
        else:
            k_eval_list = [float(args.tau)]
    else:
        k_eval_list = _require_k_eval(args)
    eval_per_k_accs   = {k: [] for k in k_eval_list}  # {K: [acc, ...]}
    eval_per_k_losses = {k: [] for k in k_eval_list}  # {K: [loss, ...]}
    # ── Primary (headline/deployment) K ──────────────────────────────────────
    # Drives: the eval_accs/eval_losses aliases, best-N checkpoint retention, the
    # mid-training single-K eval, and the FastKV legacy overlay alias.
    #
    # This used to be spelled `k_eval_list[0]`, which was harmless while K_EVAL_VALUES
    # held ONE value.  k_eval_list is sorted ASCENDING (for stable per-K logging order),
    # so the moment K_EVAL_VALUES became "2000 1000 500" that expression silently became
    # K=500 -- the headline metric and the best-checkpoint criterion would have switched
    # from the deployment K to the smallest K without anything in the logs saying so.
    # Take the first K as WRITTEN in the config instead: the per-dataset scripts list the
    # deployment K first.  Fail loudly rather than fall back if that isn't a real eval K.
    if args.gate_selection == "threshold":
        k_primary = k_eval_list[0]                      # tau runs: only one ordering
    else:
        k_primary = args.k_eval[0]
    if k_primary not in eval_per_k_accs:
        raise ValueError(f"primary K {k_primary} not among eval K values {k_eval_list}")
    # Aliases to the primary K for backward-compat with _save_curves.
    eval_accs   = eval_per_k_accs[k_primary]
    eval_losses = eval_per_k_losses[k_primary]
    # Step -> primary-K eval acc seen at-or-before each periodic checkpoint
    # save.  Drives last-N + best-N step-checkpoint retention (see
    # _prune_step_checkpoints).  Steps saved before any eval has run record
    # -inf so they can be pruned by last-N but never win best-N.
    _saved_step_accs: Dict[int, float] = {}
    # The eval group whose primary-K accuracy ranks step checkpoints for
    # --checkpoint-keep-best.  The "primary" (first) group is whatever
    # happens to be listed first -- in the isolated LongBench runs that is the
    # 10-row val split, so best-N kept the checkpoint that won on 10 samples and
    # pruned the one that won on the 190-row test partition.  Default: the group
    # with the most samples; --checkpoint-best-group names one explicitly.
    _best_group_latest: Dict[str, Any] = {"acc": float("-inf"), "group": ""}

    def _note_best_group_acc(_per_group: Dict[str, Dict[str, Any]]) -> None:
        _want = getattr(args, "checkpoint_best_group", None)
        if _want:
            if _want not in _per_group:
                raise ValueError(
                    f"--checkpoint-best-group {_want!r} is not an eval group "
                    f"(have: {sorted(_per_group)})")
            _g = _want
        else:
            _g = max(_per_group, key=lambda g: _per_group[g]["n_samples"])
        if _best_group_latest["group"] != _g:
            _best_group_latest["group"] = _g
            if is_main:
                print(f"[ckpt] --checkpoint-keep-best ranks by group {_g!r} "
                      f"(n={_per_group[_g]['n_samples']}) at K={k_primary}", flush=True)
        _best_group_latest["acc"] = float(_per_group[_g]["accuracy"])
    # On resume, seed it with the step checkpoints already on disk (acc unknown,
    # so -inf: eligible for last-N, never best-N).  Otherwise the first
    # post-resume save sees only itself as "saved" and prunes every pre-resume
    # checkpoint, including the one just resumed from.
    if _resume_state is not None:
        for _f in os.listdir(args.output_dir):
            _m = _CKPT_STEP_RE.match(_f)
            if _m is not None:
                _saved_step_accs.setdefault(int(_m.group(1)), float("-inf"))
        if is_main and _saved_step_accs:
            print(f"[train] Resume: {len(_saved_step_accs)} step checkpoint(s) on disk "
                  f"count toward --checkpoint-keep-last: {sorted(_saved_step_accs)}")

    # K source: exactly one phase-2 target mechanism must be configured
    # (--phase2-loguniform, or one of the fixed length-target lists).
    if getattr(args, "phase2_loguniform", False) and (
            getattr(args, "phase2_len_list", None)
            or getattr(args, "phase2_bucket_lists", None)):
        raise ValueError("--phase2-loguniform is mutually exclusive with the "
                         "list-based K sources (phase2-len-list/"
                         "phase2-bucket-lists)")
    if not (getattr(args, "phase2_len_list", None)
            or getattr(args, "phase2_bucket_lists", None)
            or getattr(args, "phase2_loguniform", False)):
        raise ValueError(
            "No K source configured: set one of --phase2-loguniform / "
            "--phase2-len-list / --phase2-bucket-lists.")

    def _sample_K(step: int, n_art: int) -> int:
        # The real per-step K's come from the mixed-K block (M = len(list)
        # >= 2, or 2 for loguniform). This per-item value is a poison
        # placeholder so any path that would consume it directly (packed
        # sequences, a 1-element phase2 list) fails loudly instead of
        # silently training at a legacy K.
        return -1

    # Each rank uses a different seed so they sample non-overlapping batches.
    rng = random.Random(args.seed + rank)
    if _resume_state is not None and is_main:
        # Only rank 0's `rng` was saved (see _save_train_state docstring).
        # Other ranks keep their fresh args.seed + rank initialization.
        rng.setstate(_resume_state["rng_state"])
        print("[train] Restored rank-0 Python rng state for batch sampling")

    # ── Length-bucketed sampling ──────────────────────────────────────────────
    # When --length-buckets is set (recommended for any multi-dataset run), every
    # training step has all ranks deterministically choose the SAME length
    # bucket, then each rank draws a sample uniformly within that bucket.  This
    # eliminates the DDP straggler effect caused by cross-rank seq-length
    # variance) — while keeping samples
    # mixed across datasets within each bucket.
    #
    # Bucket choice is seeded by step number (not by rank), so every rank picks
    # the same bucket at the same step.  Within-bucket sample choice uses the
    # rank-local `rng`, so ranks still see different samples.
    _length_buckets: Optional[List[Tuple[int, int]]] = None
    _tokenized_train_by_bucket: List[List[dict]] = []
    _nonempty_buckets: List[int] = []
    _bucket_weights: List[int] = []
    if getattr(args, "length_buckets", None):
        _raw_bnds = sorted({int(b.strip()) for b in args.length_buckets.split(",") if b.strip()})
        if any(b <= 0 for b in _raw_bnds):
            raise ValueError(f"--length-buckets boundaries must be positive; got {_raw_bnds}")
        # Build half-open intervals [lo, hi).  Include implicit [0, first) and
        # [last, +inf) ranges so every sample lands in exactly one bucket.
        _bnds_full = [0] + _raw_bnds + [10**9]
        _length_buckets = list(zip(_bnds_full[:-1], _bnds_full[1:]))

        # Store INDICES, not sample dicts: with the packed mmap store, holding
        # sample dicts would materialize every row's tokens (host OOM).  Lengths
        # come from the meta LEN column when available (no token copy); the sampler
        # materializes one row on draw via tokenized_train[idx].
        from tokengate.datasets import pack as _pack
        _meta_attr = getattr(tokenized_train, "_meta", None)
        if _meta_attr is not None:
            _lengths = [int(v) for v in _meta_attr[:, _pack.LEN]]
        else:
            _lengths = [len(_s["full_ids"]) for _s in tokenized_train]
        _tokenized_train_by_bucket = [[] for _ in _length_buckets]
        for _idx, _n in enumerate(_lengths):
            for _i, (_lo, _hi) in enumerate(_length_buckets):
                if _lo <= _n < _hi:
                    _tokenized_train_by_bucket[_i].append(_idx)
                    break

        # Drop buckets with no samples; their weight contribution would be 0
        # anyway, and we use the index list to avoid iterating empties.
        _nonempty_buckets = [_i for _i, _b in enumerate(_tokenized_train_by_bucket) if _b]
        if not _nonempty_buckets:
            # Should never happen given the [0, +inf) span, but guard anyway.
            raise RuntimeError("--length-buckets produced 0 non-empty buckets; check boundaries")
        _bucket_weights = [len(_tokenized_train_by_bucket[_i]) for _i in _nonempty_buckets]

        if is_main:
            print("[train] Length-bucketed sampling enabled:")
            for _i in _nonempty_buckets:
                _lo, _hi = _length_buckets[_i]
                _n_samples = len(_tokenized_train_by_bucket[_i])
                _share = _n_samples / max(len(tokenized_train), 1)
                _hi_disp = f"{_hi:>7d}" if _hi < 10**8 else "    inf"
                print(f"[train]   [{_lo:>7d}, {_hi_disp})  n={_n_samples:>6d}  ({_share:5.1%})")
    def _bucket_for_step(s: int) -> int:
        """Pick a bucket for step `s`.  Deterministic across ranks."""
        # Seed deterministically from step alone so all ranks agree.  Using a
        # large stride (1e7) avoids correlation with any per-rank rng seeding.
        _r = random.Random(args.seed * 10_000_019 + s)
        return _r.choices(_nonempty_buckets, weights=_bucket_weights, k=1)[0]

    # ── Per-epoch shuffled iteration (--shuffle-epochs) ───────────────────────
    # Replaces the default i.i.d. with-replacement draw with iteration over a
    # shuffled permutation of the pool.  The permutation is seeded identically
    # on every rank (so they agree) and then sharded — rank r consumes
    # perm[r::world_size] — so the ranks' shards are disjoint and their union
    # covers the pool exactly once per epoch (true epoch semantics, no Poisson
    # over/under-sampling).  At the end of a shard the permutation is
    # regenerated with a fresh epoch seed (reshuffle).
    _shuffle_epochs = bool(getattr(args, "shuffle_epochs", False))
    if _shuffle_epochs and _length_buckets is not None:
        raise ValueError(
            "--shuffle-epochs is mutually exclusive with --length-buckets "
            "(each defines its own sampling distribution).  Pick one."
        )
    _epoch_state = {"epoch": -1, "order": [], "cursor": 0}

    # Packed path: precompute a per-row keep-probability vector once so the
    # per-epoch permutation is a single vectorized pass instead of one dict
    # materialization per row per epoch (millions of mmap slices otherwise).
    _packed_keep_prob = None
    _target_balanced = bool(getattr(args, "target_balanced_map", None))
    if _target_balanced:
        # Target-balanced weights are realised through the shuffle-epoch permutation
        # over the packed store; every other sampling path (length buckets,
        # legacy pickle loader) would silently ignore the map.
        if not _pretrain_packed:
            raise ValueError(
                "--target-balanced-map requires --pretrain-packed (weights are "
                "realised over the packed store's per-epoch permutation).")
        if not (_shuffle_epochs or _length_buckets is not None):
            raise ValueError(
                "--target-balanced-map requires --shuffle-epochs OR --length-buckets; "
                "the legacy sampler does not consume the map.")
        _ds_keep = _target_balanced_index_weights(
            args.target_balanced_map, _pk_ds_names, _pk_ds_tasks,
            np.bincount(tokenized_train.dataset_idx(),
                        minlength=len(_pk_ds_names)).astype(np.int64), is_main)
        _packed_keep_prob = _ds_keep[tokenized_train.dataset_idx()]

    # Target-balanced × length-buckets: sample the BUCKET ∝ its target-balanced mass, then a row
    # ∝ target-balanced weight WITHIN the bucket. P(row)=P(bucket)·P(row|bucket)=(m_b/M)·(w_i/m_b)=
    # w_i/M → exact target-balanced marginal, while keeping straggler-safe per-step length homogeneity
    # (all ranks still pick the same bucket at step s). Rows with 0 weight (excluded datasets) drop out.
    _tb_bucket_arr: Dict[int, "np.ndarray"] = {}
    _tb_bucket_p: Dict[int, "np.ndarray"] = {}
    _tb_rng = np.random.default_rng(int(args.seed) * 7919 + rank)
    if _target_balanced and _length_buckets is not None and _packed_keep_prob is not None:
        _new_bw = []
        for _bi in _nonempty_buckets:
            _rows = np.asarray(_tokenized_train_by_bucket[_bi], dtype=np.int64)
            _w = _packed_keep_prob[_rows].astype(np.float64)
            _tot = float(_w.sum())
            _tb_bucket_arr[_bi] = _rows
            _tb_bucket_p[_bi] = (_w / _tot) if _tot > 0 else None
            _new_bw.append(_tot)
        _bucket_weights = _new_bw   # bucket sampled ∝ target-balanced mass (was ∝ row count)
        if is_main:
            print("[train] TARGET-BALANCED × length-bucket mass: "
                  + ", ".join(f"b{_bi}={_bw:.0f}" for _bi, _bw in zip(_nonempty_buckets, _bucket_weights)),
                  flush=True)

    def _refill_epoch_order() -> None:
        _epoch_state["epoch"] += 1
        _seed = args.seed * 1_000_003 + _epoch_state["epoch"]
        if _pretrain_packed:
            # Vectorized epoch order (numpy).  Cross-rank deterministic via the
            # shared seed; produces a numpy index array that indexes the store.
            _rng_np = np.random.default_rng(_seed)
            _n = len(tokenized_train)
            if _packed_keep_prob is not None:
                # w<1: keep with prob w. w>1: floor(w) copies + one more with
                # prob frac(w).  (w<=1 reduces to the old keep-with-prob-w.)
                _floor = np.floor(_packed_keep_prob).astype(np.int64)
                _counts = _floor + (_rng_np.random(_n)
                                    < (_packed_keep_prob - _floor)).astype(np.int64)
                _perm = np.repeat(np.arange(_n), _counts)
            else:
                _perm = np.arange(_n)
            _rng_np.shuffle(_perm)
            _perm = _perm if _data_presharded else _perm[rank::world_size]
        else:
            _g = random.Random(_seed)
            _perm = list(range(len(tokenized_train)))
            _g.shuffle(_perm)
            # When the data was pre-sharded at load (DDP), tokenized_train is already
            # this rank's [rank::world_size] slice, so iterate all of it.  Otherwise
            # (full replication, e.g. single GPU) shard the index permutation here.
            _perm = _perm if _data_presharded else _perm[rank::world_size]
        _epoch_state["order"] = _perm
        _epoch_state["cursor"] = 0
        if is_main:
            print(f"[train] shuffle-epochs: epoch {_epoch_state['epoch']} — "
                  f"{len(_perm)} {'local-shard' if _data_presharded else 'global'} "
                  f"samples this epoch ({len(_epoch_state['order'])}/rank across {world_size} ranks)"
                  + (" [target-balanced]" if _target_balanced else ""))

    def _next_shuffled() -> dict:
        if _epoch_state["cursor"] >= len(_epoch_state["order"]):
            _refill_epoch_order()
        _idx = _epoch_state["order"][_epoch_state["cursor"]]
        _epoch_state["cursor"] += 1
        return tokenized_train[_idx]

    def _sample_train(s: int = 0) -> dict:
        """One training sample.
        - Length-bucketed mode (--length-buckets): all ranks pick the same
          bucket at step s, then draw uniformly within it.  Mixes across
          datasets within a bucket.  Highest-priority routing.
        - Shuffle-epochs mode (--shuffle-epochs): iterate a per-epoch shuffled,
          rank-sharded permutation without replacement.
        - Default: uniform over the concatenated pool (i.i.d. with replacement)."""
        if _length_buckets is not None:
            # buckets hold INDICES (packed-mmap safe); materialize one row on draw.
            _b = _bucket_for_step(s)
            if _tb_bucket_p.get(_b) is not None:
                # target-balanced: draw a row ∝ its target-balanced weight within the bucket
                return tokenized_train[int(_tb_rng.choice(_tb_bucket_arr[_b], p=_tb_bucket_p[_b]))]
            return tokenized_train[rng.choice(_tokenized_train_by_bucket[_b])]
        if _shuffle_epochs:
            return _next_shuffled()
        return rng.choice(tokenized_train)

    step = 0
    if _resume_state is not None:
        step = int(_resume_state["step"])
        if is_main:
            print(f"[train] Resumed at step={step}; will run through "
                  f"step={args.steps} ({args.steps - step} steps remaining)")
    # Resume-aware throughput accounting: t0 is this PROCESS's start, so pace
    # metrics must divide by steps run since _start_step, not the cumulative step
    # counter — otherwise a resumed run reports elapsed/step (e.g. 600s/4000) and
    # train/seconds_per_step reads absurdly low (<1s) for hours after resume.
    _start_step = step
    _gate_wk_norm:     float = None   # set after each optimizer.step(); None on non-update steps
    _gate_wq_norm:     float = None
    _gate_wp_norm:     float = None
    _gate_v_attn_norm: float = None   # attnpool: learned attention query vector
    _gate_w_out_norm:  float = None   # crossattn: output projection (zero-init)
    _gate_W_Qa_norm:   float = None   # crossattn: article query projection
    _gate_W_Kc_norm:   float = None   # crossattn: instruction key projection
    _gate_W_Vc_norm:   float = None   # crossattn: instruction value projection
    _saved_v_attn_grad_norm:        float = None
    _saved_w_out_grad_norm:         float = None
    _saved_gate_scale_grad_norm:    float = None
    _gate_W_noise_norm:             float = None
    _saved_W_noise_grad_norm:       float = None
    _saved_W_proj_grad_norm:        float = None  # linear/boundary: primary scoring weight
    _saved_W_k_grad_norm:           float = None  # attnpool/multiquery: article projection
    _saved_W_q_grad_norm:           float = None  # attnpool/multiquery: query projection
    # Adapter norms keyed by group name (lora: lora_A/lora_B; delta:
    # delta/delta_bias) → logged as train/{name}_norm / train/{name}_grad_norm.
    _adapter_norms: dict = {}
    _saved_adapter_grad_norms: dict = {}
    _saved_branch_grad_norms: dict = {}
    _last_batch_data: list = []  # updated each step; used for SWA train-loss spot-check
    model.eval()  # backbone in eval mode throughout (no dropout)
    gate.train()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=args.steps, desc=f"train[r{rank}]") if is_main else None
    if pbar is not None and step > 0:
        pbar.update(step)
    t0 = time.time()
    # Profiling: wall-clock of the previous completed UPDATE step.  dt between
    # consecutive successful updates = steady-state s/it (the [profile] line is
    # emitted after `step += 1` in the success path; eval/ckpt/SWA are no-ops
    # under --profile so they don't pollute the measurement).
    _prof_t_prev = None
    # ── "other"-bucket decomposition (opt-in; PROFILE_PHASES=1, multi-rank) ────
    # Isolates the non-compute slice of s/it — the part the phase-2 homogenization
    # could recover — into: (a) STRAGGLER WAIT (a timed dist.barrier right after
    # backward: on rank 0, how long it idles for the slowest rank before any
    # collective), (b) grad ALL-REDUCE (pure NCCL comm, now that the barrier
    # aligned the ranks), (c) OPTIMIZER step (Muon Newton-Schulz on the 109M LoRA
    # + AdamW).  The residual (dt − compute − these three) is data/clip/python.
    # The barrier mostly RELABELS wait already absorbed by the first collective,
    # so perturbation is small; a few extra syncs/step make this a measurement
    # mode — take the clean throughput headline from a PROFILE_PHASES-only run.
    _prof_other = _gate_forward_mod.PROFILE_PHASES and world_size > 1
    _straggler_ms = _allreduce_ms = _optim_ms = float("nan")
    # Per-step rank-0 stats+wandb logging costs ~1s/step (measured; NOT network —
    # same online/offline).  TRAIN_LOG_FREQ throttles the whole diagnostic block
    # (heavy .item() stat syncs, ~20 param .cpu() reads, phase-2 stat collection,
    # and the wandb.log of ~50 metrics) to every-N steps; the cheap essentials
    # (loss/K/pbar) still update every step.  Default 1 = unchanged behaviour.
    _train_log_freq = max(1, int(os.environ.get("TRAIN_LOG_FREQ", "1")))
    # The per-sample gate-output stats (gate_all_mean/std, gate_topk_mean/std) are
    # the .item()-heavy inner loop; collect them far less often than the rest.
    # Only fires on steps that are ALSO log steps (multiple of TRAIN_LOG_FREQ).
    _per_sample_stats_freq = max(1, int(os.environ.get("PER_SAMPLE_STATS_FREQ", "100")))
    if is_main and _train_log_freq > 1:
        print(f"[train] TRAIN_LOG_FREQ={_train_log_freq}: full diagnostics + wandb "
              f"every {_train_log_freq} steps; per-sample gate stats every "
              f"{_per_sample_stats_freq} (essentials every step).", flush=True)

    # Batching configuration
    batch_size = getattr(args, 'batch_size', 1)
    # Engage the batched code path when EITHER we batch >1 sample per step OR we
    # run multi-K amortization (M > 1).  The multi-K branch lives inside the
    # batched block and reuses ONE Phase-1 prefill across M K values; at
    # batch_size=1 this gives M (sample,K) gradient contributions per expensive
    # prefill with no extra long-article memory — the memory-efficient way to
    # get dense mixed-K signal (esp. small-K) every step.  M is derived from
    # the K source.
    if getattr(args, "phase2_len_list", None):
        _mixed_k_m_cfg = len(args.phase2_len_list)
    elif getattr(args, "phase2_bucket_lists", None):
        _mixed_k_m_cfg = max(len(_l) for _l in args.phase2_bucket_lists)
    else:
        # phase2_loguniform (validated at startup): sampled L + full sentinel.
        _mixed_k_m_cfg = 2
    use_batching = batch_size > 1 or _mixed_k_m_cfg > 1
    if not use_batching:
        # The legacy single-sample training path was REMOVED (cleanup after the a_j
        # split landed): it predated the scoring branch and would silently score from
        # h^fwd. Every supported config trains through the batched / multi-K paths.
        raise NotImplementedError(
            "single-sample training path removed: use --batch-size > 1 or a multi-K "
            "config (phase2_loguniform / a multi-element phase2 list).")

    # ── No-compression baseline eval (per metric group, same pools as eval) ──
    # Runs once before training over the SAME pooled per-metric-group eval sets
    # the gated per-step eval uses (eval_samples_by_group), so the no-compress
    # reference is directly comparable to the gated `eval/cat_<group>` curves
    # (identical samples, identical per-sample task resolution / native metrics).
    # All ranks participate inside each per-group call so dist.all_reduce
    # synchronises.  We also bucket a per-dataset breakdown from the SAME
    # generations (return_per_ds), so `eval/<ds>/..._no_compress` overlays the
    # gated per-dataset breakdown on matching samples.  Stored values are
    # re-emitted at every eval step as flat W&B reference lines.
    _no_compress_per_group: Dict[str, Dict[str, float]] = {}     # {group: {acc, f1, loss}}
    _no_compress_per_ds: Dict[str, Dict[str, float]] = {}        # {ds: {acc, f1}}

    # Resume path: skip the (~90-min) baseline eval and restore values from the
    # checkpoint so reference lines continue to overlay the gated eval curves
    # without re-running model inference on rank 0.
    if _resume_state is not None:
        _no_compress_per_group = _resume_state.get("no_compress_per_group", {}) or {}
        _no_compress_per_ds    = _resume_state.get("no_compress_per_ds", {}) or {}
        if is_main:
            print(f"[train] Resume: skipping the step-0 no-compress baseline "
                  f"eval; restored {len(_no_compress_per_group)} no-compress "
                  f"groups from checkpoint.")

    if eval_samples and _resume_state is None and not args.profile:
        for _grp, _samps in eval_samples_by_group.items():
            if not _samps:
                continue
            torch.cuda.empty_cache()
            if is_main:
                print(f"\n[no-compress eval] group={_grp}  ({len(_samps)} samples, "
                      f"same pool as gated eval)")
            # Metric tags are homogeneous within a group; read from any member.
            _grp_task = tasks_by_name[_samps[0]["task"]]
            _nc_acc, _nc_f1, _nc_loss, _nc_predictions, _nc_per_ds = evaluate_no_compression(
                model, tokenizer, _samps, _grp_task,
                max_gen_tokens=args.max_gen_tokens,
                device=str(device),
                log_examples=args.log_examples,
                world_size=world_size,
                rank=rank,
                wandb_run=_wandb_run,
                output_dir=args.output_dir,
                label=f"no-compress eval [{_grp}]",
                tasks_by_name=tasks_by_name,
                return_per_ds=True,
            )
            if world_size > 1:
                dist.barrier()
            if is_main:
                _no_compress_per_group[_grp] = {
                    "accuracy": _nc_acc, "f1": _nc_f1, "loss": _nc_loss,
                    "acc_metric_tag": _grp_task.acc_metric_name,
                    "f1_metric_tag":  _grp_task.secondary_metric_name,
                }
                _no_compress_per_ds.update(_nc_per_ds)


    # ── Patch model with FlexAttention ───────────────────────────────────────
    # Done AFTER the no-compression baseline evals so they run on the model's
    # native SDPA attention — simpler, deterministic, no FlexAttention overhead.
    print("[train] Patching model with FlexAttention...")
    model = patch_model_with_flex_attention(model)
    print("[train] Model patched successfully")
    warmup_flex_attention(device)
    if args.attn_backends != "flex":
        set_attention_backends_for_layers(model, args.attn_backends)
        if is_main:
            print(f"[train] attention kernels: {args.attn_backends} -> causal/bias = {GATE_ATTN[args.attn_backends]}")

    # ── Optionally compile phase-2 transformer layers ────────────────────────
    # `--compile-phase2` wraps each transformer layer ABOVE the gate layer with
    # torch.compile.  FlexAttention itself is already compiled (dynamic=True,
    # fullgraph=False) inside each attention forward, so this nests another
    # compile boundary around the layer (q/k/v_proj → attn → o_proj → MLP).
    # The expected win is 10-30 % on the phase-2 forward (the MLP + norm +
    # residual + projection portion that's NOT inside FlexAttention).
    #
    # Risks:
    #   - First step after enabling will see a one-shot 10-60 s compile burst
    #     per unique seq-len bucket (dynamic=True should bound this).
    #   - With FlexAttention nested, recompiles can stack — we already bumped
    #     `_dynamo.config.recompile_limit = 64`.  May need to bump higher.
    #   - Per-layer compile interacts with LoRA: if LoRA is applied AFTER
    #     compile, the compiled graph captures the pre-LoRA module.  We thus
    #     compile AFTER LoRA injection (which happens in patch_model_for_lora
    #     called later) — see below for the correct ordering.
    # Default off until 8-GPU integration testing validates correctness.
    if args.pack_phase2_max_tokens > 0:
        if args.attn_backends != "fa2":
            raise ValueError(f"--pack-phase2-max-tokens needs --attn-backends fa2, got {args.attn_backends!r}")
        if args.batch_size != 1:
            raise ValueError(f"--pack-phase2-max-tokens needs batch size 1, got {args.batch_size}")
        from kernels.fa2bias.keybias import require_varlen
        require_varlen()
        _gate_forward_mod.PACK_PHASE2 = True
        _gate_forward_mod.PACK_PHASE2_MAX_TOKENS = args.pack_phase2_max_tokens
    if is_main:
        print(f"[train] packed Phase 2: {'<= ' + str(args.pack_phase2_max_tokens) + ' tokens' if args.pack_phase2_max_tokens > 0 else 'off'}")
    _compile_phase2 = bool(getattr(args, "compile_phase2", False))
    if _compile_phase2:
        if is_main:
            print(f"[train] --compile-phase2 enabled: will compile transformer "
                  f"layers [{args.gate_layer + 1}..{len(model.model.layers)-1}] "
                  f"AFTER LoRA injection.  Expect a one-time compile burst on "
                  f"the first eval / training step at each unique seq-len bucket.")
        # Bump recompile limit for safety: with N length buckets × FlexAttention
        # already using up to ~8 recompile slots, the default 64 can be tight.
        try:
            torch._dynamo.config.recompile_limit = max(
                getattr(torch._dynamo.config, "recompile_limit", 8), 256
            )
        except Exception:
            pass
        # nn.Module.compile: in place, no OptimizedModule wrapper, so parameter names -- and
        # with them every saved adapter key -- stay as uncompiled.  `dynamic=True` avoids
        # per-seq-len recompiles; `fullgraph=False` allows graph breaks inside the layer.
        _phase2_start = args.gate_layer + 1
        _phase2_end = len(model.model.layers)
        for _li in range(_phase2_start, _phase2_end):
            model.model.layers[_li].compile(dynamic=True, fullgraph=False)
        if is_main:
            print(f"[train]   compiled layers {_phase2_start}..{_phase2_end-1} "
                  f"(N={_phase2_end - _phase2_start}); first step will compile.")
    def _add_baselines(d: dict) -> dict:
        """Inject stored baseline metrics as flat W&B lines.

        The no-compress baseline is computed on the SAME pooled
        per-metric-group eval sets as the gated eval, so they are emitted both
        per-group (overlaying `eval/cat_<group>/...`) and per-dataset (overlaying
        the gated `eval/<ds>/...` breakdown).  Also emits legacy flat keys (from
        the primary/first group) for single-dataset W&B history continuity."""
        # Per-group no-compress lines (overlay the headline eval/cat_<group>).
        for _grp, _b in _no_compress_per_group.items():
            d[f"eval/cat_{_grp}/accuracy_{_b['acc_metric_tag']}_no_compress"] = _b["accuracy"]
            if _b["f1"] is not None and _b["f1_metric_tag"] != _b["acc_metric_tag"]:
                d[f"eval/cat_{_grp}/accuracy_{_b['f1_metric_tag']}_no_compress"] = _b["f1"]
            d[f"eval/cat_{_grp}/loss_no_compress"] = _b["loss"]
        # Per-dataset no-compress lines (breakdown; no per-ds loss).
        for _ds, _b in _no_compress_per_ds.items():
            d[f"eval/{_ds}/accuracy_{_b['acc_metric_tag']}_no_compress"] = _b["accuracy"]
            if _b["f1"] is not None and _b["f1_metric_tag"] != _b["acc_metric_tag"]:
                d[f"eval/{_ds}/accuracy_{_b['f1_metric_tag']}_no_compress"] = _b["f1"]
        # Legacy flat keys (primary/first group) for continuity.  Sourced from
        # the per-group dicts (which carry loss; the per-ds breakdown does not).
        if _no_compress_per_group:
            _primary = next(iter(_no_compress_per_group))
            _b = _no_compress_per_group[_primary]
            d[f"eval/accuracy_{_acc_metric_tag}_no_compress"] = _b["accuracy"]
            if _b["f1"] is not None and _judge_active:
                d["eval/accuracy_F1_no_compress"] = _b["f1"]
            d["eval/loss_no_compress"] = _b["loss"]
        return d

    # Release memory cached by the startup baselines (no-compress + FastKV at every
    # tsp layer) and the flex warmup before the FIRST eval, mirroring the existing
    # post-step-0 cleanup below.
    import gc as _gc_pre
    _gc_pre.collect()
    torch.cuda.empty_cache()

    # E2E_MEM_CENSUS=1: dump what is still LIVE on the GPU at this point (debugging
    # the v16a step-0 OOM: ~73 GiB allocated before the first eval forward).
    if os.environ.get("E2E_MEM_CENSUS") == "1":
        from collections import Counter as _Ctr
        _census = _Ctr()
        for _o in _gc_pre.get_objects():
            try:
                if torch.is_tensor(_o) and _o.is_cuda:
                    _census[(tuple(_o.shape), str(_o.dtype))] += _o.numel() * _o.element_size()
            except Exception:
                pass
        _tot = sum(_census.values())
        print(f"[mem-census] rank={rank} allocated={torch.cuda.memory_allocated()/2**30:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GB "
              f"python-visible={_tot/2**30:.2f}GB", flush=True)
        if is_main:
            for (_shape, _dt), _b in _census.most_common(30):
                print(f"  [mem-census] {_b/2**30:8.3f}GB  {_dt:15s} {_shape}", flush=True)
            print(torch.cuda.memory_summary(abbreviated=True), flush=True)
        if os.environ.get("E2E_CENSUS_EXIT") == "1":
            raise SystemExit("[mem-census] E2E_CENSUS_EXIT=1 — stopping after census")


    # CPU fp32 snapshot of the scoring branch at its (possibly resume-loaded)
    # starting weights, for drift-from-init telemetry (Muon moves the
    # branch at ~constant rate regardless of gradient size, so weight motion
    # alone cannot distinguish learning from churn). Rank 0 only, ~800MB CPU.
    _branch_init_cpu = None
    _drift_log_ctr = [0]
    if is_main and getattr(gate, "gate_branch", None) is not None:
        _branch_init_cpu = {
            "gate_proj": gate.gate_branch.gate_proj.weight.detach().float().cpu().clone(),
            "up_proj":   gate.gate_branch.up_proj.weight.detach().float().cpu().clone(),
            "w":         gate.gate_branch.down_proj.weight.detach().float().cpu().clone(),
        }

    # ── Pre-step-0 eval (records initialisation baseline) ────────────────────
    # All ranks participate so evaluate() can distribute inference across GPUs.
    # Skipped on resume: the step-0 init-eval point was already logged to W&B
    # in the original run; re-logging at step=0 from a resumed checkpoint would
    # either silently no-op (wandb dedupes step 0) or overwrite with a now-
    # trained-model curve, which is misleading.
    # --eval-on-resume runs it after a resume as well, so a restarted run reports
    # where it stands before spending a cycle training. It is logged at the RESUMED
    # step, not step 0: W&B dedupes a re-logged step 0, and overwriting the original
    # init point with a now-trained curve would be worse than not logging at all.
    _init_eval_step = 0 if _resume_state is None else int(_start_step)
    if (eval_samples and not args.profile
            and (_resume_state is None or args.eval_on_resume)):
        if world_size > 1:
            dist.barrier()
        torch.cuda.empty_cache()
        gate.eval()
        if is_main:
            eval_steps.append(_init_eval_step)
            _wandb_log = {}
            _gate_output_eval_buf.clear()
            _gate_eval_capture[0] = True
        for _ki, _k in enumerate(k_eval_list):
            # Threshold-mode sweep: each k_eval_list entry is a tau value; bind
            # it onto the underlying gate so the selection helper picks it up.
            # The K passed to _evaluate_per_group below is ignored by the gate
            # (threshold mode reads gate.tau exclusively).
            if args.gate_selection == "threshold":
                (gate.module if hasattr(gate, "module") else gate).tau = float(_k)
            _per_group, _resp, _per_ds = _evaluate_per_group(
                model, gate, tokenizer,
                eval_samples_by_group, tasks_by_name, ds_task,
                _k, args,
                rank=rank, world_size=world_size, device=str(device),
                log_examples=args.log_examples if _ki == 0 else 0,
            )
            if is_main:
                # Primary (first) group feeds the legacy curves.
                _primary = next(iter(_per_group))
                _pm = _per_group[_primary]
                eval_per_k_accs[_k].append(_pm["accuracy"])
                eval_per_k_losses[_k].append(_pm["loss"])
                if _k == k_primary:
                    _note_best_group_acc(_per_group)
                _summary = ", ".join(
                    f"{_g}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                    f" loss={m['loss']:.4f}"
                    f" tok={m['n_tokens_pruned']:.0f}/{m['n_tokens_full']:.0f}"
                    f" art={m['n_tokens_pruned_art']:.0f}/{m['n_tokens_full_art']:.0f}"
                    f" (n={m['n_samples']})"
                    for _g, m in _per_group.items()
                )
                _ds_summary = ", ".join(
                    f"{_d}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                    for _d, m in _per_ds.items()
                )
                _label = ("tau" if args.gate_selection == "threshold" else "K")
                print(f"\n  [eval/cat] step={_init_eval_step}  {_label}={_k}  {_summary}  (init)")
                print(f"  [eval/ds]  step={_init_eval_step}  {_label}={_k}  {_ds_summary}")
                _emit_eval_wandb_keys(_wandb_log, _per_group, _per_ds, _k,
                                      _acc_metric_tag, _judge_active)
                if _ki == 0:
                    _log_responses(_resp, 0, args.output_dir, _wandb_run)
            # ── random-K baseline at the INIT point too, so the eval/randomk_* curve
            #    starts at step 0 alongside the gated curve (same construction as the
            #    periodic block: random selection, no bias, no attn_scale, all-rank DDP). ──
            for _rk_gatemode in (True,):  # random-K on the gate model (always on)
                _gi = gate.module if hasattr(gate, "module") else gate
                _prev_rt = getattr(_gi, "random_topk", False)
                _prev_nb = getattr(_gi, "_baseline_no_bias_scale", False)
                _gi.random_topk = True
                if not _rk_gatemode:
                    _gi._baseline_no_bias_scale = True
                try:
                    _rk_pg, _, _rk_pds = _evaluate_per_group(
                        model, gate, tokenizer, eval_samples_by_group, tasks_by_name,
                        ds_task, _k, args, rank=rank, world_size=world_size,
                        device=str(device), log_examples=0)
                finally:
                    _gi.random_topk = _prev_rt
                    _gi._baseline_no_bias_scale = _prev_nb
                if is_main:
                    _rk_src = "randomkgate_" if _rk_gatemode else "randomk_"
                    _emit_eval_wandb_keys(_wandb_log, _rk_pg, _rk_pds, _k,
                                          _acc_metric_tag, _judge_active, source=_rk_src)
                    _rk_sum = ", ".join(f"{_g}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                                        for _g, m in _rk_pg.items())
                    print(f"  [eval/{_rk_src.rstrip('_')}] step={_init_eval_step}  K={_k}  {_rk_sum}", flush=True)
            # ── gate @ FULL context at the INIT point too (cycle 0), so the
            #    eval/gatefull_* curve starts alongside the gated curve. ──
            if args.gate_full_eval_freq > 0 and _ki == 0:
                _gf_pg, _, _gf_pds = _evaluate_per_group(
                    model, gate, tokenizer, eval_samples_by_group, tasks_by_name,
                    ds_task, 10_000_000, args, rank=rank, world_size=world_size,
                    device=str(device), log_examples=0)
                if is_main:
                    _emit_eval_wandb_keys(_wandb_log, _gf_pg, _gf_pds, "full",
                                          _acc_metric_tag, _judge_active, source="gatefull_")
                    _gf_sum = ", ".join(f"{_g}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                                        for _g, m in _gf_pg.items())
                    print(f"  [eval/gatefull] step={_init_eval_step}  K=full  {_gf_sum}", flush=True)
        # Restore the training τ after the eval-sweep loop.  The loop mutates
        # gate.tau to each sweep entry in turn and leaves it at the last
        # value (e.g. 0.9 for "0.3 0.5 0.7 0.9").  gate_forward._select_indices
        # reads gate.tau at every training step, so without this reset the
        # gate trains at the eval sweep's terminal τ rather than args.tau.
        if args.gate_selection == "threshold":
            (gate.module if hasattr(gate, "module") else gate).tau = float(args.tau)
        if is_main:
            _gate_eval_capture[0] = False
            if _gate_output_eval_buf:
                _agg_w_eval = (
                    (gate.module if hasattr(gate, "module") else gate).agg_weights.detach().float().cpu()
                    if hasattr(gate.module if hasattr(gate, "module") else gate, "agg_weights")
                    and (gate.module if hasattr(gate, "module") else gate).agg_weights is not None
                    else None
                )
                _pct_list = []
                for _t in _gate_output_eval_buf:
                    _ts = _t.mean(dim=(-2, -1)) if _t.dim() == 3 else ((_t * _agg_w_eval).sum(-1) if (_t.dim() == 2 and _agg_w_eval is not None) else _t)
                    _pct_list.append(
                        (torch.sigmoid(_ts) >= float(args.tau)).float().mean().item()
                    )
                _wandb_log["eval/gate_pct_above_tau"] = sum(_pct_list) / len(_pct_list)
                # b_j distribution metrics at step=0 (matryoshka init snapshot).
                _log_bj_distribution(
                    _wandb_log,
                    _gate_output_eval_buf,
                    n_K_eval=max(1, len(k_eval_list)),
                    k_eval_values=list(k_eval_list),
                    n_tracked=4,
                )
            _gate_output_eval_buf.clear()
            if _wandb_run is not None:
                _wandb_run.log(_add_baselines(_wandb_log), step=_init_eval_step)
        gate.train()
        if world_size > 1:
            dist.barrier()

    _step_loss_letter: float = float("nan")
    _step_loss_eos: float = float("nan")
    _oom_skips = 0  # cumulative count of training steps skipped due to CUDA OOM
    _phase2_loss_acc: Dict[int, list] = {}  # phase-2-length → [loss_sum, n] over the log window
    _nonfinite_grad_skips = 0  # cumulative steps skipped due to non-finite (NaN/inf) gradient
    # Release memory cached/fragmented by the step-0 baseline+gated eval before
    # training starts.  The eval inflated reserved to ~55GB while only ~30GB was
    # allocated (KV/activations from full-context no-compress generation cached
    # by the allocator), leaving training with too little headroom → ~70%
    # OOM-skips in run4/5.  Returning the cache here restores full headroom.
    import gc as _gc0
    _gc0.collect()
    torch.cuda.empty_cache()
    if is_main:
        print(f"[mem] entering training loop (post baseline-eval + flex warmup + empty_cache): "
              f"allocated={torch.cuda.memory_allocated()/1e9:.1f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.1f}GB", flush=True)

    def _do_checkpoint_save(_step):
        """Save gate / LoRA / FT-below weights + train_state at _step.

        Called from BOTH the success path (after optimizer step) AND the
        all-ranks-skip path (when any rank OOMs and ranks agree to skip).
        Without the second call site, every save boundary has a ~40% chance
        of being silently skipped (with ~6% per-rank OOM × 8 ranks, the prob
        that no rank OOMs in any given iteration is ~0.61), which produces
        runs that complete without a single intermediate step checkpoint.
        """
        if args.profile:          # throughput-only run: no checkpoint I/O
            return
        if not is_main:
            return
        _ckpt_meta = {
            "step": _step,
            "gate_layer": args.gate_layer,
            "phase2_len_list": args.phase2_len_list,
            "phase2_bucket_lists": args.phase2_bucket_lists,
            "length_buckets": getattr(args, "length_buckets", None),
            "base_model": args.base_model,
            "task": args.task,
            "dataset": args.dataset,
            "adapter_type": args.adapter_type if _lora_enabled else None,
            "lora_layer_ids": lora_layer_ids if _lora_enabled else None,
            "lora_rank": args.lora_rank if (_lora_enabled and args.adapter_type == "lora") else None,
            "lora_alpha": args.lora_alpha if (_lora_enabled and args.adapter_type == "lora") else None,
            "lora_targets": args.lora_targets if (_lora_enabled and args.adapter_type == "lora") else None,
            "lora_start_layer": lora_start_layer if _lora_enabled else None,
        }
        ckpt_path = os.path.join(args.output_dir, f"gate_step{_step}.pt")
        save_gate(gate, ckpt_path, metadata=_ckpt_meta)
        _extra_saved = []
        if _lora_enabled:
            lora_ckpt_path = os.path.join(args.output_dir, f"lora_step{_step}.pt")
            torch.save(collect_lora_state_dict(model), lora_ckpt_path)
            _extra_saved.append(lora_ckpt_path)
        # Optional: also save a DEPLOYABLE SWA checkpoint (averaged gate + adapter)
        # at every step-save once SWA has started, so any eval step can be
        # benchmarked directly without extracting swa_* from train_state. Follows
        # the same keep_last/keep_best retention as the raw step checkpoints.
        if (getattr(args, "save_swa_step_checkpoints", False)
                and swa_gate is not None and swa_n > 0):
            swa_gate_path = os.path.join(args.output_dir, f"swa_gate_step{_step}.pt")
            save_gate(swa_gate, swa_gate_path, metadata={**_ckpt_meta, "swa": True, "swa_n": swa_n})
            _extra_saved.append(swa_gate_path)
            if _lora_enabled and swa_adapter_sd is not None:
                swa_lora_path = os.path.join(args.output_dir, f"swa_lora_step{_step}.pt")
                torch.save({k: v.detach().cpu() for k, v in swa_adapter_sd.items()},
                           swa_lora_path)
                _extra_saved.append(swa_lora_path)
        train_state_path = os.path.join(args.output_dir, f"train_state_step{_step}.pt")
        _save_train_state(
            train_state_path,
            step=_step, optimizer=optimizer, scheduler=scheduler,
            swa_gate=swa_gate, swa_n=swa_n, rng=rng,
            wandb_run_id=(_wandb_run.id if _wandb_run is not None else None),
            args=args,
            swa_adapter_sd=swa_adapter_sd,
            no_compress_per_group=_no_compress_per_group,
            no_compress_per_ds=_no_compress_per_ds,
        )
        _extra_saved.append(train_state_path)
        print(f"  [ckpt] saved {ckpt_path}"
              + (" + " + " + ".join(_extra_saved) if _extra_saved else ""))
        # Ranked by the best-group accuracy seen at or before this save (-inf
        # before any eval: prunable by last-N, never a best-N winner).
        _saved_step_accs[_step] = _best_group_latest["acc"]
        _n_pruned, _kept = _prune_step_checkpoints(
            args.output_dir, _saved_step_accs,
            keep_last=args.checkpoint_keep_last,
            keep_best=args.checkpoint_keep_best,
            keep_all_after_step=(args.warmup_steps
                                 if getattr(args, "checkpoint_keep_all_after_warmup", False)
                                 else None),
        )
        if _n_pruned > 0:
            print(f"  [ckpt] pruned {_n_pruned} old step files (kept steps: {_kept})")

    def _do_eval():
        """Run an in-training eval cycle at the current `step` value.

        Defined as a closure so the same body fires from BOTH the success path
        and the all-ranks-skip path. Without the second call site, ~40% of
        eval boundaries land on all-skip iterations and silently drop the eval
        (same root cause as the checkpoint-skip bug above).

        Reads `step`, `gate`, `model`, `swa_gate`, etc. via late binding from
        the enclosing scope; the only rebound int (_swa_eval_cycle) is declared
        nonlocal so its `+= 1` increment hits the outer scope.
        """
        nonlocal _swa_eval_cycle
        # Throughput-only run: skip eval entirely.  Guarded identically on every
        # rank (the call sites are deterministic `step % eval_freq == 0`), so all
        # ranks return together — no barrier desync.
        if args.profile:
            return
        if world_size > 1:
            dist.barrier()
        if eval_samples:
            torch.cuda.empty_cache()
            # Per-rank memory snapshot: both step-10800 crashes were marginal
            # allocation failures during eval, invisible until they killed the
            # job — this line names the rank(s) entering eval near the ceiling.
            print(f"[eval-mem] rank={rank} step={step} entering eval: "
                  f"allocated={torch.cuda.memory_allocated()/2**30:.1f}GB "
                  f"reserved={torch.cuda.memory_reserved()/2**30:.1f}GB", flush=True)
            gate.eval()
            if is_main:
                eval_steps.append(step)
                _wandb_log = {"eval/epoch": step * (args.batch_size * world_size) / _n_train_global}
                _recent_train = sum(train_losses[-100:]) / max(len(train_losses[-100:]), 1)

            _eval_cycle_idx = step // args.eval_freq
            _all_eval_groups = [g for g in eval_samples_by_group
                                if eval_samples_by_group[g]]
            if getattr(args, "eval_alternate_datasets", False) and len(_all_eval_groups) > 1:
                _active_groups = [_all_eval_groups[_eval_cycle_idx % len(_all_eval_groups)]]
            else:
                _active_groups = _all_eval_groups
            _active_eval_by_group = {
                g: eval_samples_by_group[g] for g in _active_groups
            }
            _k_stride = max(1, int(getattr(args, "eval_k_stride", 1) or 1))
            if _k_stride > 1 and _eval_cycle_idx % _k_stride != 0:
                _active_k_list = [k_primary]
            else:
                _active_k_list = list(k_eval_list)

            # Enable rank-0 capture of the raw gate output during eval so we
            # can compute pct_above_tau (fraction of article tokens whose
            # sigmoid(b) >= training tau, averaged over articles).  The hook
            # writes to _gate_output_eval_buf; we drain it after the eval-tau
            # loop.  SWA eval is intentionally NOT captured (different weights).
            if is_main:
                _gate_output_eval_buf.clear()
                _gate_eval_capture[0] = True

            for _ki, _k in enumerate(_active_k_list):
                if args.gate_selection == "threshold":
                    (gate.module if hasattr(gate, "module") else gate).tau = float(_k)
                _per_group, _resp, _per_ds = _evaluate_per_group(
                    model, gate, tokenizer,
                    _active_eval_by_group, tasks_by_name, ds_task,
                    _k, args,
                    rank=rank, world_size=world_size, device=str(device),
                    log_examples=args.log_examples if _ki == 0 else 0,
                )
                if is_main:
                    _primary = next(iter(_per_group))
                    _pm = _per_group[_primary]
                    eval_per_k_accs[_k].append(_pm["accuracy"])
                    eval_per_k_losses[_k].append(_pm["loss"])
                    if _k == k_primary:
                        _note_best_group_acc(_per_group)
                    _summary = ", ".join(
                        f"{_g}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                        f" loss={m['loss']:.4f}"
                        f" tok={m['n_tokens_pruned']:.0f}/{m['n_tokens_full']:.0f}"
                        f" art={m['n_tokens_pruned_art']:.0f}/{m['n_tokens_full_art']:.0f}"
                        f" (n={m['n_samples']})"
                        for _g, m in _per_group.items()
                    )
                    _ds_summary = ", ".join(
                        f"{_d}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                        for _d, m in _per_ds.items()
                    )
                    _label = ("tau" if args.gate_selection == "threshold" else "K")
                    print(f"\n  [eval/cat] step={step}  {_label}={_k}  {_summary}  "
                          f"train_loss={_recent_train:.4f}")
                    print(f"  [eval/ds]  step={step}  {_label}={_k}  {_ds_summary}")
                    _emit_eval_wandb_keys(_wandb_log, _per_group, _per_ds, _k,
                                          _acc_metric_tag, _judge_active)
                    if _ki == 0:
                        _log_responses(_resp, step, args.output_dir, _wandb_run)
                # ── random-K checkpoint baseline (benchmark `randomk` analogue): SAME
                #    adapter+gate model, random token selection instead of gate top-K
                #    (gate.random_topk). Collective eval on all ranks. ──────────────
                for _rk_gatemode in (True,):  # random-K on the gate model (always on)
                    # _rk_gatemode False → benchmark-matched base randomk (no bias, no
                    # scale); True → random selection on the GATE model (temperature
                    # and live bias kept) under eval/randomkgate_*.
                    _gi = gate.module if hasattr(gate, "module") else gate
                    _prev_rt = getattr(_gi, "random_topk", False)
                    _gi.random_topk = True
                    _prev_nb = getattr(_gi, "_baseline_no_bias_scale", False)
                    if not _rk_gatemode:
                        _gi._baseline_no_bias_scale = True
                    try:
                        _rk_pg, _, _rk_pds = _evaluate_per_group(
                            model, gate, tokenizer, _active_eval_by_group, tasks_by_name,
                            ds_task, _k, args, rank=rank, world_size=world_size,
                            device=str(device), log_examples=0)
                    finally:
                        _gi.random_topk = _prev_rt
                        _gi._baseline_no_bias_scale = _prev_nb
                    if is_main:
                        _rk_src = "randomkgate_" if _rk_gatemode else "randomk_"
                        _emit_eval_wandb_keys(_wandb_log, _rk_pg, _rk_pds, _k,
                                              _acc_metric_tag, _judge_active, source=_rk_src)
                        _rk_sum = ", ".join(f"{_g}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                                            for _g, m in _rk_pg.items())
                        print(f"  [eval/{_rk_src.rstrip('_')}] step={step}  K={_k}  {_rk_sum}", flush=True)
                # ── gate @ FULL context (benchmark `gate_full` analogue): same model,
                #    K=10M keeps every article token, learned bias APPLIED. Every
                #    --gate-full-eval-freq'th eval cycle; K-independent → _ki==0 only. ──
                if (args.gate_full_eval_freq > 0 and _ki == 0
                        and (step // max(1, args.eval_freq)) % args.gate_full_eval_freq == 0):
                    _gf_pg, _, _gf_pds = _evaluate_per_group(
                        model, gate, tokenizer, _active_eval_by_group, tasks_by_name,
                        ds_task, 10_000_000, args, rank=rank, world_size=world_size,
                        device=str(device), log_examples=0)
                    if is_main:
                        _emit_eval_wandb_keys(_wandb_log, _gf_pg, _gf_pds, "full",
                                              _acc_metric_tag, _judge_active, source="gatefull_")
                        _gf_sum = ", ".join(f"{_g}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                                            for _g, m in _gf_pg.items())
                        print(f"  [eval/gatefull] step={step}  K=full  {_gf_sum}", flush=True)
            if args.gate_selection == "threshold":
                (gate.module if hasattr(gate, "module") else gate).tau = float(args.tau)

            # Drain eval-time gate-output capture → pct_above_tau metric +
            # b_j distribution tracking.
            # Identical b-tensors are captured _len(_active_k_list)_ times
            # (gate weights don't change across the tau-iter), so the mean is
            # the same as one pass; harmless redundancy.
            if is_main:
                _gate_eval_capture[0] = False
                if _gate_output_eval_buf:
                    _agg_w_eval = (
                        (gate.module if hasattr(gate, "module") else gate).agg_weights.detach().float().cpu()
                        if hasattr(gate.module if hasattr(gate, "module") else gate, "agg_weights")
                        and (gate.module if hasattr(gate, "module") else gate).agg_weights is not None
                        else None
                    )
                    _pct_list = []
                    for _t in _gate_output_eval_buf:
                        _ts = _t.mean(dim=(-2, -1)) if _t.dim() == 3 else ((_t * _agg_w_eval).sum(-1) if (_t.dim() == 2 and _agg_w_eval is not None) else _t)
                        _pct_list.append(
                            (torch.sigmoid(_ts) >= float(args.tau)).float().mean().item()
                        )
                    _wandb_log["eval/gate_pct_above_tau"] = sum(_pct_list) / len(_pct_list)
                    # b_j distribution metrics — fixed-subset histograms,
                    # bimodality coefficient (Sarle's), threshold-at-K markers.
                    # Skipped silently when wandb isn't running (helper guards).
                    _log_bj_distribution(
                        _wandb_log,
                        _gate_output_eval_buf,
                        n_K_eval=max(1, len(_active_k_list)),
                        k_eval_values=list(k_eval_list),
                        n_tracked=4,
                    )
                _gate_output_eval_buf.clear()

            # DDP SWA eval: when --swa-eval-ddp (default) and multi-rank, EVERY rank runs the
            # SWA eval on its [rank::world_size] shard (results gather to rank 0), instead of
            # rank 0 doing all of it while ranks 1..N-1 idle at a barrier.  The decision must be
            # identical on every rank (they call the collective _evaluate_per_group together):
            # swa_gate is created on all ranks (all-rank SWA update) and _swa_eval_cycle is
            # incremented identically, so it is.  Off / single-GPU -> the old rank-0-only path.
            _swa_ddp = bool(getattr(args, "swa_eval_ddp", True)) and world_size > 1
            if (is_main or _swa_ddp) and swa_gate is not None:
                _swa_eval_cycle += 1
                _run_swa_eval_this_cycle = (
                    args.swa_eval_freq <= 1
                    or ((_swa_eval_cycle - 1) % args.swa_eval_freq == 0)
                )
            else:
                _run_swa_eval_this_cycle = False
            if (is_main or _swa_ddp) and _run_swa_eval_this_cycle:
                if swa_gate is not None:
                    swa_gate.eval()
                    # Swap the SWA-averaged adapter (ΔW) into `model` for the SWA eval so it
                    # evaluates a TRUE average of ALL trainable params (gate + adapter), not the
                    # averaged gate bolted onto the live adapter.  Each participating rank swaps in
                    # its OWN (bit-identical) SWA copy and restores its OWN live adapter in `finally`
                    # — an unrestored adapter would silently diverge that rank's weights from the rest.
                    _live_adapter_sd = None
                    if _lora_enabled and swa_adapter_sd is not None:
                        _live_adapter_sd = {k: v.detach().clone()
                                            for k, v in collect_lora_state_dict(model).items()}
                        load_lora_state_dict(model, swa_adapter_sd)
                    try:
                        if is_main:
                            swa_eval_steps.append(step)
                        for _ki, _k in enumerate(_active_k_list):
                            if args.gate_selection == "threshold":
                                (swa_gate.module if hasattr(swa_gate, "module") else swa_gate).tau = float(_k)
                            # sharded across ranks when _swa_ddp (gathers to rank 0), else rank-0-only.
                            _per_group_swa, _, _per_ds_swa = _evaluate_per_group(
                                model, swa_gate, tokenizer,
                                _active_eval_by_group, tasks_by_name, ds_task,
                                _k, args,
                                rank=(rank if _swa_ddp else 0),
                                world_size=(world_size if _swa_ddp else 1),
                                device=str(device),
                                log_examples=0,
                            )
                            if is_main:
                                _primary_swa = next(iter(_per_group_swa))
                                _spm = _per_group_swa[_primary_swa]
                                swa_eval_per_k_accs[_k].append(_spm["accuracy"])
                                swa_eval_per_k_losses[_k].append(_spm["loss"])
                                _swa_summary = ", ".join(
                                    f"{_g}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                                    f" loss={m['loss']:.4f}"
                                    for _g, m in _per_group_swa.items()
                                )
                                _ds_summary_swa = ", ".join(
                                    f"{_d}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                                    for _d, m in _per_ds_swa.items()
                                )
                                print(f"\n  [swa eval/cat] step={step}  K={_k}  {_swa_summary}")
                                print(f"  [swa eval/ds]  step={step}  K={_k}  {_ds_summary_swa}")
                                _emit_eval_wandb_keys(_wandb_log, _per_group_swa, _per_ds_swa, _k,
                                                      _acc_metric_tag, _judge_active,
                                                      swa=True)
                        if args.gate_selection == "threshold":
                            (swa_gate.module if hasattr(swa_gate, "module") else swa_gate).tau = float(args.tau)
                        if is_main and _last_batch_data:
                            # Release eval-loop reservations first: the probe's batch shapes
                            # can trigger a fresh Triton compile, whose kernel load needs
                            # driver-side device memory outside the PyTorch allocator.
                            torch.cuda.empty_cache()
                            _swa_train_total, _swa_train_n = 0.0, 0
                            with torch.no_grad():
                                for _i in range(0, len(_last_batch_data), 4):
                                    _mini = _last_batch_data[_i: _i + 4]
                                    try:
                                        _sl, _lm, _ = forward_with_gate_batched(
                                            model, swa_gate, _mini, args.gate_layer,
                                            frozen_backbone=True,
                                        )
                                        for _b2, _it in enumerate(_mini):
                                            _mk = _lm[_b2].to(_sl.device)
                                            if not _mk.any():
                                                continue
                                            _lbl = torch.tensor(
                                                _it["input_ids"][_it["ctx_len"]: _it["ctx_len"] + _it["n_ans"]],
                                                dtype=torch.long, device=_sl.device,
                                            )
                                            if _sl[_b2, _mk, :].shape[0] == _lbl.shape[0]:
                                                _swa_train_total += F.cross_entropy(_sl[_b2, _mk, :], _lbl).item()
                                                _swa_train_n += 1
                                    except (torch.cuda.OutOfMemoryError, RuntimeError) as _oe:
                                        # Driver-level OOM at kernel launch is a plain RuntimeError,
                                        # not torch.cuda.OutOfMemoryError; both only skip this
                                        # diagnostic chunk. Anything else must still raise.
                                        if "out of memory" not in str(_oe):
                                            raise
                                        print(f"  [swa train-loss probe] OOM on chunk {_i}, skipped: {_oe}",
                                              flush=True)
                                        torch.cuda.empty_cache()
                            if _swa_train_n > 0:
                                _swa_tl = _swa_train_total / _swa_train_n
                                swa_train_losses.append(_swa_tl)
                                _wandb_log["train/swa_loss"] = _swa_tl
                    finally:
                        if _live_adapter_sd is not None:
                            load_lora_state_dict(model, _live_adapter_sd)

            if is_main and _wandb_run is not None:
                _wandb_run.log(_add_baselines(_wandb_log), step=step)
            gate.train()
            import gc as _gce
            _gce.collect()
            torch.cuda.empty_cache()
            if is_main:
                _save_curves(train_losses, k_log, eval_accs, eval_losses, eval_steps, args.output_dir,
                             bias_norm_log=bias_norm_log,
                             attn_scale_log=attn_scale_log,
                             eval_per_k_accs=eval_per_k_accs,
                             eval_per_k_losses=eval_per_k_losses,
                             swa_eval_per_k_accs=swa_eval_per_k_accs,
                             swa_eval_per_k_losses=swa_eval_per_k_losses,
                             swa_eval_steps=swa_eval_steps)
        if world_size > 1:
            dist.barrier()

    while step < args.steps:
        _skip_grad_sync = False
        # Residual-decomposition timers (opt-in): iteration start; data-prep and
        # forward/backward host-wall and clip are captured below.  perf_counter =
        # CPU wall, so fwd_bwd_wall − GPU-event compute reveals host-bound gaps.
        _t_iter = time.perf_counter() if _prof_other else 0.0
        _data_prep_ms = _fwd_bwd_wall_ms = _clip_ms = _logging_ms = float("nan")
        # Log full diagnostics + wandb this step? (essentials still log every step.)
        _do_diag = (step % _train_log_freq == 0)
        # Collect the expensive per-sample gate stats this step? (subset of diag.)
        _do_stats = _do_diag and (step % _per_sample_stats_freq == 0)
        # Collect phase-2 attention stats in the forward only on diag steps (rank 0);
        # off otherwise removes the per-forward collection cost + de-inflates rank-0.
        if is_main:
            set_phase2_stats_enabled(_do_diag)
        # ── Per-step diagnostic-buffer reset (ALL ranks) ─────────────────────
        # These three buffers feed rank-0 W&B stats and are read+cleared inside
        # the `if is_main` logging block.  On non-main ranks that block never
        # runs, so without an unconditional reset here they grow by one entry
        # per gate()/attention call every train step AND every eval forward —
        # an unbounded CPU leak across ranks 1..N that OOM-killed long runs.
        # Clearing at the top (a) bounds every rank to one step's worth and
        # (b) discards the previous iteration's eval-time appends so rank-0
        # stats reflect only this step's training forward.
        _gate_output_raw_buf.clear()
        _flex_attn_stats_buf.clear()
        if hasattr(gate, "_noise_stats_buf"):
            gate._noise_stats_buf = []
        # Reset the per-step sparsity accumulator (grad-tracked tensor) and
        # the kept-count diagnostics on the underlying gate module.  See
        # gate_forward._accumulate_gate_sparsity.  We set the accumulator to
        # None so the first sample of the next forward creates a fresh
        # grad-tracking tensor — clearing it to a leaf zero would block the
        # backward through sum-of-sigmoid means.
        _gate_inner = gate.module if hasattr(gate, "module") else gate
        _gate_inner._sparsity_term = None
        _gate_inner._kept_counts = []
        _gate_inner._n_art_counts = []
        _gate_capture[0] = False
        # Capture gate outputs only on steps whose per-sample stats we'll log
        # (rank 0) — skips the buffer append + downstream .item() loop otherwise.
        _gate_capture[0] = is_main and _do_stats
        if use_batching:
            # ── Batched training ──────────────────────────────────────────────────
            # Sample a batch of examples
            batch_samples = []
            batch_data = []
            
            _batch_seq_cap = int(os.environ.get("TRAIN_SEQ_CAP", "0")) or (
                (args.max_article_tokens + 5000) if args.max_article_tokens is not None
                else 20000
            )
            for _ in range(batch_size):
                # RESAMPLE bad/too-long draws instead of dropping them, so the
                # batch is always filled and every rank runs one forward + one
                # collective per step.  Dropping (the old behaviour) could leave
                # batch_data empty on some ranks and not others → the `continue`
                # below would skip this step's skip-flag all-reduce and desync
                # DDP (NCCL hang).  See the single-sample path for the rationale.
                sample = _sample_train(step)
                for _ in range(64):
                    n_art = sample["article_end"] - sample["article_start"]
                    if n_art > 0 and sample["n_ans"] > 0 \
                            and len(sample["full_ids"]) <= _batch_seq_cap:
                        break
                    sample = _sample_train(step)
                n_art = sample["article_end"] - sample["article_start"]
                K = _sample_K(step, n_art)

                batch_data.append({
                    "input_ids": sample["full_ids"],
                    "article_start": sample["article_start"],
                    "article_end": sample["article_end"],
                    "ctx_len": sample["ctx_len"],
                    "n_ans": sample["n_ans"],
                    "K": K,
                })
                batch_samples.append(sample)

            # batch_data is now full (resampled); this guard is a defensive
            # no-op.  The in-forward `n_valid == 0` path handles any pathological
            # residue sync-safely, so we never reach a desyncing main-loop skip.
            if not batch_data:
                _skip_grad_sync = True

            # §1b: set True by the detach multi-K branch, which does its own
            # backward; the finalization below then skips its .backward().
            _mk_detach_done = False
            _p12_len_log = None  # (phase-1 len, phase-2 len stats) for W&B

            try:
                gate.train()
                if getattr(args, "use_packed_sequences", False):
                    # Pack the per-step batch into packed element(s) with
                    # segment-causal attention.  Numerically equivalent to the
                    # padded batched path (argmax-match verified); eliminates
                    # the intra-batch padding waste when batch_size > 1.
                    from tokengate.e2e_gate.train.packed_sequences import (
                        pack_samples as _pack_samples,
                        forward_with_gate_packed as _fwd_packed,
                    )
                    _packing_input = [
                        {"full_ids": it["input_ids"],
                         **{k: it[k] for k in ("article_start", "article_end",
                                               "ctx_len", "n_ans", "K")}}
                        for it in batch_data
                    ]
                    packed_elems = _pack_samples(
                        _packing_input, max_total_len=args.packed_max_total_len,
                    )
                    logits, loss_mask, padding_mask = _fwd_packed(
                        model, gate, packed_elems,
                        gate_layer=args.gate_layer,
                        prune_phase1_kv=args.prune_phase1_kv,
                    )
                    batch_loss = 0.0
                    batch_loss_letter = 0.0
                    batch_loss_eos = 0.0
                    n_valid = 0
                    for _pi, _pe in enumerate(packed_elems):
                        if not loss_mask[_pi].any():
                            continue
                        ans_positions_all = loss_mask[_pi].nonzero(as_tuple=True)[0]
                        cursor = 0
                        for _seg in _pe.segments:
                            _n = _seg.n_ans
                            if cursor + _n > ans_positions_all.shape[0]:
                                break
                            seg_pos = ans_positions_all[cursor:cursor + _n]
                            seg_logits = logits[_pi, seg_pos, :]
                            seg_full_ids = _pe.input_ids[_seg.token_start:_seg.token_end]
                            local_ctx_end = _seg.ctx_end - _seg.token_start
                            seg_ans_ids = seg_full_ids[local_ctx_end:local_ctx_end + _n]
                            seg_labels = torch.tensor(
                                seg_ans_ids, dtype=torch.long, device=logits.device,
                            )
                            if seg_logits.shape[0] != seg_labels.shape[0]:
                                cursor += _n
                                continue
                            seg_loss = F.cross_entropy(seg_logits, seg_labels)
                            batch_loss = (batch_loss + seg_loss
                                          if isinstance(batch_loss, torch.Tensor) else seg_loss)
                            if seg_logits.shape[0] >= 2:
                                batch_loss_letter += F.cross_entropy(seg_logits[:-1], seg_labels[:-1]).item()
                                batch_loss_eos += F.cross_entropy(seg_logits[-1:], seg_labels[-1:]).item()
                            n_valid += 1
                            cursor += _n
                else:
                    # Multi-K-per-step amortization branch.
                    # M phase-2 targets per step (shared across batch): run Phase 1
                    # + gate ONCE, then Phase 2 M times.  Each (sample, K) pair
                    # contributes one CE; n_valid counts all surviving pairs so the
                    # final mean is (1/batch_size)·mean_over_M.  M is derived from
                    # the K source.
                    if getattr(args, "phase2_len_list", None):
                        _mixed_k_m = len(args.phase2_len_list)
                    elif getattr(args, "phase2_bucket_lists", None):
                        _mixed_k_m = len(args.phase2_bucket_lists[_bucket_for_step(step)])
                    else:
                        # phase2_loguniform: sampled L + full sentinel
                        _mixed_k_m = 2
                    if _mixed_k_m > 1:
                        if getattr(args, "phase2_loguniform", False):
                            # TWO targets — full-context sentinel
                            # + ONE L ~ LogUniform(128, seq_len/2). Stateless RNG on
                            # (seed, step): identical across ranks, resume-safe. W&B
                            # lanes use the nearest power of two of L (bounded keys);
                            # the ACTUAL sampled L is exact.
                            import math as _math
                            import random as _rnd2
                            _item0 = batch_data[0]
                            _nart0 = _item0["article_end"] - _item0["article_start"]
                            _seq0 = len(_item0["input_ids"])
                            _plen0 = _seq0 - _nart0
                            _lo = 128.0
                            _hi = max(_lo + 1.0, _seq0 / 2.0)
                            _rlu = _rnd2.Random((args.seed + 1) * 1000003 + step)
                            _L2 = int(round(_math.exp(_rlu.uniform(
                                _math.log(_lo), _math.log(_hi)))))
                            _K2 = max(1, min(_nart0, _L2 - _plen0))
                            _warn_scaffold_infeasible([_L2], _plen0, _nart0,
                                                      "phase2_loguniform", is_main)
                            _p2lu_lane = max(128, min(8192,
                                2 ** int(round(_math.log2(max(1, _L2))))))
                            K_list_step = [_K2, _nart0]
                            _p2lu_L_log = [_p2lu_lane, -1]
                            if step == 0 and is_main:
                                print(f"[train] phase2_loguniform (M={len(K_list_step)}): "
                                      f"L2={_L2} (lane L{_p2lu_lane}) + full sentinel "
                                      f"(seq={_seq0}, n_art={_nart0}, "
                                      f"prompt_len={_plen0}, K={K_list_step})",
                                      flush=True)
                        elif getattr(args, "phase2_len_list", None):
                            # Length-HOMOGENIZED: fix each sample's Phase-2 ctx to L_i via
                            # K_i = L_i - prompt_len (prompt_len = seq_len - n_art, the non-
                            # article tokens kept in Phase 2).  DROP targets the sample can't
                            # hit EXACTLY: feasible iff 1 <= K_i <= n_art (i.e. prompt_len <
                            # L_i <= seq_len).  Survivors run at Phase-2 length == L_i (true
                            # homogenization); a short sample simply skips the high-L targets
                            # (variable M this step).  batch_size=1 → exact per-sample.
                            _item0 = batch_data[0]
                            _seq0 = len(_item0["input_ids"])
                            _nart0 = _item0["article_end"] - _item0["article_start"]
                            _plen0 = _seq0 - _nart0
                            K_list_step = [(_nart0 if int(_L) == -1 else int(_L) - _plen0)
                                           for _L in args.phase2_len_list
                                           if int(_L) == -1 or 0 < int(_L) - _plen0 <= _nart0]
                            _warn_scaffold_infeasible(args.phase2_len_list, _plen0,
                                                      _nart0, "phase2_len_list", is_main)
                            if not K_list_step:
                                # sample shorter than the smallest target (or prompt >= all):
                                # keep one clamped index so the step stays valid.
                                K_list_step = [max(1, min(int(_nart0),
                                                          int(args.phase2_len_list[0]) - _plen0))]
                            if step == 0 and is_main:
                                print(f"[train] phase2_len_list (fixed, M={len(args.phase2_len_list)}, "
                                      f"drop-infeasible): {args.phase2_len_list} → K={K_list_step} "
                                      f"(n_art={_nart0}, prompt_len={_plen0}, kept={len(K_list_step)})",
                                      flush=True)
                        elif getattr(args, "phase2_bucket_lists", None):
                            # Length-BUCKETED homogenization: the sampler routed this step's
                            # sample into bucket _bucket_for_step(step) (all ranks agree, so
                            # every rank's sample is in the same seq-len band).  Use that
                            # bucket's PHASE_2_LEN_LIST; since the top is chosen ≤ the bucket's
                            # floor, all samples reach every target → homogeneous, no straggler,
                            # dense gradients on the long bucket.  drop-infeasible is a safety net.
                            _blk = args.phase2_bucket_lists[_bucket_for_step(step)]
                            _item0 = batch_data[0]
                            _nart0 = _item0["article_end"] - _item0["article_start"]
                            _plen0 = len(_item0["input_ids"]) - _nart0
                            # L == -1 is the FULL-CONTEXT sentinel: K_i = n_art
                            # (phase 2 keeps every article token — the
                            # fix for full-context erosion).
                            K_list_step = [(_nart0 if int(_L) == -1 else int(_L) - _plen0)
                                           for _L in _blk
                                           if int(_L) == -1 or 0 < int(_L) - _plen0 <= _nart0]
                            _warn_scaffold_infeasible(_blk, _plen0, _nart0,
                                                      "phase2_bucket_lists", is_main)
                            if not K_list_step:
                                K_list_step = [max(1, min(_nart0, int(_blk[0]) - _plen0))]
                            if step == 0 and is_main:
                                print(f"[train] phase2_bucket_lists: {len(args.phase2_bucket_lists)} "
                                      f"buckets; step-0 bucket={_bucket_for_step(step)} list={_blk} → "
                                      f"K={K_list_step} (n_art={_nart0}, prompt_len={_plen0})", flush=True)
                        else:
                            raise RuntimeError(
                                "unreachable: a K-source list is validated at startup")
                        # W&B length telemetry: sample length before pruning
                        # (phase-1 ctx) vs the step's post-prune phase-2 ctx
                        # lengths (prompt + K_i per target).
                        _li = batch_data[0]
                        _li_nart = _li["article_end"] - _li["article_start"]
                        _li_plen = _li["ctx_len"] - _li_nart
                        _p12_len_log = {
                            "p1": float(_li["ctx_len"]),
                            "p2_mean": float(sum(_li_plen + _k_ for _k_ in K_list_step)
                                             / max(1, len(K_list_step))),
                            "p2_min": float(_li_plen + min(K_list_step)),
                            "p2_max": float(_li_plen + max(K_list_step)),
                            # K = kept PRUNABLE (article) tokens; phase2_len =
                            # non-prunable scaffold + K. Sentinel already
                            # resolved (K == n_art on full-context targets).
                            "k_mean": float(sum(K_list_step) / max(1, len(K_list_step))),
                            "k_min": float(min(K_list_step)),
                            "k_max": float(max(K_list_step)),
                        }
                        # §1b: fused forward + per-K backward + ONE Phase-1
                        # backward — the only multi-K
                        # backward (the amortized variant was removed).  Returns
                        # the same per-step accumulators; the backward is already
                        # done, so the finalization below skips its .backward().
                        from tokengate.e2e_gate.gate_module.gate_forward import (
                            forward_backward_with_gate_multi_K_detach,
                        )
                        if _prof_other:
                            _t_fb0 = time.perf_counter()
                            _data_prep_ms = (_t_fb0 - _t_iter) * 1000.0
                        batch_loss, n_valid, batch_loss_letter, batch_loss_eos = (
                            forward_backward_with_gate_multi_K_detach(
                                model, gate, batch_data, args.gate_layer,
                                K_list=K_list_step,
                                prune_phase1_kv=args.prune_phase1_kv,
                                lora_start_layer=lora_start_layer,
                                grad_accum=args.grad_accum,
                                phase2_grad_checkpoint=bool(args.grad_checkpoint),
                            )
                        )
                        if _prof_other:
                            _fwd_bwd_wall_ms = (time.perf_counter() - _t_fb0) * 1000.0
                        _mk_detach_done = True
                        logits = loss_mask = padding_mask = None
                        # per-phase-2-length train loss (accumulate this rank's per-K CE,
                        # keyed by the phase-2 LENGTH target; flushed at the log cadence).
                        if is_main:
                            from tokengate.e2e_gate.gate_module import gate_forward as _gfmod
                            _pk = _gfmod.LAST_PER_K_TRAIN_LOSS
                            if _pk is not None:
                                if getattr(args, "phase2_bucket_lists", None):
                                    _Ls = args.phase2_bucket_lists[_bucket_for_step(step)]
                                elif getattr(args, "phase2_loguniform", False):
                                    _Ls = _p2lu_L_log
                                elif getattr(args, "phase2_len_list", None):
                                    _Ls = args.phase2_len_list
                                else:
                                    _Ls = _pk["K"]
                                for _i, (_pls, _pln) in enumerate(zip(_pk["loss_sum"], _pk["n"])):
                                    if _i < len(_Ls) and _pln > 0:
                                        _pa = _phase2_loss_acc.setdefault(int(_Ls[_i]), [0.0, 0])
                                        _pa[0] += _pls
                                        _pa[1] += _pln
                    else:
                        logits, loss_mask, padding_mask = forward_with_gate_batched(
                            model, gate, batch_data,
                            args.gate_layer,
                            prune_phase1_kv=args.prune_phase1_kv,
                            lora_start_layer=lora_start_layer,
                        )

                        # Compute loss for each sample
                        batch_loss = 0.0
                        batch_loss_letter = 0.0
                        batch_loss_eos = 0.0
                        n_valid = 0

                        for b_idx, (sample, item) in enumerate(zip(batch_samples, batch_data)):
                            # Get answer tokens for this sample
                            ctx_len = item["ctx_len"]
                            n_ans = item["n_ans"]
                            full_ids = item["input_ids"]

                            # Extract logits and labels for answer positions
                            sample_loss_mask = loss_mask[b_idx].to(logits.device)
                            if not sample_loss_mask.any():
                                continue

                            sample_logits = logits[b_idx, sample_loss_mask, :]  # (n_ans, vocab)
                            answer_ids = full_ids[ctx_len:ctx_len + n_ans]
                            sample_labels = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)

                            if sample_logits.shape[0] != sample_labels.shape[0]:
                                continue

                            sample_loss = F.cross_entropy(sample_logits, sample_labels)
                            batch_loss += sample_loss
                            if sample_logits.shape[0] >= 2:
                                batch_loss_letter += F.cross_entropy(sample_logits[:-1], sample_labels[:-1]).item()
                                batch_loss_eos += F.cross_entropy(sample_logits[-1:], sample_labels[-1:]).item()
                            n_valid += 1

                if n_valid == 0:
                    optimizer.zero_grad(set_to_none=True)
                    _skip_grad_sync = True
                else:
                    # Average loss over valid samples in batch
                    loss = batch_loss / n_valid
                    # Sparsity penalty: lambda * mean(sigmoid(b)) averaged across
                    # the n_valid samples whose gate forwards accumulated into
                    # gate._sparsity_term.  Always sums into `loss` even when
                    # lambda=0 would be a no-op — keeping the addition explicit
                    # lets us log the term independently.
                    _spars_term = getattr(_gate_inner, "_sparsity_term", None)
                    _step_sparsity_loss = 0.0
                    if (_spars_term is not None) and (args.sparsity_lambda != 0.0):
                        _n_acc = max(1, len(getattr(_gate_inner, "_kept_counts", []) or [1]))
                        _sparsity_loss = args.sparsity_lambda * (_spars_term / _n_acc)
                        _step_sparsity_loss = float(_sparsity_loss.detach().item())
                        loss = loss + _sparsity_loss
                    _step_loss_letter = batch_loss_letter / n_valid
                    _step_loss_eos = batch_loss_eos / n_valid
                    # §1b detach path already did its own per-K + Phase-1 backward
                    # (and scaled by 1/grad_accum); `loss` here is detached.
                    if not _mk_detach_done:
                        (loss / args.grad_accum).backward()

                # Log average K for this batch
                avg_K_sampled = sum(item["K"] for item in batch_data) / len(batch_data)
                avg_K_clamped = sum(
                    max(1, min(item["K"], item["article_end"] - item["article_start"]))
                    for item in batch_data
                ) / len(batch_data)
                avg_n_art = sum(item["article_end"] - item["article_start"] for item in batch_data) / len(batch_data)
                # Threshold-mode kept-count diagnostics (accumulator populated by
                # _accumulate_gate_sparsity).  These mirror avg_K_clamped but
                # reflect the effective kept count, which is sample-dependent
                # in threshold mode and identical to K_clamped in top-K mode.
                _kept_counts = getattr(_gate_inner, "_kept_counts", None) or []
                _n_art_counts = getattr(_gate_inner, "_n_art_counts", None) or []
                if _kept_counts:
                    _step_mean_kept_K = sum(_kept_counts) / len(_kept_counts)
                else:
                    _step_mean_kept_K = float("nan")
                if _kept_counts and _n_art_counts and len(_kept_counts) == len(_n_art_counts):
                    _step_mean_kept_frac = (
                        sum(k / max(1, n) for k, n in zip(_kept_counts, _n_art_counts))
                        / len(_kept_counts)
                    )
                else:
                    _step_mean_kept_frac = float("nan")
                _last_batch_data = batch_data
                
            except (torch.cuda.OutOfMemoryError, RuntimeError) as _oom_err:
                # See single-sample handler for the rationale: FlexAttention
                # kernel-launch driver OOMs surface as plain RuntimeError, not
                # torch.cuda.OutOfMemoryError, so we widen the catch.  Re-raise
                # non-OOM RuntimeErrors.
                if not isinstance(_oom_err, torch.cuda.OutOfMemoryError):
                    _msg = str(_oom_err).lower()
                    if ("out of memory" not in _msg
                            and "cuda driver error" not in _msg):
                        raise
                # CRITICAL: drop references to the failed step's tensors + autograd
                # graph BEFORE gc/empty_cache (see single-sample handler below for
                # the full rationale — otherwise the retained graph keeps the rank
                # pinned and every later step cascade-OOMs).
                logits = loss = batch_loss = loss_mask = padding_mask = None
                # Drop the BlockMask cache to free CUDA-resident masks; gc.collect()
                # before empty_cache so freed tensors release before coalescing.
                # Without this, post-OOM steps cascade-OOM even on small sequences
                # because the allocator stays fragmented.
                import gc as _gc
                from tokengate.e2e_gate.attn.flex_attention_utils import (
                    clear_block_mask_cache as _cbmc,
                )
                _cbmc()
                _gc.collect()
                torch.cuda.empty_cache()
                max_seq = max(len(item["input_ids"]) for item in batch_data)
                _oom_skips += 1
                print(f"\n[OOM] step={step}, batch_size={len(batch_data)}, max_seq={max_seq}, "
                      f"skipping (cumulative OOM-skips: {_oom_skips}).", flush=True)
                optimizer.zero_grad(set_to_none=True)
                _skip_grad_sync = True
        
        _gate_capture[0] = False

        grad_norm = 0.0
        if (step + 1) % args.grad_accum == 0:
            # ── STRAGGLER WAIT probe ──────────────────────────────────────────
            # Timed barrier BEFORE the first collective: sync this rank's own
            # compute, then measure how long it blocks for the slowest rank to
            # arrive.  On rank 0 this is the idle time the length-heterogeneity
            # across the 4 samples costs — the homogenization target.  All ranks
            # call it (gated on env, identical everywhere) so no deadlock.
            if _prof_other:
                _straggler_ms = _allreduce_ms = _optim_ms = float("nan")
                torch.cuda.synchronize()
                _t_barrier = time.perf_counter()
                dist.barrier()
                _straggler_ms = (time.perf_counter() - _t_barrier) * 1000.0
            # Sync the skip flag across all ranks before any collective.
            # If any rank hit OOM or had no valid samples this step, all ranks
            # skip the gradient all-reduce — otherwise the ranks that continued
            # early miss the collective and NCCL times out.
            if world_size > 1:
                _skip_tensor = torch.tensor(int(_skip_grad_sync), dtype=torch.long, device=device)
                dist.all_reduce(_skip_tensor, op=dist.ReduceOp.MAX)
                _skip_grad_sync = bool(_skip_tensor.item())
            if not _skip_grad_sync:
                # Gate / LoRA / (optionally) backbone are NOT wrapped in DDP
                # because the gate's attn_scale and gate_scale receive gradients
                # from Phase-2 attention layers outside gate.forward(), which
                # breaks DDP's bucket discovery.  We therefore manually average
                # gradients across ranks.
                #
                # Coalesced all-reduce: instead of issuing one NCCL call per
                # parameter tensor (~5–50 tiny launches per step), we flatten
                # all gradients into a single contiguous buffer, do ONE big
                # all-reduce, then copy the averaged result back into each
                # tensor.  Saves the per-launch overhead (NCCL kernel launch +
                # CUDA stream sync) and lets the NCCL kernel run efficiently
                # over a single large message.  Falls back to the per-tensor
                # loop when world_size == 1 (no comm needed) or when no
                # parameter has a grad (skip step).
                if world_size > 1:
                    _sync_params: List[torch.nn.Parameter] = list(gate.parameters())
                    if _lora_enabled:
                        _sync_params.extend(lora_params)

                    _grads = [p.grad for p in _sync_params if p.grad is not None]
                    if _grads:
                        # Flatten into a single contiguous buffer for one NCCL
                        # call.  `_flatten_dense_tensors` returns a new tensor;
                        # we all_reduce it, then unflatten back into per-grad
                        # views with the same shape.
                        from torch._utils import (
                            _flatten_dense_tensors,
                            _unflatten_dense_tensors,
                        )
                        _flat = _flatten_dense_tensors(_grads)
                        if _prof_other:
                            torch.cuda.synchronize()
                            _t_ar = time.perf_counter()
                        dist.all_reduce(_flat, op=dist.ReduceOp.AVG)
                        if _prof_other:
                            torch.cuda.synchronize()
                            _allreduce_ms = (time.perf_counter() - _t_ar) * 1000.0
                        for _g, _new in zip(_grads,
                                            _unflatten_dense_tensors(_flat, _grads)):
                            _g.copy_(_new)
            else:
                # All ranks agreed to skip this step — advance step and move on.
                step += 1
                if pbar is not None:
                    pbar.update(1)
                # Run eval + save checkpoint even on all-skip steps so
                # eval_freq / checkpoint_freq aren't probabilistically dropped
                # by OOM-skip races. With ~6% per-rank OOM × 8 ranks, ~40% of
                # save/eval boundaries land on all-skip iterations and would
                # otherwise never fire their eval or write a checkpoint.
                if step % args.eval_freq == 0:
                    _do_eval()
                if step % args.checkpoint_freq == 0:
                    _do_checkpoint_save(step)
                continue

            all_params = list(gate.parameters()) + lora_params
            # Standard gradient clipping.  clip_grad_norm_ returns the total norm;
            # if any grad element is non-finite (a true NaN/inf, which the biodex
            # extreme-small-K STE regime produces on ~80% of steps), the norm is
            # inf → clip_coef = max_norm/(inf+eps) = 0 → grads are zeroed.  That
            # zero-update is a PROTECTIVE skip: the downstream skip-guard then
            # advances the step counter without an optimizer.step(), so the gate
            # only learns from the finite-grad steps.  Do NOT replace this with an
            # fp64-accumulated norm — that "rescues" large-but-genuinely-divergent
            # grads into the update and drifts b_j to inf → CUDA illegal access.
            if _prof_other:
                torch.cuda.synchronize()
                _t_clip = time.perf_counter()
            _grad_norm_t = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            if _prof_other:
                torch.cuda.synchronize()
                _clip_ms = (time.perf_counter() - _t_clip) * 1000.0

            # Save per-tensor grad norms BEFORE zero_grad clears them.
            #
            # Previously this block issued ~10–18 independent `.grad.norm().item()`
            # calls per update step.  Each `.item()` forces a host sync (waits for
            # all enqueued GPU work to finish), so the cumulative cost is
            # 5–20 ms / step on 8×H100.  We collect every norm into a single
            # tensor of scalars staying on GPU, then do ONE host transfer via
            # `.cpu().tolist()`, dropping the per-step sync count from ~18 → 1.
            _gn_names: list = ["clip"]
            _gn_tensors: list = [_grad_norm_t]

            def _add_gn(name: str, t):
                if t is not None and t.grad is not None:
                    _gn_names.append(name)
                    _gn_tensors.append(t.grad.norm())

            _add_gn("attn_scale", getattr(gate, "attn_scale", None))
            _add_gn("gate_scale", getattr(gate, "gate_scale", None))
            _add_gn("W_noise",    getattr(gate, "W_noise", None))
            _add_gn("v_attn",     getattr(gate, "v_attn", None))
            _add_gn("w_out",      getattr(gate, "w_out", None))

            # gate.proj may be Sequential[Linear, ...] or bare Linear.
            if hasattr(gate, "proj"):
                _proj_w = (gate.proj[0].weight
                           if isinstance(gate.proj, torch.nn.Sequential)
                              and hasattr(gate.proj[0], "weight")
                           else (gate.proj.weight if hasattr(gate.proj, "weight") else None))
                _add_gn("W_proj", _proj_w)

            if hasattr(gate, "W_k") and hasattr(gate.W_k, "weight"):
                _add_gn("W_k", gate.W_k.weight)
            if hasattr(gate, "W_q") and hasattr(gate.W_q, "weight"):
                _add_gn("W_q", gate.W_q.weight)
            _br_gn = getattr(gate, "gate_branch", None)
            if _br_gn is not None:
                _add_gn("branch_gate_proj", _br_gn.gate_proj.weight)
                _add_gn("branch_up_proj", _br_gn.up_proj.weight)
                _add_gn("readout_w", _br_gn.down_proj.weight)

            # Adapter grad norms: mean per named group (lora_A/lora_B for the
            # factored adapter, delta/delta_bias for the additive one).
            for _ag_name, _ag_params in _adapter_norm_groups.items():
                _ag_gnorms = [p.grad.norm() for p in _ag_params
                              if p.grad is not None]
                if _ag_gnorms:
                    _gn_names.append(_ag_name)
                    _gn_tensors.append(torch.stack(_ag_gnorms).mean())

            # Single GPU→CPU sync for ALL grad-norm scalars.
            _gn_vals = torch.stack(_gn_tensors).detach().cpu().tolist()
            _gn = dict(zip(_gn_names, _gn_vals))

            grad_norm = _gn["clip"]
            _saved_attn_scale_grad_norm = _gn.get("attn_scale")
            _saved_gate_scale_grad_norm = _gn.get("gate_scale")
            _saved_W_noise_grad_norm    = _gn.get("W_noise")
            _saved_v_attn_grad_norm     = _gn.get("v_attn")
            _saved_w_out_grad_norm      = _gn.get("w_out")
            _saved_W_proj_grad_norm     = _gn.get("W_proj")
            _saved_W_k_grad_norm        = _gn.get("W_k")
            _saved_W_q_grad_norm        = _gn.get("W_q")
            _saved_branch_grad_norms    = {_k: _gn.get(_k) for _k in
                ("branch_gate_proj", "branch_up_proj", "readout_w")}
            _saved_adapter_grad_norms   = {k: _gn[k] for k in _adapter_norm_groups
                                           if k in _gn}

            # Non-finite-gradient guard.  clip_grad_norm_ returns the total grad
            # norm; if any grad is NaN/inf the total norm is NaN, AND
            # clip_grad_norm_ has already multiplied every grad by the resulting
            # NaN clip coefficient — so applying this step would poison EVERY
            # weight (NaN spreads through the whole model in one update).  Skip
            # the optimizer step entirely on a non-finite grad norm and drop the
            # poisoned grads.  This bounds the damage of a single bad batch to a
            # skipped step instead of an unrecoverable NaN model.
            if not bool(torch.isfinite(_grad_norm_t)):
                optimizer.zero_grad(set_to_none=True)
                _nonfinite_grad_skips += 1
                if is_main and (_nonfinite_grad_skips <= 20
                                or _nonfinite_grad_skips % 50 == 0):
                    print(f"\n[nonfinite-grad] step={step}: grad norm non-finite, "
                          f"skipping optimizer step "
                          f"(cumulative: {_nonfinite_grad_skips}).", flush=True)
                step += 1
                if pbar is not None:
                    pbar.update(1)
                if step % args.eval_freq == 0:
                    _do_eval()
                if step % args.checkpoint_freq == 0:
                    _do_checkpoint_save(step)
                continue

            if _prof_other:
                torch.cuda.synchronize()
                _t_opt = time.perf_counter()
            optimizer.step()
            if _prof_other:
                torch.cuda.synchronize()
                _optim_ms = (time.perf_counter() - _t_opt) * 1000.0
            scheduler.step()
            # set_to_none=True frees the .grad tensors instead of writing zeros,
            # which is faster (no memset) and reduces peak working memory.  Reads
            # of `.grad` on non-update steps already guard with `is not None`.
            optimizer.zero_grad(set_to_none=True)
            # Capture per-matrix norms immediately after the weight update so
            # they reflect the new values (not stale values from grad_accum-1
            # non-update steps, which would produce a staircase in W&B).
            #
            # Batched into ONE GPU→CPU sync (see grad-norm comment above for the
            # rationale).  We stack all data.norm() tensors on GPU, then issue
            # a single `.cpu().tolist()`.  LoRA gets the mean of its A / B sets.
            _wn_names: list = []
            _wn_tensors: list = []

            def _add_wn(name: str, t):
                if t is not None:
                    _wn_names.append(name)
                    _wn_tensors.append(t.data.norm())

            for _ag_name, _ag_params in _adapter_norm_groups.items():
                _wn_names.append(f"{_ag_name}_mean")
                # fp32 norm: bf16 base weights (norm ~O(100)) have a 0.5 grid, which
                # pins the logged norm to a flat integer and hides the (small,
                # mostly-orthogonal) training drift. Cast to fp32 so the metric moves.
                _wn_tensors.append(
                    torch.stack([p.data.float().norm() for p in _ag_params]).mean()
                )
            if is_main:
                _add_wn("W_k",    gate.W_k.weight    if hasattr(gate, "W_k")    else None)
                _add_wn("W_q",    gate.W_q.weight    if hasattr(gate, "W_q")    else None)
                _add_wn("W_proj", gate.W_proj.weight if hasattr(gate, "W_proj") else None)
                _add_wn("v_attn", getattr(gate, "v_attn", None))
                _add_wn("w_out",  getattr(gate, "w_out", None))
                _add_wn("W_Qa",   gate.W_Qa.weight   if hasattr(gate, "W_Qa")   else None)
                _add_wn("W_Kc",   gate.W_Kc.weight   if hasattr(gate, "W_Kc")   else None)
                _add_wn("W_Vc",   gate.W_Vc.weight   if hasattr(gate, "W_Vc")   else None)
                _add_wn("W_noise", getattr(gate, "W_noise", None))

            if _wn_tensors:
                _wn_vals = torch.stack(_wn_tensors).detach().cpu().tolist()
                _wn = dict(zip(_wn_names, _wn_vals))
            else:
                _wn = {}

            _adapter_norms = {k: _wn[f"{k}_mean"] for k in _adapter_norm_groups
                              if f"{k}_mean" in _wn}
            if is_main:
                _gate_wk_norm      = _wn.get("W_k")
                _gate_wq_norm      = _wn.get("W_q")
                _gate_wp_norm      = _wn.get("W_proj")
                _gate_v_attn_norm  = _wn.get("v_attn")
                _gate_w_out_norm   = _wn.get("w_out")
                _gate_W_Qa_norm    = _wn.get("W_Qa")
                _gate_W_Kc_norm    = _wn.get("W_Kc")
                _gate_W_Vc_norm    = _wn.get("W_Vc")
                _gate_W_noise_norm = _wn.get("W_noise")

            # ── SWA update ───────────────────────────────────────────────────
            if swa_enabled and step >= swa_start_step and (step - swa_start_step) % args.swa_freq == 0:
                _swa_just_inited = False
                # With --swa-eval-ddp (default), EVERY rank maintains its own SWA average so the
                # SWA eval can shard like the live eval; the averages stay bit-identical because
                # live weights are all_reduce-synced each step.  Otherwise rank 0 only (legacy).
                if is_main or (bool(getattr(args, "swa_eval_ddp", True)) and world_size > 1):
                    _gate_raw_swa = gate
                    if swa_gate is None:
                        # Clear transient threshold-mode diagnostic state before
                        # snapshotting: `_sparsity_term` is a non-leaf tensor
                        # (graph-tracked sum-of-sigmoid-means) populated by the
                        # most recent forward — `copy.deepcopy` refuses non-leaf
                        # tensor attributes, so we drop it.  These are also
                        # reset at the top of every train step.
                        _gs = _gate_raw_swa.module if hasattr(_gate_raw_swa, "module") else _gate_raw_swa
                        _gs._sparsity_term = None
                        _gs._kept_counts = []
                        _gs._n_art_counts = []
                        # First update: clone current weights as the starting average.
                        # fp32 storage (asserted): the running mean is exact only in fp32.
                        swa_gate = copy.deepcopy(_gate_raw_swa)
                        assert_fp32_params(swa_gate, "swa_gate")
                        swa_gate.eval()
                        swa_n = 1
                        _swa_just_inited = True
                        # Seed the averaged adapter (delta ΔW / LoRA) from the live weights:
                        # an fp32 clone on the same device, so later updates never leave it.
                        if _lora_enabled:
                            swa_adapter_sd = {k: p.detach().to(torch.float32).clone()
                                              for k, p in collect_lora_named_params(model)}
                        if is_main:
                            print(f"  [swa] initialised at step={step} (swa_start_step={swa_start_step})")
                    else:
                        # Average the adapter with the SAME pre-increment count as _swa_update
                        # uses for the gate: avg += (cur - avg)/(swa_n+1), fp32, in place on the
                        # device straight from the live params (no D2H copy).  Do it BEFORE
                        # _swa_update (which increments swa_n) so both use the same n.
                        if _lora_enabled and swa_adapter_sd is not None:
                            swa_inplace_update(swa_adapter_sd.items(),
                                               collect_lora_named_params(model), swa_n)
                        _swa_update(_gate_raw_swa)

                # No broadcast/barrier/init-eval here.  The old code ran two full
                # evals (base + swa deepcopy) on rank 0 alone while non-zero ranks
                # waited at a barrier — for slow datasets (govreport, large K) this
                # exceeded the 2-hour NCCL watchdog timeout.  Since swa_gate is
                # always deepcopy(gate) at init, the results are identical to base;
                # the next eval_freq step logs SWA metrics through the normal path.

        # Log rank 0's loss directly.  A cross-rank all_reduce here would require
        # all ranks to reach this line simultaneously, but ranks can diverge when
        # any rank hits a `continue` (e.g. OOM skip).  The small per-step variance
        # from logging a single rank's loss is acceptable.
        #
        # OOM-skip guard: both OOM handlers above (batched at ~3819, single at
        # ~3973) set ``loss = None`` to drop the retained autograd graph.  When
        # grad_accum == 1 the post-OOM continue at the bottom of the gradient-
        # step block (~4053) skips past this line; but with grad_accum > 1, OOM
        # on a non-grad-step bypasses that block entirely and we land here with
        # loss=None → AttributeError.  Fall back to NaN so the training loop
        # advances; the next successful step reassigns loss to a real tensor.
        if loss is None:
            logged_loss = float("nan")
        else:
            logged_loss = loss.detach().item()

        # Residual probe: the rank-0-only stats/logging block below (per-sample
        # .item() syncs, topk/softmax/entropy, param norms, wandb.log) is NOT
        # training compute but runs every step on rank 0 → inflates its s/it.
        _t_log0 = time.perf_counter() if _prof_other else 0.0
        if is_main:
            train_losses.append(logged_loss)
            k_log.append((step, avg_K_sampled, avg_K_clamped, avg_n_art))
            pbar.set_postfix({"loss": f"{logged_loss:.4f}", "K": f"{avg_K_clamped:.0f}", "step": step, "oom": _oom_skips})
            pbar.update(1)
            # DIAGNOSTIC memtrace: allocated climbing => tensor leak; reserved
            # climbing w/ allocated flat => fragmentation; bmcache growing =>
            # block-mask/compile-cache growth.  (Tracking run6's ~80GB creep.)
            if step % 100 == 0:
                try:
                    from tokengate.e2e_gate.attn.flex_attention_utils import (
                        get_block_mask_cache_stats as _bmcs,
                    )
                    _bm = _bmcs()
                except Exception:
                    _bm = {}
                print(f"[memtrace] step={step} "
                      f"allocated={torch.cuda.memory_allocated()/1e9:.1f}GB "
                      f"reserved={torch.cuda.memory_reserved()/1e9:.1f}GB "
                      f"max_alloc={torch.cuda.max_memory_allocated()/1e9:.1f}GB "
                      f"bmcache={_bm}", flush=True)

            if _do_diag:
                # Gate parameter norm (cheap scalar; every diag step).
                if getattr(gate, "proj", None) is not None:
                    proj = gate.proj
                    if hasattr(proj, "weight"):
                        gate_weight_norm = proj.weight.data.norm().item()
                    else:
                        gate_weight_norm = sum(p.data.norm()**2 for p in proj.parameters()).sqrt().item()
                else:
                    gate_weight_norm = sum(p.data.norm()**2 for p in gate.parameters()).sqrt().item()

                # Per-sample gate-output stats (gate_all_mean/std, gate_topk_mean/std):
                # the .item()-heavy inner loop → only on PER_SAMPLE_STATS_FREQ steps.
                # Capture is gated on _do_stats too, so the buffer is empty otherwise
                # and these stay None (omitted from the log).
                gate_output_mean = gate_output_std = None
                gate_topk_mean = gate_topk_std = None
                if _do_stats and _gate_output_raw_buf:
                    if use_batching:
                        _k_per_sample = [max(1, min(item["K"], item["article_end"] - item["article_start"]))
                                         for item in batch_data]
                    else:
                        _k_per_sample = [K_clamped]
                    _agg_w_cpu = (gate.agg_weights.detach().float().cpu()
                                  if hasattr(gate, "agg_weights") and gate.agg_weights is not None else None)
                    _all_means, _all_stds, _topk_means, _topk_stds = [], [], [], []
                    for _t, _k in zip(_gate_output_raw_buf, _k_per_sample):
                        _all_means.append(_t.mean().item())
                        _all_stds.append(_t.std().item() if _t.numel() > 1 else 0.0)
                        _kc = max(1, min(_k, _t.shape[0]))
                        _t_scores = _t.mean(dim=(-2, -1)) if _t.dim() == 3 else ((_t * _agg_w_cpu).sum(-1) if (_t.dim() == 2 and _agg_w_cpu is not None) else _t)
                        _topk_vals = torch.topk(_t_scores, _kc).values
                        _topk_means.append(_topk_vals.mean().item())
                        _topk_stds.append(_topk_vals.std().item() if _kc > 1 else 0.0)
                    _n = max(len(_all_means), 1)
                    gate_output_mean = sum(_all_means) / _n
                    gate_output_std  = sum(_all_stds)  / _n
                    gate_topk_mean   = sum(_topk_means) / _n
                    gate_topk_std    = sum(_topk_stds)  / _n
                _gate_output_raw_buf.clear()

                # Phase-2 attention stats: b_normalized / qk_normalized / softmax_input.
                _p2 = list(_flex_attn_stats_buf)
                _flex_attn_stats_buf.clear()
                if _p2:
                    _b_h  = torch.stack([s['b']  for s in _p2]).mean(dim=0)
                    _qk_h = torch.stack([s['qk'] for s in _p2]).mean(dim=0)
                    _si_h = torch.stack([s['si'] for s in _p2]).mean(dim=0)
                    _b_norm_stats  = {'mean': _b_h.mean().item(),  'max': _b_h.max().item(),  'min': _b_h.min().item()}
                    _qk_norm_stats = {'mean': _qk_h.mean().item(), 'max': _qk_h.max().item(), 'min': _qk_h.min().item()}
                    _si_stats      = {'mean': _si_h.mean().item(), 'max': _si_h.max().item(), 'min': _si_h.min().item()}
                    # Signed formula-term scalars; averaged across
                    # buffer entries (per-entry variances averaged — a
                    # monitoring approximation that ignores between-entry
                    # mean spread).
                    _term_stats = {}
                    for _k_ in ("b_m", "b_v", "qk_m", "qk_v", "si_m", "si_v",
                                "se_m", "se_v"):
                        _vals = [s_[_k_] for s_ in _p2 if _k_ in s_]
                        _term_stats[_k_] = (sum(_vals) / len(_vals)) if _vals else None
                else:
                    _b_norm_stats = _qk_norm_stats = _si_stats = None
                    _term_stats = {}

                # Learned scalars: attn_scale, gate_scale (mean/min/max).
                def _param_stats(p):
                    d = p.detach().float().cpu()
                    if d.numel() == 1:
                        v = d.item()
                        return {"mean": v, "min": v, "max": v}
                    return {"mean": d.mean().item(), "min": d.min().item(), "max": d.max().item()}

                attn_scale_stats = (_param_stats(gate.attn_scale)
                    if hasattr(gate, "attn_scale") and gate.attn_scale is not None else None)
                # log the EFFECTIVE magnitude softplus(beta), not the raw parameter
                gate_scale_stats = _param_stats(
                    torch.nn.functional.softplus(gate.gate_scale.detach()))
                # effective temperature S_eff = softplus(raw sigma)
                _attn_eff_stats = _param_stats(
                    torch.nn.functional.softplus(gate.attn_scale.detach()))
                if attn_scale_stats is not None:
                    attn_scale_log.append((step, attn_scale_stats["mean"]))

                if _wandb_run is not None:
                    _avg_n_ans = (sum(item["n_ans"] for item in _last_batch_data) / max(len(_last_batch_data), 1)
                                  if _last_batch_data else 0.0)
                    log_dict = {
                        "train/loss": logged_loss,
                        **({"train/phase1_len": _p12_len_log["p1"],
                            "train/phase2_len_mean": _p12_len_log["p2_mean"],
                            "train/phase2_len_min": _p12_len_log["p2_min"],
                            "train/phase2_len_max": _p12_len_log["p2_max"],
                            "train/K_mean": _p12_len_log["k_mean"],
                            "train/K_min": _p12_len_log["k_min"],
                            "train/K_max": _p12_len_log["k_max"]}
                           if _p12_len_log is not None else {}),
                        "train/sparsity_loss": locals().get("_step_sparsity_loss", 0.0),
                        "train/sparsity_lambda": args.sparsity_lambda,
                        "train/mean_kept_K": locals().get("_step_mean_kept_K", float("nan")),
                        "train/mean_kept_frac": locals().get("_step_mean_kept_frac", float("nan")),
                        "train/tau": getattr(_gate_inner, "tau", float("nan")),
                        "train/epoch": (step + 1) * (args.batch_size * world_size) / _n_train_global,
                        "train/oom_skips": _oom_skips,
                        "train/K_clamped": avg_K_clamped,
                        "train/K_sampled": avg_K_sampled,
                        "train/n_art": avg_n_art,
                        "train/avg_n_ans": _avg_n_ans,
                        "train/grad_norm": grad_norm,
                        "train/gate_weight_norm": gate_weight_norm,
                        "train/lr": scheduler.get_last_lr()[0],
                        "system/elapsed_min": (time.time() - t0) / 60,
                        "train/elapsed_seconds": time.time() - t0,
                        # steps run by THIS process (resume-aware; see _start_step)
                        "train/seconds_per_step": (time.time() - t0)
                                                  / max(step - _start_step, 1),
                    }
                    # per-phase-2-length train loss (mean over this log window), then reset
                    if _phase2_loss_acc:
                        for _L, (_lsum, _ln) in sorted(_phase2_loss_acc.items()):
                            if _ln > 0:
                                log_dict[f"train/phase2_loss_L{_L}"] = _lsum / _ln
                        _phase2_loss_acc.clear()
                    if gate_output_mean is not None:
                        log_dict["train/gate_all_mean"]  = gate_output_mean
                        log_dict["train/gate_all_std"]   = gate_output_std
                        log_dict["train/gate_topk_mean"] = gate_topk_mean
                        log_dict["train/gate_topk_std"]  = gate_topk_std
                    if _lora_enabled and lora_params:
                        for _ag_name, _ag_val in _adapter_norms.items():
                            log_dict[f"train/{_ag_name}_norm"] = _ag_val
                        for _ag_name, _ag_val in _saved_adapter_grad_norms.items():
                            log_dict[f"train/{_ag_name}_grad_norm"] = _ag_val
                    _br_t = gate.gate_branch if getattr(gate, "gate_branch", None) is not None else None
                    if _br_t is not None:
                        for _bk, _bv in _saved_branch_grad_norms.items():
                            if _bv is not None:
                                log_dict[f"train/{_bk}_grad_norm"] = _bv
                        log_dict["train/readout_w_norm"] = float(_br_t.down_proj.weight.detach().norm())
                        log_dict["train/branch_gate_proj_fro"] = float(_br_t.gate_proj.weight.detach().norm())
                        log_dict["train/branch_up_proj_fro"] = float(_br_t.up_proj.weight.detach().norm())
                        if _branch_init_cpu is not None:
                            _drift_log_ctr[0] += 1
                            if _drift_log_ctr[0] % 25 == 1:
                                with torch.no_grad():
                                    _c_g = _br_t.gate_proj.weight.detach().float().cpu()
                                    _c_u = _br_t.up_proj.weight.detach().float().cpu()
                                    _c_w = _br_t.down_proj.weight.detach().float().cpu().flatten()
                                    _i_g, _i_u = _branch_init_cpu["gate_proj"], _branch_init_cpu["up_proj"]
                                    _i_w = _branch_init_cpu["w"].flatten()
                                    log_dict["train/branch_drift_from_init"] = float(
                                        ((_c_g - _i_g).norm() / _i_g.norm()
                                         + (_c_u - _i_u).norm() / _i_u.norm()) / 2)
                                    if float(_i_w.norm()) > 0:
                                        log_dict["train/readout_cos_from_init"] = float(
                                            torch.nn.functional.cosine_similarity(_c_w, _i_w, dim=0))
                    if _b_norm_stats is not None:
                        log_dict["train/b_normalized/mean"] = _b_norm_stats["mean"]
                        log_dict["train/b_normalized/max"]  = _b_norm_stats["max"]
                        log_dict["train/b_normalized/min"]  = _b_norm_stats["min"]
                        log_dict["train/qk_normalized/mean"] = _qk_norm_stats["mean"]
                        log_dict["train/qk_normalized/max"]  = _qk_norm_stats["max"]
                        log_dict["train/qk_normalized/min"]  = _qk_norm_stats["min"]
                        log_dict["train/softmax_input/mean"] = _si_stats["mean"]
                        log_dict["train/softmax_input/max"]  = _si_stats["max"]
                        log_dict["train/softmax_input/min"]  = _si_stats["min"]
                    # Formula panels (mode-agnostic: the buffer
                    # collects the same signed terms for tanh and normed heads):
                    #   softmax input = scale_term * [ logit term + gate term ]
                    if _term_stats.get("qk_m") is not None:
                        log_dict["train/logit_term/mean"] = _term_stats["qk_m"]
                        log_dict["train/logit_term/var"]  = _term_stats["qk_v"]
                        log_dict["train/gate_term/mean"]  = _term_stats["b_m"]
                        log_dict["train/gate_term/var"]   = _term_stats["b_v"]
                        log_dict["train/softmax_input_term/mean"] = _term_stats["si_m"]
                        log_dict["train/softmax_input_term/var"]  = _term_stats["si_v"]
                    if _term_stats.get("se_m") is not None:
                        log_dict["train/attn_scale_term/mean"] = _term_stats["se_m"]
                        log_dict["train/attn_scale_term/var"]  = _term_stats["se_v"]
                    _g_n = gate.module if hasattr(gate, "module") else gate
                    _rs_ = getattr(_g_n, "_last_rms_stats", None)
                    if _rs_ is not None:
                        log_dict["train/rms_bj/mean"] = _rs_["mean"]
                        log_dict["train/rms_bj/var"] = _rs_["var"]
                        log_dict["train/rms_bj/abs_mean"] = _rs_["abs_mean"]
                    _sse_ = torch.nn.functional.softplus(
                        _g_n.attn_scale.detach().float())
                    log_dict["train/attn_scale_softplus/mean"] = _sse_.mean().item()
                    log_dict["train/attn_scale_softplus/var"] = _sse_.var(unbiased=False).item()
                    _ge_ = torch.nn.functional.softplus(
                        _g_n.gate_scale.detach().float())
                    log_dict["train/G_eff/mean"] = _ge_.mean().item()
                    log_dict["train/G_eff/var"] = _ge_.var(unbiased=False).item()
                    if attn_scale_stats is not None:
                        log_dict["train/b_norm_gain"] = float(_g_n.b_norm.weight.item())
                        if _attn_eff_stats is not None:
                            log_dict["train/attn_scale_eff"] = _attn_eff_stats["mean"]
                            log_dict["train/attn_scale_eff_min"] = _attn_eff_stats["min"]
                            log_dict["train/attn_scale_eff_max"] = _attn_eff_stats["max"]
                        log_dict["train/attn_scale"] = attn_scale_stats["mean"]
                        log_dict["train/attn_scale_min"] = attn_scale_stats["min"]
                        log_dict["train/attn_scale_max"] = attn_scale_stats["max"]
                    if gate_scale_stats is not None:
                        log_dict["train/gate_scale"] = gate_scale_stats["mean"]
                        log_dict["train/gate_scale_min"] = gate_scale_stats["min"]
                        log_dict["train/gate_scale_max"] = gate_scale_stats["max"]
                    _wandb_run.log(log_dict, step=step + 1)
            else:
                # Throttled: drop this step's accumulated stat buffers so they
                # neither leak nor go stale before the next diagnostics step.
                _gate_output_raw_buf.clear()
                _flex_attn_stats_buf.clear()
                if getattr(gate, "_noise_stats_buf", None):
                    gate._noise_stats_buf = []

        if _prof_other:
            _logging_ms = (time.perf_counter() - _t_log0) * 1000.0

        step += 1

        # ── Throughput profile line (one per successful UPDATE step) ───────────
        # dt = wall-clock since the previous successful update = steady-state s/it.
        # Reported fields let profile_multinode.sh derive global samples/sec and
        # an approximate tokens/sec: per step the cluster processes
        # world_size × batch_size samples; per-rank tokens ≈ n_art (phase-1) +
        # M × K_mean (phase-2).  Only rank 0 prints; multiply by world_size for
        # the global rate (ranks see the same length distribution in expectation).
        if args.profile and os.environ.get("TORCH_PROFILE_AT") and is_main:
            _tp_at = int(os.environ["TORCH_PROFILE_AT"])
            if step == _tp_at:
                _torch_prof = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    with_stack=True)
                _torch_prof.__enter__()
            elif step == _tp_at + 3:
                torch.cuda.synchronize()
                _torch_prof.__exit__(None, None, None)
                print(_torch_prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=45), flush=True)
                print(_torch_prof.key_averages(group_by_stack_n=6).table(sort_by="self_cpu_time_total", row_limit=25, max_src_column_width=160), flush=True)
                _torch_prof.export_chrome_trace(os.environ.get("TORCH_PROFILE_OUT", "/tmp/gate_step_trace.json"))
        if args.profile:
            _now = time.perf_counter()
            if _prof_t_prev is not None and is_main:
                _dt = _now - _prof_t_prev
                _samples = world_size * batch_size
                print(f"[profile] step={step} dt_s={_dt:.4f} "
                      f"samples={_samples} world_size={world_size} "
                      f"n_art={avg_n_art:.0f} K_mean={avg_K_sampled:.0f} "
                      f"M={_mixed_k_m_cfg} "
                      f"seq={len(batch_data[0]['input_ids'])} "
                      f"K_list={','.join(str(int(k)) for k in K_list_step)}", flush=True)
            _prof_t_prev = _now

            # ── Per-phase breakdown (opt-in; PROFILE_PHASES=1) ─────────────────
            # Read the events stashed by the detach forward.  ONE sync (rank 0
            # only) resolves all elapsed_time reads; emits Phase-1 fwd/bwd, gate,
            # and per-frac Phase-2 fwd+bwd times + the Phase-1/Phase-2 context
            # lengths.  Per-K fields follow this step's K_list order.
            _pt = getattr(_gate_forward_mod, "PHASE_TIMING", None)
            if _pt is not None and is_main:
                _ev = _pt["events"]
                torch.cuda.synchronize()
                _p1fwd = _ev["start"].elapsed_time(_ev["p1fwd"])
                _gate_ms = _ev["p1fwd"].elapsed_time(_ev["gate"])
                # Per-K Phase-2 split: prev→k_fwd = fwd (layers+lm_head+loss),
                # k_fwd→k = the per-K backward.
                _prev = _ev["gate"]
                _p2, _p2f, _p2b = [], [], []
                for _ekf, _ek in zip(_ev["k_fwd"], _ev["k"]):
                    _fwd = _prev.elapsed_time(_ekf)
                    _bwd = _ekf.elapsed_time(_ek)
                    _p2f.append(_fwd); _p2b.append(_bwd); _p2.append(_fwd + _bwd)
                    _prev = _ek
                _p1bwd = _ev["p1bwd_s"].elapsed_time(_ev["p1bwd_e"])
                _fmt = lambda xs: ",".join(f"{x:.3f}" for x in xs)
                print(f"[phase_profile] step={step} "
                      f"p1_fwd_ms={_p1fwd:.3f} gate_ms={_gate_ms:.3f} "
                      f"p1_bwd_ms={_p1bwd:.3f} p1_ctx={_pt['p1_ctx']} "
                      f"n_art={_pt['n_art']:.0f} "
                      f"K={','.join(str(k) for k in _pt['K'])} "
                      f"p2_ms={_fmt(_p2)} "
                      f"p2_fwd_ms={_fmt(_p2f)} p2_bwd_ms={_fmt(_p2b)} "
                      f"p2_ctx={','.join(str(c) for c in _pt['p2_ctx'])} "
                      f"straggler_ms={_straggler_ms:.3f} "
                      f"allreduce_ms={_allreduce_ms:.3f} "
                      f"optim_ms={_optim_ms:.3f} "
                      f"dataprep_ms={_data_prep_ms:.3f} "
                      f"fwdbwd_wall_ms={_fwd_bwd_wall_ms:.3f} "
                      f"clip_ms={_clip_ms:.3f} "
                      f"logging_ms={_logging_ms:.3f}",
                      flush=True)
            _gate_forward_mod.PHASE_TIMING = None

        # No periodic cache clearing: single-dataset runs have a naturally
        # bounded shape distribution.

        # ── Eval ─────────────────────────────────────────────────────────────
        # All ranks participate so evaluate() can distribute inference across GPUs.
        # _do_eval is a closure defined above the loop; called from both the
        # success path here AND the all-skip path so eval_freq isn't
        # probabilistically dropped by OOM-skip races (same fix as checkpoint).
        if step % args.eval_freq == 0:
            _do_eval()

        # ── Checkpoint ───────────────────────────────────────────────────────
        # _do_checkpoint_save guards on is_main internally; defined as a closure
        # before the loop so the all-skip branch (line ~4063) can call it too.
        if step % args.checkpoint_freq == 0:
            _do_checkpoint_save(step)

    if is_main:
        pbar.close()
    elapsed = time.time() - t0
    if is_main:
        print(f"\n[train] Done. {step} steps in {elapsed / 60:.1f} min.")

    if is_main and not args.profile:
        # ── Save final gate ───────────────────────────────────────────────────
        # Skipped under --profile so throughput/boundary probes don't overwrite a
        # real run's gate_final.pt / lora_final.pt / params.json when they share an
        # output dir (dir name doesn't encode frac_max_k or profile flags).
        final_path = os.path.join(args.output_dir, "gate_final.pt")
        save_gate(gate, final_path, metadata={
            "step": step,
            "gate_layer": args.gate_layer,
            "phase2_len_list": args.phase2_len_list,
            "phase2_bucket_lists": args.phase2_bucket_lists,
            "length_buckets": getattr(args, "length_buckets", None),
            "base_model": args.base_model,
            "task": args.task,
            "dataset": args.dataset,
            "train_samples": len(tokenized_train),
            "adapter_type": args.adapter_type if _lora_enabled else None,
            "lora_layer_ids": lora_layer_ids if _lora_enabled else None,
            "lora_rank": args.lora_rank if (_lora_enabled and args.adapter_type == "lora") else None,
            "lora_alpha": args.lora_alpha if (_lora_enabled and args.adapter_type == "lora") else None,
            "lora_targets": args.lora_targets if (_lora_enabled and args.adapter_type == "lora") else None,
            "lora_start_layer": lora_start_layer if _lora_enabled else None,
        })
        print(f"[train] Final gate saved to {final_path}")
        if _lora_enabled:
            lora_final_path = os.path.join(args.output_dir, "lora_final.pt")
            torch.save(collect_lora_state_dict(model), lora_final_path)
            print(f"[train] LoRA weights saved to {lora_final_path}")

    # Final eval (all ranks participate).  Skipped under --profile (this decode-
    # heavy sweep is pure overhead for a throughput run and adds ~10 min after
    # the loop).  Guard is identical on every rank → no barrier desync.
    if eval_samples and not args.profile:
        if world_size > 1:
            dist.barrier()
        gate.eval()
        if is_main:
            eval_steps.append(step)
            _wandb_log = {"eval/epoch": step * (args.batch_size * world_size) / len(tokenized_train)}
            _gate_output_eval_buf.clear()
            _gate_eval_capture[0] = True
        for _ki, _k in enumerate(k_eval_list):
            if args.gate_selection == "threshold":
                (gate.module if hasattr(gate, "module") else gate).tau = float(_k)
            _per_group, _resp, _per_ds = _evaluate_per_group(
                model, gate, tokenizer,
                eval_samples_by_group, tasks_by_name, ds_task,
                _k, args,
                rank=rank, world_size=world_size, device=str(device),
                log_examples=args.log_examples if _ki == 0 else 0,
            )
            if is_main:
                _primary = next(iter(_per_group))
                _pm = _per_group[_primary]
                eval_per_k_accs[_k].append(_pm["accuracy"])
                eval_per_k_losses[_k].append(_pm["loss"])
                if _k == k_primary:
                    _note_best_group_acc(_per_group)
                _summary = ", ".join(
                    f"{_g}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                    f" loss={m['loss']:.4f}"
                    f" tok={m['n_tokens_pruned']:.0f}/{m['n_tokens_full']:.0f}"
                    f" art={m['n_tokens_pruned_art']:.0f}/{m['n_tokens_full_art']:.0f}"
                    f" (n={m['n_samples']})"
                    for _g, m in _per_group.items()
                )
                _ds_summary = ", ".join(
                    f"{_d}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                    for _d, m in _per_ds.items()
                )
                _label = ("tau" if args.gate_selection == "threshold" else "K")
                print(f"[train] Final eval/cat  {_label}={_k}  {_summary}")
                print(f"[train] Final eval/ds   {_label}={_k}  {_ds_summary}")
                _emit_eval_wandb_keys(_wandb_log, _per_group, _per_ds, _k,
                                      _acc_metric_tag, _judge_active)
                if _ki == 0:
                    _log_responses(_resp, step, args.output_dir, _wandb_run)
        # Restore the training τ after the final eval-sweep loop.  This is
        # the last opportunity the live gate is used before save, so the
        # checkpointed gate carries args.tau (not the last sweep entry).
        if args.gate_selection == "threshold":
            (gate.module if hasattr(gate, "module") else gate).tau = float(args.tau)

        # Drain final-eval gate-output capture → pct_above_tau.  SWA eval below
        # is intentionally not captured (different weights).
        if is_main:
            _gate_eval_capture[0] = False
            if _gate_output_eval_buf:
                _agg_w_eval = (
                    (gate.module if hasattr(gate, "module") else gate).agg_weights.detach().float().cpu()
                    if hasattr(gate.module if hasattr(gate, "module") else gate, "agg_weights")
                    and (gate.module if hasattr(gate, "module") else gate).agg_weights is not None
                    else None
                )
                _pct_list = []
                for _t in _gate_output_eval_buf:
                    _ts = _t.mean(dim=(-2, -1)) if _t.dim() == 3 else ((_t * _agg_w_eval).sum(-1) if (_t.dim() == 2 and _agg_w_eval is not None) else _t)
                    _pct_list.append(
                        (torch.sigmoid(_ts) >= float(args.tau)).float().mean().item()
                    )
                _wandb_log["eval/gate_pct_above_tau"] = sum(_pct_list) / len(_pct_list)
                # b_j distribution metrics for the final eval.
                _log_bj_distribution(
                    _wandb_log,
                    _gate_output_eval_buf,
                    n_K_eval=max(1, len(k_eval_list)),
                    k_eval_values=list(k_eval_list),
                    n_tracked=4,
                )
            _gate_output_eval_buf.clear()

        # ── Final SWA eval + save (rank 0 only) ───────────────────────────────
        if is_main:
            if swa_gate is not None:
                swa_gate.eval()
                # Swap the averaged adapter in for the final SWA eval too (lora_final.pt =
                # the LIVE adapter was already saved above, so this is safe); restore after.
                _live_adapter_sd = None
                if _lora_enabled and swa_adapter_sd is not None:
                    _live_adapter_sd = {k: v.detach().clone()
                                        for k, v in collect_lora_state_dict(model).items()}
                    load_lora_state_dict(model, swa_adapter_sd)
                try:
                    swa_eval_steps.append(step)
                    for _ki, _k in enumerate(k_eval_list):
                        if args.gate_selection == "threshold":
                            (swa_gate.module if hasattr(swa_gate, "module") else swa_gate).tau = float(_k)
                        _per_group_swa, _, _per_ds_swa = _evaluate_per_group(
                            model, swa_gate, tokenizer,
                            eval_samples_by_group, tasks_by_name, ds_task,
                            _k, args,
                            rank=0, world_size=1, device=str(device),
                            log_examples=0,
                        )
                        _primary_swa = next(iter(_per_group_swa))
                        _spm = _per_group_swa[_primary_swa]
                        swa_eval_per_k_accs[_k].append(_spm["accuracy"])
                        swa_eval_per_k_losses[_k].append(_spm["loss"])
                        _swa_summary = ", ".join(
                            f"{_g}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                            f" loss={m['loss']:.4f}"
                            for _g, m in _per_group_swa.items()
                        )
                        _ds_summary_swa = ", ".join(
                            f"{_d}: {m['acc_metric_tag']}={m['accuracy']:.3f}"
                            for _d, m in _per_ds_swa.items()
                        )
                        print(f"[train] Final SWA eval/cat  K={_k}  {_swa_summary}  (n={swa_n} updates)")
                        print(f"[train] Final SWA eval/ds   K={_k}  {_ds_summary_swa}")
                        _emit_eval_wandb_keys(_wandb_log, _per_group_swa, _per_ds_swa, _k,
                                              _acc_metric_tag, _judge_active,
                                              swa=True)
                    # Restore the SWA gate's τ before save so the persisted
                    # swa_gate_final.pt carries args.tau, not the last sweep entry.
                    if args.gate_selection == "threshold":
                        (swa_gate.module if hasattr(swa_gate, "module") else swa_gate).tau = float(args.tau)
                finally:
                    if _live_adapter_sd is not None:
                        load_lora_state_dict(model, _live_adapter_sd)

            # Single wandb.log call so base + SWA metrics appear in the same step
            if _wandb_run is not None:
                _wandb_run.log(_add_baselines(_wandb_log), step=step)

            if swa_gate is not None:
                swa_final_path = os.path.join(args.output_dir, "swa_gate_final.pt")
                save_gate(swa_gate, swa_final_path, metadata={
                    "step": step,
                    "swa_n": swa_n,
                    "gate_layer": args.gate_layer,
                    "phase2_len_list": args.phase2_len_list,
                    "phase2_bucket_lists": args.phase2_bucket_lists,
                    "length_buckets": getattr(args, "length_buckets", None),
                    "base_model": args.base_model,
                    "task": args.task,
                    "dataset": args.dataset,
                    "adapter_type": args.adapter_type if _lora_enabled else None,
                    "lora_layer_ids": lora_layer_ids if _lora_enabled else None,
                    "lora_rank": args.lora_rank if (_lora_enabled and args.adapter_type == "lora") else None,
                    "lora_alpha": args.lora_alpha if (_lora_enabled and args.adapter_type == "lora") else None,
                    "lora_targets": args.lora_targets if (_lora_enabled and args.adapter_type == "lora") else None,
                    "lora_start_layer": lora_start_layer if _lora_enabled else None,
                })
                print(f"[train] SWA gate saved to {swa_final_path}  (n={swa_n} updates)")
                # Deployable SWA needs the averaged adapter too (delta ΔW / LoRA), not just
                # the averaged gate — pair swa_gate_final.pt with swa_lora_final.pt at eval.
                if _lora_enabled and swa_adapter_sd is not None:
                    swa_lora_path = os.path.join(args.output_dir, "swa_lora_final.pt")
                    torch.save({k: v.detach().cpu() for k, v in swa_adapter_sd.items()},
                               swa_lora_path)
                    print(f"[train] SWA adapter saved to {swa_lora_path}  (n={swa_n} updates)")

        gate.train()
        if world_size > 1:
            dist.barrier()

    if is_main:
        _gate_hook_handle.remove()
        _save_curves(train_losses, k_log, eval_accs, eval_losses, eval_steps, args.output_dir,
                     bias_norm_log=bias_norm_log,
                     attn_scale_log=attn_scale_log,
                     eval_per_k_accs=eval_per_k_accs,
                     eval_per_k_losses=eval_per_k_losses,
                     swa_eval_per_k_accs=swa_eval_per_k_accs,
                     swa_eval_per_k_losses=swa_eval_per_k_losses,
                     swa_eval_steps=swa_eval_steps)
        print(f"[train] Curves saved to {args.output_dir}/")

        if _wandb_run is not None:
            _wandb_run.finish()

    if world_size > 1:
        dist.destroy_process_group()


def _save_curves(
    train_losses: List[float],
    k_log: List[tuple],          # list of (step, K_sampled, K_clamped, n_art)
    eval_accs: List[float],
    eval_losses: List[float],
    eval_steps: List[int],
    output_dir: str,
    bias_norm_log: Optional[List[tuple]] = None,    # list of (step, gate_output_norm)
    attn_scale_log: Optional[List[tuple]] = None,   # list of (step, attn_scale_val)
    eval_per_k_accs: Optional[dict] = None,      # {K: [acc, ...]}
    eval_per_k_losses: Optional[dict] = None,    # {K: [loss, ...]}
    swa_eval_per_k_accs: Optional[dict] = None,  # {K: [acc, ...]} SWA
    swa_eval_per_k_losses: Optional[dict] = None,# {K: [loss, ...]} SWA
    swa_eval_steps: Optional[list] = None,        # steps where SWA eval ran (subset of eval_steps)
) -> None:
    """Write loss/accuracy/K CSVs and (if matplotlib available) plots."""
    with open(os.path.join(output_dir, "train_loss.csv"), "w") as f:
        f.write("step,loss\n")
        for i, l in enumerate(train_losses):
            f.write(f"{i},{l}\n")

    with open(os.path.join(output_dir, "eval_acc.csv"), "w") as f:
        f.write("step,accuracy,eval_loss\n")
        for s, a, el in zip(eval_steps, eval_accs, eval_losses):
            f.write(f"{s},{a},{el}\n")

    # Per-K CSV (written whenever multi-K data is available)
    if eval_per_k_accs and len(eval_per_k_accs) > 0:
        k_vals_sorted = sorted(eval_per_k_accs.keys())
        _has_swa_csv = bool(swa_eval_per_k_accs)
        with open(os.path.join(output_dir, "eval_acc_per_k.csv"), "w") as f:
            cols = [f"accuracy_K{k},loss_K{k}" for k in k_vals_sorted]
            if _has_swa_csv:
                cols += [f"swa_accuracy_K{k},swa_loss_K{k}" for k in k_vals_sorted]
            f.write("step," + ",".join(cols) + "\n")
            for i, s in enumerate(eval_steps):
                row = [str(s)]
                for k in k_vals_sorted:
                    accs_k   = eval_per_k_accs.get(k, [])
                    losses_k = (eval_per_k_losses or {}).get(k, [])
                    row += [
                        f"{accs_k[i]:.6f}"   if i < len(accs_k)   else "",
                        f"{losses_k[i]:.6f}" if i < len(losses_k) else "",
                    ]
                if _has_swa_csv:
                    for k in k_vals_sorted:
                        s_accs   = swa_eval_per_k_accs.get(k, [])
                        s_losses = (swa_eval_per_k_losses or {}).get(k, [])
                        row += [
                            f"{s_accs[i]:.6f}"   if i < len(s_accs)   else "",
                            f"{s_losses[i]:.6f}" if i < len(s_losses) else "",
                        ]
                f.write(",".join(row) + "\n")

    with open(os.path.join(output_dir, "k_log.csv"), "w") as f:
        f.write("step,K_sampled,K_clamped,n_art,compression_ratio\n")
        for entry in k_log:
            step, ks, kc = entry[0], entry[1], entry[2]
            na = entry[3] if len(entry) > 3 else ""
            compression = f"{kc / entry[3]:.4f}" if len(entry) > 3 and entry[3] > 0 else ""
            f.write(f"{step},{ks},{kc},{na},{compression}\n")

    if bias_norm_log:
        with open(os.path.join(output_dir, "bias_norm.csv"), "w") as f:
            f.write("step,bias_norm\n")
            for step, bn in bias_norm_log:
                f.write(f"{step},{bn}\n")

    if attn_scale_log:
        with open(os.path.join(output_dir, "attn_scale.csv"), "w") as f:
            f.write("step,attn_scale\n")
            for step, val in attn_scale_log:
                f.write(f"{step},{val}\n")

    if not _HAVE_MPL:
        return

    has_eval = bool(eval_steps)
    has_bias = bool(bias_norm_log)
    has_attn_scale = bool(attn_scale_log)
    ncols = (4 if has_eval else 2) + (1 if has_bias else 0) + (1 if has_attn_scale else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

    def _smooth(vals, window):
        return [
            sum(vals[max(0, i - window): i + 1]) / len(vals[max(0, i - window): i + 1])
            for i in range(len(vals))
        ]

    # ── Training loss (smoothed) ─────────────────────────────────────────────
    if train_losses:
        window = max(1, len(train_losses) // 100)
        smoothed = _smooth(train_losses, window)
        loss_steps = [e[0] for e in k_log] if k_log else list(range(len(train_losses)))
        axes[0].plot(loss_steps[:len(smoothed)], smoothed, linewidth=1)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss (smoothed)")
    axes[0].grid(True, alpha=0.3)

    # ── K and article size (single y-axis, tokens) ───────────────────────────
    # Both K_clamped and n_art are in tokens, so share the same axis.
    # Each point is the per-step average across all samples in the batch.
    if k_log:
        k_steps  = [e[0] for e in k_log]
        k_vals   = [e[2] for e in k_log]          # K_clamped (avg over batch)
        art_vals = [e[3] if len(e) > 3 else None for e in k_log]
        k_win    = max(1, len(k_vals) // 100)

        axes[1].plot(k_steps, _smooth(k_vals, k_win),
                     linewidth=1, color="tab:orange", label="K kept (avg)")

        if any(v is not None for v in art_vals):
            art_clean = [v if v is not None else 0 for v in art_vals]
            axes[1].plot(k_steps, _smooth(art_clean, k_win),
                         linewidth=1, color="tab:blue", linestyle="--",
                         label="n_art (avg)", alpha=0.8)

            compression_ratios = [k / a if a > 0 else 0
                                   for k, a in zip(k_vals, art_clean)]
            avg_compression = sum(compression_ratios) / len(compression_ratios)
            axes[1].text(0.02, 0.98, f"Avg K/n_art: {avg_compression:.2%}",
                         transform=axes[1].transAxes, fontsize=9,
                         verticalalignment="top",
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Tokens")
        axes[1].set_title("K kept vs Article Size (avg per step)")
        axes[1].legend(fontsize=8, loc="upper right")
        axes[1].grid(True, alpha=0.3)

    # ── Eval accuracy ────────────────────────────────────────────────────────
    _multi_k  = eval_per_k_accs is not None and len(eval_per_k_accs) > 1
    _has_swa  = bool(swa_eval_per_k_accs)
    _k_colors = plt.cm.tab10.colors if _HAVE_MPL else []
    # SWA eval only starts after warmup, so it has fewer points than eval_steps.
    # Use a dedicated list so SWA points are plotted at their correct x positions.
    _swa_eval_steps = swa_eval_steps if swa_eval_steps else eval_steps
    if has_eval:
        if _multi_k:
            for _ci, (_k, _accs_k) in enumerate(sorted(eval_per_k_accs.items())):
                _col = _k_colors[_ci % len(_k_colors)]
                axes[2].plot(eval_steps[:len(_accs_k)], _accs_k,
                             marker="o", linewidth=1.5, color=_col, label=f"K={_k}")
                if _has_swa:
                    _s_accs = swa_eval_per_k_accs.get(_k, [])
                    axes[2].plot(_swa_eval_steps[:len(_s_accs)], _s_accs,
                                 marker="s", linewidth=1.5, color=_col,
                                 linestyle="--", alpha=0.7, label=f"SWA K={_k}")
            axes[2].legend(fontsize=7, loc="lower right")
        else:
            axes[2].plot(eval_steps, eval_accs, marker="o", linewidth=1.5,
                         color="tab:green", label="base")
            if _has_swa:
                # `k_eval_list` is a local var of train() not in scope here; derive
                # the primary K from the dict we already have.
                _primary_k = next(iter(eval_per_k_accs)) if eval_per_k_accs else None
                _s_accs = swa_eval_per_k_accs.get(_primary_k, [])
                if _s_accs:
                    axes[2].plot(_swa_eval_steps[:len(_s_accs)], _s_accs,
                                 marker="s", linewidth=1.5, color="tab:green",
                                 linestyle="--", alpha=0.7, label="SWA")
            axes[2].legend(fontsize=8, loc="lower right")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Task Accuracy")
        axes[2].set_title("Eval Accuracy (per K, solid=base dashed=SWA)")
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)

    # ── Eval loss ────────────────────────────────────────────────────────────
    if has_eval and eval_losses:
        if _multi_k and eval_per_k_losses:
            for _ci, (_k, _losses_k) in enumerate(sorted(eval_per_k_losses.items())):
                _col = _k_colors[_ci % len(_k_colors)]
                axes[3].plot(eval_steps[:len(_losses_k)], _losses_k,
                             marker="o", linewidth=1.5, color=_col, label=f"K={_k}")
                if _has_swa:
                    _s_losses = (swa_eval_per_k_losses or {}).get(_k, [])
                    axes[3].plot(_swa_eval_steps[:len(_s_losses)], _s_losses,
                                 marker="s", linewidth=1.5, color=_col,
                                 linestyle="--", alpha=0.7, label=f"SWA K={_k}")
            axes[3].legend(fontsize=7, loc="upper right")
        else:
            axes[3].plot(eval_steps, eval_losses, marker="o", linewidth=1.5,
                         color="tab:red", label="base")
            if _has_swa:
                # `k_eval_list` is a local var of train() not in scope here; derive
                # the primary K from the dict we already have.
                _primary_k = next(iter(eval_per_k_accs)) if eval_per_k_accs else None
                _s_losses = (swa_eval_per_k_losses or {}).get(_primary_k, [])
                if _s_losses:
                    axes[3].plot(_swa_eval_steps[:len(_s_losses)], _s_losses,
                                 marker="s", linewidth=1.5, color="tab:red",
                                 linestyle="--", alpha=0.7, label="SWA")
            axes[3].legend(fontsize=8, loc="upper right")
        axes[3].set_xlabel("Step")
        axes[3].set_ylabel("Cross-Entropy Loss")
        axes[3].set_title("Eval Loss (teacher-forced, solid=base dashed=SWA)")
        axes[3].grid(True, alpha=0.3)

    # ── Gate bias norm ───────────────────────────────────────────────────────
    bias_ax_idx = (4 if has_eval else 2)
    if has_bias:
        bn_steps = [e[0] for e in bias_norm_log]
        bn_vals  = [e[1] for e in bias_norm_log]
        bn_win   = max(1, len(bn_vals) // 100)
        axes[bias_ax_idx].plot(bn_steps, _smooth(bn_vals, bn_win),
                               linewidth=1, color="tab:purple")
        axes[bias_ax_idx].set_xlabel("Step")
        axes[bias_ax_idx].set_ylabel("L2 Norm")
        axes[bias_ax_idx].set_title("Gate Output Norm (‖b_j‖ avg)")
        axes[bias_ax_idx].grid(True, alpha=0.3)

    # ── Learned attention scale ───────────────────────────────────────────────
    if has_attn_scale:
        as_steps = [e[0] for e in attn_scale_log]
        as_vals  = [e[1] for e in attn_scale_log]
        as_win   = max(1, len(as_vals) // 100)
        ax_as = axes[bias_ax_idx + (1 if has_bias else 0)]
        ax_as.plot(as_steps, _smooth(as_vals, as_win), linewidth=1, color="tab:brown")
        ax_as.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_as.set_xlabel("Step")
        ax_as.set_ylabel("Scale value")
        ax_as.set_title("Learned Attention Scale (attn_scale)")
        ax_as.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=100)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """The trainer's full CLI. Exposed so tokengate.trainer (the Python
    training interface) can validate the argument lists it generates."""
    parser = argparse.ArgumentParser(
        description="Train E2E gate for layer-based KV token pruning."
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", type=str, default="biodex",
                        help="Dataset name, or comma-separated list of datasets for "
                             "multitask training (must align 1:1 with --task).")
    parser.add_argument("--task", type=str, default="biodex_reactions",
                        help="Task name, or comma-separated list of tasks for multitask "
                             "training (must align 1:1 with --dataset).")
    parser.add_argument("--shuffle-epochs", action="store_true",
                        help="Sample training data by iterating a per-epoch shuffled "
                             "permutation of the pool (without replacement) instead of the "
                             "default i.i.d. with-replacement draw.  The permutation is "
                             "seeded identically across ranks and sharded (rank r takes "
                             "perm[r::world_size]) so the ranks' shards are disjoint and "
                             "their union covers the pool exactly once per epoch; the "
                             "permutation is regenerated (reshuffled) at each epoch "
                             "boundary.  Mutually exclusive with --length-buckets "
                             "(it defines its own sampling distribution).")
    parser.add_argument("--length-buckets", type=str, default=None,
                        help="Comma-separated full-sequence length boundaries, e.g. "
                             "\"4000,8000,16000,32000\".  Creates length buckets at "
                             "[0,4000), [4000,8000), [8000,16000), [16000,32000), [32000,+inf).  "
                             "Every training step, all ranks deterministically pick the same "
                             "bucket (weighted by per-bucket sample count) and then draw a "
                             "sample uniformly within it.  Eliminates the DDP straggler "
                             "effect caused by cross-rank sequence-length variance in "
                             "multitask / pretrain runs while keeping samples mixed across "
                             "datasets.  Default None disables bucketing — preserves the "
                             "previous behaviour (uniform-over-pool / dataset-weights / "
                             "task-switch).  Recommended for any multitask or pretrain run "
                             "with > 1 dataset.")
    parser.add_argument("--eval-alternate-datasets", action="store_true",
                        help="Rotate which dataset is evaluated each eval cycle, instead "
                             "of evaluating ALL datasets every cycle.  At eval cycle k, "
                             "only dataset k%%N is evaluated (N = num datasets).  Each "
                             "dataset's W&B history is sparser by Nx but the wall-clock "
                             "per eval cycle drops by Nx.  Big win for pretrain (N=7) and "
                             "multitask (N=2).  Default False = eval all datasets every cycle.")
    parser.add_argument("--eval-k-stride", type=int, default=1,
                        help="At every eval cycle, always evaluate at K_eval[0] (the "
                             "primary K).  Every N cycles, evaluate at ALL --k-eval values.  "
                             "Default 1 = eval all K values every cycle.  Set to e.g. 5 to "
                             "cut K-side eval cost ~3x (5 cycles of 1 K + 1 cycle of 3 K) "
                             "with sparser non-primary-K W&B lines.")
    parser.add_argument("--attn-backends", type=str, default="fa2", choices=["flex", "flash", "fa2"],
                        help="attention kernels for the patched layers: fa2 (the DEFAULT and the final-run "
                             "configuration: Phase 1 as FlashAttention-2 through SDPA, Phase 2 on the "
                             "FlashAttention-2 fork with a per-key bias, kernels/fa2bias); flash "
                             "(FlashAttention-2 for Phase 1, the bias folded into extra q/k columns for "
                             "Phase 2; same math, no external kernel); flex (FlexAttention score_mod bias — "
                             "training's original kernel, slow, and degraded at phase-2 lengths below its "
                             "128-token block)")
    parser.add_argument("--pack-phase2-max-tokens", type=int, default=0,
                        help="Packed Phase 2: run the Phase-2 targets of a sample as ONE variable-length "
                             "launch per layer when their packed length is <= this many tokens (0 = off, "
                             "the targets run one after the other).  Needs --attn-backends fa2 with the "
                             "varlen key-bias build (kernels/fa2bias), batch size 1, no Phase-2 "
                             "activation checkpointing.  Same losses and gradients as the sequential path "
                             "(tests/test_packed_phase2_grad.py).")
    parser.add_argument("--compile-phase2", action="store_true",
                        help="EXPERIMENTAL: wrap each phase-2 transformer layer with "
                             "torch.compile(dynamic=True, fullgraph=False).  FlexAttention "
                             "stays compiled internally; the outer compile attempts to "
                             "fuse the QKV projection, MLP and norm.  First step at each "
                             "unique seq-len bucket will pay a 10-60 s compile cost; steady "
                             "state is expected 10-30 %% faster on the phase-2 forward.  "
                             "Default off until 8-GPU integration testing validates "
                             "correctness with LoRA + FlexAttention nesting.")
    parser.add_argument("--use-packed-sequences", action="store_true",
                        help="EXPERIMENTAL: pack the per-step batch into a single packed "
                             "batch element with segment-causal attention (instead of "
                             "padded B=N batched).  Eliminates padding waste when batch_size "
                             "> 1 with variable lengths.  Numerically equivalent to "
                             "forward_with_gate_batched (argmax-match verified) modulo bf16 "
                             "accumulation noise.  Production scope (Phase-B v1): "
                             "frozen_backbone=True, no LoRA, scalar gate bias.  Default off "
                             "pending wider validation on the real training configs.")
    parser.add_argument("--packed-max-total-len", type=int, default=30000,
                        help="When --use-packed-sequences is set, the maximum total token "
                             "count per packed batch element.  FFD packer ensures no "
                             "packed element exceeds this length.  Default 30000.")
    # ── Pre-prepped tokenized data (the only data path) ──────────────────────
    # The trainer loads tokenized .pkl / packed stores written by
    # datasets/<ds>/<task>/prepare.py; it never builds data itself.
    parser.add_argument("--pretrain-data-root", type=str,
                        default="datasets",
                        help="Root for the pre-prepped tokenized data.")
    parser.add_argument("--pretrain-packed", action="store_true",
                        help="Load training tokens from the packed, memory-mapped "
                             "store (tokenized/<model>/packed/{train_tokens,train_meta}"
                             ".npy, written by tokengate.datasets.pack) instead of "
                             "unpickling train.pkl.  The "
                             "packed store loads in milliseconds, uses ~10x less RAM, "
                             "and is shared across ranks via the OS page cache.  Eval "
                             "(val.pkl) is unaffected.")
    parser.add_argument("--target-balanced-map", type=str, default=None,
                        help="Path to an eval-target→training-pool YAML "
                             "(eval_target_train_map_v16_2wiki_only.yaml). Enables TARGET-BALANCED "
                             "sampling (packed path): each draw picks an eval target uniformly, "
                             "then a (dataset,task) uniformly from that target's pool, then a row "
                             "uniformly — so every eval target gets equal training mass regardless "
                             "of pool row-counts. Datasets in no pool are never sampled. Overrides "
                             "uniform-over-all-rows.")
    parser.add_argument("--gate-full-eval-freq", type=int, default=0,
                        help="every Nth eval cycle, ALSO evaluate the gated model at FULL "
                             "context (K=10M -> all article tokens kept, learned bias applied "
                             "-- the in-training analogue of the benchmark gate_full anchor). "
                             "Logged as eval/gatefull_cat_<grp>/..._Kfull. 0 (default) = off.")
    parser.add_argument("--pretrain-eval-only-pairs", type=str, default="",
                        help="Comma-separated dataset:task pairs to EVALUATE but never "
                             "train on (held-out subsets), e.g. "
                             "'longbench-datasets:longbench_trec,...'.  Their val.pkl is "
                             "loaded into the eval pool and reported in separate "
                             "'<category>_heldout' groups (or their own bare category "
                             "if no trained subset shares that metric family).  NOT "
                             "added to --dataset/--task, so train + pack are untouched.")
    parser.add_argument("--eval-split-manifest", type=str, default="",
                        help="Path to a longbench_eval_split JSON "
                             "({task: {group, val:[ids], test:[ids]}}).  When set, the "
                             "in-training eval is restricted to ONLY the subsets in the "
                             "manifest (LongBench), keeping only each subset's `val` row "
                             "ids; every other eval row (non-LongBench subsets AND the "
                             "reserved `test` rows) is dropped.  This gives a clean "
                             "val/test split of the test-only LongBench benchmark so "
                             "in-training monitoring never touches the final-benchmark "
                             "rows.  REQUIRED whenever eval rows are loaded: the "
                             "manifest is the explicit per-dataset val/test size "
                             "mechanism.")
    parser.add_argument("--eval-split-keep-untracked", action="store_true",
                        help="Modifier for --eval-split-manifest: also KEEP the val rows "
                             "of trained subsets that are NOT in the split manifest (e.g. "
                             "the few-shot classification packs), instead of dropping them. "
                             "LongBench subsets still use only their `val` partition (test "
                             "reserved).  Lets an in-distribution eval (a held-out val split "
                             "of the trained data) run ALONGSIDE the exact LongBench val "
                             "partition for held-out subsets like trec.")
    parser.add_argument("--eval-test-full-tasks", type=str, nargs="+", default=None,
                        help="Modifier for --eval-split-manifest: ALSO evaluate the full "
                             "reserved `test` partition of each listed subset, as an extra "
                             "eval group. Entry form `task` or `task:wandb_group` — the "
                             "group defaults to `<task>_test`; a named group must not "
                             "already exist (free the name via --eval-group-override). "
                             "The whole test set is scored at "
                             "every eval point, with the same randomk/SWA/baseline "
                             "overlays as every other group.  Deliberately breaks the "
                             "'test rows never touched during training' rule for the "
                             "listed subsets — use only on diagnostic runs whose final "
                             "numbers will not be reported on those subsets.")
    parser.add_argument("--eval-ind-train-groups", type=str, nargs="+", default=None,
                        help="Extra IN-DISTRIBUTION eval group(s) built from TRAINING rows. "
                             "Entry form `group:N:ds/task[,ds/task...]` — pools the listed "
                             "pairs' tokenized train rows (within the eval article range), "
                             "seeded-samples N uniformly over the pooled rows (mirroring the "
                             "within-target training draw), and evaluates them as an uncapped "
                             "group at every eval point. The sampled rows REMAIN in training "
                             "(this measures train-distribution generation quality incl. "
                             "memorization, not held-out generalization).")
    parser.add_argument("--eval-group-override", type=str, default="",
                        help="Pin eval subsets to reported metric groups regardless of "
                             "what is trained: 'group:task1,task2;group2:task3'.  Lets a "
                             "harness reproduce v10's exact groups (e.g. "
                             "'qa:longbench_hotpotqa,longbench_2wikimqa,longbench_musique,"
                             "longbench_qasper,longbench_narrativeqa;qa_heldout:"
                             "longbench_multifieldqa_en,longbench_triviaqa') while those "
                             "subsets are held OUT of training to measure transfer.")
    parser.add_argument("--gate-layer", type=int, required=True,
                        help="Transformer layer at which to apply the gate.")
    parser.add_argument("--k-eval", type=int, nargs="+", default=None,
                        help="K values to use during evaluation. Each value runs a separate "
                             "eval pass and is logged as eval/accuracy_{judge|F1}_K{k} and eval/loss_K{k} "
                             "in W&B. Required (no default).")
    parser.add_argument("--phase2-bucket-lists", type=str, default=None,
                        help="Per-length-bucket PHASE_2_LEN_LISTs (requires --length-"
                             "buckets).  Semicolon-separated, one list per bucket in "
                             "--length-buckets interval order (incl. the implicit [0,first) "
                             "and [last,inf) ranges), each comma-separated ints.  Each step "
                             "the sampler draws one bucket (all ranks agree), so all samples "
                             "reach that bucket's targets → homogeneous Phase-2 with a "
                             "bucket-appropriate top (dense gradients on the long bucket, no "
                             "straggler).  Example (--length-buckets 2000,4000): "
                             "'130,210,330,460,580,700;250,400,650,1000,1500,2000;"
                             "350,600,1000,1600,2300,3000'.  Mutually exclusive with "
                             "--phase2-len-list / --phase2-loguniform.")
    parser.add_argument("--phase2-len-list", type=int, nargs="+", default=None,
                        help="FIXED list of M TARGET PHASE-2 SEQUENCE LENGTHS.  Per sample, "
                             "K_i = clamp(L_i - prompt_len, 1, n_art) where prompt_len = "
                             "seq_len - n_art (the non-article tokens: prefix + post-article "
                             "+ answer), so every sample's Phase-2 context (prompt+K+answer) "
                             "equals L_i regardless of article length.  Equalizes Phase-2 "
                             "compute across DP ranks (straggler fix) and inherently caps "
                             "Phase-2 memory at max(L).  M = len(list).  Mutually exclusive "
                             "with --phase2-bucket-lists / --phase2-loguniform.  Eval K "
                             "stays absolute (--k-eval).  "
                             "Example: --phase2-len-list 157 188 250 375 625 1072.")
    parser.add_argument("--long-ctx-factor", type=float, default=1.0,
                        help="YaRN RoPE scaling factor for the base model.  1.0 (default) = no "
                             "scaling (legacy behaviour, max ctx = model.config.max_position_embeddings, "
                             "32768 for Qwen3-8B).  F>1 sets config.rope_scaling = "
                             "{type: yarn, factor: F, original_max_position_embeddings: <model "
                             "default>} before model instantiation, extending effective max ctx to "
                             "F × original.  Use F=4.0 for 128k ctx (LongBench convention).  "
                             "")
    parser.add_argument("--epochs", type=float, default=None,
                        help="Number of epochs (passes over the training set).  "
                             "When set, overrides --steps; actual step count is "
                             "ceil(epochs * n_train / (batch_size * world_size)).")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Total gradient steps.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Hard cap on optimizer steps for cheap profiling/"
                             "smoke runs.  When set, the training loop exits "
                             "cleanly (barrier + destroy_process_group) after "
                             "this many UPDATE steps, regardless of --steps/"
                             "--epochs.  Combine with --profile.")
    parser.add_argument("--profile", action="store_true",
                        help="Throughput-profiling mode: print a per-step "
                             "[profile] timing line (rolling dt, samples, tokens) "
                             "and SKIP eval/checkpoint/SWA so steady-state train "
                             "step time is measured cleanly.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "muon"],
                        help="'adamw' (default) or 'muon' (Muon for 2D weight "
                             "matrices — LoRA A/B — with aux-AdamW for everything "
                             "else; muon.py). Muon always runs with "
                             "adjust_lr_fn='match_rms_adamw', so it shares the "
                             "per-group AdamW lrs (--lr / --lora-lr); there is no "
                             "separate Muon LR.")
    parser.add_argument("--muon-gate", action="store_true",
                        help="With --optimizer muon: also put the gate's own 2D linear "
                             "maps (mlp-branch gate_proj/up_proj, per-head proj) on Muon "
                             "instead of the aux AdamW.  Scale grids, vectors and the "
                             "(1, d) readout stay on AdamW.")
    parser.add_argument("--no-decay-scalars", action="store_true",
                        help="weight_decay=0 for gate_scale, attn_scale, v_attn, "
                             "log_temp_ca and every <2D gate tensor; --weight-decay "
                             "then applies to the weight matrices only.")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Decoupled weight decay applied to EVERY optimizer "
                             "param group (AdamW, and both Muon + aux-AdamW when "
                             "--optimizer muon). Unset keeps the legacy defaults: "
                             "1e-4 for AdamW, 0.0 for the Muon path.")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable HuggingFace gradient checkpointing on the backbone "
                             "for the gated trainer path. Recomputes layer activations on "
                             "backward instead of storing them, dropping peak memory ~30x "
                             "on long sequences. Required when training LoRA on a frozen "
                             "backbone at high K (e.g. K>=8000) where every layer's "
                             "activations would otherwise exceed 80GB. The LoRA-only "
                             "(--no-gate) path already enables this unconditionally.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for training. Use >1 to enable batched training "
                             "with FlexAttention. Recommended: 2-8 for better GPU utilization.")
    parser.add_argument("--max-article-tokens", type=int, default=None,
                        help="Truncate articles to this many tokens when building the "
                             "SFT dataset.  Default: None (no cap; model context window "
                             "is the only limit).  Phase 2 memory scales with K, not "
                             "article length, so very long articles are fine.")
    parser.add_argument("--min-article-tokens", type=int, default=None,
                        help="Drop SFT samples whose tokenized compressible article is "
                             "shorter than this many tokens.  Default: None (no minimum). "
                             "Used in multitask training to filter out trivially short "
                             "articles that would give weak compression signal.")
    parser.add_argument("--eval-min-article-tokens", type=int, default=None,
                        help="Article-length floor applied to EVAL (val) rows only. "
                             "Default None → fall back to --min-article-tokens (coupled). "
                             "Set explicitly to DECOUPLE the eval set from the training "
                             "length window, e.g. train on [0,16000] but keep the eval "
                             "frozen on the full length range for cross-run comparability.")
    parser.add_argument("--eval-max-article-tokens", type=int, default=None,
                        help="Article-length cap applied to EVAL (val) rows only. "
                             "Default None → fall back to --max-article-tokens (coupled). "
                             "Set explicitly (e.g. 30000) to keep the in-training eval set "
                             "frozen while the training length window (--max-article-tokens) "
                             "is tighter.  Dropped, never truncated (same policy).")
    parser.add_argument("--max-answer-tokens", type=int, default=None,
                        help="Drop (never truncate) SFT samples whose tokenized target "
                             "answer exceeds this many tokens.  Default: None (no cap). "
                             "Summarization datasets have long, highly-variable answers; "
                             "the compiled Phase-2 sequence is ~K_kept + n_ans, so that "
                             "variability floods torch.compile/FlexAttention with distinct "
                             "shapes and stalls the first-step compile on multi-GPU.  "
                             "Capping n_ans collapses the shape spread.  Distinct from "
                             "--max-gen-tokens, which only bounds eval-time generation.")
    parser.add_argument("--checkpoint-freq", type=int, default=500,
                        help="Save gate checkpoint every N steps.")
    parser.add_argument("--checkpoint-keep-all-after-warmup", action="store_true",
                        default=False,
                        help="exempt gate_/lora_ step checkpoints with "
                             "step >= --warmup-steps from retention pruning "
                             "(train_state files still pruned by keep-last).")
    parser.add_argument("--checkpoint-keep-last", type=int, default=3,
                        help="Retain only the N most-recent periodic step "
                             "checkpoints (gate_step*.pt + lora_step*.pt).  Older step files are "
                             "deleted at every save to bound disk usage.  "
                             "*_final.pt and swa_gate_final.pt are always "
                             "kept.  Set to a large number to disable last-N "
                             "pruning.")
    parser.add_argument("--checkpoint-keep-best", type=int, default=1,
                        help="Additionally retain the N step-checkpoints with "
                             "the highest eval accuracy (primary K, group per "
                             "--checkpoint-best-group) seen so far (union with "
                             "--checkpoint-keep-last).  0 disables best tracking; "
                             "only last-N are kept.  Default 1 = always preserve "
                             "the single best intermediate checkpoint.")
    parser.add_argument("--checkpoint-best-group", type=str, default=None,
                        help="Eval group whose accuracy ranks --checkpoint-keep-best. "
                             "Default: the group with the most samples (the held-out "
                             "test partition in the isolated runs), NOT the first-"
                             "listed group, which may be a 10-row val split.  An "
                             "unknown name fails at the first eval.")
    parser.add_argument("--save-swa-step-checkpoints", action="store_true",
                        help="At every periodic step-save, ALSO write a deployable "
                             "SWA checkpoint (swa_gate_step*.pt + swa_lora_step*.pt) "
                             "once SWA has started — so any post-warmup eval step can "
                             "be benchmarked directly without extracting swa_* from "
                             "train_state.  Follows the same keep-last/keep-best "
                             "retention as the raw step checkpoints (set "
                             "--checkpoint-keep-last high to keep them all).  ~772MB "
                             "each (delta adapter) — size the checkpoint-freq for disk.")
    parser.add_argument("--eval-freq", type=int, default=200,
                        help="Run eval every N steps.")
    parser.add_argument("--log-examples", type=int, default=5,
                        help="Number of eval responses to log per eval run (0 = none).")
    parser.add_argument("--max-gen-tokens", type=int, default=10,
                        help="Max tokens to generate per sample during eval. "
                             "10 is sufficient for single-word answers (male/female/unknown).")
    parser.add_argument("--eval-gen-batch-size", type=int, default=1,
                        help="Mini-batch size for generation during eval. "
                             "1 = sequential (default). Increase to 4-8 for "
                             "faster eval when K is fixed and GPU memory allows.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-gate", type=str, default=None,
                        help="Path to a gate checkpoint to resume weights from "
                             "(weights only — does NOT restore optimizer / SWA "
                             "/ scheduler / RNG / step counter).  Use "
                             "--resume-from for a true mid-run resume.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from a saved train_state_step*.pt. "
                             "Accepts either an exact file path or a run output "
                             "directory (the latest train_state_step*.pt inside "
                             "is used).  Restores optimizer + scheduler + SWA + "
                             "RNG + W&B run id + step counter, plus the gate / "
                             "FT-backbone / LoRA weights from the sibling "
                             "*_step{N}.pt files.  Critical args (gate_layer, "
                             "K-source lists, dataset, base_model, LoRA config) "
                             "must match the saved checkpoint or training fails "
                             "with an arg-mismatch error.  Mutually exclusive "
                             "with --resume-gate.")
    parser.add_argument("--prune-phase1-kv", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Prune Phase 1 KV cache before Phase 2 runs. "
                             "False (default): Phase 2 sees full Phase 1 KV, "
                             "matching inference behavior.")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Number of linear LR warmup steps before the post-warmup schedule. "
                             "0 = no warmup (default). Typical: 5-10%% of total steps.")
    parser.add_argument("--lr-schedule", type=str, default="cosine",
                        choices=["cosine", "constant"],
                        help="LR schedule after warmup: 'cosine' (default, decays to 0.1×LR) "
                             "or 'constant' (holds LR fixed after warmup).")
    parser.add_argument("--swa-freq", type=int, default=1,
                        help="Stochastic Weight Averaging update frequency (steps). "
                             "SWA starts at --swa-start (default: end of warmup) and averages "
                             "gate weights every N steps thereafter. "
                             "0 or -1 = disabled. "
                             "Works best with --lr-schedule=constant.")
    parser.add_argument("--swa-start", type=int, default=0,
                        help="Absolute training step at which SWA averaging begins. "
                             "0 (default) = start immediately after warmup. "
                             "Set to e.g. half of total steps to let the gate stabilise "
                             "before averaging begins. "
                             "Clamped to max(warmup_steps, swa_start) so SWA never fires "
                             "before warmup ends.")
    parser.add_argument("--swa-eval-freq", type=int, default=1,
                        help="Run the in-training SWA eval only every Nth regular eval cycle. "
                             "1 (default) = every eval (current behavior); 4 = every 4th. "
                             "Under --no-swa-eval-ddp the SWA eval is rank-0 only (ranks 1..N-1 idle "
                             "at a barrier), so on multi-GPU it can dominate per-eval wall time; "
                             "throttling buys back most of that overhead without losing the SWA "
                             "curve. The final post-training SWA eval is unaffected by this flag.")
    parser.add_argument("--swa-eval-ddp", action=argparse.BooleanOptionalAction, default=True,
                        help="Shard the in-training SWA eval across ALL ranks (DP), like the live "
                             "eval, instead of running it on rank 0 alone while ranks 1..N-1 idle at "
                             "a barrier. Every rank holds a bit-identical SWA average (live weights "
                             "are all_reduce-synced each step, the same invariant the live sharded "
                             "eval already relies on), so each rank evals its [rank::world_size] "
                             "shard and the results gather to rank 0 — ~world_size x faster SWA eval. "
                             "Default ON; pass --no-swa-eval-ddp to restore rank-0-only. Single-GPU "
                             "runs are unaffected (world_size==1 -> identical to the old path).")
    parser.add_argument("--scalar-lr-include-gate-scale", action="store_true", default=False,
                        help="Put gate_scale (G) into the scalar-LR group (lr * scalar_lr_mult). "
                             "Default OFF per the documented 100x-era exclusion; opt in for "
                             "modest mults where amplifying G's trust calibration is the goal.")
    parser.add_argument("--lora-muon", action="store_true", default=False,
                        help="Route factored LoRA A/B matrices to Muon. Default OFF: "
                             "skinny low-rank factors go to AdamW (standard practice; "
                             "rank-16 Muon lagged AdamW, project_muon_lags_adamw_gate). "
                             "Set for full-rank LoRA (large r). Delta adapters always "
                             "use Muon.")
    parser.add_argument("--lora-layers", type=str, default=None,
                        help="LoRA layer selection: 'all' or comma-separated ids. "
                             "Default (unset) applies LoRA at the gate layer only. "
                             "LoRA-only.")
    parser.add_argument("--phase2-loguniform", action="store_true", default=False,
                        help="per step, TWO phase-2 targets — the "
                             "full-context sentinel plus one L ~ LogUniform(128, "
                             "seq_len/2). Stateless per-step RNG (seed, step): rank-"
                             "identical, resume-safe. Mutually exclusive with the "
                             "list-based K sources.")
    parser.add_argument("--scalar-lr-mult", type=float, default=100.0,
                        help="LR multiplier for scalar gate parameters (attn_scale, gate_scale). "
                             "These live in a separate optimizer param group with lr * scalar_lr_mult. "
                             "Default 100 compensates for the fact that AdamW's adaptive normalisation "
                             "gives a per-step update of ~±lr regardless of gradient magnitude — "
                             "scalar parameters need a higher nominal LR to move meaningfully.")
    parser.add_argument("--eval-on-resume", action="store_true",
                        help="Run the pre-step-0 eval after a --resume-from as well, so a "
                             "restarted run reports where it stands before training "
                             "further. Logged at the resumed step rather than step 0.")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Weights & Biases project name. If not set, W&B logging is disabled.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name. Defaults to the output directory basename.")
    # ── Adapter (always applied at the gate layer) ────────────────────────────
    parser.add_argument("--adapter-type", type=str, default="lora",
                        choices=["lora", "delta"],
                        help="'lora' (default; factored A·B, enabled via "
                             "--lora-rank > 0) or 'delta' (additive full-matrix "
                             "ΔW on attn q/k/v/o + MLP gate/up/down of the gate "
                             "layer; rank/alpha-free, Muon-native). "
                             "delta is incompatible with --lora-rank/--lora-alpha/"
                             "--lora-targets.")
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="LoRA adapter rank (r). 0 (default) = LoRA disabled.")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                        help="LoRA alpha scaling factor. Effective scale = alpha/rank. Default: 16.")
    parser.add_argument("--lora-lr", type=float, default=None,
                        help="Learning rate for LoRA adapter parameters. "
                             "Default: None (uses base --lr).")
    parser.add_argument("--lora-targets", type=str, nargs="+", default=None,
                        help="Attention projection names to apply LoRA to. "
                             "Default: q_proj k_proj v_proj o_proj.")
    parser.add_argument("--resume-lora", type=str, default=None,
                        help="Path to a lora_step*.pt checkpoint to resume LoRA weights from.")
    parser.add_argument("--random-topk", action="store_true", default=False,
                        help="Ablation baseline: replace the gate's top-K selection with a "
                             "uniform random K-subset of article tokens, sampled fresh each "
                             "step.  Freezes ALL gate parameters (excluded from optimizer); "
                             "the gate still forwards (its zero-init linear weights produce "
                             "zero bias into Phase 2), so the Phase 1/Phase 2 split is "
                             "preserved but selection is random.  Lets a downstream LoRA / "
                             "FT-before-gate adapt to a random K-subset, isolating 'gate "
                             "picks better tokens' from 'layer 0 adapts to incomplete "
                             "context'.  Requires LoRA (--lora-rank > 0) as the trainable "
                             "param source.")
    parser.add_argument("--gate-selection", type=str, default="topk",
                        choices=["topk", "threshold"],
                        help="Article-token selection mode.  'topk' (default) preserves "
                             "the prior fixed-K behaviour.  'threshold' keeps every "
                             "article token whose sigmoid(b) > --tau, producing a "
                             "sample-dependent kept count.  Combine with "
                             "--sparsity-lambda to drive the gate to aggressively drop "
                             "irrelevant tokens.")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="Threshold for --gate-selection=threshold (and the default "
                             "tau at inference unless overridden).  Article token j is "
                             "kept iff sigmoid(b_j) > tau.  Ignored when "
                             "--gate-selection=topk.")
    parser.add_argument("--sparsity-lambda", type=float, default=0.0,
                        help="Coefficient on the sparsity penalty added to the task loss "
                             "before backward.  Setting to 0 disables the penalty.  "
                             "Active in both topk and threshold selection modes.  "
                             "The exact form of the penalty is controlled by "
                             "--sparsity-mode (see below).")
    parser.add_argument("--sparsity-mode",
                        choices=["soft_mean", "soft_count", "soft_count_frac",
                                 "soft_count_frac_l2"],
                        default="soft_mean",
                        help="How the sparsity penalty is computed.\n"
                             "  soft_mean       : lambda * mean(sigmoid(b_j)) over "
                             "article tokens (averaged across batch). LEGACY. Per-token "
                             "gradient is diluted by 1/n_art — uniformly pulls all "
                             "scores down so sigmoid(b)<tau everywhere, and "
                             "_select_threshold falls back to argmax(b) on every "
                             "article (bypasses the threshold).\n"
                             "  soft_count      : lambda * sum(sigmoid((b - logit(tau)) * "
                             "sparsity_sharpness)) — soft, differentiable count of "
                             "tokens above tau. Per-token gradient is largest AT the "
                             "decision boundary and saturates far from it, naturally "
                             "pushing the score distribution bimodal. Total magnitude "
                             "scales with n_art, which can swamp CE on long articles. "
                             "Pair with --sparsity-sharpness.\n"
                             "  soft_count_frac : same per-token shape as soft_count, "
                             "but normalised by n_art (mean instead of sum). The term "
                             "is a soft estimate of the FRACTION of article tokens "
                             "above tau, bounded in [0,1] regardless of article "
                             "length. Use this when n_art varies across datasets/"
                             "batches and the count form would dominate CE.\n"
                             "  soft_count_frac_l2 : lambda * p^2 where "
                             "p = mean(sigmoid((b - logit(tau)) * sharpness)) — squared "
                             "soft-frac-kept. Per-token gradient scales linearly with "
                             "p instead of being constant, so sparsity pressure "
                             "DECREASES as the kept fraction shrinks. Doesn't add a "
                             "restoring force vs argmax-1 collapse but delays it, "
                             "leaving more room for CE to develop differential signal "
                             "before sparsity dominates.")
    parser.add_argument("--sparsity-sharpness", type=float, default=5.0,
                        help="Sharpness `s` in the soft-count penalty: "
                             "sigmoid((b - logit_tau) * s). Larger → closer to a hard "
                             "1[sigmoid(b)>tau] step (harder to optimise but more accurate "
                             "as a count); smaller → smoother gradient. 5–10 is a "
                             "reasonable starting range. Used by both soft_count and "
                             "soft_count_frac; ignored by soft_mean.")
    parser.add_argument("--tau-eval", type=float, nargs="+", default=None,
                        help="Sweep of tau values for the in-training eval cycle when "
                             "--gate-selection=threshold.  Each tau gets its own "
                             "per-K-style curve in W&B "
                             "(acc_tau<val>, mean_kept_K_tau<val>, ...).  When None, "
                             "falls back to evaluating at --tau only.  Ignored when "
                             "--gate-selection=topk.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.attn_backends == "fa2":
        # Fail at startup, not at the first attention call: the fa2 training
        # step (and packed Phase 2) has no runtime fallback, unlike inference.
        from kernels.fa2bias.keybias import available as _fa2_bias_available
        if not _fa2_bias_available():
            parser.error(
                "--attn-backends fa2 (the Qwen3 default) needs the key-bias "
                "FlashAttention-2 kernel, which is not built on this machine: "
                "build it with `python kernels/fa2bias/build.py`, or train on the "
                "no-build flex kernels (TRAIN_ATTN_BACKENDS=flex PACK_PHASE2=0 "
                "COMPILE_PHASE2=0 for the shell drivers).")
    # Parse --phase2-bucket-lists "L,L,..;L,L,..;.." into a list of int lists.
    if getattr(args, "phase2_bucket_lists", None):
        args.phase2_bucket_lists = [
            [int(x) for x in grp.split(",") if x.strip()]
            for grp in args.phase2_bucket_lists.split(";") if grp.strip()]
        if not getattr(args, "length_buckets", None):
            parser.error("--phase2-bucket-lists requires --length-buckets.")
    if args.phase2_len_list and args.phase2_bucket_lists:
        parser.error("--phase2-len-list and --phase2-bucket-lists are "
                     "mutually exclusive.")
    _validate_random_topk_args(parser, args)
    if args.adapter_type == "delta" and (args.lora_rank > 0
                                         or args.lora_targets is not None):
        parser.error("--adapter-type delta is rank/alpha-free with fixed "
                     "attn+MLP targets; unset --lora-rank / --lora-targets.")
    # §1b detach backward (the only multi-K backward): retain_graph across the
    # per-K backwards is incompatible with FlexAttention's compiled donated
    # buffers; disable them process-wide.
    import torch._functorch.config as _ft_cfg
    _ft_cfg.donated_buffer = False
    # M > 1 always routes through the detach backward, which does not add the
    # sparsity penalty term.
    _m_from_source = (len(args.phase2_len_list) if args.phase2_len_list
                      else (max(len(_l) for _l in args.phase2_bucket_lists)
                            if args.phase2_bucket_lists else 2))
    if _m_from_source > 1 and args.sparsity_lambda != 0.0:
        parser.error(
            "multi-K training (M > 1) uses the detach backward, which "
            "requires --sparsity-lambda 0.")
    train(args)


def _validate_random_topk_args(parser, args):
    """Fail loudly on incompatible --random-topk combinations.

    --random-topk freezes the gate and replaces top-K with a uniform random
    K-subset, so trainable params MUST come from LoRA; otherwise the run is
    a no-op.
    """
    if not args.random_topk:
        return
    if args.lora_rank <= 0 and args.adapter_type != "delta":
        parser.error(
            "--random-topk requires an adapter (--lora-rank > 0 or "
            "--adapter-type delta); otherwise nothing is trained "
            "(the gate is frozen)."
        )


if __name__ == "__main__":
    main()
