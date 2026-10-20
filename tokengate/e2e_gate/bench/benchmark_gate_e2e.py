"""End-to-end latency / throughput / accuracy benchmark for a trained E2E gate.

Single-GPU worker.  Compares the **gated** inference path (KV cache pruned to K
kept article tokens, FlexAttention with gate bias) against the **vanilla** model
on the standard no-gate code path (full context, SDPA, plain ``model.generate``).

For a list of (dataset, task) pairs it loads the *already-tokenized pretraining
val.pkl* eval samples (the same ones training evaluated on) and, for each K in an
ascending sweep, records:

  * single-stream latency (batch size 1, warmup + CUDA-synced):
      prefill_s, decode_s, total_s, n_gen_tokens, n_ctx_tokens, n_kept_tokens
  * batched throughput (batch size B, OOM-fallback halving):
      wall_s, total_gen_tokens, throughput_tok_s, throughput_samples_s
  * accuracy + f1 per sample (from the batched generations, task.evaluate)

Accuracy + throughput use all ``--n-samples`` rows; latency uses the cheaper
``--n-latency-samples`` subset (latency is low-variance, the per-token Python
decode loop is the cost driver).  Vanilla is K-independent → measured once.

Results are appended to per-dataset CSVs under ``--out-dir/raw`` and the worker is
**resumable**: any (mode, dataset, K) already present is skipped.

Run order per worker: vanilla pass first (so reference lines appear immediately),
then the gate K-sweep ascending (so low-K points — the interesting end — land
first and the live plotter fills curves left-to-right).

Example
-------
    python -m tokengate.e2e_gate.bench.benchmark_gate_e2e \
        --gpu 0 \
        --gate-path .../baseline/L0/gate_step137700.pt --gate-layer 0 \
        --datasets longbench_2wikimqa,longbench_hotpotqa \
        --pretrain-data-root datasets \
        --k-values 100 250 500 1000 2000 4000 \
        --n-samples 50 --n-latency-samples 10 \
        --batch-size 8 --max-new-tokens 256 \
        --max-article-tokens 30000 --max-answer-tokens 1024 \
        --out-dir benchmarks/Qwen3-8B/<tag>
"""

import argparse
import csv
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

# Pin the GPU *before* torch initialises CUDA.  The driver also sets this; doing
# it here makes the worker correct when launched standalone.
_args_gpu = None
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        _args_gpu = sys.argv[_i + 1]
if _args_gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _args_gpu
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokengate.e2e_gate.gate_module.e2e_gate_module import load_gate  # noqa: E402
from tokengate.e2e_gate.gate_module.gate_forward import (  # noqa: E402
    apply_chat_template,
    find_article_boundaries,
    prefill_for_decode,
)
from tokengate.e2e_gate.attn.qwen3_flex_attention import (  # noqa: E402
    GATE_ATTN,
    patch_model_with_flex_attention,
    set_attention_backends_for_layers,
    set_gate_bias_for_layers,
    get_full_attn_layer_indices,
)
from tokengate.e2e_gate.attn.flex_attention_utils import (  # noqa: E402
    warmup_flex_attention,
)
from tokengate.generators import _strip_thinking  # noqa: E402
from tokengate.e2e_gate.decode.gate_decode import generate_batch  # noqa: E402
from tokengate.datasets.tasks import get_task  # noqa: E402


DEVICE = "cuda"


# ── timing ────────────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """CUDA-synced wall-clock timer (context manager)."""

    def __enter__(self):
        _sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync()
        self.elapsed = time.perf_counter() - self._t0


# ── val-sample loading (mirrors train_e2e_gate.py pretrain-mix eval) ───────────

def load_val_samples(
    pretrain_data_root: str,
    dataset: str,
    task_name: str,
    model_name: str,
    n_samples: int,
    max_article_tokens: Optional[int],
    max_answer_tokens: Optional[int],
) -> List[dict]:
    """Load the first ``n_samples`` rows from the tokenized val.pkl, applying the
    same drop-not-truncate article/answer filter training uses.

    Returns rows with at least: article, target, input, dataset, task.
    """
    val_path = os.path.join(
        pretrain_data_root, dataset, task_name, "tokenized", model_name, "val.pkl"
    )
    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"val.pkl not found for {dataset}/{task_name} at {val_path}. "
            f"It must exist in the data root (the pretraining val set)."
        )
    with open(val_path, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, list):
        raw = raw.get("eval_samples", [])
    raw = raw[:n_samples]

    def _in_range(row) -> bool:
        n_art = row["article_end"] - row["article_start"]
        if max_article_tokens is not None and n_art > max_article_tokens:
            return False
        if max_answer_tokens is not None and row.get("n_ans", 0) > max_answer_tokens:
            return False
        return True

    task = get_task(task_name)
    rows = [r for r in raw if _in_range(r)]
    for r in rows:
        r.setdefault("input", task.format_prompt(r["article"]))
        r.setdefault("dataset", dataset)
        r.setdefault("task", task_name)
    return rows


def compute_boundaries(tokenizer, task, sample) -> Tuple[List[int], int, int]:
    """Re-derive prompt token ids + article boundaries exactly like eval does."""
    prompt_ids = apply_chat_template(
        tokenizer, sample["input"], use_chat_template=task.use_chat_template
    )
    nc = task.get_non_compressible_context(sample["article"])
    empty_ids = apply_chat_template(
        tokenizer, task.format_prompt(nc), use_chat_template=task.use_chat_template
    )
    pre, post = find_article_boundaries(prompt_ids, empty_ids)
    art_start = pre
    art_end = len(prompt_ids) - post
    return prompt_ids, art_start, art_end


# ── gate generation ────────────────────────────────────────────────────────────

@torch.no_grad()
def gate_generate_single(
    model, gate, tokenizer, prompt_ids, art_start, art_end, gate_layer, K,
    max_new_tokens, input_all_prev_layers,
) -> Tuple[str, int, int, int, float, float]:
    """Single-sample gate prefill + greedy decode, timed.

    Returns (answer, n_gen, n_kept, n_ctx, prefill_s, decode_s).
    """
    inner = model.model
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    with Timer() as t_pf:
        first_tok, cache, kept_pos, n_kept, _kc = prefill_for_decode(
            model, gate, input_ids, art_start, art_end, gate_layer, K,
        )
    set_gate_bias_for_layers(model, 0, len(inner.layers), None)

    gen = [first_tok]
    next_pos = (kept_pos[0, -1] + 1).to(DEVICE).view(1, 1)
    cache_next = n_kept
    with Timer() as t_dec:
        for _ in range(max_new_tokens - 1):
            if gen[-1].item() == tokenizer.eos_token_id:
                break
            cur = gen[-1]
            hs = inner.embed_tokens(cur)
            pe = inner.rotary_emb(hs, next_pos)
            cp = torch.tensor([cache_next], device=hs.device)
            for li in range(len(inner.layers)):
                hs = inner.layers[li](
                    hs, attention_mask=None, position_ids=next_pos,
                    past_key_values=cache, use_cache=True, cache_position=cp,
                    position_embeddings=pe,
                )
            hs = inner.norm(hs)
            tk = model.lm_head(hs)[:, -1].argmax(-1, keepdim=True).to(DEVICE)
            gen.append(tk)
            next_pos = next_pos + 1
            cache_next += 1
    if gen and gen[-1].item() == tokenizer.eos_token_id:
        gen = gen[:-1]
    n_gen = len(gen)
    ids = torch.cat(gen, dim=1)[0].cpu().tolist() if gen else []
    answer = _strip_thinking(tokenizer.decode(ids, skip_special_tokens=True)) if ids else ""
    return answer, n_gen, n_kept, len(prompt_ids), t_pf.elapsed, t_dec.elapsed


@torch.no_grad()
def gate_generate_batch(
    model, gate, tokenizer, metas, gate_layer, K, max_new_tokens,
    input_all_prev_layers, decode_phase2_form=False, prune_below_gate=False,
):
    """Thin compat wrapper around gate_decode.generate_batch — the single
    shared decode implementation, also used by the trainer's in-training
    eval.  Kept because 15+ callers import this name with this signature.

    metas: list of dicts with prompt_ids / article_start / article_end.
    input_all_prev_layers is vestigial (accepted, never read).
    Returns (answers, n_gen_list, n_kept_list, prefill_s, decode_s, decode_steps).
    """
    res = generate_batch(
        model, gate, tokenizer,
        [m["prompt_ids"] for m in metas],
        [m["article_start"] for m in metas],
        [m["article_end"] for m in metas],
        gate_layer, K, max_new_tokens,
        decode_phase2_form=decode_phase2_form,
        prune_below_gate=prune_below_gate,
    )
    return res.answers, res.n_gen, res.n_kepts, res.prefill_s, res.decode_s, res.steps


# ── vanilla generation (standard no-gate code path) ────────────────────────────

@torch.no_grad()
def vanilla_generate_single(
    model, tokenizer, prompt_ids, max_new_tokens,
) -> Tuple[str, int, int, float, float]:
    """Standard prefill + model.generate, timed.  prefill_s measured separately;
    decode_s = total - prefill.  Returns (answer, n_gen, n_ctx, prefill_s, decode_s).
    """
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    attn = torch.ones_like(input_ids)
    with Timer() as t_pf:
        model(input_ids, attention_mask=attn, use_cache=True)
    with Timer() as t_tot:
        out = model.generate(
            input_ids, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0, input_ids.shape[1]:]
    # strip trailing eos/pad for token count + text
    gen_list = gen_ids.cpu().tolist()
    if tokenizer.eos_token_id in gen_list:
        gen_list = gen_list[:gen_list.index(tokenizer.eos_token_id)]
    answer = _strip_thinking(tokenizer.decode(gen_list, skip_special_tokens=True)) if gen_list else ""
    decode_s = max(0.0, t_tot.elapsed - t_pf.elapsed)
    return answer, len(gen_list), len(prompt_ids), t_pf.elapsed, decode_s


@torch.no_grad()
def vanilla_generate_batch(model, tokenizer, prompt_ids_list, max_new_tokens,
                           prefill_model=None):
    """Standard batched generation (left-padded), TIMED with a manual decode loop
    (same structure as the gate path → fair prefill/decode split).

    ``prefill_model`` (if given) runs ONLY the prefill forward — used to time a
    torch.compile'd prefill while keeping the decode loop on the eager ``model``
    (the decode's growing DynamicCache is not compile-friendly).
    Returns (answers, n_gen, prefill_s, decode_s, decode_steps)."""
    # the position_ids below assume LEFT padding; a right-padding tokenizer would
    # silently corrupt every padded row (garbage from the first generated token)
    assert tokenizer.padding_side == "left", \
        f"vanilla_generate_batch requires tokenizer.padding_side='left', got {tokenizer.padding_side!r}"
    enc = tokenizer.pad({"input_ids": prompt_ids_list}, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attn = enc["attention_mask"].to(DEVICE)
    B, L = input_ids.shape
    # LEFT-padded batch: real tokens must get RoPE positions 0..real_len-1 (pad slots
    # masked). Without explicit position_ids HF assigns 0..L-1 incl. pads, shifting
    # every padded row and breaking the prefill↔decode position handoff (only the
    # single longest row is correct) → degraded generation. Derive from the mask.
    prefill_pos = (attn.long().cumsum(-1) - 1).clamp_min(0)
    prefill_pos = prefill_pos.masked_fill(attn == 0, 0)
    _pf_model = prefill_model if prefill_model is not None else model
    with Timer() as t_pf:
        out = _pf_model(input_ids, attention_mask=attn, position_ids=prefill_pos, use_cache=True)
    cache = out.past_key_values
    cur = out.logits[:, -1].argmax(-1, keepdim=True)
    all_gen = [[cur[b:b + 1]] for b in range(B)]
    eos = (cur[:, 0] == tokenizer.eos_token_id)
    pos = attn.sum(dim=1, keepdim=True)  # next position per row (= real length, consistent with prefill_pos)
    steps = 0
    with Timer() as t_dec:
        for _ in range(max_new_tokens - 1):
            if eos.all():
                break
            attn = torch.cat([attn, torch.ones((B, 1), dtype=attn.dtype, device=DEVICE)], dim=1)
            out = model(cur, attention_mask=attn, past_key_values=cache, use_cache=True,
                        position_ids=pos)
            cache = out.past_key_values
            cur = out.logits[:, -1].argmax(-1, keepdim=True)
            for b in range(B):
                all_gen[b].append(cur[b:b + 1])
            eos |= (cur[:, 0] == tokenizer.eos_token_id)
            pos = pos + 1
            steps += 1
    answers, n_gen = [], []
    for b in range(B):
        glist = torch.cat(all_gen[b], dim=1)[0].cpu().tolist()
        if tokenizer.eos_token_id in glist:
            glist = glist[:glist.index(tokenizer.eos_token_id)]
        n_gen.append(len(glist))
        answers.append(_strip_thinking(tokenizer.decode(glist, skip_special_tokens=True)) if glist else "")
    return answers, n_gen, t_pf.elapsed, t_dec.elapsed, steps


# ── CSV helpers ────────────────────────────────────────────────────────────────

# All measurements are BATCHED at batch_size; latency = the per-batch wall a
# request experiences when batched with batch_size-1 others.  One row per config.
LAT_FIELDS = ["mode", "K", "dataset", "task", "batch_size", "n_batches", "n_samples",
              "n_ctx_tokens", "n_kept_tokens", "n_gen_tokens", "decode_steps",
              "prefill_s", "decode_s", "total_s", "oom"]
ACC_FIELDS = ["mode", "K", "dataset", "task", "sample_idx", "accuracy", "f1",
              "acc_metric", "f1_metric", "n_ctx_tokens", "n_kept_tokens", "n_gen_tokens"]
THR_FIELDS = ["mode", "K", "dataset", "task", "batch_size", "n_samples",
              "wall_s", "total_gen_tokens", "throughput_tok_s",
              "throughput_samples_s", "mean_ctx_tokens", "mean_kept_tokens", "oom"]


def _csv_path(out_dir, kind, dataset):
    return os.path.join(out_dir, "raw", f"{kind}_{dataset}.csv")


def _load_done_keys(path, key_cols) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            done.add(tuple(row[c] for c in key_cols))
    return done


def _append_rows(path, fields, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ── measurement orchestration ──────────────────────────────────────────────────

def _config_has_real_result(thr_path, mode, k_str, dataset) -> bool:
    """True iff a NON-oom throughput row exists for (mode, K, dataset).  A stale
    oom=1 row does NOT count as done, so previously-OOM'd configs get retried."""
    if not os.path.exists(thr_path):
        return False
    with open(thr_path, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("mode"), row.get("K"), row.get("dataset")) == (mode, k_str, dataset) \
               and str(row.get("oom", "0")) != "1":
                return True
    return False


def _is_oom(e) -> bool:
    """True for both the torch caching-allocator OOM AND the driver-level OOM
    (RuntimeError: CUDA driver error: out of memory / CUDA out of memory), which
    the gate pass on very long (20k+ tok) contexts can raise.  The latter is NOT
    a torch.cuda.OutOfMemoryError, so without this it slips past the batch-fallback
    and crashes the worker."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def measure_config(
    out_dir, mode, K, dataset, task_name, tokenizer, task,
    metas, gen_batch_timed_fn, batch_size, max_new_tokens,
):
    """Batched accuracy + (eager) timing pass with OOM batch-fallback.

    A batch that OOMs is split in half and retried, recursively down to size 1, so
    accuracy is ALWAYS captured for every sample (a long-context vanilla batch that
    won't fit at bs=4 still yields per-sample accuracy at a smaller effective bs).
    Only a single sample that OOMs even alone is dropped (recorded, rare).

    NOTE: the per-batch wall times here are eager + confounded by how many tokens
    each answer generates and by the fallback batch size — they are NOT the
    benchmark's latency/throughput numbers (those come from benchmarks/run_benchmark.py).
    This harness exists to reproduce TRAINING-EVAL ACCURACY.

    gen_batch_timed_fn(batch) -> (answers, n_gen_list, n_kept_list, prefill_s,
                                  decode_s, decode_steps)  — raises OOM on failure.
    """
    k_str = str(K)
    # Key raw files by TASK (unique per subset) not dataset: a shared-dataset-name
    # mix (LongBench-matched → every subset is "longbench-datasets") would otherwise
    # make all workers write the SAME file concurrently and clobber each other.
    thr_path = _csv_path(out_dir, "throughput", task_name)
    if _config_has_real_result(thr_path, mode, k_str, dataset):
        print(f"  [skip] {mode} K={K} {dataset} (already done)", flush=True)
        return

    acc_metric = getattr(task, "acc_metric_name", "acc")
    f1_metric = getattr(task, "secondary_metric_name", "f1")

    acc_rows = []
    pf_walls, dec_walls, step_counts = [], [], []
    total_gen = total_wall = 0.0
    ctx_sum = kept_sum = gen_sum = n = 0
    n_oom_singletons = 0
    eff_bs_min = batch_size

    # Warmup at the MEASUREMENT batch size — FlexAttention's BlockMask cache
    # and torch.compile keys are shape-dependent, so a bs=1 warmup leaves a
    # bs=N measurement to pay the per-shape compile cost on its first batch
    # (~0.4s for a 6k-token sequence). When two baselines run back-to-back,
    # the second one inherits a warm cache from the first → artifactual
    # speedup. Warming at the actual measurement batch_size eliminates both
    # confounders (verified: same-selection runs at K=100 and K=-3 in
    # threshold mode previously diverged 0.96s vs 0.50s on biodex — should
    # now match).
    _warmup_batch = metas[:batch_size] if len(metas) >= batch_size else metas
    if _warmup_batch:
        try:
            gen_batch_timed_fn(_warmup_batch)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as _e:
            if not _is_oom(_e):
                raise
            torch.cuda.empty_cache()
        # Drop the cache state added by warmup so the measurement loop starts
        # with the same allocator state on every baseline.
        torch.cuda.empty_cache()

    def _run_with_fallback(batch):
        """Yield (meta, ans, n_gen, n_kept, pf, dec, steps) for every meta in batch,
        splitting on OOM down to singletons."""
        nonlocal n_oom_singletons, eff_bs_min
        try:
            answers, n_gen, n_kept, pf, dec, steps = gen_batch_timed_fn(batch)
            eff_bs_min = min(eff_bs_min, len(batch))
            for m, ans, ng, nk in zip(batch, answers, n_gen, n_kept):
                yield (m, ans, ng, nk, pf, dec, steps)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as _e:
            if not _is_oom(_e):
                raise
            torch.cuda.empty_cache()
            if len(batch) == 1:
                n_oom_singletons += 1
                print(f"  [OOM-drop] {mode} K={K} {dataset} sample {batch[0]['idx']} "
                      f"(ctx={batch[0]['n_ctx']}) OOMs even at bs=1", flush=True)
                return
            mid = len(batch) // 2
            print(f"  [OOM-split] {mode} K={K} {dataset} bs={len(batch)} -> "
                  f"{mid}+{len(batch)-mid}", flush=True)
            yield from _run_with_fallback(batch[:mid])
            yield from _run_with_fallback(batch[mid:])

    for i in range(0, len(metas), batch_size):
        batch = metas[i:i + batch_size]
        for m, ans, ng, nk, pf, dec, steps in _run_with_fallback(batch):
            pf_walls.append(pf); dec_walls.append(dec); step_counts.append(steps)
            total_wall += pf + dec
            s = m["sample"]
            acc_i = task.evaluate(ans, s["target"], sample=s)
            f1_i = task.evaluate_f1(ans, s["target"], sample=s)
            acc_rows.append({
                "mode": mode, "K": k_str, "dataset": dataset, "task": task_name,
                "sample_idx": m["idx"], "accuracy": f"{acc_i:.6f}", "f1": f"{f1_i:.6f}",
                "acc_metric": acc_metric, "f1_metric": f1_metric,
                "n_ctx_tokens": m["n_ctx"], "n_kept_tokens": nk, "n_gen_tokens": ng,
            })
            total_gen += ng; ctx_sum += m["n_ctx"]; kept_sum += nk; gen_sum += ng; n += 1

    if n == 0:
        # Whole config OOM'd even at bs=1 — record a genuine OOM marker.
        _append_rows(thr_path, THR_FIELDS, [{
            "mode": mode, "K": k_str, "dataset": dataset, "task": task_name,
            "batch_size": 1, "n_samples": 0, "wall_s": "", "total_gen_tokens": "",
            "throughput_tok_s": "", "throughput_samples_s": "", "mean_ctx_tokens": "",
            "mean_kept_tokens": "", "oom": 1,
        }])
        return
    batch_size = eff_bs_min  # report the effective (post-fallback) batch size
    nb = len(pf_walls)
    mean_pf = sum(pf_walls) / nb
    mean_dec = sum(dec_walls) / nb
    _append_rows(_csv_path(out_dir, "latency", task_name), LAT_FIELDS, [{
        "mode": mode, "K": k_str, "dataset": dataset, "task": task_name,
        "batch_size": batch_size, "n_batches": nb, "n_samples": n,
        "n_ctx_tokens": f"{ctx_sum/n:.1f}", "n_kept_tokens": f"{kept_sum/n:.1f}",
        "n_gen_tokens": f"{gen_sum/n:.2f}", "decode_steps": f"{sum(step_counts)/nb:.1f}",
        "prefill_s": f"{mean_pf:.6f}", "decode_s": f"{mean_dec:.6f}",
        "total_s": f"{mean_pf+mean_dec:.6f}", "oom": 0,
    }])
    _append_rows(_csv_path(out_dir, "accuracy", task_name), ACC_FIELDS, acc_rows)
    thr_tok = total_gen / total_wall if total_wall > 0 else 0.0
    thr_samp = n / total_wall if total_wall > 0 else 0.0
    _append_rows(thr_path, THR_FIELDS, [{
        "mode": mode, "K": k_str, "dataset": dataset, "task": task_name,
        "batch_size": batch_size, "n_samples": n, "wall_s": f"{total_wall:.4f}",
        "total_gen_tokens": int(total_gen), "throughput_tok_s": f"{thr_tok:.3f}",
        "throughput_samples_s": f"{thr_samp:.4f}",
        "mean_ctx_tokens": f"{ctx_sum/n:.1f}", "mean_kept_tokens": f"{kept_sum/n:.1f}", "oom": 0,
    }])
    mean_acc = sum(float(r["accuracy"]) for r in acc_rows) / max(len(acc_rows), 1)
    print(f"  [done] {mode} K={K} {dataset}: {acc_metric}={mean_acc:.3f} "
          f"lat={mean_pf+mean_dec:.2f}s (pf {mean_pf:.2f}+dec {mean_dec:.2f}) "
          f"thr={thr_tok:.0f} tok/s ({thr_samp:.2f} samp/s) bs={batch_size} n={n}", flush=True)


# ── model loading ──────────────────────────────────────────────────────────────

def compile_layer_forward(layer, cuda_graphs: bool = False, idx=None) -> None:
    """torch.compile a decoder layer's forward under a code object of its own.

    Dynamo caches compiled graphs per code object, and every layer's graph is
    distinct (dynamo guards on the layer's slot in the KV cache), so with the
    shared Qwen3DecoderLayer.forward the cache behind one code object holds a
    graph per layer -- 28 per model, 56 with a length-dependent kernel choice,
    twice that with two models resident -- and each call walks that chain of
    guard sets until its own matches.  At 1k tokens that walk cost more than
    the layer's kernels: Phase 2 of the pruned prefill read a flat 9.5 ms for
    26 layers whatever the context.  A copy of the code object per layer gives
    each its own cache of one or two entries.
    """
    import types
    fn = type(layer).forward
    # idx names the per-layer code object; DeltaNet layers of the Qwen3.5
    # hybrid have linear_attn instead of self_attn, so never touch
    # layer.self_attn directly here
    if idx is None:
        _attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
        idx = getattr(_attn, "layer_idx", "")
    code = fn.__code__.replace(co_name=f"{fn.__code__.co_name}_l{idx}")
    fresh = types.FunctionType(code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    fresh.__kwdefaults__ = fn.__kwdefaults__
    layer.forward = torch.compile(types.MethodType(fresh, layer), dynamic=True, fullgraph=False,
                                  mode="reduce-overhead" if cuda_graphs else None)


def load_vanilla_model(base_model):
    from transformers import AutoModelForCausalLM
    # FlashAttention-2 explicitly (not "sdpa"): at bs>1 the SDPA path pads every
    # sequence to the batch-max length and builds a 4-D additive float mask, which
    # (a) drops off SDPA's flash backend onto the slower mem-efficient kernel and
    # (b) wastes memory/compute on pad tokens (this is what OOM-halved bs=8 → 2).
    # HF's flash_attention_2 path unpads the batch (varlen / cu_seqlens), so each
    # sequence runs at its true length on FA2 — the fair, intended vanilla kernel.
    # No silent sdpa fallback: if FA2 is unavailable we want a loud error, not a
    # quiet kernel swap that would invalidate the timing comparison.
    m = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to(DEVICE).eval()
    return m


def _infer_lora_shape(sd, path):
    """(layer_ids, dotted_targets, rank) read off a LoRA state dict itself.

    The artifact is self-describing: keys carry the layer index and the dotted
    module path relative to the decoder layer (e.g. 'self_attn.q_proj'), and
    lora_B is stored (rank, out).  Metadata is NOT consulted — old checkpoints
    with adapter layers ≠ [lora_start_layer .. gate_layer] (e.g. the ll=all
    runs) load correctly this way.  Only alpha is non-inferable (see
    _lora_alpha_for).
    """
    import re as _re
    layer_ids = sorted({int(m.group(1)) for k in sd
                        for m in [_re.search(r"\.layers\.(\d+)\.", k)] if m})
    targets = sorted({m.group(1) for k in sd
                      for m in [_re.search(r"\.layers\.\d+\.(.+?)\.lora_[AB]$", k)] if m})
    ranks = {v.shape[0] for k, v in sd.items() if k.endswith(".lora_B")}
    if not layer_ids or not targets or len(ranks) != 1:
        raise ValueError(f"cannot infer LoRA shape from {path}: layers={layer_ids} "
                         f"targets={targets} ranks={ranks}")
    return layer_ids, targets, ranks.pop()


def _lora_alpha_for(gate_path, meta):
    """LoRA alpha for a gate checkpoint: gate meta, else the run's params.json
    (written next to the checkpoints), else fail — alpha is not stored in the
    adapter file and a silently-wrong default rescales every adapter output."""
    if meta.get("lora_alpha"):
        return float(meta["lora_alpha"])
    pj = os.path.join(os.path.dirname(os.path.abspath(gate_path)), "params.json")
    if os.path.exists(pj):
        import json as _json
        with open(pj) as f:
            la = _json.load(f).get("lora_alpha")
        if la:
            return float(la)
    raise ValueError(f"lora_alpha not in gate meta and no params.json with it next to "
                     f"{gate_path}")


GATE_COMPILE = ("none", "p2", "all")


def load_gate_model(base_model, gate_path, gate_layer, gate_attn="fa2",
                    merge_adapters_above_gate=False, compile_phase2="none", cuda_graphs=False):
    """gate_attn names the inference attention kernels
    (qwen3_flex_attention.GATE_ATTN); "fa2" (FA2 flash causal + the
    FlashAttention-2 key-bias kernel) is the default and matches the final
    training runs' kernels; "flex" is training's original (slow) kernel.
    merge_adapters_above_gate folds the LoRA / delta adapters of the layers
    ABOVE gate_layer into their base weights (lora.merge_adapters_into_layers);
    the gate layer and below keep the unmerged, training-identical path so the
    selection is bit-for-bit what the checkpoint selects.
    compile_phase2="p2" torch.compiles every decoder layer above gate_layer
    (dynamic shapes; the Phase-1 layers stay eager for the same reason), with
    inductor's emulate_precision_casts so fused kernels keep eager's bf16
    roundings (without it a near-tie first token can flip)."""
    from transformers import AutoModelForCausalLM
    from tokengate.e2e_gate.gate_module.lora import find_adapter_path
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(DEVICE).eval()

    full_attn = get_full_attn_layer_indices(model)
    # The gate READS hidden states at gate_layer — any layer type works
    # (run-17 gates sit on DeltaNet layers 1/2 of the Qwen3.5 hybrid); what
    # must exist is at least one full-attention layer ABOVE it for the bias
    # to land on.  (On dense Qwen3 every layer is full-attn, so this accepts
    # exactly the layers the old full-attn-only check did, minus the last.)
    if not 0 <= gate_layer < len(model.model.layers):
        raise ValueError(f"gate_layer={gate_layer} out of range (n_layers={len(model.model.layers)})")
    if not any(i > gate_layer for i in full_attn):
        raise ValueError(
            f"gate_layer={gate_layer}: no full-attention layer above it to bias ({full_attn})"
        )

    gate, meta = load_gate(gate_path)
    gate = gate.to(DEVICE).to(torch.float32).eval()
    input_all_prev_layers = False

    # Adapter pairing by checkpoint name (lora.find_adapter_path): gate_step{N} ->
    # lora_step{N}, gate_final -> lora_final, swa_gate_* -> swa_lora_*.  Never a live
    # adapter for an SWA gate and never lora_final for a step gate -- the old fallbacks
    # silently scored a gate with an adapter it was not averaged / saved with.
    lora_path = find_adapter_path(gate_path)
    if meta.get("adapter_type") and lora_path is None:
        raise FileNotFoundError(
            f"{gate_path} was trained with adapter_type={meta['adapter_type']!r} but no "
            f"paired adapter file exists next to it (see lora.find_adapter_path)")
    if lora_path is not None:
        from tokengate.e2e_gate.gate_module.lora import (
            apply_delta_to_layers, apply_lora_to_layers, load_lora_state_dict,
        )
        # meta["adapter_type"] disambiguates delta vs lora (absent in pre-delta
        # checkpoints → "lora").
        adapter_type = meta.get("adapter_type") or "lora"
        ls = meta.get("lora_start_layer") or 0
        layer_ids = list(range(ls, gate_layer + 1))
        if adapter_type == "delta":
            _sd = torch.load(lora_path, map_location=DEVICE)
            # The adapter may cover layers ABOVE gate_layer (e.g. --unfreeze-layers, which
            # applies delta to those layers). Infer ALL layers present in the checkpoint so
            # apply_delta wraps them; else load_lora_state_dict fails on unmatched tensors.
            import re as _re
            _ck = sorted({int(m.group(1)) for k in _sd for m in [_re.search(r"\.layers\.(\d+)\.", k)] if m})
            layer_ids = sorted(set(layer_ids) | set(_ck))
            apply_delta_to_layers(model, layer_ids)
            load_lora_state_dict(model, _sd)
            print(f"[gate] loaded delta adapter {lora_path} (layers={layer_ids})", flush=True)
        else:
            _sd = torch.load(lora_path, map_location=DEVICE)
            layer_ids, lt, lr = _infer_lora_shape(_sd, lora_path)
            la = _lora_alpha_for(gate_path, meta)
            apply_lora_to_layers(model, layer_ids, lr, la, lt)
            load_lora_state_dict(model, _sd)
            print(f"[gate] loaded LoRA {lora_path} (rank={lr}, alpha={la}, targets={lt}, "
                  f"layers={layer_ids[0]}..{layer_ids[-1]})", flush=True)
    else:
        print(f"[gate] WARNING: no LoRA file found next to {gate_path}", flush=True)

    if merge_adapters_above_gate:
        # True / 1: layers above the gate only (the selection stays bit-exact);
        # 2: every layer (kernel parity with the baseline; changes the selection
        # at bf16 near-ties, see check_gate_equiv).
        from tokengate.e2e_gate.gate_module.lora import merge_adapters_into_layers
        _first = 0 if merge_adapters_above_gate == 2 else gate_layer + 1
        _layers = list(range(_first, len(model.model.layers)))
        _n = merge_adapters_into_layers(model, _layers)
        print(f"[gate] merged {_n} adapter modules into base weights on layers "
              f"{_layers[0]}..{_layers[-1]}"
              + ("" if _first == 0 else f"; layers 0..{gate_layer} unmerged"), flush=True)
    model = patch_model_with_flex_attention(model)
    set_attention_backends_for_layers(model, gate_attn)
    print(f"[gate] attention kernels: {gate_attn} -> causal/bias = {GATE_ATTN[gate_attn]}", flush=True)
    if compile_phase2 in ("p2", "all"):
        # "all" also compiles the Phase-1 layers 0..gate_layer (layer gate_layer
        # runs through attn_only_forward / gate_layer_mlp, which call its
        # submodules, so its layer.forward compile is inert; the submodules stay
        # eager). Whether that keeps the selection bit-identical is what
        # check_gate_equiv decides.
        import torch._inductor.config as _ic
        _ic.emulate_precision_casts = True
        # Every decoder layer compiles to its own graph (dynamo guards on the
        # layer's index into the cache), times two once a length-dependent
        # kernel choice adds a specialisation; with a second compiled model
        # resident (the TTFT bench) the default limit of 8 -- or the 64 set
        # for training -- is crossed and dynamo silently runs the rest EAGER
        # (the baseline read 27-32 ms where it measures 16).  Raise the limit
        # and make crossing it an error, never a fallback.
        import torch._dynamo.config as _dc
        _dc.recompile_limit = max(_dc.recompile_limit, 512)
        _dc.accumulated_recompile_limit = max(_dc.accumulated_recompile_limit, 4096)
        _dc.fail_on_recompile_limit_hit = True
        # cuda_graphs: the Phase-2 layers run under CUDA graphs (inductor's
        # reduce-overhead mode, one recording per distinct shape) and the
        # pruned sequence is padded to a multiple of 64 tokens so the shapes
        # are few.  Phase 2 on ~1k kept tokens was 26 x 0.33 ms of launch
        # overhead, whatever the context length.  Phase 1 stays as it is.
        from tokengate.e2e_gate.gate_module.gate_forward import set_phase2_bucket
        from tokengate.e2e_gate.attn.qwen3_flex_attention import set_static_inputs_for_layers
        # cuda_graphs: 0 none; 1 inductor's reduce-overhead mode per layer above
        # the gate; 2 the whole Phase-2 loop captured once per shape
        # (phase2_graph) with the layers compiled in the default mode.
        cuda_graphs = int(cuda_graphs)
        set_phase2_bucket(64 if cuda_graphs else None, cuda_graphs=cuda_graphs)
        set_static_inputs_for_layers(model, gate_layer + 1, len(model.model.layers), bool(cuda_graphs))
        _first = 0 if compile_phase2 == "all" else gate_layer + 1
        for _i in range(_first, len(model.model.layers)):
            compile_layer_forward(model.model.layers[_i], cuda_graphs=(cuda_graphs == 1 and _i > gate_layer), idx=_i)
        # The gate layer runs as two halves (attention in Phase 1, MLP on the kept
        # tokens at Phase-2 entry) through module-level functions, not
        # layer.forward; compile those too when Phase 1 is compiled.
        from tokengate.e2e_gate.gate_module.gate_forward import (
            compile_gate_branch, set_split_layer_compile,
        )
        set_split_layer_compile(compile_phase2 == "all")
        # The scoring branch too, in the parity configuration: its norm,
        # activation and casts fuse (their roundings change, so not for "p2").
        if compile_phase2 == "all" and getattr(gate, "gate_branch", None) is not None:
            compile_gate_branch(gate, model, gate_layer)
        print(f"[gate] torch.compile on layers {_first}..{len(model.model.layers) - 1} "
              f"(dynamic, emulate_precision_casts"
              f"{', CUDA graphs above the gate layer, Phase 2 padded to 64' if cuda_graphs == 1 else ''}"
              f"{', Phase 2 as one CUDA graph per shape, padded to 64' if cuda_graphs == 2 else ''})"
              + ("; scoring branch compiled" if getattr(gate, "_compiled_branch_hs", None) is not None else "")
              + ("; gate layer's split halves compiled" if compile_phase2 == "all" else ""), flush=True)
    elif compile_phase2 != "none":
        raise ValueError(f"compile_phase2 {compile_phase2!r} not in {GATE_COMPILE}")
    warmup_flex_attention(DEVICE)
    # load_gate() reconstructs the scoring branch itself: fp32-storage CastLinears
    # that compute in the input dtype (the bf16 backbone activations), exactly the
    # modules training used -- no rebuild from the base model, no fp32-compute drift
    # vs the training-time [eval] / [swa eval] numbers.  It fails loudly when a
    # checkpoint declares a branch it does not carry.
    _gb_mode = getattr(gate, "gate_branch_mode", None)
    if _gb_mode and gate.gate_branch is None:
        raise RuntimeError(f"gate_branch_mode={_gb_mode!r} but load_gate attached no "
                           f"scoring branch: {gate_path}")
    if _gb_mode:
        from tokengate.e2e_gate.gate_module.e2e_gate_module import freeze_cast_linears
        _nf = freeze_cast_linears(gate.gate_branch, torch.bfloat16)
        print(f"[gate] scoring branch ({_gb_mode}): "
              f"{sum(p.numel() for p in gate.gate_branch.parameters())/1e6:.2f}M params, "
              f"fp32 storage, compute in the backbone dtype ({_nf} CastLinears' bf16 copies frozen)", flush=True)
    print(f"[gate] ready (layer={gate_layer}, d_model={gate.d_model})", flush=True)
    return model, gate, input_all_prev_layers


# ── main ───────────────────────────────────────────────────────────────────────

def build_metas(tokenizer, task, samples):
    """Compute prompt_ids + boundaries once per sample (shared by both modes)."""
    metas = []
    for idx, s in enumerate(samples):
        prompt_ids, a_s, a_e = compute_boundaries(tokenizer, task, s)
        if a_e - a_s <= 0 or a_s >= a_e:
            continue  # no prunable article — skip (same as eval 'valid' filter)
        metas.append({
            "idx": idx, "sample": s, "prompt_ids": prompt_ids,
            "article_start": a_s, "article_end": a_e, "n_ctx": len(prompt_ids),
        })
    return metas


# Held-out dataset configs.
# Tuple: (dataset_key, task_name, loader_kwargs, display_name, seed, max_n)
# seed/max_n=None → use the caller's defaults.
HELDOUT_DATASETS = [
    # LongBench OOD eval (matches FastKV's eval). English subsets only,
    # non-overlapping with the trained pairs. Each subset is loaded
    # from THUDM/LongBench via the `longbench_*` dataset key in load_samples.
    # Split=test (LongBench's only split). seed=0 for determinism; max_n=None
    # → uses default n_samples cap (50) from the heldout driver.
    ("longbench_multifieldqa_en",      "longbench_multifieldqa_en",
        {"split": "test"}, "longbench_multifieldqa_en",      0, None),
    ("longbench_hotpotqa",             "longbench_hotpotqa",
        {"split": "test"}, "longbench_hotpotqa",             0, None),
    ("longbench_2wikimqa",             "longbench_2wikimqa",
        {"split": "test"}, "longbench_2wikimqa",             0, None),
    ("longbench_trec",                 "longbench_trec",
        {"split": "test"}, "longbench_trec",                 0, None),
    ("longbench_samsum",               "longbench_samsum",
        {"split": "test"}, "longbench_samsum",               0, None),
    ("longbench_passage_count",        "longbench_passage_count",
        {"split": "test"}, "longbench_passage_count",        0, None),
    ("longbench_passage_retrieval_en", "longbench_passage_retrieval_en",
        {"split": "test"}, "longbench_passage_retrieval_en", 0, None),
]


def load_heldout_samples(ds_key, task_name, loader_kwargs, n_samples,
                         max_article_tokens, max_answer_tokens, tokenizer=None,
                         seed=42, max_n=None):
    """Load held-out eval samples via build_sft_samples, then apply the same
    drop-not-truncate filter as the pretrain val.pkl path.
    seed/max_n: per-dataset overrides from HELDOUT_DATASETS."""
    from tokengate.e2e_gate.train.sft_dataset import build_sft_samples

    task = get_task(task_name)
    if hasattr(task, "use_judge"):
        task.use_judge = False
    fetch_n = max_n if max_n is not None else n_samples * 8
    raw = build_sft_samples(task, ds_key, n=fetch_n, seed=seed, **loader_kwargs)

    def _in_range(r):
        art_ids = tokenizer.encode(r["article"], add_special_tokens=False) if tokenizer else r["article"].split()
        n_art = len(art_ids)
        if max_article_tokens and n_art > max_article_tokens:
            return False
        return True

    rows = []
    for r in raw:
        if not _in_range(r):
            continue
        r.setdefault("dataset", ds_key)
        r.setdefault("task", task_name)
        rows.append(r)
        # When max_n is set (per-dataset cap), take all of them; otherwise cap at n_samples
        cap = max_n if max_n is not None else n_samples
        if len(rows) >= cap:
            break
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True, help="GPU index (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--gate-path", required=True)
    p.add_argument("--gate-layer", type=int, required=True)
    p.add_argument("--datasets", required=True,
                   help="comma list of dataset:task pairs, e.g. trivia_qa:trivia_qa,cuad:cuad_qa")
    p.add_argument("--pretrain-data-root", required=True)
    p.add_argument("--k-values", type=int, nargs="+", required=True)
    p.add_argument("--n-samples", type=int, required=True)
    p.add_argument("--n-latency-samples", type=int, default=0,
                   help="(deprecated/ignored — latency is now measured batched)")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--max-article-tokens", type=int, required=True)
    p.add_argument("--max-answer-tokens", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-source", choices=["pretrain", "heldout"], default="pretrain",
                   help="pretrain: load from val.pkl; heldout: load via build_sft_samples.")
    p.add_argument("--skip-vanilla", action="store_true")
    p.add_argument("--skip-gate", action="store_true")
    p.add_argument("--compile-vanilla", action="store_true",
                   help="torch.compile the vanilla model (mode=default, dynamic=True) "
                        "before timing; rows are labeled mode=vanilla_compiled. dynamic=True "
                        "is required — the manual decode loop calls model() with a varying "
                        "seqlen / growing cache, so a static compile would recompile per shape.")
    p.add_argument("--gate-first", action="store_true",
                   help="run the gate K-sweep before the (slow) vanilla pass so "
                        "accuracy-at-budget curves land first; resumable.")
    p.add_argument("--avg-ctx-point", action="store_true",
                   help="also measure the gate at K = each dataset's mean context "
                        "length (tagged K=-2; plotted at the avg-context line).")
    p.add_argument("--max-ctx-point", action="store_true",
                   help="also measure the gate at K = each dataset's MAX context "
                        "length, i.e. no pruning (tagged K=-3; plotted at max-context).")
    args = p.parse_args()

    model_name = os.path.basename(args.base_model)
    pairs = [tuple(x.split(":", 1)) for x in args.datasets.split(",") if x]
    k_sorted = sorted(set(args.k_values))  # ascending

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # for batched vanilla generate

    # Pre-load samples + tasks + metas per dataset (metas reused across K).
    if args.data_source == "heldout":
        # Respect --datasets as a (ds_key[:task]) FILTER over HELDOUT_DATASETS so
        # each worker handles only its assigned dataset (clean per-GPU parallelism,
        # no redundant work / CSV write races).  Empty / "all" → every heldout set.
        want = set()
        for p in pairs:
            want.add(p if len(p) == 2 else (p[0], None))
        def _wanted(ds_key, tk):
            if not want or ("all", None) in want:
                return True
            return (ds_key, tk) in want or (ds_key, None) in want
        pairs_to_load = [(ds_key, tk, kwargs, disp, seed, max_n)
                         for ds_key, tk, kwargs, disp, seed, max_n in HELDOUT_DATASETS
                         if _wanted(ds_key, tk)]
        if not pairs_to_load:
            raise ValueError(f"--datasets {args.datasets!r} matched no HELDOUT_DATASETS "
                             f"({[h[0] for h in HELDOUT_DATASETS]})")
    else:
        pairs_to_load = [(ds, tk, None, f"{ds}/{tk}", None, None) for ds, tk in pairs]

    print(f"[worker gpu={args.gpu}] source={args.data_source} datasets={[d for ds,tk,_,d,_s,_n in pairs_to_load]} "
          f"K={k_sorted}", flush=True)
    ds_state = {}
    for ds, tk, loader_kwargs, disp, ds_seed, ds_max_n in pairs_to_load:
        if args.data_source == "heldout":
            seed = ds_seed if ds_seed is not None else 42
            max_n = ds_max_n  # None → use n_samples cap
            samples = load_heldout_samples(ds, tk, loader_kwargs, args.n_samples,
                                           args.max_article_tokens, args.max_answer_tokens,
                                           tokenizer=tokenizer, seed=seed, max_n=max_n)
        else:
            samples = load_val_samples(
                args.pretrain_data_root, ds, tk, model_name, args.n_samples,
                args.max_article_tokens, args.max_answer_tokens,
            )
        task = get_task(tk)
        # Disable LLM judge for benchmarking — use token-level F1/ROUGE/acc instead.
        if hasattr(task, "use_judge"):
            task.use_judge = False
        metas = build_metas(tokenizer, task, samples)
        # Use display name as the dataset key in CSVs.
        csv_ds = disp if args.data_source == "heldout" else ds
        ds_state[(csv_ds, tk)] = (task, metas)
        print(f"  {disp}: {len(metas)} usable samples "
              f"(mean ctx {sum(m['n_ctx'] for m in metas)/max(len(metas),1):.0f} tok)",
              flush=True)

    bs = args.batch_size

    def run_vanilla_pass():
        # vanilla (full context, K-independent; recorded under K=-1)
        _label = "vanilla_compiled" if args.compile_vanilla else "vanilla"
        _tag = " +torch.compile(prefill)" if args.compile_vanilla else ""
        print(f"[worker] === vanilla pass (batched bs={bs}{_tag}) ===", flush=True)
        vmodel = load_vanilla_model(args.base_model)
        # PREFILL-ONLY compile: the prefill is a single cache-free forward, so dynamic=True
        # gives one shape-generic kernel for all article lengths. The DECODE loop's
        # DynamicCache changes stride every step → recompiles to the limit → eager fallback
        # with overhead, so decode stays eager. Prefill dominates here (ctx ~5k vs 50 gen
        # tokens) so this captures the real compile win cleanly. (StaticCache would be needed
        # for a true compiled decode; mode=default — reduce-overhead/CUDA-graphs crashed before.)
        vmodel_pf = (torch.compile(vmodel, mode="default", dynamic=True)
                     if args.compile_vanilla else None)
        for (ds, tk), (task, metas) in ds_state.items():
            def _vb(batch):
                ans, n_gen, pf, dec, st = vanilla_generate_batch(
                    vmodel, tokenizer, [m["prompt_ids"] for m in batch], args.max_new_tokens,
                    prefill_model=vmodel_pf)
                return ans, n_gen, [m["n_ctx"] for m in batch], pf, dec, st  # kept==ctx
            measure_config(args.out_dir, _label, -1, ds, tk, tokenizer, task, metas,
                           _vb, bs, args.max_new_tokens)
        del vmodel
        torch.cuda.empty_cache()

    def run_gate_pass():
        # gate K-sweep ascending, then avg-ctx (K=-2) + max-ctx (K=-3)
        print(f"[worker] === gate pass (batched bs={bs}) ===", flush=True)
        gmodel, gate, iapl = load_gate_model(args.base_model, args.gate_path, args.gate_layer)
        # Threshold-mode gates IGNORE the K argument (gate_forward._select_for_gate
        # routes to threshold mode), so the K-sweep produces N identical points.
        # Honor the K-sweep as a τ=trained-tau measurement once; for the
        # max-ctx point (K=-3), force-keep-all by temporarily flipping
        # selection_mode to topk so the K-value is actually used (measures the
        # gate-model overhead vs vanilla at full context).
        _is_threshold = (getattr(gate, "selection_mode", "topk") == "threshold")
        if _is_threshold:
            print(f"[worker] gate.selection_mode == 'threshold' (tau={getattr(gate, 'tau', 0.5)}); "
                  f"K-sweep collapses to a single τ-only measurement; max-ctx point "
                  f"temporarily forces selection_mode=topk so all tokens are kept.",
                  flush=True)

        def _gate_point(K_label, k_of_ds):
            for (ds, tk), (task, metas) in ds_state.items():
                if not metas:
                    continue
                K = k_of_ds(metas)
                def _gb(batch, K=K):
                    return gate_generate_batch(gmodel, gate, tokenizer, batch,
                                               args.gate_layer, K, args.max_new_tokens, iapl)
                measure_config(args.out_dir, "gate", K_label, ds, tk, tokenizer, task, metas,
                               _gb, bs, args.max_new_tokens)

        if _is_threshold:
            # One timing point only — at the trained tau. Label with K=-4 so it
            # doesn't collide with the top-K label range and is easy to filter.
            _tau = float(getattr(gate, "tau", 0.5))
            print(f"[worker] === gate @ τ={_tau} (threshold mode, single point, "
                  f"label K=-4) ===", flush=True)
            _gate_point(-4, lambda metas: 10_000_000)  # K is ignored in threshold mode
        else:
            for K in k_sorted:
                _gate_point(K, lambda metas, K=K: K)
        if args.avg_ctx_point:
            print("[worker] === gate avg-context point (K = mean ctx len) ===", flush=True)
            _gate_point(-2, lambda metas: max(1, round(sum(m["n_ctx"] for m in metas) / len(metas))))
        if args.max_ctx_point:
            print("[worker] === gate max-context point (K = max ctx len, no pruning) ===", flush=True)
            if _is_threshold:
                # Force top-K so K is honored and ALL article tokens are kept.
                # Otherwise the threshold dispatch would again collapse to the
                # ~49-token argmax-fallback set and the max-ctx point would
                # measure the same thing as the τ measurement above.
                _saved_mode = gate.selection_mode
                gate.selection_mode = "topk"
                try:
                    _gate_point(-3, lambda metas: max(m["n_ctx"] for m in metas))
                finally:
                    gate.selection_mode = _saved_mode
            else:
                _gate_point(-3, lambda metas: max(m["n_ctx"] for m in metas))
        del gmodel, gate
        torch.cuda.empty_cache()

    # --gate-first runs the (fast, high-priority) gate K-sweep before the slow
    # vanilla pass so the accuracy-at-budget curves land first.  Resumable, so a
    # restart with --gate-first just fills in gate configs and skips done vanilla.
    order = []
    if not args.skip_gate:
        order.append(run_gate_pass)
    if not args.skip_vanilla:
        order.append(run_vanilla_pass)
    if not args.gate_first:
        order.reverse()
    for p in order:
        p()

    print(f"[worker gpu={args.gpu}] DONE", flush=True)


if __name__ == "__main__":
    main()
