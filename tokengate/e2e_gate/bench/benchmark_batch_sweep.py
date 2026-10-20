"""Batch-size sweep: decode throughput + memory vs batch size, gate vs vanilla.

Answers two questions the per-request latency benchmark can't:
  (a) how far a given batch size is from saturating the GPU (decode tok/s vs B), and
  (b) the largest batch the gate fits vs vanilla at a fixed context length
      (the gate's pruned KV — B×K vs B×L — lets more sequences fit).

Controlled micro-benchmark: synthetic prompts of a fixed length L (so every
sequence in the batch is the same length — no padding), article = middle 80 %.
For each (mode, batch size): prefill the batch, then time a FIXED number of decode
steps (no EOS), and record decode tokens/sec and peak memory in the prefill phase
and (separately) the decode phase (KV-dominated steady state).  Sweeps B upward
until OOM and records the largest batch that fit.

Single-GPU by design (memory/throughput must be measured in isolation).

    python -m tokengate.e2e_gate.bench.benchmark_batch_sweep \
        --gpu 0 --gate-path .../gate_step137700.pt --gate-layer 0 \
        --context-len 8000 --k-values 500 2000 --decode-tokens 64 \
        --batch-sizes 1 2 4 8 16 32 64 128 --out-dir benchmarks/.../batch_sweep
"""

import argparse
import csv
import os
import sys
import time

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

from tokengate.e2e_gate.bench.benchmark_gate_e2e import (  # noqa: E402
    load_gate_model, load_vanilla_model,
)
from tokengate.e2e_gate.gate_module.gate_forward import (  # noqa: E402
    prefill_batch_for_decode,
)
from tokengate.e2e_gate.attn.qwen3_flex_attention import (  # noqa: E402
    set_gate_bias_for_layers,
)

DEVICE = "cuda"
GB = 1024 ** 3


def _sync():
    torch.cuda.synchronize()


def synth_prompts(B, L, vocab):
    """B random prompt-id lists of length L (uniform length → no padding)."""
    arr = torch.randint(1, vocab, (B, L), dtype=torch.long)
    return arr.tolist()


@torch.no_grad()
def gate_decode(model, gate, prompts, gate_layer, K, n_decode, iapl):
    """Gate prefill (prune to K) + timed fixed-length batched decode.
    Returns (decode_s, prefill_peak_gb, decode_peak_gb)."""
    inner = model.model
    a_s = [len(p) // 10 for p in prompts]
    a_e = [len(p) - len(p) // 10 for p in prompts]

    torch.cuda.reset_peak_memory_stats()
    first_toks, cache, kept_pos_b, n_kepts, _ = prefill_batch_for_decode(
        model, gate, prompts, a_s, a_e, gate_layer, K)
    _sync()
    prefill_peak = torch.cuda.max_memory_allocated() / GB
    set_gate_bias_for_layers(model, 0, len(inner.layers), None)

    B = first_toks.shape[0]
    last_valid = torch.tensor([n - 1 for n in n_kepts], device=DEVICE)
    next_pos = (kept_pos_b[torch.arange(B), last_valid] + 1).unsqueeze(1).to(DEVICE)
    cache_next = kept_pos_b.shape[1]
    cur = first_toks.to(DEVICE)

    torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_decode):
        hs = inner.embed_tokens(cur)
        pe = inner.rotary_emb(hs, next_pos)
        cp = torch.tensor([cache_next], device=hs.device)
        for li in range(len(inner.layers)):
            hs = inner.layers[li](hs, attention_mask=None, position_ids=next_pos,
                                  past_key_values=cache, use_cache=True,
                                  cache_position=cp, position_embeddings=pe)
        hs = inner.norm(hs)
        cur = model.lm_head(hs)[:, -1].argmax(-1, keepdim=True).to(DEVICE)
        next_pos = next_pos + 1
        cache_next += 1
    _sync()
    dt = time.perf_counter() - t0
    return dt, prefill_peak, torch.cuda.max_memory_allocated() / GB


@torch.no_grad()
def vanilla_decode(model, prompts, n_decode):
    """Vanilla full-context prefill + timed fixed-length batched decode (SDPA)."""
    input_ids = torch.tensor(prompts, dtype=torch.long, device=DEVICE)
    attn = torch.ones_like(input_ids)
    torch.cuda.reset_peak_memory_stats()
    out = model(input_ids, attention_mask=attn, use_cache=True)
    _sync()
    prefill_peak = torch.cuda.max_memory_allocated() / GB
    cache = out.past_key_values
    cur = out.logits[:, -1].argmax(-1, keepdim=True)

    torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_decode):
        out = model(cur, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
    _sync()
    dt = time.perf_counter() - t0
    return dt, prefill_peak, torch.cuda.max_memory_allocated() / GB


FIELDS = ["mode", "K", "context_len", "decode_tokens", "batch_size", "ok",
          "decode_s", "decode_tok_s", "prefill_peak_gb", "decode_peak_gb", "note"]


def run_mode(label, fn, k_label, args, vocab, writer, fout):
    print(f"\n=== {label} ===", flush=True)
    # warmup (smallest batch) to trigger compilation, not recorded
    try:
        fn(synth_prompts(1, args.context_len, vocab), args.decode_tokens)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    for B in args.batch_sizes:
        torch.cuda.empty_cache()
        try:
            dt, ppk, dpk = fn(synth_prompts(B, args.context_len, vocab), args.decode_tokens)
            tok_s = B * args.decode_tokens / dt
            row = {"mode": label, "K": k_label, "context_len": args.context_len,
                   "decode_tokens": args.decode_tokens, "batch_size": B, "ok": 1,
                   "decode_s": f"{dt:.4f}", "decode_tok_s": f"{tok_s:.1f}",
                   "prefill_peak_gb": f"{ppk:.2f}", "decode_peak_gb": f"{dpk:.2f}", "note": ""}
            print(f"  B={B:<4} decode {dt*1000/args.decode_tokens:6.1f} ms/step  "
                  f"{tok_s:8.1f} tok/s  prefill_peak {ppk:5.1f} GB  decode_peak {dpk:5.1f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {"mode": label, "K": k_label, "context_len": args.context_len,
                   "decode_tokens": args.decode_tokens, "batch_size": B, "ok": 0,
                   "decode_s": "", "decode_tok_s": "", "prefill_peak_gb": "",
                   "decode_peak_gb": "", "note": "OOM"}
            print(f"  B={B:<4} OOM — largest batch that fit was below this", flush=True)
            writer.writerow(row); fout.flush()
            break
        writer.writerow(row); fout.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--gate-path", required=True)
    p.add_argument("--gate-layer", type=int, required=True)
    p.add_argument("--context-len", type=int, required=True)
    p.add_argument("--k-values", type=int, nargs="+", required=True)
    p.add_argument("--decode-tokens", type=int, required=True)
    p.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    vocab = tok.vocab_size

    csv_path = os.path.join(args.out_dir, f"batch_sweep_L{args.context_len}.csv")
    fout = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDS)
    writer.writeheader()

    print(f"[batch-sweep] gpu={args.gpu} L={args.context_len} "
          f"decode_tokens={args.decode_tokens} batch_sizes={args.batch_sizes}", flush=True)

    # Vanilla (full context, K-independent) first.
    vmodel = load_vanilla_model(args.base_model)
    run_mode("vanilla@full", lambda pr, n: vanilla_decode(vmodel, pr, n), -1, args, vocab, writer, fout)
    del vmodel
    torch.cuda.empty_cache()

    # Gate at each K.
    gmodel, gate, iapl = load_gate_model(args.base_model, args.gate_path, args.gate_layer)
    for K in args.k_values:
        run_mode(f"gate@K={K}", lambda pr, n, K=K: gate_decode(gmodel, gate, pr, args.gate_layer, K, n, iapl),
                 K, args, vocab, writer, fout)

    fout.close()
    print(f"\n[batch-sweep] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
