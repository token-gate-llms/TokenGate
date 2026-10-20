#!/usr/bin/env python3
"""Benchmark a TokenGate model against standard full-context prefill.

Measures, on the same samples and the same base model:
  * accuracy  — the task's own metric on greedy generations;
  * TTFT      — prefill-to-first-token latency, per sample at batch size 1.

Three arms: `full` (the unmodified base model prefilling every token — the
TTFT reference), `tokengate@full` (the trained TokenGate model with nothing
pruned — the accuracy reference), and `tokengate@K` for each configured K
(the gate keeps K state tokens; the layers above the gate never see — or
cache — the rest).

Dataset, checkpoint and K values come from a YAML config:

    python benchmarks/run_benchmark.py --config benchmarks/configs/niah.yaml
    python benchmarks/run_benchmark.py --config ... --split test   # full held-out set
    python benchmarks/run_benchmark.py --config ... --n 4 --k 500 1000

Config keys: base_model, task, gate_layer, checkpoint (HF repo id or a local
run dir holding the gate + name-paired adapter), k_values, max_new_tokens,
gate_attn (fa2|flash|flex), and one sample source — either samples (jsonl of
{id, article, target} rows) or dataset + eval_split (a prepared tokenized
store + split manifest; --split picks the val partition or the full reserved
test partition).
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _resolve_checkpoint(spec: str) -> str:
    """A local dir is used as-is; anything else is a HF Hub repo id."""
    if os.path.isdir(spec):
        return spec
    from huggingface_hub import snapshot_download
    print(f"[bench] downloading checkpoint from hf.co/{spec} ...")
    return snapshot_download(spec)


def _gate_path(ckpt_dir: str) -> str:
    for pat in ("swa_gate_*.pt", "gate_final.pt", "gate_step*.pt"):
        hits = sorted(glob.glob(os.path.join(ckpt_dir, pat)))
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"no gate checkpoint (gate_*.pt) in {ckpt_dir}")


@torch.no_grad()
def _full_prefill_arm(base_model: str, metas, task, max_new_tokens: int):
    """Standard prefill baseline: unmodified model, every token cached."""
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    def _run(meta, timed: bool):
        ids = torch.tensor([meta["prompt_ids"]], device="cuda")
        cache = DynamicCache(config=model.config)
        if timed:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(ids, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        if timed:
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t0
        generated = [tok]
        for _ in range(max_new_tokens - 1):
            if generated[-1].item() == _EOS_ID:
                break
            out = model(generated[-1], past_key_values=cache, use_cache=True)
            generated.append(out.logits[:, -1].argmax(-1, keepdim=True))
        if generated and generated[-1].item() == _EOS_ID:
            generated = generated[:-1]
        text = _TOK.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True)
        return _strip_thinking(text), ttft

    _run(metas[0], timed=False)  # warmup (kernel autotune, allocator)
    rows = []
    for m in metas:
        ans, ttft = _run(m, timed=True)
        rows.append({"id": m["sample"]["id"], "ttft_s": ttft,
                     "acc": task.evaluate(ans, m["sample"]["target"]),
                     "answer": ans})
    del model
    torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def _tokengate_arm(base_model, gate_path, gate_layer, gate_attn, metas, task,
                   k_values, max_new_tokens):
    from tokengate.e2e_gate.bench.benchmark_gate_e2e import (
        gate_generate_batch, load_gate_model)
    model, gate, _ = load_gate_model(
        base_model, gate_path, gate_layer, gate_attn=gate_attn)

    results = {}
    for k in ["full"] + list(k_values):
        k = 10_000_000 if k == "full" else k
        gate_generate_batch(model, gate, _TOK, metas[:1], gate_layer, k,
                            max_new_tokens, input_all_prev_layers=False)  # warmup
        rows = []
        for m in metas:  # batch size 1: honest per-sample TTFT
            answers, _, n_kept, prefill_s, _, _ = gate_generate_batch(
                model, gate, _TOK, [m], gate_layer, k,
                max_new_tokens, input_all_prev_layers=False)
            rows.append({"id": m["sample"]["id"], "ttft_s": prefill_s,
                         "acc": task.evaluate(answers[0], m["sample"]["target"]),
                         "n_kept": n_kept[0], "answer": answers[0]})
        results["full" if k == 10_000_000 else k] = rows
    del model, gate
    torch.cuda.empty_cache()
    return results


def _load_split_samples(cfg, split: str):
    """Samples from a prepared store's eval pool, filtered to one partition of
    the eval-split manifest (`val` = in-training eval rows, `test` = the
    reserved held-out partition)."""
    import pickle
    store = os.path.join(_REPO_ROOT, cfg.get("data_root", "datasets"),
                         cfg["dataset"], cfg["task"], "tokenized",
                         os.path.basename(cfg["base_model"]), "val.pkl")
    if not os.path.exists(store):
        raise FileNotFoundError(
            f"{store} not found — prepare the dataset first "
            f"(datasets/{cfg['dataset']}/{cfg['task']}/prepare.py)")
    split_path = cfg["eval_split"] if os.path.isabs(cfg["eval_split"]) else \
        os.path.join(_REPO_ROOT, cfg["eval_split"])
    ids = set(json.load(open(split_path))[cfg["task"]][split])
    rows = [r for r in pickle.load(open(store, "rb")) if r["id"] in ids]
    if len(rows) != len(ids):
        raise ValueError(f"{store}: found {len(rows)} of {len(ids)} "
                         f"'{split}' ids from {split_path}")
    return [{"id": r["id"], "article": r["article"], "target": r["target"]}
            for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="benchmark YAML config")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="eval-split partition to run (store-backed configs): "
                         "'val' = the in-training eval rows, 'test' = the full "
                         "reserved held-out partition (default: val)")
    ap.add_argument("--n", type=int, default=None, help="samples to run (default all)")
    ap.add_argument("--k", type=int, nargs="+", default=None,
                    help="override the config's k_values")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    required = ("base_model", "task", "gate_layer", "checkpoint",
                "k_values", "max_new_tokens", "gate_attn")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"benchmark config {args.config} is missing keys: {missing}")
    if ("samples" in cfg) == ("eval_split" in cfg):
        raise KeyError(f"benchmark config {args.config}: set exactly one sample "
                       "source — 'samples' (a jsonl of {id, article, target} "
                       "rows) or 'dataset' + 'eval_split' (a prepared store + "
                       "split manifest, run with --split val|test)")
    k_values = args.k or cfg["k_values"]

    from transformers import AutoTokenizer
    from tokengate.datasets.tasks import get_task
    from tokengate.e2e_gate.bench.benchmark_gate_e2e import build_metas
    from tokengate.generators import _strip_thinking as _st

    global _TOK, _EOS_ID, _strip_thinking
    _TOK = AutoTokenizer.from_pretrained(cfg["base_model"])
    _EOS_ID = _TOK.eos_token_id
    _strip_thinking = _st

    task = get_task(cfg["task"])
    if "samples" in cfg:
        samples_path = cfg["samples"] if os.path.isabs(cfg["samples"]) else \
            os.path.join(_REPO_ROOT, cfg["samples"])
        samples = [json.loads(l) for l in open(samples_path)]
        src = os.path.basename(samples_path)
    else:
        samples = _load_split_samples(cfg, args.split)
        src = f"{args.split} split ({os.path.basename(cfg['eval_split'])})"
    samples = samples[: args.n]
    print(f"[bench] samples: {src}")
    for s in samples:
        s["input"] = task.format_prompt(s["article"])
    metas = build_metas(_TOK, task, samples)
    n_ctx = [len(m["prompt_ids"]) for m in metas]
    print(f"[bench] {cfg['task']}: {len(metas)} samples, "
          f"{min(n_ctx)}-{max(n_ctx)} prompt tokens, arms: full + "
          f"tokengate@K={k_values}")

    full_rows = _full_prefill_arm(cfg["base_model"], metas, task,
                                  cfg["max_new_tokens"])
    gate_rows = _tokengate_arm(
        cfg["base_model"], _gate_path(_resolve_checkpoint(cfg["checkpoint"])),
        cfg["gate_layer"], cfg["gate_attn"], metas, task, k_values,
        cfg["max_new_tokens"])

    def _mean(rows, key):
        return sum(r[key] for r in rows) / max(len(rows), 1)

    full_ttft = _mean(full_rows, "ttft_s")
    tag = task.acc_metric_name
    print(f"\n{'arm':<12} {'K':>6} {'acc(' + tag + ')':>10} "
          f"{'TTFT(ms)':>10} {'speedup':>8}")
    print("-" * 50)
    print(f"{'full':<12} {'-':>6} {_mean(full_rows, 'acc'):>10.3f} "
          f"{full_ttft * 1e3:>10.1f} {'1.00x':>8}")
    for k in ["full"] + list(k_values):
        rows = gate_rows[k]
        ttft = _mean(rows, "ttft_s")
        print(f"{'tokengate':<12} {str(k):>6} {_mean(rows, 'acc'):>10.3f} "
              f"{ttft * 1e3:>10.1f} {full_ttft / ttft:>7.2f}x")
    print()


if __name__ == "__main__":
    main()
