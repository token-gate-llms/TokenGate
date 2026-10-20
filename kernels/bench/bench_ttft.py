"""The two TTFT numbers the gate's prefill is optimized against.

    baseline : normal inference, full context  (vanilla Qwen3, FA2)
    ours     : the gate's prefill at K=full    (same tokens, same layers,
                                                nothing pruned)

Both do the SAME work, so their ratio is the gate machinery's overhead with the
pruning win factored out -- the one number an optimization has to move. Timing
a pruned K instead would mix the win in and hide a regression.

ONE SEQUENCE AT A TIME. TTFT is a per-request latency; a batch finishes
together, so a batch-amortized figure is throughput wearing a latency label. At
batch 1 there is also nothing to pad, so the two paths see literally the same
tokens -- no padding, no attention mask, no varlen-vs-padded difference to
confound the ratio, and nothing OOMs, so no adaptive-batch machinery.

Prefill only: no generation, no scoring. Both arms stay resident and are timed
sequence by sequence, one arm's repeats then the other's; median of --repeats after --warmup,
which is not optional -- the first prefill pays FlexAttention compilation.
"""

import argparse
import csv
import json
import os
import pickle
import random
import statistics
import sys
import time

_gpu = None
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        _gpu = sys.argv[_i + 1]
if _gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      os.path.expanduser("~/.cache/tokengate/inductor"))

import torch  # noqa: E402
torch.set_float32_matmul_precision("high")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokengate.e2e_gate.bench.benchmark_gate_e2e import (  # noqa: E402
    build_metas, load_gate_model, load_val_samples, load_vanilla_model, GATE_COMPILE, compile_layer_forward,
)
from tokengate.e2e_gate.attn.flex_attention_utils import (  # noqa: E402
    attention_causal, resolve_backends, set_attn_crossover,
)
from tokengate.e2e_gate.gate_module.gate_forward import (  # noqa: E402
    prefill_batch_for_decode, prefill_phase_ms, set_prefill_phase_timing,
)
from tokengate.e2e_gate.attn.kv_cache import PrefillCache  # noqa: E402
from tokengate.e2e_gate.attn.qwen3_flex_attention import GATE_ATTN  # noqa: E402
from tokengate.datasets.tasks import get_task  # noqa: E402

DEVICE = "cuda"
FULL = 10 ** 7


BASELINE_ATTN = ("fa2", "cudnn", "fa4", "auto")
BASELINE_COMPILE = ("none", "layers", "model")


def _causal_attention_fn(backend):
    """An HF attention function (AttentionInterface) running flex_attention_utils'
    plain causal kernels, for the baseline arm: one unpadded prefill, no mask."""
    def fn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        if attention_mask is not None or query.shape[2] == 1 or dropout:
            raise RuntimeError("the baseline's causal attention handles one unpadded prefill only")
        if scaling is not None and abs(scaling - query.shape[-1] ** -0.5) > 1e-12:
            raise RuntimeError(f"scaling {scaling} is not 1/sqrt(head_dim)")
        o = attention_causal(query, key, value, backend)          # (B, H, S, D)
        return o.transpose(1, 2).contiguous(), None
    return fn


def configure_attn_crossover(a):
    """'auto' (per-length kernel choice) in either arm needs --attn-crossover."""
    needs = a.baseline_attn == "auto" or GATE_ATTN[a.gate_attn][0] == "auto"
    if needs and a.attn_crossover is None:
        raise SystemExit("--attn-crossover N (tokens; cuDNN below, FA4 at and above) is required "
                         "when the baseline or the gate arm's Phase 1 attention is 'auto'")
    if a.attn_crossover is not None:
        set_attn_crossover(a.attn_crossover)


def load_baseline(base_model, attn, compile_mode="none", adapter=None, graphs_from_layer=None):
    """The vanilla model exactly as the baseline arm times it. check_gate_equiv
    loads it through here too, so a faster baseline is also checked to still
    emit the same first token.

    adapter=(lora_state_dict_path, alpha) folds an explicit LoRA (e.g. the SFT
    baseline the FastKV lanes run on) into every layer's weights before the
    layers are compiled -- retime_ttft's FastKV lanes.  graphs_from_layer=L
    compiles layers >= L in CUDA-graph mode (FastKV's pruned Phase 2, the
    counterpart of the gate's); the caller pads that phase's length.

    attn selects the causal-attention kernel:
      fa2    HF flash_attention_2 (the kernels-hub FA2), what the batched
             accuracy harness runs;
      cudnn  torch SDPA pinned to the cuDNN fused-attention backend -- on a
             b200 3.2-3.8x faster than FA2 at 4k-16k tokens (FA2 is an
             Ampere-era kernel; cuDNN 9 has Blackwell kernels).
    """
    if attn in ("fa4", "auto"):
        attn = resolve_backends(attn, "flex")[0]
    if attn == "fa2":
        m = load_vanilla_model(base_model)
    elif attn == "cudnn":
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE).eval()
    elif attn in ("fa4", "auto"):
        # fa4   FlashAttention's CuTe Blackwell kernel (flex_attention_utils._fa4_causal,
        #       the same op the gate arm's Phase 2 runs with its bias columns);
        # auto  cuDNN below --attn-crossover tokens, fa4 at and above: the
        #       faster of the two at each length.
        from transformers import AttentionInterface, AutoModelForCausalLM
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        name = f"rkvs_causal_{attn}"
        if name not in ALL_ATTENTION_FUNCTIONS.valid_keys():
            AttentionInterface.register(name, _causal_attention_fn(attn))
        m = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.bfloat16, attn_implementation=name).to(DEVICE).eval()
    else:
        raise ValueError(f"baseline attn {attn!r} not in {BASELINE_ATTN}")
    if adapter is not None:
        from tokengate.e2e_gate.bench.benchmark_gate_e2e import _infer_lora_shape
        from tokengate.e2e_gate.gate_module.lora import (
            apply_lora_to_layers, load_lora_state_dict, merge_adapters_into_layers,
        )
        path, alpha = adapter
        sd = torch.load(path, map_location=DEVICE)
        layer_ids, targets, rank = _infer_lora_shape(sd, path)
        apply_lora_to_layers(m, layer_ids, rank, float(alpha), targets)
        load_lora_state_dict(m, sd)
        n = merge_adapters_into_layers(m, layer_ids)
        print(f"[ttft] baseline + adapter {os.path.basename(path)} (rank {rank}, alpha {alpha}, "
              f"{targets}, layers {layer_ids[0]}..{layer_ids[-1]}): {n} modules folded", flush=True)
    if compile_mode != "none":
        # Fused kernels keep eager's intermediate bf16 roundings: without
        # this a near-tie first token can flip vs eager.
        import torch._inductor.config as _ic
        _ic.emulate_precision_casts = True
        # See load_gate_model: one graph per layer (and per length regime), so
        # the recompile limit must hold both arms' graphs, and crossing it must
        # raise rather than run eager.
        import torch._dynamo.config as _dc
        _dc.recompile_limit = max(_dc.recompile_limit, 512)
        _dc.accumulated_recompile_limit = max(_dc.accumulated_recompile_limit, 4096)
        _dc.fail_on_recompile_limit_hit = True
    if compile_mode == "layers":
        # torch.compile each decoder layer (dynamic shapes: the prompts run
        # 1k-39k tokens); the embedding, cache bookkeeping and lm_head stay
        # eager. Fuses RMSNorm, rotary and SiLU*up, which eager runs as ~10
        # memory-bound kernels per layer.
        for i, layer in enumerate(m.model.layers):
            compile_layer_forward(layer, cuda_graphs=graphs_from_layer is not None and i >= graphs_from_layer)
    elif compile_mode == "model":
        m.forward = torch.compile(m.forward, dynamic=True, fullgraph=False)
    elif compile_mode != "none":
        raise ValueError(f"baseline compile mode {compile_mode!r} not in {BASELINE_COMPILE}")
    return m


def baseline_prefill(vmodel, ids):
    """One vanilla prefill of one unpadded sequence; returns the last position's
    logits, (1, 1, vocab).

    logits_to_keep=1: a prefill needs the LAST position's logits and nothing
    else. Without it lm_head runs over all ~10.5k positions -- a (1, 10.5k,
    151936) tensor, ~3.2 GB and ~6.5 TFLOPs of matmul that no serving stack
    pays and that the gate path does not pay either (it applies lm_head to one
    hidden vector). Left in, it inflates the baseline and understates our
    overhead.
    """
    # PrefillCache: the same K/V store the gate prefill uses -- the first write
    # per layer keeps the tensor instead of torch.cat-copying it (kv_cache.py).
    kw = dict(past_key_values=PrefillCache(vmodel.config), use_cache=True, logits_to_keep=1)
    with torch.no_grad():
        return vmodel(ids, **kw).logits


def _phase(label, t0):
    print(f"[ttft] {label}: {time.perf_counter() - t0:.1f}s", flush=True)
    return time.perf_counter()


def _meta_key(a):
    return (a.dataset, a.task, a.model_name, a.split, a.seed,
            a.eval_samples_per_group, a.n_samples, a.max_article_tokens,
            a.base_model)


def _metas(a, tokenizer):
    if a.cache_metas and os.path.exists(a.cache_metas):
        with open(a.cache_metas, "rb") as f:
            blob = pickle.load(f)
        if blob["key"] == _meta_key(a):
            print(f"[ttft] metas from cache {a.cache_metas}", flush=True)
            return blob["metas"]
    rows = load_val_samples(a.pretrain_data_root, a.dataset, a.task,
                            a.model_name, 10 ** 7, a.max_article_tokens, None)
    with open(a.eval_split_manifest) as f:
        keep = set(json.load(f)[a.task][a.split])
    rows = [r for r in rows if r.get("id") in keep]
    if not rows:
        raise RuntimeError(f"no rows of {a.task} in split {a.split!r}")
    # bench_grouped_eval's own draw for this group, then a prefix of it: same
    # seed, pool and group size, so these are that run's rows.
    rows = random.Random(a.seed).sample(
        rows, min(a.eval_samples_per_group, len(rows)))[:a.n_samples]
    for r in rows:
        r.setdefault("task", a.task)
    metas = build_metas(tokenizer, task := get_task(rows[0]["task"]), rows)
    del task
    if a.cache_metas:
        os.makedirs(os.path.dirname(os.path.abspath(a.cache_metas)), exist_ok=True)
        with open(a.cache_metas, "wb") as f:
            pickle.dump({"key": _meta_key(a), "metas": metas}, f)
    return metas


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--gate", required=True, help="/path/to/gate.pt@LAYER")
    p.add_argument("--pretrain-data-root", required=True)
    p.add_argument("--eval-split-manifest", required=True)
    p.add_argument("--dataset", default="longbench-datasets")
    p.add_argument("--task", default="longbench_repobench_p")
    p.add_argument("--model-name", default="Qwen3-8B")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--max-article-tokens", type=int, default=39000)
    p.add_argument("--eval-samples-per-group", type=int, default=475)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--phases", action="store_true", help="also print CUDA-event phase timings per sequence for the gate arm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-metas", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--gate-attn", required=True, choices=tuple(GATE_ATTN),
                   help="the gate arm's attention kernels (qwen3_flex_attention.GATE_ATTN)")
    p.add_argument("--gate-compile", required=True, choices=GATE_COMPILE,
                   help="torch.compile of the gate arm: none | p2 (the decoder layers above the gate)")
    p.add_argument("--gate-graphs", type=int, required=True, choices=[0, 1, 2],
                   help="1: the layers above the gate under per-layer CUDA graphs; 2: the whole Phase-2 loop as one graph "
                        "per shape; Phase 2 padded to 64 tokens either way (benchmark_gate_e2e.load_gate_model cuda_graphs)")
    p.add_argument("--merge-adapters", type=int, required=True, choices=[0, 1, 2],
                   help="1: fold the LoRA/delta adapters above the gate layer into the base weights; 2: every layer")
    p.add_argument("--baseline-attn", required=True, choices=BASELINE_ATTN,
                   help="causal-attention kernel of the baseline arm (see load_baseline)")
    p.add_argument("--baseline-compile", required=True, choices=BASELINE_COMPILE,
                   help="torch.compile of the baseline: none | layers (each decoder layer) | model")
    p.add_argument("--attn-crossover", type=int, default=None,
                   help="tokens at which plain causal attention switches from cuDNN to FA4 "
                        "(required by 'auto' in either arm; measure both kernels on the GPU at hand)")
    p.add_argument("--k", default="full",
                   help="the gate arm's K (kept article tokens): 'full' for the overhead measurement, "
                        "or an integer to time the pruned prefill against the full-context baseline")
    a = p.parse_args()
    configure_attn_crossover(a)
    k_run = FULL if a.k == "full" else int(a.k)
    k_label = "full" if k_run >= FULL else str(k_run)

    t0 = time.perf_counter()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    metas = _metas(a, tokenizer)
    ctx = statistics.mean(len(m["prompt_ids"]) for m in metas)
    print(f"[ttft] {len(metas)} sequences, one at a time, both arms paired per sequence, mean ctx {ctx:.0f} tok, "
          f"repeats={a.repeats} warmup={a.warmup}", flush=True)
    t0 = _phase("data", t0)

    # Both arms stay resident and are timed sequence by sequence, the two arms'
    # blocks of repeats back to back: the GPU's clock and the node's other
    # tenants drift on the scale of a sweep (the same baseline read 17.1 and
    # 19.9 ms for one sequence in two back-to-back sweeps), and pairing the
    # arms within seconds puts both under the same drift, so the ratio cancels it.
    vmodel = load_baseline(a.base_model, a.baseline_attn, a.baseline_compile)
    t0 = _phase(f"load baseline ({a.baseline_attn}, compile={a.baseline_compile})", t0)
    gpath, glayer = a.gate.rsplit("@", 1)
    gmodel, gate = load_gate_model(a.base_model, gpath, int(glayer), a.gate_attn,
                                   merge_adapters_above_gate=a.merge_adapters,
                                   compile_phase2=a.gate_compile, cuda_graphs=a.gate_graphs)[:2]
    t0 = _phase(f"load ours ({a.gate_attn}, merge={a.merge_adapters}, compile={a.gate_compile})", t0)
    resident_mb = torch.cuda.memory_allocated() / 2 ** 20

    # One unpadded sequence: no attention mask and no explicit position_ids,
    # because the defaults (all-attend, arange) are exactly right for it.  The
    # prompt tokens live on the device already, for both arms.
    ids = {id(m): torch.tensor([m["prompt_ids"]], device=DEVICE) for m in metas}
    gids = {id(m): torch.tensor(m["prompt_ids"], device=DEVICE) for m in metas}

    def baseline(m):
        baseline_prefill(vmodel, ids[id(m)])

    def ours(m):
        prefill_batch_for_decode(
            gmodel, gate, [gids[id(m)]], [m["article_start"]],
            [m["article_end"]], int(glayer), k_run)

    def _once(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    for m in metas:
        for _ in range(a.warmup):
            baseline(m)
            ours(m)
    from torch._dynamo.utils import counters as _dyn_counters
    _fr = dict(_dyn_counters["frames"])
    print(f"[ttft] dynamo frames after warmup: {_fr} (recompile_limit "
          f"{torch._dynamo.config.recompile_limit}; every frame must be 'ok')", flush=True)
    if _fr.get("total", 0) != _fr.get("ok", 0):
        raise RuntimeError(f"dynamo fell back to eager on {_fr.get('total', 0) - _fr.get('ok', 0)} frames: {_fr}")
    t0 = _phase(f"warmup ({a.warmup} per arm per sequence)", t0)

    # Per sequence: all repeats of one arm, then all of the other.  Alternating
    # single calls put the baseline at 27 ms where it measures 16 alone at 4.5k
    # tokens: each arm's call pattern leaves the caching allocator's free blocks
    # shaped for it, and the other arm then maps fresh memory inside the timed
    # region (device allocs are counted below; steady state is zero).  A block
    # of repeats pays that once and the median discards it.
    base, mine = [], []
    peak = {"baseline": 0.0, "ours": 0.0}
    allocs = {"baseline": 0, "ours": 0}

    def _n_alloc():
        return torch.cuda.memory_stats().get("num_device_alloc", 0)

    for m in metas:
        tb, to = [], []
        torch.cuda.reset_peak_memory_stats()
        n0 = _n_alloc()
        for _ in range(a.repeats):
            tb.append(_once(lambda: baseline(m)))
        allocs["baseline"] += _n_alloc() - n0
        peak["baseline"] = max(peak["baseline"], torch.cuda.max_memory_allocated() / 2 ** 20)
        torch.cuda.reset_peak_memory_stats()
        n0 = _n_alloc()
        for _ in range(a.repeats):
            to.append(_once(lambda: ours(m)))
        allocs["ours"] += _n_alloc() - n0
        peak["ours"] = max(peak["ours"], torch.cuda.max_memory_allocated() / 2 ** 20)
        base.append(statistics.median(tb) * 1000)
        mine.append(statistics.median(to) * 1000)
    print(f"[ttft] device allocations inside the timed repeats: baseline {allocs['baseline']}, "
          f"ours {allocs['ours']} (non-zero means the allocator mapped memory in the timed region)", flush=True)
    # Peaks are measured with both arms' weights resident; the other arm's
    # weights are subtracted so each number is that arm alone, as before.
    other = {"baseline": sum(p.numel() * p.element_size() for p in gmodel.parameters()) / 2 ** 20
             + sum(p.numel() * p.element_size() for p in gate.parameters()) / 2 ** 20,
             "ours": sum(p.numel() * p.element_size() for p in vmodel.parameters()) / 2 ** 20}
    base_mem = peak["baseline"] - other["baseline"]
    ours_mem = peak["ours"] - other["ours"]
    t0 = _phase(f"measure both arms, paired per sequence ({a.repeats} repeats)", t0)
    if a.phases:
        # One CUDA-event-timed run per sequence, after the timed ones: where the
        # gate arm's wall time sits, per phase, at every context length.
        set_prefill_phase_timing(True)
        print("  phases per sequence (ms):", flush=True)
        for m in sorted(metas, key=lambda m: len(m["prompt_ids"])):
            runs = []
            for _ in range(3):
                ours(m)
                runs.append(prefill_phase_ms())
            med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
            print(f"    {len(m['prompt_ids']):6d}  " + "  ".join(f"{k} {v:6.2f}" for k, v in med.items())
                  + f"  total {sum(med.values()):6.2f}", flush=True)
        set_prefill_phase_timing(False)

    def _sd(v):
        return statistics.stdev(v) if len(v) > 1 else 0.0

    base_ms, ours_ms = statistics.mean(base), statistics.mean(mine)
    # The per-arm spread is context-length spread across sequences, not timer
    # noise -- these prompts run 1k-39k tokens. The number to read is the PAIRED
    # ratio: both arms run the same sequences, so per-sequence ours/baseline
    # cancels the length and its spread is the real uncertainty on the target.
    ratios = [o / b for o, b in zip(mine, base)]
    print("\n  per sequence (tokens, baseline ms, ours ms, ratio):", flush=True)
    for m, bm, om, r in sorted(zip(metas, base, mine, ratios), key=lambda t: len(t[0]["prompt_ids"])):
        print(f"    {len(m['prompt_ids']):6d}  {bm:8.2f}  {om:8.2f}  {r:5.2f}x", flush=True)
    print(f"\n  baseline (full ctx, {a.baseline_attn}, compile={a.baseline_compile}) "
          f"{base_ms:8.2f} +- {_sd(base):6.2f} ms  peak={base_mem:.0f}MB", flush=True)
    print(f"  ours     (K={k_label}, {a.gate_attn}, merge={a.merge_adapters}, compile={a.gate_compile}) "
          f"{ours_ms:8.2f} +- {_sd(mine):6.2f} ms  peak={ours_mem:.0f}MB", flush=True)
    print(f"  (+- is spread over {len(base)} sequences of different lengths, not timer noise)", flush=True)
    print(f"  {'OVERHEAD' if k_run >= FULL else 'RATIO   '}  paired         {statistics.mean(ratios):8.2f}x +- {_sd(ratios):.2f}  "
          f"[{min(ratios):.2f}, {max(ratios):.2f}]  " + ("(target 1.00x)" if k_run >= FULL else f"(ours at K={k_label} / full-context baseline)") + "\n", flush=True)

    if a.out:
        # One row per run, appended: the file is the log of every measurement
        # (any GPU, any configuration), not the last one's scratch.
        header = ["time", "gpu", "n_seq", "mean_ctx", "gate",
                  "baseline_attn", "baseline_compile", "baseline_ms", "baseline_ms_std", "baseline_peak_mb",
                  "gate_attn", "merge_adapters", "gate_compile", "ours_ms", "ours_ms_std", "ours_peak_mb",
                  "overhead_paired", "overhead_std", "overhead_min", "overhead_max"]
        row = [time.strftime("%Y-%m-%d %H:%M:%S"), torch.cuda.get_device_name(), len(metas), round(ctx),
               os.path.basename(gpath) + f"@{glayer}" + ("" if k_run >= FULL else f"#K={k_label}"),
               a.baseline_attn, a.baseline_compile, round(base_ms, 3), round(_sd(base), 3), round(base_mem, 1),
               a.gate_attn, a.merge_adapters, a.gate_compile, round(ours_ms, 3), round(_sd(mine), 3),
               round(ours_mem, 1), round(statistics.mean(ratios), 4), round(_sd(ratios), 4),
               round(min(ratios), 4), round(max(ratios), 4)]
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        if os.path.exists(a.out):
            with open(a.out, newline="") as f:
                first = next(csv.reader(f), None)
            if first != header:
                raise RuntimeError(f"{a.out} has a different header ({first}); move it aside -- "
                                   f"this bench appends rows with {header}")
        with open(a.out, "a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(header)
            w.writerow(row)
        print(f"[ttft] appended to {a.out}", flush=True)


if __name__ == "__main__":
    main()
