"""Where one prefill's time goes, kernel by kernel, in each arm of bench_ttft.

Same data and models as bench_ttft; one sequence (--seq, default the one of
median length), one profiled prefill per arm after a warmup. For each arm:
wall time, the sum of device kernel time, the idle gap between them, the
biggest gaps with the kernel that ended each, host-side sync points, and the
top kernels by device time.
"""

import argparse
import os
import statistics
import sys
import time
from collections import defaultdict

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
from torch.profiler import ProfilerActivity, profile  # noqa: E402
torch.set_float32_matmul_precision("high")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.bench.bench_ttft import (  # noqa: E402
    _metas, BASELINE_ATTN, BASELINE_COMPILE, FULL, baseline_prefill, load_baseline, configure_attn_crossover,
)
from tokengate.e2e_gate.bench.benchmark_gate_e2e import (  # noqa: E402
    load_gate_model, GATE_COMPILE,
)
from tokengate.e2e_gate.gate_module.gate_forward import (  # noqa: E402
    prefill_batch_for_decode, prefill_phase_ms, set_prefill_phase_timing,
)
from tokengate.e2e_gate.attn.qwen3_flex_attention import GATE_ATTN  # noqa: E402

DEVICE = "cuda"
_SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync",
               "cudaMemcpy", "aten::_local_scalar_dense", "aten::nonzero",
               "aten::item")


def _device_events(prof):
    evs = []
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            evs.append((e.time_range.start, e.time_range.end, e.name))
    evs.sort()
    return evs


def _phase_table(prof):
    """Device time inside each prefill/* annotation: kernels whose start falls
    inside the annotation's GPU span (the annotations themselves and inductor's
    '## Call CompiledFxGraph' markers are excluded from the sums)."""
    spans = [(e.time_range.start, e.time_range.end, e.name) for e in prof.events()
             if e.device_type == torch.autograd.DeviceType.CUDA and e.name.startswith("prefill/")]
    if not spans:
        return
    kernels = [(e.time_range.start, e.time_range.end) for e in prof.events()
               if e.device_type == torch.autograd.DeviceType.CUDA
               and not e.name.startswith("prefill/") and not e.name.startswith("## ")]
    named = [(e.time_range.start, e.time_range.end, e.name) for e in prof.events()
             if e.device_type == torch.autograd.DeviceType.CUDA
             and not e.name.startswith("prefill/") and not e.name.startswith("## ")]
    print("  prefill phases (device ms, wall ms), with each phase's heaviest kernels:")
    for st, en, name in sorted(spans):
        inside = [(ks, ke, kn) for ks, ke, kn in named if st <= ks < en]
        busy = sum(ke - ks for ks, ke, _ in inside) / 1000
        print(f"    {name:22s} {busy:7.2f}  {(en - st)/1000:7.2f}   ({len(inside)} kernels)")
        if name.startswith("prefill/phase"):
            continue
        agg = defaultdict(lambda: [0.0, 0])
        for ks, ke, kn in inside:
            agg[kn][0] += (ke - ks) / 1000
            agg[kn][1] += 1
        for kn, (ms, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:8]:
            print(f"        {ms:6.3f}  {c:3d}  {kn[:90]}")


def _report(label, prof, wall_ms, top):
    evs = _device_events(prof)
    evs = [e for e in evs if not e[2].startswith("prefill/")]
    busy = sum(en - st for st, en, _ in evs) / 1000
    t0 = evs[0][0]
    gaps = []
    for (s0, e0, _), (s1, e1, n1) in zip(evs, evs[1:]):
        if s1 > e0:
            gaps.append((s1 - e0, e0 - t0, n1))
    gaps.sort(reverse=True)
    idle = sum(g for g, _, _ in gaps) / 1000
    print(f"\n=== {label}: wall {wall_ms:.1f} ms | device busy {busy:.1f} ms | "
          f"idle gaps {idle:.1f} ms ({len(gaps)} gaps, {len(evs)} device events)")
    print("  biggest gaps (ms, at ms, kernel that ended it):")
    for g, at, n in gaps[:12]:
        print(f"    {g/1000:7.2f}  @{at/1000:7.1f}  {n[:90]}")
    syncs = defaultdict(int)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CPU and e.name in _SYNC_NAMES:
            syncs[e.name] += 1
    print("  host sync points:", dict(syncs) or "none")
    agg = defaultdict(lambda: [0.0, 0])
    for st, en, n in evs:
        agg[n][0] += (en - st) / 1000
        agg[n][1] += 1
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    print(f"  top {top} kernels by device time (ms, count):")
    for n, (ms, c) in rows[:top]:
        print(f"    {ms:8.2f}  {c:5d}  {n[:100]}")


def _sync_stacks(prof, limit=12):
    """Python call sites of the host-side sync points (item / nonzero), so a
    stray .item() in a layer loop can be found without guessing."""
    from collections import Counter
    sites = Counter()
    for e in prof.events():
        if e.name in ("aten::_local_scalar_dense", "aten::nonzero") and e.stack:
            frames = [f for f in e.stack if "site-packages" not in f and "/torch/" not in f][:3]
            sites[" <- ".join(frames) or e.stack[0]] += 1
    print("  sync call sites:")
    for site, n in sites.most_common(limit):
        print(f"    {n:4d}  {site[:220]}")


def _profile_one(fn, warmup, with_stack=False):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=with_stack) as prof:
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000
    return prof, wall_ms


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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-metas", default=None)
    p.add_argument("--seq", type=int, default=None, help="index into the metas; default median length")
    p.add_argument("--seq-len", type=int, default=None, help="pick the sequence closest to this many tokens")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--arm", default="both", choices=["both", "baseline", "ours"])
    p.add_argument("--sync-stacks", action="store_true",
                   help="record Python stacks and print where the host syncs come from")
    p.add_argument("--gate-attn", required=True, choices=tuple(GATE_ATTN),
                   help="the gate arm's attention kernels (qwen3_flex_attention.GATE_ATTN)")
    p.add_argument("--gate-compile", required=True, choices=GATE_COMPILE,
                   help="torch.compile of the gate arm: none | p2 (the decoder layers above the gate)")
    p.add_argument("--gate-graphs", type=int, required=True, choices=[0, 1, 2],
                   help="1: the layers above the gate under per-layer CUDA graphs; 2: the whole Phase-2 loop as one graph "
                        "per shape; Phase 2 padded to 64 tokens either way (benchmark_gate_e2e.load_gate_model cuda_graphs)")
    p.add_argument("--merge-adapters", type=int, required=True, choices=[0, 1, 2],
                   help="1: fold the LoRA/delta adapters above the gate layer into the base weights; 2: every layer")
    p.add_argument("--baseline-attn", required=True, choices=BASELINE_ATTN)
    p.add_argument("--baseline-compile", required=True, choices=BASELINE_COMPILE)
    p.add_argument("--attn-crossover", type=int, default=None,
                   help="tokens at which plain causal attention switches from cuDNN to FA4 (required by 'auto')")
    a = p.parse_args()
    configure_attn_crossover(a)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    metas = _metas(a, tokenizer)
    lens = [len(m["prompt_ids"]) for m in metas]
    if a.seq_len is not None:
        a.seq = min(range(len(lens)), key=lambda i: abs(lens[i] - a.seq_len))
    elif a.seq is None:
        a.seq = sorted(range(len(lens)), key=lambda i: lens[i])[len(lens) // 2]
    m = metas[a.seq]
    print(f"[profile] seq {a.seq}: {lens[a.seq]} tok (all: {sorted(lens)})", flush=True)

    if a.arm in ("both", "baseline"):
        vmodel = load_baseline(a.base_model, a.baseline_attn, a.baseline_compile)
        ids = torch.tensor([m["prompt_ids"]], device=DEVICE)

        def baseline():
            baseline_prefill(vmodel, ids)

        prof, wall = _profile_one(baseline, a.warmup, a.sync_stacks)
        _report(f"baseline ({a.baseline_attn})", prof, wall, a.top)
        if a.sync_stacks:
            _sync_stacks(prof)
        del vmodel, ids, prof
        torch.cuda.empty_cache()

    if a.arm in ("both", "ours"):
        gpath, glayer = a.gate.rsplit("@", 1)
        gmodel, gate = load_gate_model(a.base_model, gpath, int(glayer), a.gate_attn,
                                   merge_adapters_above_gate=a.merge_adapters,
                                   compile_phase2=a.gate_compile, cuda_graphs=a.gate_graphs)[:2]

        gids = torch.tensor(m["prompt_ids"], device=DEVICE)

        def ours():
            with torch.no_grad():
                prefill_batch_for_decode(
                    gmodel, gate, [gids], [m["article_start"]],
                    [m["article_end"]], int(glayer), FULL)

        prof, wall = _profile_one(ours, a.warmup, a.sync_stacks)
        _report("ours (K=full)", prof, wall, a.top)
        _phase_table(prof)
        # CUDA-event phase timing, median of 5 unprofiled runs (the profiler's
        # per-kernel overhead is out of these numbers).
        set_prefill_phase_timing(True)
        runs = []
        for _ in range(5):
            ours()
            runs.append(prefill_phase_ms())
        set_prefill_phase_timing(False)
        print("  prefill phases by CUDA events (median ms of 5 runs):")
        for name in runs[0]:
            print(f"    {name:14s} {statistics.median(r[name] for r in runs):7.2f}")
        print(f"    {'total':14s} {statistics.median(sum(r.values()) for r in runs):7.2f}")
        if a.sync_stacks:
            _sync_stacks(prof)


if __name__ == "__main__":
    main()
