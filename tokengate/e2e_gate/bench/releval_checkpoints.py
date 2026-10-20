"""Offline re-evaluation of a training run's SAVED gate checkpoints.

Rebuilds per-step eval curves from what is actually on disk: every saved
(gate_step{N}.pt, lora_step{N}.pt) pair is scored with the BENCHMARK eval path
at the requested K values plus the full-context anchor, so checkpoint selection
keys on reproducible from-disk numbers rather than the in-training eval series.

Protocol notes:
  * K is the TRAINING convention: kept-article-token count (nominal), clamped
    per sample to n_art; --include-full adds K=10M ("keep everything, bias on").
  * decode uses the exact phase-2 softmax form (the benchmark/deployment
    standard).
  * adapters stay UNMERGED (merge=0) so the checkpoint pair can be hot-swapped
    into the resident model; merge=0 vs merge=2 was verified numerically
    equivalent.
  * accuracy (primary `acc` column) = task.evaluate — SQuAD-normalized
    containment, identical to the bench harness AND the in-training eval; the
    raw-substring `acc_strict` column is secondary (raw containment fails
    generations that insert an article or punctuation).

Sharding: --num-shards/--shard-index split the CHECKPOINT list round-robin so
one lane fans out over GPUs; each shard appends to its own CSV.

Usage (one shard):
  python -m tokengate.e2e_gate.bench.releval_checkpoints \
      --gpu 0 --run-dir <.../baseline/L1> --gate-layer 1 \
      --k-values 128 2000 --include-full \
      --num-shards 8 --shard-index 0 --out releval_l1.shard0of8.csv
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

_gpu = None
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        _gpu = sys.argv[_i + 1]
if _gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tokengate.e2e_gate.bench.benchmark_gate_e2e import (  # noqa: E402
    build_metas, gate_generate_batch, load_gate_model, load_val_samples,
)
from tokengate.e2e_gate.gate_module.e2e_gate_module import load_gate  # noqa: E402
from tokengate.e2e_gate.gate_module.lora import load_lora_state_dict  # noqa: E402
from tokengate.datasets.tasks import get_task  # noqa: E402

DEVICE = "cuda"
FULL = 10_000_000
_STEP_RE = re.compile(r"gate_step(\d+)\.pt$")


def checkpoint_steps(run_dir):
    """Sorted steps having BOTH gate_step{N}.pt and lora_step{N}.pt."""
    steps = []
    for p in glob.glob(os.path.join(run_dir, "gate_step*.pt")):
        m = _STEP_RE.search(p)
        if not m:
            continue
        n = int(m.group(1))
        if os.path.isfile(os.path.join(run_dir, f"lora_step{n}.pt")):
            steps.append(n)
        else:
            raise FileNotFoundError(
                f"{run_dir}: gate_step{n}.pt has no lora_step{n}.pt sibling — "
                f"adapter pairing is by name only (CLAUDE.md), refusing to score "
                f"a mismatched pair")
    if not steps:
        raise FileNotFoundError(f"no gate_step*.pt under {run_dir}")
    return sorted(steps)


def load_pool(tokenizer, task, base_model, pretrain_data_root, split_manifest,
              max_article_tokens):
    """The niah val+test pool with per-meta group tags (niah / niah_heldout) —
    the same 16+184 rows the training eval and the benchmark lanes score."""
    with open(split_manifest) as f:
        sp = json.load(f)
    val_ids = set(sp["niah_kamradt"]["val"])
    test_ids = set(sp["niah_kamradt"]["test"])
    rows = load_val_samples(pretrain_data_root, "niah_kamradt", "niah_kamradt",
                            os.path.basename(base_model), 10_000_000,
                            max_article_tokens, None)
    rows = [r for r in rows if r.get("id") in val_ids | test_ids]
    metas = build_metas(tokenizer, task, rows)
    for m in metas:
        rid = m["sample"]["id"]
        m["group"] = "niah" if rid in val_ids else "niah_heldout"
    n_val = sum(1 for m in metas if m["group"] == "niah")
    print(f"[releval] pool: {len(metas)} metas ({n_val} niah/val, "
          f"{len(metas) - n_val} niah_heldout/test)", flush=True)
    return metas


@torch.no_grad()
def score_arm(model, gate, tokenizer, metas, gate_layer, K, iapl,
              batch_size, max_new_tokens, task):
    """(group -> (n_task, n_strict, n)) for one (checkpoint, K) arm.

    PRIMARY score = task.evaluate (SQuAD-normalized containment for
    niah_kamradt: lowercase, strip punctuation, drop articles) — identical to
    the bench harness and the in-training eval, so releval curves compose with
    both. The raw lowercased-substring 'strict' score is kept as a secondary
    column: the 2026-09-13 instrument hunt traced a phantom 0.995-vs-0.950
    "kernel/node/code" discrepancy to exactly this normalization difference
    (~9/200 generations differ only by punctuation/articles)."""
    hits = {}
    for i in range(0, len(metas), batch_size):
        mb = metas[i:i + batch_size]
        answers, *_ = gate_generate_batch(
            model, gate, tokenizer, mb, gate_layer, K, max_new_tokens, iapl,
            decode_phase2_form=True)
        for m, a in zip(mb, answers):
            tgt = m["sample"].get("target") or ""
            ok_task = bool(tgt.strip()) and task.evaluate(a, tgt) >= 1.0
            ok_strict = bool(tgt.strip()) and tgt.strip().lower() in a.strip().lower()
            c, cs, n = hits.get(m["group"], (0, 0, 0))
            hits[m["group"]] = (c + int(ok_task), cs + int(ok_strict), n + 1)
    return hits


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", required=True)
    p.add_argument("--run-dir", required=True, help=".../baseline/L{n} training output dir")
    p.add_argument("--gate-layer", type=int, required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--pretrain-data-root", default=os.path.join(
        REPO, "datasets"))
    p.add_argument("--split-manifest", default=os.path.join(
        REPO, "tokengate", "datasets", "eval_splits",
        "longbench_eval_split_niah_iso_v1.json"))
    p.add_argument("--max-article-tokens", type=int, default=17000)
    p.add_argument("--k-values", type=int, nargs="+", required=True,
                   help="kept-article-token K arms (training convention), e.g. 128 2000")
    p.add_argument("--include-full", action="store_true",
                   help="add the K=10M full-context arm (the corrected gatefull)")
    p.add_argument("--gate-attn", default="fa2",
                   help="benchmark-validated kernels; fa2 needs the fa2bias hdim256 build "
                        "(FA2BIAS_SITE), cudnn-p2 is the no-build fallback")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--out", required=True, help="CSV (appended; header on create)")
    a = p.parse_args()
    if not (0 <= a.shard_index < a.num_shards):
        p.error(f"--shard-index {a.shard_index} outside [0, {a.num_shards})")

    steps = checkpoint_steps(a.run_dir)
    my_steps = steps[a.shard_index::a.num_shards]
    lane = f"gate_l{a.gate_layer}"
    print(f"[releval] {lane} {a.run_dir}\n[releval] {len(steps)} ckpts total, "
          f"shard {a.shard_index}/{a.num_shards} -> {len(my_steps)}: {my_steps}", flush=True)
    arms = list(a.k_values) + ([FULL] if a.include_full else [])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    task = get_task("niah_kamradt")
    if hasattr(task, "use_judge"):
        task.use_judge = False
    metas = load_pool(tokenizer, task, a.base_model, a.pretrain_data_root,
                      a.split_manifest, a.max_article_tokens)

    # Resident model built from the FIRST shard checkpoint (unmerged adapters so
    # later checkpoints hot-swap via load_lora_state_dict + a fresh load_gate).
    first = os.path.join(a.run_dir, f"gate_step{my_steps[0]}.pt")
    model, gate, iapl = load_gate_model(
        a.base_model, first, a.gate_layer, a.gate_attn,
        merge_adapters_above_gate=0, compile_phase2="all", cuda_graphs=0)[:3]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new_file = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    out_f = open(a.out, "a", newline="")
    writer = csv.writer(out_f)
    if new_file:
        writer.writerow(["lane", "step", "K", "group", "acc", "acc_strict",
                         "n_correct", "n"])

    for si, step in enumerate(my_steps):
        if si > 0:  # hot-swap the pair (first ckpt is already resident)
            sd = torch.load(os.path.join(a.run_dir, f"lora_step{step}.pt"),
                            map_location=DEVICE)
            load_lora_state_dict(model, sd)
            gate = load_gate(os.path.join(a.run_dir, f"gate_step{step}.pt"))[0]
            gate = gate.to(DEVICE).float().eval()
        for K in arms:
            hits = score_arm(model, gate, tokenizer, metas, a.gate_layer, K,
                             iapl, a.batch_size, a.max_new_tokens, task)
            for grp, (c, cs, n) in sorted(hits.items()):
                writer.writerow([lane, step, K, grp, f"{c / n:.6f}",
                                 f"{cs / n:.6f}", c, n])
            out_f.flush()
            k_lab = "full" if K == FULL else K
            print(f"[releval] {lane} step={step} K={k_lab}  " +
                  "  ".join(f"{g}={c}/{n}(strict {cs})"
                            for g, (c, cs, n) in sorted(hits.items())),
                  flush=True)
    out_f.close()
    print(f"[releval] shard {a.shard_index}/{a.num_shards} DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
