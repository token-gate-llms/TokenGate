"""Unified accuracy benchmark on the EXACT training-eval set.

Every baseline is evaluated on the SAME samples — training's pooled-per-metric-group
eval set (`random.Random(seed).sample(eval_samples_per_group)` over each dataset's
first-N val rows, reconstructing train_e2e_gate.py:2649-2665). This is the only way
the numbers are comparable to the training run's W&B `eval/cat_<group>` lines AND to
each other.

Baselines, all on the identical per-group sample set:
  * no-comp  — vanilla Qwen3 (no gate, no LoRA), full context. This is exactly the
               model training's no-compression baseline uses (LoRA-B is zero-init and
               the baseline runs before any training step → adapter is a no-op).
  * gate@K   — the learned gate pruned to K, for each K in --k-values.
  * gate@full— the gate model with no pruning (K = ∞), i.e. the LoRA ceiling.

Builds metas ONCE per group (same article-boundary filter for all baselines), so
no-comp / gate@K / gate@full score the IDENTICAL rows. Writes per-sample rows to
{out-dir}/grouped_eval.csv and prints a per-group + overall table per baseline.

Usage:
  python -m tokengate.e2e_gate.bench.bench_grouped_eval --gpu 6 \
    --gate-path <...> --gate-layer 0 --manifest v3_accepted28 \
    --pretrain-data-root <...> --k-values 100 250 500 1000 2000 4000 \
    --eval-samples-per-group 50 --seed 0 --max-new-tokens 1024 --batch-size 4 \
    --out-dir benchmarks/Qwen3-8B/grouped_eval
"""

import argparse
import csv
import json
import os
import random
import sys

_args_gpu = None
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        _args_gpu = sys.argv[_i + 1]
if (_args_gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ
        and "WORLD_SIZE" not in os.environ):   # under torchrun each rank uses LOCAL_RANK, not --gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = _args_gpu
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokengate.e2e_gate.attn.qwen3_flex_attention import GATE_ATTN  # noqa: E402
from tokengate.e2e_gate.bench.benchmark_gate_e2e import (  # noqa: E402
    load_val_samples, build_metas, load_gate_model, gate_generate_batch,
    load_vanilla_model, vanilla_generate_batch, _is_oom,
)
from tokengate.e2e_gate.bench.benchmark_sanity_check import (  # noqa: E402
    manifest_pairs,
)
# Use the SAME category map the trainer uses (tasks.get_task_category) so the
# pooled groups match the in-training eval exactly.
from tokengate.datasets.tasks import get_task, get_task_category  # noqa: E402
from tokengate.datasets.manifest import (  # noqa: E402
    eval_only_pairs_csv,
)
from tokengate.e2e_gate.bench.answer_type import classify_answer  # noqa: E402

csv.field_size_limit(10 ** 8)
FIELDS = ["baseline", "K", "k_frac", "group", "dataset", "task", "sample_idx",
          # answer_type: what the gold answer REQUIRES (extractive / abstractive /
          # yes_no / unanswerable) — dominates accuracy-at-budget far more than the
          # selection rule, so the plots break out one panel per type.
          "answer_type",
          "accuracy", "f1", "n_gen_tokens", "n_kept_tokens", "n_ctx_tokens",
          # actual-run timing (the ACTUAL generation that produced this row's answer),
          # batch-amortized to per-sample: prefill_s/decode_s = batch time / batch size.
          "prefill_s", "decode_s", "decode_steps",
          # bs: the ACTUAL batch size this sample's generation ran at (= the
          # amortization denominator of prefill_s/decode_s). Under
          # --adaptive-batch-size an OOMing batch retries at half size, so bs
          # can be < --batch-size; plots must mark such points distinctly for
          # fair TTFT reads. Under --fixed-batch-size bs is constant.
          "bs",
          # oom: 1 = this sample's batch OOMed at the mode's floor (fixed: the
          # target bs; adaptive: bs=1) — metrics empty; plots mark, never average.
          "oom"]


def _batched(metas, bs):
    for i in range(0, len(metas), bs):
        yield metas[i:i + bs]




def _free_graph_state(model):
    """Adaptive-halving support: drop the captured Phase-2 CUDA graphs and
    their per-batch-size static buffers before retrying at a smaller batch —
    otherwise the retry allocates a fresh buffer set ON TOP of the retained
    one and every halving step makes the OOM strictly worse (the full-context
    arms then die at the bs=1 floor; see phase2_graph.release)."""
    from tokengate.e2e_gate.gate_module.phase2_graph import release
    n = release(model)
    if n:
        print(f"  [oom] released {n} captured graphs + their static buffers before the retry", flush=True)


def _gate_attn(args):
    if args.gate_attn is None:
        raise ValueError(f"--gate-attn is required to load a gate model ({' | '.join(GATE_ATTN)})")
    return args.gate_attn


def _merge_adapters(args):
    if args.merge_adapters is None:
        raise ValueError("--merge-adapters 0|1|2 is required to load a gate model")
    return args.merge_adapters


def _gate_load_kwargs(args):
    """The inference configuration of a gate model, every knob explicit (the
    TTFT bench and the accuracy harness must score what they time)."""
    from tokengate.e2e_gate.attn.flex_attention_utils import set_attn_crossover
    if args.gate_compile is None or args.gate_graphs is None:
        raise ValueError("--gate-compile none|p2|all and --gate-graphs 0|1|2 are required to load a gate model")
    if GATE_ATTN[_gate_attn(args)][0] == "auto":
        if args.attn_crossover is None:
            raise ValueError(f"--gate-attn {args.gate_attn} picks the plain kernel by length: --attn-crossover N is required")
        set_attn_crossover(args.attn_crossover)
    return dict(merge_adapters_above_gate=_merge_adapters(args), compile_phase2=args.gate_compile,
                cuda_graphs=args.gate_graphs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--gate-path", default=None,
                   help="gate checkpoint; required for the gate lanes "
                        "(gate / gate-full / gate-random-topk)")
    p.add_argument("--gate-attn", choices=tuple(GATE_ATTN), default="fa2",
                   help="the gate arm's attention kernels (qwen3_flex_attention.GATE_ATTN). "
                        "Default fa2 = FA2 flash causal + the FlashAttention-2 key-bias "
                        "kernel (kernels/fa2bias) — the final-run configuration, numerically "
                        "matching training's kernels")
    p.add_argument("--gate-compile", choices=["none", "p2", "all"], default=None,
                   help="torch.compile of the gate model (benchmark_gate_e2e.load_gate_model); required with a gate")
    p.add_argument("--gate-graphs", type=int, choices=[0, 1, 2], default=None,
                   help="CUDA graphs for the layers above the gate (0 none, 1 per layer, 2 the whole Phase-2 loop); "
                        "required with a gate")
    p.add_argument("--attn-crossover", type=int, default=None,
                   help="tokens at which an 'auto' causal backend switches from cuDNN to FA4")
    p.add_argument("--merge-adapters", type=int, choices=[0, 1, 2], default=None,
                   help="1: fold the LoRA/delta adapters above the gate layer into the base weights "
                        "(benchmark_gate_e2e.load_gate_model); required whenever a gate model is loaded")
    p.add_argument("--gate-layer", type=int, default=None,
                   help="required alongside --gate-path (see there)")
    p.add_argument("--data-source", choices=["grouped", "heldout"], default="grouped",
                   help="grouped: training's pooled-per-metric-group val set. "
                        "heldout: each OOD dataset (HELDOUT_DATASETS) as its own group.")
    p.add_argument("--manifest", default=None, help="(grouped mode) pretraining manifest")
    p.add_argument("--pretrain-data-root", default=None, help="(grouped mode) val.pkl root")
    p.add_argument("--groups", default=None,
                   help="comma list to restrict to specific groups/datasets (for GPU sharding)")
    p.add_argument("--group-by", choices=["category", "subset"], default="category",
                   help="category: pool by metric group (training-parity, 50/group). "
                        "subset: each LongBench subset is its own group (50/subset) "
                        "→ per-dataset accuracy-vs-K breakdown across all baselines.")
    p.add_argument("--eval-split-manifest", default=None,
                   help="(grouped mode) longbench_eval_split JSON. When set, eval is "
                        "restricted to that manifest's LongBench subsets and to the "
                        "rows of the chosen --split partition (see below), so the "
                        "benchmark uses the SAME val/test partition as training.")
    p.add_argument("--split", choices=["val", "test", "all"], default=None,
                   help="which partition of --eval-split-manifest to evaluate. "
                        "'val' reproduces the in-training eval (50/group); 'test' is "
                        "the reserved held-out benchmark set (use a large "
                        "--eval-samples-per-group to cover all test rows).")
    p.add_argument("--k-values", type=int, nargs="*", default=None,
                   help="absolute K token budgets (one curve point per K).")
    p.add_argument("--k-frac-list", type=float, nargs="+", default=None,
                   help="fractional K budgets: per-sample K = round(frac × n_ctx), "
                        "so points align by compression ratio across variable-length "
                        "subsets. Mutually exclusive with --k-values; x-axis becomes "
                        "the fraction (no-comp anchor at frac=1.0).")
    p.add_argument("--decode-phase2-form", action=argparse.BooleanOptionalAction, default=True,
                   help="gate lanes DECODE with the exact phase-2 softmax form "
                        "(learned scales + per-key gate bias over the pruned "
                        "cache) instead of plain attention.  DEFAULT ON since "
                        "2026-09-09 (EXPT C: +1.3-2.0 pts at compressed arms, "
                        "TTFT unchanged); --no-decode-phase2-form restores the "
                        "plain decode (how the pre-EXPT-C lanes were scored)")
    p.add_argument("--prune-below-gate", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="prune the phase-1 KV cache at layers <= the gate/TSP layer "
                        "down to the kept tokens after prefill (smaller decode cache; "
                        "the pre-2026-09 behavior).  DEFAULT OFF: those layers keep "
                        "the FULL phase-1 KV through decode (decode routes a "
                        "full-width pad mask to them; accuracy-identical per the "
                        "noprunebelowL ablation)")
    p.add_argument("--phase2-lengths", type=int, nargs="*", default=None,
                   help="target TOTAL phase-2 prompt lengths T (kept article + "
                        "non-prunable scaffold): per-sample K = max(1, T − scaffold), "
                        "then clamped to the article length as usual, so the pruned "
                        "prompt hits T exactly wherever achievable. The CSV K column "
                        "records T (the arm id). Mutually exclusive with --k-values / "
                        "--k-frac-list.")
    p.add_argument("--eval-samples-per-group", type=int, required=True)
    p.add_argument("--sample-slice", default=None,
                   help="A/N: score only the drawn samples whose index is A modulo N (indices are kept, so "
                        "the N slices' CSVs merge into the one run; kernels/bench/bench_merge.py). With an empty "
                        "--k-values only the full-context arms run: one GPU per (lane, K, slice).")
    p.add_argument("--val-samples-per-dataset", type=int, default=50)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--max-article-tokens", type=int, default=39000,
                   help="39k = the standing eval protocol (Qwen3-8B 40,960 ctx minus prompt overhead + 1,024 gen)")
    p.add_argument("--max-answer-tokens", type=int, default=1024)
    p.add_argument("--adaptive-batch-size", action="store_true",
                   help="halve the batch and retry when a batch OOMs (8->4->2->1), "
                        "resetting to --batch-size for each new batch; every row "
                        "records the ACTUAL bs used (bs column) so plots can mark "
                        "reduced-batch points; a sample that still OOMs at bs=1 is "
                        "recorded as oom=1. Mutually exclusive with --fixed-batch-size.")
    p.add_argument("--timing-warmup", type=int, default=1,
                   help="discarded prefill+decode passes before each (baseline, K) "
                        "sweep is timed, so compile/autotune cost stays out of "
                        "prefill_s. 0 restores the un-warmed timings.")
    p.add_argument("--fixed-batch-size", action="store_true",
                   help="disable every OOM fallback: all arms run at exactly "
                        "--batch-size so TTFT is batch-fair across every point; "
                        "a batch that OOMs is recorded per-sample as oom=1 rows "
                        "(empty metrics) and skipped, and plots must mark the "
                        "point as OOM rather than average it")
    p.add_argument("--full-ctx-k", type=int, default=10_000_000,
                   help="K for the gate@full baseline (clamped to n_art per sample → no pruning)")
    p.add_argument("--full-ctx-only", action="store_true",
                   help="score only the full-context arms (checkpoint re-scoring); "
                        "takes no K source")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--skip-no-comp", action="store_true")
    p.add_argument("--no-comp-adapter", default=None)
    p.add_argument("--no-comp-adapter-alpha", type=float, default=None)
    p.add_argument("--no-comp-tag", default="no_compress")
    p.add_argument("--skip-gate", action="store_true")
    p.add_argument("--gate-tag", default="gate",
                   help="CSV baseline tag for the gate rows (full point gets '<tag>_full'). "
                        "Lets two gate checkpoints (e.g. regular vs SWA) coexist as distinct "
                        "baselines in one benchmark experiment.")
    p.add_argument("--skip-gate-full", action="store_true",
                   help="skip the gate full-ctx anchor point (for K-top-up runs "
                        "merging into a CSV that already has gate_full rows)")
    p.add_argument("--gate-random-topk", action="store_true",
                   help="replace the gate's top-K selection with uniform random selection "
                        "INSIDE the gate path (bias/attn_scale behavior otherwise unchanged; "
                        "combine with --zero-gate-bias for the selection-isolating control "
                        "twin of gate_nobias)")
    p.add_argument("--zero-gate-bias", action="store_true",
                   help="Zero the gate's additive phase-2 attention bias at eval "
                        "(gate.zero_gate_bias; build_gate_bias emits zeros). Selection "
                        "is UNCHANGED (topk runs on the raw scores) — this isolates "
                        "the gate's SELECTION from its bias.")
    p.add_argument("--incremental-plot", action="store_true",
                   help="re-render the combined accuracy figure after EACH baseline finishes "
                        "(rank 0). Needs --plot-baselines; --plot-note/--manifest optional.")
    p.add_argument("--plot-baselines", nargs="+", default=None,
                   help="baseline list handed to plot_accuracy_vs_k.py for --incremental-plot")
    p.add_argument("--plot-note", default="", help="title note for --incremental-plot")
    p.add_argument("--merge-existing-csv", action="store_true",
                   help="seed the output CSV from an existing grouped_eval.csv (rank 0) so a run "
                        "of only NEW baselines is MERGED into it instead of overwriting. Each "
                        "baseline that finishes replaces any prior rows with its tag, so re-running "
                        "a baseline updates it in place. Use to add a baseline to a completed split.")
    args = p.parse_args()
    _k_sources = [args.k_values, args.k_frac_list, args.phase2_lengths]
    if args.full_ctx_only:
        if any(s is not None for s in _k_sources):
            p.error("--full-ctx-only takes no --k-values, --k-frac-list or --phase2-lengths")
    elif sum(s is not None for s in _k_sources) != 1:
        p.error("provide exactly one of --k-values, --k-frac-list or --phase2-lengths")
    _slice = None
    if args.sample_slice:
        _a, _n = (int(x) for x in args.sample_slice.split("/"))
        if not (0 <= _a < _n):
            p.error(f"--sample-slice {args.sample_slice}: need 0 <= A < N")
        _slice = (_a, _n)
    if args.fixed_batch_size and args.adaptive_batch_size:
        p.error("--fixed-batch-size and --adaptive-batch-size are mutually exclusive")
    _needs_gate = (not args.skip_gate or not args.skip_gate_full
                   or args.gate_random_topk)
    if _needs_gate and (args.gate_path is None or args.gate_layer is None):
        p.error("--gate-path/--gate-layer are required for the requested lanes "
                "(pass --skip-gate --skip-gate-full to run only no-comp)")
    frac_mode = args.k_frac_list is not None
    # ── DDP (torchrun): shard eval samples across ranks, gather to rank 0. Every
    #    baseline runs on ALL ranks (same resources); single-process (WORLD_SIZE
    #    unset) is byte-identical to the pre-DDP path. ─────────────────────────
    _WS = int(os.environ.get("WORLD_SIZE", "1"))
    _RANK = int(os.environ.get("RANK", "0"))
    _DDP = _WS > 1
    _is_main = _RANK == 0
    if _DDP:
        import torch.distributed as dist
        from datetime import timedelta
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        # Large collective timeout (default 600s is far too tight). The per-baseline
        # all_gather_object is a single barrier at the END of a baseline, so ranks with
        # heavier [rank::world_size] shards (more long-context samples) arrive much later;
        # on the big test split that cumulative skew exceeded 600s and aborted the whole
        # job (seen on base/test + swa/test mid-gate). 6h >> any single job's runtime, so
        # the barrier never false-fires within a run; a true hang is caught by the monitor.
        dist.init_process_group("nccl", timeout=timedelta(hours=6))

    # K "specs" that the per-K baselines (gate / random-K / FastKV) iterate.  Each
    # spec yields a per-batch K (scalar in absolute mode; a per-sample list in
    # fractional / phase2-length mode), a display Klabel written to the CSV
    # K column, and the k_frac written to the CSV k_frac column ("" in abs mode).
    # phase2-length mode: the arm targets a TOTAL pruned-prompt length T, so
    # K = T − scaffold per sample (scaffold = n_ctx − article length, the
    # non-prunable prefix+suffix); the CSV K column records T itself.
    def _k_specs():
        if args.full_ctx_only:
            return []
        if frac_mode:
            return [("frac", fr) for fr in args.k_frac_list]
        if args.phase2_lengths is not None:
            return [("p2len", T) for T in args.phase2_lengths]
        return [("abs", K) for K in args.k_values]

    def _batch_k(spec, batch):
        kind, v = spec
        if kind == "frac":
            return [max(1, round(v * m["n_ctx"])) for m in batch]
        if kind == "p2len":
            return [max(1, v - (m["n_ctx"] - (m["article_end"] - m["article_start"])))
                    for m in batch]
        return v

    def _labels(spec):
        # (K_label_for_csv, k_frac_for_csv); for fractional, the per-row K is the
        # per-sample budget (filled in by _score_and_record), so Klabel is unused.
        kind, v = spec
        return (None, f"{v:g}") if kind == "frac" else (v, "")

    os.makedirs(args.out_dir, exist_ok=True)
    # Provenance: snapshot the resolved args next to the results (the training
    # loop's params.json analogue — the bench previously recorded nothing).
    if _is_main:
        with open(os.path.join(args.out_dir, "args.json"), "w") as _af:
            json.dump(vars(args), _af, indent=1, default=str)
    out_csv = os.path.join(args.out_dir, "grouped_eval.csv")
    model_name = os.path.basename(args.base_model)
    want = set(args.groups.split(",")) if args.groups else None

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ── Assemble eval_by_group ────────────────────────────────────────────────
    if args.data_source == "grouped":
        # Training's exact pooled-per-metric-group val set.  Mirrors the trainer's
        # grouping (train_e2e_gate.py): category is resolved PER TASK (subsets in a
        # shared-dataset-name mix like LongBench-matched would otherwise collapse),
        # the held-out (eval-only) subsets are included, and a subset whose metric
        # family also has a trained member goes to "<category>_heldout".
        assert args.manifest and args.pretrain_data_root, "grouped mode needs --manifest + --pretrain-data-root"
        trained_pairs = list(manifest_pairs(args.manifest))
        _eo_csv = eval_only_pairs_csv(args.manifest)
        eo_pairs = [tuple(p.split(":", 1)) for p in _eo_csv.split(",") if p]
        eo_set = set(eo_pairs)
        trained_cats = {get_task_category(tk) for _, tk in trained_pairs}

        def _group_label(tk, is_eo):
            c = get_task_category(tk)
            return f"{c}_heldout" if (is_eo and c in trained_cats) else c

        # Optional LongBench val/test split: restrict eval to the manifest's subsets
        # and to the chosen partition's row ids (same split training uses).
        _split_ids = None
        if args.eval_split_manifest:
            assert args.split, "--eval-split-manifest requires --split {val,test,all}"
            with open(args.eval_split_manifest) as _sf:
                _sp = json.load(_sf)
            # "all" = the union of the val and test partitions in ONE run (all
            # available benchmark rows, timed together — no posthoc join)
            _split_ids = {tk: (set(v["val"]) | set(v["test"]) if args.split == "all"
                               else set(v[args.split]))
                          for tk, v in _sp.items()}

        grp_pool = {}
        for ds, tk in trained_pairs + eo_pairs:
            if _split_ids is not None and tk not in _split_ids:
                continue  # LongBench-only when a split manifest is active
            # When splitting, load ALL val rows (not just the first N) so every
            # partition id is present before we filter to it.
            _ncap = args.val_samples_per_dataset if _split_ids is None else 10_000_000
            # Eval must score EVERY manifest row. The article/answer-length caps are
            # TRAINING-data filters (bound the phase-2 loss), NOT eval filters: a long
            # reference answer is still scored against the (max_new_tokens-capped)
            # generation. In split-manifest (eval) mode we therefore NEVER drop by
            # reference-answer length, and keep the article cap only as a model-context
            # safeguard (set it to the model's usable context, not the training value).
            _ans_cap = None if _split_ids is not None else args.max_answer_tokens
            rows = load_val_samples(args.pretrain_data_root, ds, tk, model_name,
                                    _ncap, args.max_article_tokens, _ans_cap)
            if _split_ids is not None:
                _keep = _split_ids[tk]
                rows = [r for r in rows if r.get("id") in _keep]
            # Per-subset identity = task: rewrite "dataset" so the per-dataset
            # breakdown buckets per subset (matches the trainer).
            for r in rows:
                r["dataset"] = tk
            # category mode → pooled metric groups (training parity); subset mode →
            # each subset is its own group (per-dataset breakdown for all baselines).
            _key = tk if args.group_by == "subset" else _group_label(tk, (ds, tk) in eo_set)
            grp_pool.setdefault(_key, []).extend(rows)
        if _split_ids is not None and not grp_pool:
            raise RuntimeError(
                f"--eval-split-manifest tracks task(s) {sorted(_split_ids)} but none of "
                f"them appear in --manifest {args.manifest!r} (datasets or "
                f"eval_only_datasets) — the bench would silently score nothing.")
        grng = random.Random(args.seed)
        eval_by_group = {g: grng.sample(pool, min(args.eval_samples_per_group, len(pool)))
                         for g, pool in grp_pool.items()}
    else:
        # held-out OOD datasets — each dataset is its own group (same loader as
        # benchmark_gate_e2e's heldout path). All baselines score identical rows.
        from tokengate.e2e_gate.bench.benchmark_gate_e2e import (
            HELDOUT_DATASETS, load_heldout_samples)
        eval_by_group = {}
        for ds_key, tk, kwargs, disp, seed, max_n in HELDOUT_DATASETS:
            if want and disp not in want:
                continue
            samples = load_heldout_samples(
                ds_key, tk, kwargs, args.eval_samples_per_group,
                args.max_article_tokens, args.max_answer_tokens, tokenizer=tokenizer,
                seed=seed if seed is not None else args.seed,
                max_n=args.eval_samples_per_group)
            for s in samples:
                s.setdefault("task", tk)
            eval_by_group[disp] = samples

    if want:
        eval_by_group = {g: s for g, s in eval_by_group.items() if g in want}

    # ── Build metas ONCE per group (same article-boundary filter for ALL
    #    baselines → identical rows scored everywhere) ─────────────────────────
    metas_by_group = {}
    for g, samples in eval_by_group.items():
        metas = []
        for idx, s in enumerate(samples):
            if _slice is not None and idx % _slice[1] != _slice[0]:
                continue
            task = get_task(s["task"])
            # Use token-level metrics (F1/ROUGE/acc), not an LLM judge — matches the
            # training-eval protocol (judge_model=null) and avoids a live API call.
            if hasattr(task, "use_judge"):
                task.use_judge = False
            ml = build_metas(tokenizer, task, [s])
            if ml:
                m = ml[0]; m["idx"] = idx; m["task_obj"] = task
                metas.append(m)
        metas_by_group[g] = metas
    if _is_main:
        print("[grouped-eval] eval set (identical for every baseline):", flush=True)
        for g, metas in metas_by_group.items():
            src = {}
            for m in metas:
                src[m["sample"]["dataset"]] = src.get(m["sample"]["dataset"], 0) + 1
            print(f"  {g:16s}: n={len(metas)}  from {src}", flush=True)

    if _DDP:
        # Each rank scores a disjoint [rank::world_size] stride of every group's
        # samples (deterministic metas order → disjoint + exhaustive coverage).
        metas_by_group = {g: m[_RANK::_WS] for g, m in metas_by_group.items()}

    rows_out = []
    _master_rows = []        # rank 0: cumulative gathered rows (the CSV source)
    _flushed = [0]           # local rows_out index already flushed to rank 0

    if _is_main and args.merge_existing_csv and os.path.exists(out_csv):
        # Seed from the prior CSV so a run of only NEW baselines merges into it. Each
        # baseline replaces its own tag's rows on finish (below), so this never dups.
        with open(out_csv, newline="") as _f:
            _master_rows.extend(list(csv.DictReader(_f)))
        print(f"[grouped-eval] merge: seeded {len(_master_rows)} existing rows from {out_csv}",
              flush=True)

    def _write_csv(rows):
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    def _replot():
        if not args.plot_baselines:
            return
        _repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        png = os.path.join(args.out_dir, f"accuracy_vs_K_{args.split or 'eval'}.png")
        cmd = [sys.executable, os.path.join(_repo, "benchmarks", "plot_accuracy_vs_k.py"),
               "--csv", out_csv, "--by", "both", "--out", png,
               "--baselines", *args.plot_baselines, "--title-note", args.plot_note]
        if args.manifest:
            cmd += ["--manifest", args.manifest]
        try:
            import subprocess
            subprocess.call(cmd)
        except Exception as _e:
            print(f"[grouped-eval] incremental replot skipped: {_e}", flush=True)

    def _finish_baseline(name):
        """Gather this baseline's new rows to rank 0; rank 0 writes the cumulative
        CSV and (if --incremental-plot) re-renders the figure. Called by EVERY rank
        the same number of times → the all_gather collective stays balanced."""
        new_local = rows_out[_flushed[0]:]
        _flushed[0] = len(rows_out)
        if _DDP:
            import torch.distributed as dist
            gathered = [None] * _WS
            dist.all_gather_object(gathered, new_local)
            new_all = [r for shard in gathered for r in shard]
        else:
            new_all = new_local
        if _is_main:
            if new_all:  # replace any prior rows with this baseline's tags (merge/re-run safe)
                _new_tags = {r["baseline"] for r in new_all}
                _master_rows[:] = [r for r in _master_rows if r["baseline"] not in _new_tags]
            _master_rows.extend(new_all)
            _write_csv(_master_rows)
            if args.incremental_plot:
                _replot()
            print(f"[grouped-eval] baseline '{name}' done: +{len(new_all)} rows, "
                  f"{len(_master_rows)} total → {out_csv}", flush=True)
        if _DDP:
            import torch.distributed as dist
            dist.barrier()

    def _score_and_record(baseline, K, g, batch, answers, n_gen, n_kept, k_frac="",
                          prefill_s=0.0, decode_s=0.0, decode_steps=0):
        # K may be a scalar (absolute mode / no-comp / full-ctx) or a per-sample
        # list (fractional mode); the recorded per-row K is the per-sample budget.
        accs = []
        _nb = max(1, len(batch))
        _pf_i, _dec_i = prefill_s / _nb, decode_s / _nb   # batch time → per-sample amortized
        for j, (m, ans, ng, nk) in enumerate(zip(batch, answers, n_gen, n_kept)):
            t, s = m["task_obj"], m["sample"]
            a = t.evaluate(ans, s["target"], sample=s)
            f = t.evaluate_f1(ans, s["target"], sample=s)
            accs.append(a)
            k_i = K[j] if isinstance(K, (list, tuple)) else K
            rows_out.append({
                "baseline": baseline, "K": k_i, "k_frac": k_frac, "group": g,
                "dataset": s["dataset"], "task": s["task"], "sample_idx": m["idx"],
                "answer_type": classify_answer(s["target"], s.get("article", "")),
                "accuracy": f"{a:.6f}", "f1": f"{f:.6f}", "n_gen_tokens": ng,
                "n_kept_tokens": nk, "n_ctx_tokens": m["n_ctx"],
                "prefill_s": f"{_pf_i:.6f}", "decode_s": f"{_dec_i:.6f}",
                "decode_steps": decode_steps, "bs": _nb, "oom": 0,
            })
        return accs

    def _record_oom(baseline, K, g, batch, k_frac=""):
        # --fixed-batch-size mode: a batch that OOMs is recorded, never retried
        # at a smaller batch — rows carry oom=1 with empty metrics so the plot
        # marks the point instead of averaging over a silently changed batch
        # size (unfair TTFT) or silently dropped samples.
        for j, m in enumerate(batch):
            k_i = K[j] if isinstance(K, (list, tuple)) else K
            s = m["sample"]
            rows_out.append({
                "baseline": baseline, "K": k_i, "k_frac": k_frac, "group": g,
                "dataset": s["dataset"], "task": s["task"], "sample_idx": m["idx"],
                "answer_type": classify_answer(s["target"], s.get("article", "")),
                "accuracy": "", "f1": "", "n_gen_tokens": "",
                "n_kept_tokens": "", "n_ctx_tokens": m["n_ctx"],
                "prefill_s": "", "decode_s": "", "decode_steps": "", "bs": "",
                "oom": 1,
            })
        print(f"  [oom] {baseline} K={K if not isinstance(K, (list, tuple)) else 'frac'}"
              f" group={g}: {len(batch)} samples recorded as OOM "
              f"(floor of {'fixed bs ' + str(args.batch_size) if args.fixed_batch_size else 'adaptive halving (bs=1)'})",
              flush=True)

    def _summary(baseline, K, per_group):
        allv = [a for v in per_group.values() for a in v]
        line = "  ".join(f"{g}={sum(v)/len(v):.3f}" for g, v in per_group.items() if v)
        ov = sum(allv) / max(len(allv), 1)
        # group-balanced (macro) = mean of per-group means (matches training's headline)
        macro = sum(sum(v)/len(v) for v in per_group.values() if v) / max(
            sum(1 for v in per_group.values() if v), 1)
        if _is_main:   # under DDP this is rank 0's shard only (progress); CSV/plot are exact
            print(f"[{baseline} K={K}] {line}  | micro={ov:.3f} macro={macro:.3f} "
                  f"n={len(allv)}{' [rank0 shard]' if _DDP else ''}", flush=True)

    # max_new_tokens per sample = min(CLI cap, task's per-task cap). Batches mix
    # tasks within a metric group, so use the max across the batch (longest-cap
    # sample drives the decode loop) — task evaluators consume only the first
    # cap-many tokens. This caps biodex (4 tokens) and friends without affecting
    # free-form tasks (default 1024).
    def _batch_cap(batch):
        return max(min(args.max_new_tokens, m["task_obj"].max_new_tokens) for m in batch)

    def _warm(fn):
        """Discarded prefill(s) before a (baseline, K) sweep is timed.

        prefill_s is a TTFT measurement, and the first prefill of a sweep pays
        costs no later one does: FlexAttention/torch.compile compilation for
        this K's phase-2 shape, the BlockMask cache miss, cuBLAS autotuning and
        the allocator's first grab of a cache-sized block. Left in, all of that
        lands on the first batch's rows and inflates the arm's mean TTFT by the
        compile divided by the sample count -- an artifact that shrinks as the
        eval set grows, which is the opposite of a measurement.

        An OOM here is not fatal: the sweep's own adaptive halving handles it on
        the real batch, and a warmup that could not run just leaves the first
        timed batch warm, which is what this code did before.
        """
        for _ in range(args.timing_warmup):
            try:
                fn()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as _e:
                if not _is_oom(_e):
                    raise
                torch.cuda.empty_cache()
                return
        torch.cuda.empty_cache()

    # ── no-compression: vanilla Qwen3, full context ──────────────────────────
    if not args.skip_no_comp:
        print("\n[grouped-eval] === no-compression (vanilla Qwen3, full ctx) ===", flush=True)
        vmodel = load_vanilla_model(args.base_model)
        _nc_tag, _nc_k = args.no_comp_tag, -1
        if args.no_comp_adapter:
            from tokengate.e2e_gate.bench.benchmark_gate_e2e import _infer_lora_shape
            from tokengate.e2e_gate.gate_module.lora import (
                apply_lora_to_layers, load_lora_state_dict, merge_adapters_into_layers)
            _sd = torch.load(args.no_comp_adapter, map_location="cuda")
            _lids, _tg, _r = _infer_lora_shape(_sd, args.no_comp_adapter)
            apply_lora_to_layers(vmodel, _lids, _r, float(args.no_comp_adapter_alpha), _tg)
            load_lora_state_dict(vmodel, _sd)
            _n = merge_adapters_into_layers(vmodel, _lids)
            _nc_k = -3
            print(f"[grouped-eval] {_nc_tag}: {args.no_comp_adapter} folded into {_n} modules", flush=True)
        per_group = {}
        for g, metas in metas_by_group.items():
            accs = []
            bs = args.batch_size
            i = 0
            _wb = metas[:bs]
            _warm(lambda: vanilla_generate_batch(vmodel, tokenizer, [m["prompt_ids"] for m in _wb], 2))
            while i < len(metas):
                batch = metas[i:i + bs]
                cap = _batch_cap(batch)
                try:
                    ans, ng, pf, dec, st = vanilla_generate_batch(
                        vmodel, tokenizer, [m["prompt_ids"] for m in batch], cap)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as _e:
                    if not _is_oom(_e):
                        raise
                    torch.cuda.empty_cache()
                    if args.fixed_batch_size:
                        _record_oom(_nc_tag, _nc_k, g, batch)
                        i += bs; continue
                    if args.adaptive_batch_size:
                        if bs == 1:
                            _record_oom(_nc_tag, _nc_k, g, batch)
                            i += 1; bs = args.batch_size; continue
                        bs = bs // 2; continue     # retry the same offset, half batch
                    for m in batch:  # legacy: one at a time
                        cap1 = min(args.max_new_tokens, m["task_obj"].max_new_tokens)
                        ans, ng, pf, dec, st = vanilla_generate_batch(
                            vmodel, tokenizer, [m["prompt_ids"]], cap1)
                        accs += _score_and_record(_nc_tag, _nc_k, g, [m], ans, ng,
                                                  [m["n_ctx"]], prefill_s=pf, decode_s=dec,
                                                  decode_steps=st)
                    i += bs; continue
                accs += _score_and_record(_nc_tag, _nc_k, g, batch, ans, ng,
                                          [m["n_ctx"] for m in batch], prefill_s=pf,
                                          decode_s=dec, decode_steps=st)
                i += bs
                if args.adaptive_batch_size:
                    bs = args.batch_size       # each batch re-attempts the target bs
            per_group[g] = accs
        _summary(_nc_tag, _nc_k, per_group)
        del vmodel; torch.cuda.empty_cache()
        _finish_baseline(_nc_tag)

    # ── gate: K-sweep + full-ctx, on the SAME metas ───────────────────────────
    if not args.skip_gate:
        print("\n[grouped-eval] === gate (same samples) ===", flush=True)
        gmodel, gate, iapl = load_gate_model(args.base_model, args.gate_path, args.gate_layer, _gate_attn(args),
                                                **_gate_load_kwargs(args))
        if args.gate_random_topk:
            gate.random_topk = True
            print("[grouped-eval] --gate-random-topk: uniform random selection replaces "
                  "the gate's top-K (gate path otherwise unchanged)", flush=True)
        if args.zero_gate_bias:
            gate.zero_gate_bias = True
            print("[grouped-eval] --zero-gate-bias: additive phase-2 attention "
                  "bias zeroed; SELECTION unchanged", flush=True)
        # Threshold-mode gates IGNORE the K argument at inference (see
        # gate_forward._select_for_gate). Running a K-sweep on a threshold gate
        # silently produces N identical copies of the tau-cutoff point — that's
        # the bug the biodex eval surfaced. Refuse early and point the user to
        # the right tool. Keep the K-sweep path for top-K-trained gates.
        _sel = getattr(gate, "selection_mode", "topk")
        if _sel == "threshold":
            raise RuntimeError(
                f"--gate-path {args.gate_path} was trained with selection_mode="
                f"'threshold' (tau={getattr(gate, 'tau', 0.5)}). A K-sweep through "
                "bench_grouped_eval is a no-op for this gate: gate_forward._select_"
                "for_gate routes to threshold mode and ignores K, so every K value "
                "you pass collapses to the same kept set. Threshold gates need a "
                "tau sweep, not a K sweep (the bench_tau_sweep tool was removed "
                "with the research scripts). Or pass --skip-gate to "
                "bench_grouped_eval if you only want the no-comp baseline."
            )
        _gate_specs = _k_specs() + ([] if args.skip_gate_full else [("full", args.full_ctx_k)])
        for spec in _gate_specs:
            _kind, _kv = spec
            tag = f"{args.gate_tag}_full" if _kind == "full" else args.gate_tag
            k_frac = f"{_kv:g}" if _kind == "frac" else ""
            per_group = {}
            for g, metas in metas_by_group.items():
                accs = []
                bs = args.batch_size
                i = 0
                _wb = metas[:bs]
                _wK = _batch_k(spec, _wb)
                _warm(lambda: gate_generate_batch(gmodel, gate, tokenizer, _wb, args.gate_layer, _wK, 2, iapl,
                                                  decode_phase2_form=args.decode_phase2_form,
                                                  prune_below_gate=args.prune_below_gate))
                while i < len(metas):
                    batch = metas[i:i + bs]
                    cap = _batch_cap(batch)
                    Kbatch = _batch_k(spec, batch)  # scalar (abs/full) or per-sample list (frac)
                    try:
                        ans, ng, nk, pf, dec, st =gate_generate_batch(
                            gmodel, gate, tokenizer, batch, args.gate_layer, Kbatch,
                            cap, iapl, decode_phase2_form=args.decode_phase2_form,
                            prune_below_gate=args.prune_below_gate)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as _e:
                        if not _is_oom(_e):
                            raise
                        torch.cuda.empty_cache()
                        _free_graph_state(gmodel)
                        if args.fixed_batch_size:
                            _rowK = (Kbatch if _kind == "frac"
                                     else (-3 if _kind == "full" else _kv))
                            _record_oom(tag, _rowK, g, batch, k_frac)
                            i += bs; continue
                        if args.adaptive_batch_size:
                            if bs == 1:
                                _rowK = (Kbatch if _kind == "frac"
                                         else (-3 if _kind == "full" else _kv))
                                _record_oom(tag, _rowK, g, batch, k_frac)
                                i += 1; bs = args.batch_size; continue
                            bs = bs // 2; continue   # retry same offset, half batch
                        if bs == 1:
                            i += 1; continue
                        bs = max(1, bs // 2); continue
                    rowK = Kbatch if _kind == "frac" else (-3 if _kind == "full" else _kv)
                    accs += _score_and_record(tag, rowK, g, batch, ans, ng, nk, k_frac,
                                              prefill_s=pf, decode_s=dec, decode_steps=st)
                    i += bs
                    if args.adaptive_batch_size:
                        bs = args.batch_size   # each batch re-attempts the target bs
                per_group[g] = accs
            slabel = k_frac if _kind == "frac" else (-3 if _kind == "full" else _kv)
            _summary(tag, slabel, per_group)
        del gmodel, gate; torch.cuda.empty_cache()
        _finish_baseline(args.gate_tag)


    # rank 0 already wrote the CSV incrementally in each _finish_baseline; do a final
    # authoritative write from the gathered master rows.
    if _is_main:
        _write_csv(_master_rows)
        print(f"\n[grouped-eval] wrote {len(_master_rows)} rows → {out_csv}", flush=True)
        print("[grouped-eval] DONE", flush=True)
    if _DDP:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
