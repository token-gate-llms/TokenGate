"""Correctness sanity check: does our benchmark inference reproduce the gate's
*training-time* eval accuracy?

It reconstructs the EXACT pretraining eval set (the deterministic
``random.Random(seed).sample`` per task-category, drawn from each dataset's
first-N val.pkl rows in manifest order — see train_e2e_gate.py:2649-2665), runs
**our** benchmark gate inference (gate_generate_batch from benchmark_gate_e2e) at
the training K, and reports per-category + overall accuracy.  Compare the printed
numbers against the training run's ``eval_responses.csv`` (final step).

Match (within batching/numerical noise) ⇒ the gate inference path is correct.

    python -m tokengate.e2e_gate.bench.benchmark_sanity_check \
        --gpu 0 \
        --gate-path .../baseline/L0/gate_step137700.pt --gate-layer 0 \
        --manifest v3_accepted28 \
        --pretrain-data-root /future/.../datasets \
        --k 2000 --eval-samples-per-group 50 --seed 0 \
        --max-new-tokens 1024 --batch-size 4
"""

import argparse
import os
import random
import subprocess
import sys

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv) and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]

import torch  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokengate.datasets.tasks import get_task, get_task_category  # noqa: E402
from tokengate.e2e_gate.bench.benchmark_gate_e2e import (  # noqa: E402
    load_val_samples, build_metas, load_gate_model, gate_generate_batch,
)


def manifest_pairs(manifest: str):
    def _q(field):
        return subprocess.check_output([
            sys.executable, "-m", "tokengate.datasets.manifest",
            manifest, field,
        ]).decode().strip().split(",")
    return list(zip(_q("dataset"), _q("task")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--gate-path", required=True)
    p.add_argument("--gate-layer", type=int, required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--pretrain-data-root", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--eval-samples-per-group", type=int, required=True)
    p.add_argument("--val-samples-per-dataset", type=int, default=50)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--max-article-tokens", type=int, default=30000)
    p.add_argument("--max-answer-tokens", type=int, default=1024)
    p.add_argument("--only-groups", default=None,
                   help="comma list to restrict groups, e.g. qa,code_completion")
    p.add_argument("--dump-csv", default=None,
                   help="optional path: write per-sample (group,dataset,target,prediction,acc,f1)")
    args = p.parse_args()

    import csv as _csv
    _dump_rows = []

    model_name = os.path.basename(args.base_model)
    pairs = manifest_pairs(args.manifest)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Reconstruct the exact pretraining eval set (train_e2e_gate.py:2649-2665) ─
    # Per-dataset first-N val rows (manifest order), filtered identically, then
    # pooled by task-category and sub-sampled with random.Random(seed).
    eval_by_dataset = {}
    cat_of_ds = {}
    for ds, tk in pairs:
        rows = load_val_samples(
            args.pretrain_data_root, ds, tk, model_name,
            args.val_samples_per_dataset, args.max_article_tokens, args.max_answer_tokens,
        )
        eval_by_dataset[ds] = rows
        cat_of_ds[ds] = get_task_category(tk)

    grp_pool = {}
    for ds, rows in eval_by_dataset.items():
        grp_pool.setdefault(cat_of_ds[ds], []).extend(rows)
    grng = random.Random(args.seed)
    eval_by_group = {}
    for g, pool in grp_pool.items():
        n = min(args.eval_samples_per_group, len(pool))
        eval_by_group[g] = grng.sample(pool, n)

    only = set(args.only_groups.split(",")) if args.only_groups else None
    print("[sanity] reconstructed eval set:")
    for g, s in eval_by_group.items():
        src = {}
        for r in s:
            src[r["dataset"]] = src.get(r["dataset"], 0) + 1
        print(f"  {g:16s}: n={len(s)}  from {src}")

    # ── Load gate model ──────────────────────────────────────────────────────
    gmodel, gate, iapl = load_gate_model(args.base_model, args.gate_path, args.gate_layer)

    # ── Run our gate inference per group at K, batched ───────────────────────
    print(f"\n[sanity] running gate inference at K={args.k}, "
          f"max_new_tokens={args.max_new_tokens}, batch_size={args.batch_size}\n")
    overall_acc = overall_n = 0
    results = {}
    for g, samples in eval_by_group.items():
        if only and g not in only:
            continue
        # one task per sample (mixed within a category) → group metas by task
        metas = []
        for idx, s in enumerate(samples):
            task = get_task(s["task"])
            m_list = build_metas(tokenizer, task, [s])
            if m_list:
                m = m_list[0]
                m["idx"] = idx
                m["task_obj"] = task
                metas.append(m)
        acc_sum = f1_sum = 0.0
        n = 0
        i = 0
        bs = args.batch_size
        while i < len(metas):
            batch = metas[i:i + bs]
            try:
                answers, _ngen, _nkept, *_ = gate_generate_batch(
                    gmodel, gate, tokenizer, batch, args.gate_layer, args.k,
                    args.max_new_tokens, iapl)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    i += 1
                    continue
                bs = max(1, bs // 2)
                continue
            for m, ans in zip(batch, answers):
                t = m["task_obj"]
                s = m["sample"]
                a_i = t.evaluate(ans, s["target"], sample=s)
                f_i = t.evaluate_f1(ans, s["target"], sample=s)
                acc_sum += a_i
                f1_sum += f_i
                n += 1
                if args.dump_csv:
                    _dump_rows.append({
                        "group": g, "dataset": s["dataset"], "target": s["target"][:300],
                        "prediction": ans[:300], "acc": f"{a_i:.6f}", "f1": f"{f_i:.6f}",
                    })
            i += bs
        acc = acc_sum / max(n, 1)
        f1 = f1_sum / max(n, 1)
        results[g] = (acc, f1, n)
        overall_acc += acc_sum
        overall_n += n
        print(f"  {g:16s}: acc={acc:.3f}  f1={f1:.3f}  n={n}", flush=True)

    if args.dump_csv and _dump_rows:
        with open(args.dump_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["group", "dataset", "target", "prediction", "acc", "f1"])
            w.writeheader()
            w.writerows(_dump_rows)
        print(f"[sanity] wrote {len(_dump_rows)} per-sample rows → {args.dump_csv}")

    print(f"\n  {'OVERALL':16s}: acc={overall_acc/max(overall_n,1):.3f}  n={overall_n}")
    print("\n[sanity] compare these against the training run's eval_responses.csv "
          "(final step, same K). A close match validates the gate inference path.")


if __name__ == "__main__":
    main()
