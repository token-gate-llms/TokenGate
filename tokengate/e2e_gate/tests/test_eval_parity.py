"""Trainer-eval vs bench-eval parity test.

Same checkpoint, same val rows, one process: PATH A = the trainer's
``train_e2e_gate.evaluate`` (the in-training generation eval), PATH B = the
benchmark harness's ``gate_generate_batch``.  Both now route through the
single shared decode utility (``gate_decode.generate_batch``), so the
answers must match EXACTLY — any divergence is a drift bug in meta building
(chat template / prunable span) or in one caller's wiring, which is exactly
what this test exists to catch.

Run (single GPU, needs a real checkpoint + the tokenized val store):

  python -m tokengate.e2e_gate.tests.test_eval_parity \\
      --gate-path <run_dir>/baseline/L1/gate_stepN.pt --gate-layer 1

Not a pytest test (repo convention): prints a pass/fail summary and exits
non-zero on failure.
"""
import argparse
import os
import statistics as st
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from transformers import AutoTokenizer  # noqa: E402

from tokengate.e2e_gate.bench.benchmark_gate_e2e import (  # noqa: E402
    build_metas, gate_generate_batch, load_gate_model, load_val_samples)
from tokengate.datasets.tasks import get_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-path", required=True,
                    help="gate checkpoint (gate_stepN.pt / gate_final.pt); the "
                         "paired adapter is auto-discovered by filename")
    ap.add_argument("--gate-layer", type=int, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dataset", default="longbench-datasets")
    ap.add_argument("--task-name", default="longbench_2wikimqa")
    ap.add_argument("--pretrain-data-root",
                    default=os.path.join(REPO, "datasets"))
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gate-attn", default=None,
                    help="override the kernel config (default: load_gate_model's)")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base_model)
    model_dir = os.path.basename(a.base_model.rstrip("/"))
    rows = load_val_samples(a.pretrain_data_root, a.dataset, a.task_name,
                            model_dir, a.n_samples, 39000, None)
    for r in rows:
        r["dataset"] = a.task_name
    task = get_task(rows[0]["task"])
    tasks_by_name = {rows[0]["task"]: task}
    print(f"[parity] rows={len(rows)} task={rows[0]['task']} ckpt={a.gate_path}",
          flush=True)

    kw = {} if a.gate_attn is None else {"gate_attn": a.gate_attn}
    model, gate, iapl = load_gate_model(a.base_model, a.gate_path, a.gate_layer, **kw)

    # ── PATH B: bench wrapper (gate_generate_batch -> generate_batch) ──
    metas = build_metas(tok, task, rows)
    ansB = []
    for i in range(0, len(metas), a.batch_size):
        ans, *_ = gate_generate_batch(model, gate, tok, metas[i:i + a.batch_size],
                                      a.gate_layer, a.k, a.max_new_tokens, iapl,
                                      decode_phase2_form=True)
        ansB += list(ans)
    f1B = st.mean(task.evaluate_f1(x, m["sample"]["target"], sample=m["sample"])
                  for m, x in zip(metas, ansB))
    print(f"[parity] PATH B (bench)   F1 = {f1B:.4f}  n={len(ansB)}", flush=True)
    torch.cuda.empty_cache()

    # ── PATH A: trainer evaluate() (batched chunks -> generate_batch) ──
    from tokengate.e2e_gate.train.train_e2e_gate import (
        evaluate as train_evaluate)
    out = train_evaluate(
        model, gate, tok, rows, a.gate_layer, a.k, task,
        max_gen_tokens=a.max_new_tokens, device="cuda",
        log_examples=len(rows), baseline_predictions=None, rank=0,
        world_size=1, eval_gen_batch_size=a.batch_size,
        tasks_by_name=tasks_by_name, sparsity_lambda=0.0)
    f1A, logged = out[1], out[5]
    print(f"[parity] PATH A (trainer) F1 = {f1A:.4f}", flush=True)

    ansA = [r["prediction"] for r in logged]
    n_cmp = min(len(ansA), len(ansB))
    mism = [(i, x, y) for i, (x, y) in enumerate(zip(ansA[:n_cmp], ansB[:n_cmp]))
            if x != y]
    for i, x, y in mism:
        print(f"[parity]   DIFF sample {i}: trainer={x!r}  bench={y!r}")
    ok = not mism and n_cmp == len(ansB) == len(ansA) and abs(f1A - f1B) < 1e-9
    print(f"[parity] exact matches: {n_cmp - len(mism)}/{n_cmp}  "
          f"F1 delta = {f1B - f1A:+.6f}")
    print(f"[parity] {'PASSED' if ok else 'FAILED'}: trainer eval "
          f"{'==' if ok else '!='} bench eval")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
