"""niah_kamradt_32k — the 32k-token tier of the Kamradt NIAH benchmark,
TEST-ONLY (200 rows: the same 20 held-out cities x 10 depths as
niah_kamradt's 16k tier, same seed/needles, ~32000-token haystacks — 2x the
training tier, inside Qwen3-1.7B's 40960 window).  Length-generalization
benchmark for the SAME niah checkpoints; no train split.

Reads the jsonl produced by
`generate_niah_kamradt.py --n-train 0 --test-target 32000 --test-jit 300`;
sample geometry identical to niah_kamradt/prepare.py.

Usage:
    python datasets/niah_kamradt_32k/niah_kamradt_32k/prepare.py
"""
import json
import os
from pathlib import Path

from tokengate.datasets.prepare import prepare_dataset

DATASET = "niah_kamradt_32k"
TASK = "niah_kamradt_32k"
_HERE = Path(__file__).resolve()
JSONL = _HERE.parents[2] / "niah_kamradt" / "niah_kamradt_32k.jsonl"
OUTPUT_ROOT = Path(os.environ.get("PRETRAIN_DATA_ROOT", _HERE.parents[2]))
MODEL = os.environ.get("PREPARE_MODEL", "Qwen/Qwen3-1.7B")  # e.g. Qwen/Qwen3.5-2B for the 32k tier of the hybrid gates


def _rows(split):
    n = 0
    with open(JSONL) as f:
        for line in f:
            r = json.loads(line)
            if r["split"] != split:
                continue
            n += 1
            yield {
                "id": r["id"], "dataset": DATASET, "task": TASK,
                "article": f"{r['body']}\n\nQuestion: {r['question']}",
                "answer": r["answer"],
                "metadata": {"answers": [r["answer"]], "city": r["city"],
                             "depth": r["depth_actual"], "all_classes": None},
            }
    print(f"[niah_kamradt_32k] {split}: {n} rows", flush=True)


def load_train():
    return _rows("train")      # empty by construction (--n-train 0)


def load_val():
    return _rows("test")


if __name__ == "__main__":
    if not JSONL.exists():
        raise SystemExit(f"[niah_kamradt_32k] missing generator output: {JSONL}\n"
                         f"Run generate_niah_kamradt.py --n-train 0 --test-target 32000 first.")
    prepare_dataset(dataset_name=DATASET, task_name=TASK,
                    load_train_fn=load_train, load_val_fn=load_val,
                    model=MODEL, output_root=OUTPUT_ROOT,
                    max_article_tokens=36000, run_sanity=False)
