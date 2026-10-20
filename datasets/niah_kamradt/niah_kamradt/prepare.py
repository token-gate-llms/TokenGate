"""niah_kamradt — Needle-in-a-Haystack (Kamradt, 2023) with a generated train split.

Reads the jsonl produced by
`datasets/niah_kamradt/generate_niah_kamradt.py` (PG-essay
haystack + canonical "The best thing to do in {city} is {activity}." needle;
see that file's docstring for the split-hygiene contract) and renders the
repo's standard sample geometry:

    article = haystack_with_needle + "\n\nQuestion: What is the best thing to do in {city}?"
    answer  = the activity phrase

DECONTAM: none needed beyond construction — every needle fact is generated,
essay pools and needle cities are hash-split train/test in the generator, and
the canonical San Francisco needle is test-only.

Usage:
    python datasets/niah_kamradt/niah_kamradt/prepare.py
"""
import json
import os
from pathlib import Path

from tokengate.datasets.prepare import prepare_dataset

DATASET = "niah_kamradt"
TASK = "niah_kamradt"
_HERE = Path(__file__).resolve()
JSONL = Path(os.environ.get("NIAH_DATA_DIR", _HERE.parents[1])) / "niah_kamradt_all.jsonl"
OUTPUT_ROOT = Path(os.environ.get("PRETRAIN_DATA_ROOT", _HERE.parents[2]))
# PREPARE_MODEL overrides the tokenizer target (e.g. Qwen/Qwen3.5-2B for the
# hybrid-attention runs); tokenized output lands under tokenized/<basename>,
# so re-running for a new model never touches an existing model's mmaps.
MODEL = os.environ.get("PREPARE_MODEL", "Qwen/Qwen3-1.7B")


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
    print(f"[niah_kamradt] {split}: {n} rows", flush=True)


def load_train():
    return _rows("train")


def load_val():
    return _rows("test")


if __name__ == "__main__":
    if not JSONL.exists():
        raise SystemExit(f"[niah_kamradt] missing generator output: {JSONL}\n"
                         f"Run generate_niah_kamradt.py first.")
    prepare_dataset(dataset_name=DATASET, task_name=TASK,
                    load_train_fn=load_train, load_val_fn=load_val,
                    model=MODEL, output_root=OUTPUT_ROOT, run_sanity=False)
