"""niah_kamradt_64k — the 64k-token tier of the Kamradt NIAH benchmark,
TEST-ONLY (200 rows: the same 20 held-out cities x 10 depths as
niah_kamradt's 16k tier, same seed/needles, ~64000-token haystacks — 4x the
training tier).  Length-generalization benchmark for the SAME niah
checkpoints; no train split.

64000 tokens exceeds Qwen3-1.7B's native 40960 window, so the Qwen3-1.7B arm
(bench `niah64k`) evaluates against the YaRN-scaled model dir
`the make_yarn64k.sh-built model dir` (rope_scaling yarn factor 2.0
over original 32768, max_position_embeddings 65536 — the Qwen-official recipe
for >32768 context).  Qwen3.5-2B (bench `niah35_64k`) has a 262144 native
window and needs no scaling.  Tokenization is per-model (PREPARE_MODEL); the
raw jsonl is model-agnostic (Qwen3-1.7B sizing tokenizer).

Reads the jsonl produced by
`generate_niah_kamradt.py --n-train 0 --test-target 64000 --test-jit 300`;
sample geometry identical to niah_kamradt_32k/prepare.py.

Usage:
    python datasets/niah_kamradt_64k/niah_kamradt_64k/prepare.py
    PREPARE_MODEL=Qwen/Qwen3.5-2B python -m ...niah_kamradt_64k.prepare
"""
import json
import os
from pathlib import Path

from tokengate.datasets.prepare import prepare_dataset

DATASET = "niah_kamradt_64k"
TASK = "niah_kamradt_64k"
_HERE = Path(__file__).resolve()
JSONL = _HERE.parents[2] / "niah_kamradt" / "niah_kamradt_64k.jsonl"
OUTPUT_ROOT = Path(os.environ.get("PRETRAIN_DATA_ROOT", _HERE.parents[2]))
MODEL = os.environ.get("PREPARE_MODEL", "Qwen/Qwen3-1.7B")  # e.g. Qwen/Qwen3.5-2B for the hybrid gates


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
    print(f"[niah_kamradt_64k] {split}: {n} rows", flush=True)


def load_train():
    return _rows("train")      # empty by construction (--n-train 0)


def load_val():
    return _rows("test")


if __name__ == "__main__":
    if not JSONL.exists():
        raise SystemExit(f"[niah_kamradt_64k] missing generator output: {JSONL}\n"
                         f"Run generate_niah_kamradt.py --n-train 0 --test-target 64000 first.")
    prepare_dataset(dataset_name=DATASET, task_name=TASK,
                    load_train_fn=load_train, load_val_fn=load_val,
                    model=MODEL, output_root=OUTPUT_ROOT,
                    max_article_tokens=68000, run_sanity=False)
