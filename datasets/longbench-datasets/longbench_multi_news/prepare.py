"""Matched-train prepare for LongBench `multi_news` → Multi-News (train).

Trains on multi_news train split (~45k): the source `document` packs the cluster's
articles separated by " ||||| ". LongBench formats these as
"Passage {i}:\n{doc}" with newlines encoded as the literal token NEWLINE_CHAR;
we reproduce that exactly. answer = the multi-doc summary. Deduped against the
LongBench `multi_news` eval rows. ROUGE-L.
"""
from datasets import load_dataset

from tokengate.datasets.prepare import prepare_dataset
from tokengate.datasets.matched_prepare import (
    dedup_loader,
    longbench_eval_val_loader,
)

DATASET = "longbench-datasets"
TASK = "longbench_multi_news"
SUBSET = "multi_news"


def _build_context(document: str) -> str:
    docs = [d.strip() for d in document.split("|||||")]
    docs = [d for d in docs if d]
    blocks = []
    for i, d in enumerate(docs):
        # LongBench encodes newlines as the literal NEWLINE_CHAR token.
        d_enc = d.replace("\n", " NEWLINE_CHAR")
        blocks.append(f"Passage {i + 1}:\n{d_enc}")
    return "\n".join(blocks)


def _raw(split: str):
    ds = load_dataset("multi_news", split=split, trust_remote_code=True)
    for i, ex in enumerate(ds):
        document = (ex.get("document") or "").strip()
        summ = (ex.get("summary") or "").strip().lstrip("– ").strip()
        if not document or not summ:
            continue
        context = _build_context(document)
        if not context.strip():
            continue
        yield {
            "id": f"multi_news_{split}_{i}", "dataset": DATASET, "task": TASK,
            "article": context, "answer": summ,
            "metadata": {"source_split": split, "longbench_subset": SUBSET},
        }


load_train = dedup_loader(SUBSET, lambda: _raw("train"))
load_val = longbench_eval_val_loader(SUBSET, DATASET, TASK)


if __name__ == "__main__":
    prepare_dataset(dataset_name=DATASET, task_name=TASK,
                    load_train_fn=load_train, load_val_fn=load_val)
