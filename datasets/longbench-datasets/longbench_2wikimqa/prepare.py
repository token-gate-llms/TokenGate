"""Matched-train prepare for LongBench `2wikimqa` → 2WikiMultihopQA (train).

Trains on the 2WikiMultihopQA train split (~167k) via the framolfese mirror,
formatted to mirror LongBench's `Passage {i}:\n{title}\n{body}` context. Deduped
against the LongBench `2wikimqa` eval rows.

PARITY NOTE: LongBench's 2wikimqa context carries full multi-section Wikipedia
article bodies, while the framolfese mirror only ships decomposed sentence lists.
We reproduce LongBench's *structure* (Passage headers + titles, identical task)
but the passage body text is sentence-joined, so it differs in surface detail.
This is the closest faithful reconstruction available from the mirror.
"""
from datasets import load_dataset

from tokengate.datasets.prepare import prepare_dataset
from tokengate.datasets.matched_prepare import (
    dedup_loader,
    longbench_eval_val_loader,
)

DATASET = "longbench-datasets"
TASK = "longbench_2wikimqa"
SUBSET = "2wikimqa"


def _build_context(titles, sentences_lists) -> str:
    blocks = []
    for i, (title, sents) in enumerate(zip(titles, sentences_lists)):
        if not isinstance(sents, list):
            raise ValueError(f"2wikimqa: paragraph {title!r} sentences not a list")
        body = " ".join(s for s in sents if isinstance(s, str))
        blocks.append(f"Passage {i + 1}:\n{title}\n{body}")
    return "\n".join(blocks)


def _raw(split: str):
    ds = load_dataset("framolfese/2WikiMultihopQA", split=split, trust_remote_code=True)
    for ex in ds:
        ans = (ex.get("answer") or "").strip()
        if not ans:
            continue
        ctx = ex["context"]
        titles, sentences = ctx["title"], ctx["sentences"]
        if len(titles) != len(sentences):
            raise ValueError(f"2wikimqa id={ex.get('id')!r}: title/sentence length mismatch")
        context = _build_context(titles, sentences)
        if not context.strip():
            continue
        yield {
            "id": ex["id"], "dataset": DATASET, "task": TASK,
            "article": f"{context}\n\nQuestion: {ex['question']}", "answer": ans,
            "metadata": {"type": ex.get("type"), "source_split": split, "longbench_subset": SUBSET},
        }


load_train = dedup_loader(SUBSET, lambda: _raw("train"))
load_val = longbench_eval_val_loader(SUBSET, DATASET, TASK)


if __name__ == "__main__":
    prepare_dataset(dataset_name=DATASET, task_name=TASK,
                    load_train_fn=load_train, load_val_fn=load_val)
