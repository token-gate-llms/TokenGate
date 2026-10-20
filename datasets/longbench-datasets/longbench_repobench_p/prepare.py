"""Matched-train prepare for LongBench `repobench-p` → RepoBench-P (cross_file_first).

Trains on RepoBench v1.1 (python + java) `cross_file_first` split. LongBench
repobench-p puts cross-file snippets in the context and the current-file prefix
after, predicting the next line. We mirror that:
  article = <cross-file snippets, "{path}\n{snippet}"> + "\n\nQuestion: "
            + <import_statement + "\n" + cropped_code>
  answer  = next_line
The "\n\nQuestion: " marker also makes the current-file prefix non-compressible
(it must never be pruned — it's the immediate completion context). Deduped against
the LongBench `repobench-p` eval rows. ES.
"""
from datasets import load_dataset

from tokengate.datasets.prepare import prepare_dataset
from tokengate.datasets.matched_prepare import (
    dedup_loader,
    longbench_eval_val_loader,
)

DATASET = "longbench-datasets"
TASK = "longbench_repobench_p"
SUBSET = "repobench-p"
_HF = ("tianyang/repobench_python_v1.1", "tianyang/repobench_java_v1.1")


def _cross_file(context) -> str:
    blocks = []
    for c in context or []:
        if not isinstance(c, dict):
            continue
        path = (c.get("path") or "").strip()
        snippet = c.get("snippet") or ""
        if snippet.strip():
            blocks.append(f"{path}\n{snippet}" if path else snippet)
    return "\n\n".join(blocks)


def _raw(split: str):
    for hf in _HF:
        lang = "python" if "python" in hf else "java"
        # verification_mode=no_checks: the repobench java mirror's recorded split
        # sizes don't match its cached metadata (benign HF quirk) — skip the check.
        ds = load_dataset(hf, split=split, trust_remote_code=True,
                          verification_mode="no_checks")
        for i, ex in enumerate(ds):
            nxt = (ex.get("next_line") or "").rstrip("\n")
            cropped = ex.get("cropped_code") or ""
            if not nxt.strip() or not cropped.strip():
                continue
            cross = _cross_file(ex.get("context"))
            imports = ex.get("import_statement") or ""
            cur_file = f"{imports}\n{cropped}" if imports.strip() else cropped
            article = f"{cross}\n\nQuestion: {cur_file}" if cross.strip() else \
                f"\n\nQuestion: {cur_file}"
            yield {
                "id": f"repobench_{lang}_{split}_{i}", "dataset": DATASET, "task": TASK,
                "article": article, "answer": nxt,
                "metadata": {"lang": lang, "repo_name": ex.get("repo_name"),
                             "level": ex.get("level"), "source_split": split,
                             "longbench_subset": SUBSET},
            }


load_train = dedup_loader(SUBSET, lambda: _raw("cross_file_first"))
load_val = longbench_eval_val_loader(SUBSET, DATASET, TASK)


if __name__ == "__main__":
    prepare_dataset(dataset_name=DATASET, task_name=TASK,
                    load_train_fn=load_train, load_val_fn=load_val)
