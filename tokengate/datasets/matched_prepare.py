"""Helpers for the `longbench-datasets` matched pretraining datasets.

A "matched" subset trains on the upstream *original* source dataset of a
LongBench subset, formatted to mirror the LongBench eval context (prompt parity),
with every row that exactly- or near-duplicates a LongBench eval row removed
(direct-contamination guarantee).
Each matched `prepare.py` builds raw rows in the LongBench sample shape
  {"id", "dataset", "task", "article", "answer", "metadata"}
where `article == <longbench-style context> + "\n\nQuestion: " + <question>`
(or just the context for summarization subsets with no question), then wraps its
train/val loaders with `dedup_loader(subset, ...)` so dedup happens inline before
tokenization.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

from tokengate.datasets.longbench_index import (
    DEFAULT_INDEX_PATH,
    DedupResult,
    LongBenchEvalIndex,
    dedup_against_subset,
    load_index,
)


def dedup_rows(
    subset: str,
    rows: List[Dict],
    index: Optional[LongBenchEvalIndex] = None,
    index_path: str = DEFAULT_INDEX_PATH,
    article_key: str = "article",
) -> tuple[List[Dict], DedupResult]:
    """Drop rows whose `article` exact/near-duplicates a LongBench `subset` eval row.

    Returns (kept_rows, DedupResult). Logs a one-line summary. Fails loudly if
    the index or subset is missing (a silent skip would void the contamination
    guarantee).
    """
    if index is None:
        index = load_index(index_path)
    res = dedup_against_subset(index, subset, (r.get(article_key, "") for r in rows))
    kept = [r for r, keep in zip(rows, res.keep_mask) if keep]
    print(
        f"[matched-dedup] {subset}: {res.n_total} candidate rows → "
        f"dropped {res.n_exact} exact + {res.n_neardup} near-dup "
        f"({res.dup_fraction:.2%}) → {len(kept)} kept",
        flush=True,
    )
    return kept, res


def longbench_eval_val_loader(
    subset: str,
    dataset: str,
    task: str,
) -> Callable[[], List[Dict]]:
    """Val loader that yields the official LongBench eval rows for `subset`.

    The matched dataset's in-training val set IS the benchmark we report (locked
    decision): training-time eval curves track real LongBench, and since train is
    deduped against exactly these rows there is no leakage. Rows are emitted in
    the prepare schema; `article` already carries the LongBench
    "context + \n\nQuestion: input" shape, so tokenizing with the matched task
    (== eval task) gives train/val prompt parity.
    """
    def _loader() -> List[Dict]:
        from datasets import load_dataset

        raw = load_dataset("THUDM/LongBench", subset, split="test")
        out: List[Dict] = []
        for i, r in enumerate(raw):
            ctx = r.get("context", "") or ""
            inp = (r.get("input", "") or "").strip()
            article = f"{ctx}\n\nQuestion: {inp}" if inp else ctx
            answers = r.get("answers") or []
            ans = answers[0] if answers else ""
            if not (article.strip() and str(ans).strip()):
                continue
            out.append({
                "id": r.get("_id", str(i)),
                "dataset": dataset,
                "task": task,
                "article": article,
                "answer": ans,
                "metadata": {
                    "answers": list(answers),
                    "all_classes": r.get("all_classes"),
                    "longbench_subset": subset,
                    "source_split": "longbench_test",
                },
            })
        print(f"[matched-val] {subset}: {len(out)} LongBench eval rows as val", flush=True)
        return out

    return _loader


def prepare_eval_only(
    subset: str,
    dataset: str,
    task: str,
    model: str = "Qwen/Qwen3-8B",
    output_root=None,
    max_article_tokens: int = 30000,
) -> int:
    """Tokenize a held-out LongBench subset's eval rows → val.pkl ONLY (no train).

    For subsets we evaluate but never train on (no clean upstream original, or
    deferred few-shot subsets), there is no train data to dedup against — the
    rows ARE the LongBench eval set.  This mirrors `prepare_dataset`'s val path
    (raw_val.jsonl + tokenized/<model>/val.pkl + prompts/version.txt) but skips
    the train write, the train-only stats, and the sanity check (all of which
    require train rows).  Returns the number of tokenized val rows.
    """
    import os
    from pathlib import Path

    from transformers import AutoTokenizer

    from tokengate.datasets.tasks import get_task
    from tokengate.datasets.prepare import (
        DEFAULT_OUTPUT_ROOT, tokenize_samples, write_pkl, write_raw_jsonl,
    )

    if output_root is None:
        output_root = DEFAULT_OUTPUT_ROOT
    task_obj = get_task(task)
    prompt_version = task_obj.get_prompt_version()

    base = Path(output_root) / dataset / task
    base.mkdir(parents=True, exist_ok=True)

    val_raw = list(longbench_eval_val_loader(subset, dataset, task)())
    if not val_raw:
        raise ValueError(
            f"prepare_eval_only({subset!r}): LongBench eval loader returned 0 rows — "
            f"check the subset name and split.")
    write_raw_jsonl(val_raw, base / "raw_val.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_basename = os.path.basename(model)
    tok_val = tokenize_samples(
        val_raw, tokenizer, task_obj, prompt_version,
        max_article_tokens=max_article_tokens,
    )
    write_pkl(tok_val, base / "tokenized" / model_basename / "val.pkl")

    prompts_dir = base / "prompts" / model_basename
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "version.txt").write_text(prompt_version + "\n")

    print(
        f"[prepare-eval-only] {dataset}/{task} (subset={subset}): "
        f"{len(val_raw)} LongBench eval rows → {len(tok_val)} tokenized val rows "
        f"(dropped {len(val_raw) - len(tok_val)} over {max_article_tokens}-tok cap)",
        flush=True,
    )
    return len(tok_val)


def dedup_loader(
    subset: str,
    raw_loader: Callable[[], Iterable[Dict]],
    index_path: str = DEFAULT_INDEX_PATH,
    article_key: str = "article",
) -> Callable[[], List[Dict]]:
    """Wrap a raw row-generator into a deduped list-returning loader.

    Usable directly as `load_train_fn` / `load_val_fn` for `prepare_dataset`.
    The index is loaded once per call.
    """
    def _loader() -> List[Dict]:
        rows = list(raw_loader())
        kept, _ = dedup_rows(subset, rows, index_path=index_path, article_key=article_key)
        return kept

    return _loader
