"""Generic preparation pipeline for a single (dataset, task) pair.

Per-dataset prepare.py scripts implement a loader that yields canonical raw
samples, then call prepare_dataset(...) which handles: schema validation,
raw JSONL write, token-length stats, tokenization to .pkl for the target
model, and (optionally) the two sanity checks.

The output layout under output_root/<dataset>/<task>/ is:

    raw_train.jsonl
    raw_val.jsonl                       (if load_val_fn is provided)
    stats.json
    prompts/<model_basename>/version.txt
    tokenized/<model_basename>/train.pkl
    tokenized/<model_basename>/val.pkl
    sanity/<model_basename>/results.json
    sanity/<model_basename>/full_context.jsonl
    sanity/<model_basename>/init_gated.jsonl
"""
import json
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np


RAW_FIELDS = ("id", "dataset", "task", "article", "answer", "metadata")
DEFAULT_OUTPUT_ROOT = Path("datasets")


def validate_raw_sample(sample: Dict) -> None:
    missing = [k for k in RAW_FIELDS if k not in sample]
    if missing:
        raise ValueError(
            f"raw sample missing required fields {missing}; got keys={list(sample.keys())}"
        )
    if not isinstance(sample["article"], str) or not sample["article"]:
        raise ValueError(f"raw sample id={sample.get('id')!r}: 'article' must be a non-empty string")
    if not isinstance(sample["answer"], str):
        raise ValueError(f"raw sample id={sample.get('id')!r}: 'answer' must be a string")
    if not isinstance(sample["metadata"], dict):
        raise ValueError(f"raw sample id={sample.get('id')!r}: 'metadata' must be a dict")


def write_raw_jsonl(samples: Iterable[Dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for s in samples:
            validate_raw_sample(s)
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_raw_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _length_dist(arr: np.ndarray) -> Dict:
    if arr.size == 0:
        return {"n": 0}
    counts, edges = np.histogram(arr, bins=20)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "histogram": {"counts": counts.tolist(),
                      "edges": [float(x) for x in edges.tolist()]},
    }


def compute_stats(
    samples: List[Dict],
    tokenizer,
    task,
    max_article_tokens: int = 30000,
    min_article_tokens: int = 0,
) -> Dict:
    """Token-length distributions over compressible regions (raw + filtered)
    AND over answer strings.  Answer length feeds the DATASETS.md review table
    so the user can compare answer cost across tasks at a glance."""
    raw_lens = []
    answer_lens = []
    for s in samples:
        nc = task.get_non_compressible_context(s["article"])
        compressible = s["article"][: len(s["article"]) - len(nc)] if nc else s["article"]
        ids = tokenizer.encode(compressible, add_special_tokens=False)
        raw_lens.append(len(ids))
        ans = s.get("answer", "") or ""
        ans_ids = tokenizer.encode(ans, add_special_tokens=False) if ans else []
        answer_lens.append(len(ans_ids))
    raw_arr = np.array(raw_lens, dtype=np.int64)
    ans_arr = np.array(answer_lens, dtype=np.int64)
    mask = (raw_arr >= min_article_tokens) & (raw_arr <= max_article_tokens)
    return {
        "n_total": int(raw_arr.size),
        "n_after_filter": int(mask.sum()),
        "filter": {
            "max_article_tokens": max_article_tokens,
            "min_article_tokens": min_article_tokens,
        },
        "token_lengths": {
            "raw": _length_dist(raw_arr),
            "filtered": _length_dist(raw_arr[mask]),
        },
        "answer_lengths": _length_dist(ans_arr),
    }


def tokenize_samples(
    raw_samples: List[Dict],
    tokenizer,
    task,
    prompt_version: str,
    max_article_tokens: Optional[int] = 30000,
) -> List[Dict]:
    """Tokenize raw samples into the same dict shape train_e2e_gate.py expects.

    Output keys: full_ids, ctx_len, article_start, article_end, n_ans,
    target, article, dataset, task, id, prompt_version.

    Per project policy we NEVER truncate a sample: when max_article_tokens is
    set, rows whose compressible article exceeds it are DROPPED, not sliced.
    Truncating a long doc loses the content its target depends on, and the full
    context would later OOM the no-compress baseline eval (which re-tokenizes the
    article).  Datasets that are mostly longer than the cap therefore shrink a
    lot here — that is intended.
    """
    from tokengate.e2e_gate.gate_module.gate_forward import (
        apply_chat_template, find_article_boundaries,
    )
    out: List[Dict] = []
    n_dropped = 0
    for s in raw_samples:
        prompt = task.format_prompt(s["article"])
        prompt_ids = apply_chat_template(tokenizer, prompt)
        answer_ids = tokenizer.encode(
            s["answer"] + tokenizer.eos_token, add_special_tokens=False,
        )

        non_comp = task.get_non_compressible_context(s["article"])
        empty_prompt = task.format_prompt(non_comp)
        empty_ids = apply_chat_template(tokenizer, empty_prompt)
        pre_len, post_len = find_article_boundaries(prompt_ids, empty_ids)
        article_start = pre_len
        article_end = len(prompt_ids) - post_len
        n_art = article_end - article_start
        if n_art <= 0:
            continue

        # NEVER truncate: drop rows whose compressible article exceeds the cap.
        if max_article_tokens is not None and n_art > max_article_tokens:
            n_dropped += 1
            continue
        full_ids = prompt_ids + answer_ids

        out.append({
            "full_ids": full_ids,
            "ctx_len": len(prompt_ids),
            "article_start": article_start,
            "article_end": article_end,
            "n_ans": len(answer_ids),
            "target": s["answer"],
            "article": s["article"],
            "dataset": s["dataset"],
            "task": s["task"],
            "id": s.get("id"),
            "prompt_version": prompt_version,
        })
    if n_dropped:
        print(f"[tokenize] dropped {n_dropped} samples with article > "
              f"max_article_tokens={max_article_tokens} (never truncated); "
              f"kept {len(out)}")
    return out


def write_pkl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(rows, f)


def prepare_dataset(
    *,
    dataset_name: str,
    task_name: str,
    load_train_fn: Callable[[], Iterable[Dict]],
    load_val_fn: Optional[Callable[[], Iterable[Dict]]] = None,
    model: str = "Qwen/Qwen3-8B",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_article_tokens: int = 30000,
    min_article_tokens: int = 0,
    run_sanity: bool = True,
    sanity_k: int = 2000,
    sanity_n: int = 20,
    sanity_device: str = "cuda:0",
) -> Dict:
    """End-to-end preparation for one (dataset, task) pair.

    The loader functions yield dicts matching the canonical RAW_FIELDS schema.
    The Task class (looked up from tasks/__init__.py by task_name) must define
    a class-level `prompt_version` string.

    Returns the stats dict (also written to stats.json).
    """
    from tokengate.datasets.tasks import get_task
    task = get_task(task_name)
    prompt_version = task.get_prompt_version()

    base = Path(output_root) / dataset_name / task_name
    base.mkdir(parents=True, exist_ok=True)

    train_raw = list(load_train_fn())
    n_train = write_raw_jsonl(train_raw, base / "raw_train.jsonl")
    print(f"[prepare] {dataset_name}/{task_name}: wrote {n_train} train samples → raw_train.jsonl")

    val_raw: List[Dict] = []
    if load_val_fn is not None:
        val_raw = list(load_val_fn())
        n_val = write_raw_jsonl(val_raw, base / "raw_val.jsonl")
        print(f"[prepare] {dataset_name}/{task_name}: wrote {n_val} val samples → raw_val.jsonl")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    stats = compute_stats(
        train_raw, tokenizer, task,
        max_article_tokens=max_article_tokens,
        min_article_tokens=min_article_tokens,
    )
    with (base / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    rd = stats["token_lengths"]["raw"]
    fd = stats["token_lengths"]["filtered"]
    if stats["n_total"] == 0:
        # eval-only datasets (e.g. niah_kamradt_32k) have no train split
        print(f"[prepare] {dataset_name}/{task_name}: train split EMPTY — eval-only dataset")
    else:
        print(
            f"[prepare] {dataset_name}/{task_name}: stats — "
            f"n_total={stats['n_total']} → after_filter={stats['n_after_filter']} "
            f"(max={max_article_tokens}, min={min_article_tokens})  "
            f"mean tokens raw={rd.get('mean'):.0f} filtered={fd.get('mean'):.0f}"
        )

    model_basename = os.path.basename(model)
    tok_train = tokenize_samples(train_raw, tokenizer, task, prompt_version, max_article_tokens=max_article_tokens)
    write_pkl(tok_train, base / "tokenized" / model_basename / "train.pkl")
    print(f"[prepare] {dataset_name}/{task_name}: tokenized {len(tok_train)} train rows")
    if val_raw:
        tok_val = tokenize_samples(val_raw, tokenizer, task, prompt_version, max_article_tokens=max_article_tokens)
        write_pkl(tok_val, base / "tokenized" / model_basename / "val.pkl")
        print(f"[prepare] {dataset_name}/{task_name}: tokenized {len(tok_val)} val rows")

    prompts_dir = base / "prompts" / model_basename
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "version.txt").write_text(prompt_version + "\n")

    if run_sanity:
        # Defer the heavy torch / FlexAttention imports until actually running.
        # We pass the full train_raw + sanity_n separately: sanity.py applies
        # a length filter first, then takes the first sanity_n. This prevents
        # OOM on datasets whose first samples blow past the model context
        # (e.g. NarrativeQA's first story is 199K tokens > Qwen3-8B's 131K).
        #
        # Scale the generation budget to the dataset's answer length: a fixed
        # small max_gen (the old default of 50) crushed ROUGE for summarization
        # datasets whose references run 100-700 tokens — the oracle was scored
        # on a 50-token truncation.  Use 1.5× the answer p95, clamped to
        # [64, 1024].  Short-answer QA/code emit EOS early so the extra budget
        # is harmless; long-summary tasks finally get room to finish.
        _ans_p95 = stats["answer_lengths"].get("p95", 0) or 0
        sanity_max_gen = int(min(1024, max(64, round(1.5 * _ans_p95))))
        from tokengate.datasets.sanity import run_sanity_check
        run_sanity_check(
            model_name=model,
            task=task,
            raw_samples=train_raw,
            sanity_n=sanity_n,
            K=sanity_k,
            output_dir=base / "sanity" / model_basename,
            device=sanity_device,
            max_gen=sanity_max_gen,
        )

    return stats
