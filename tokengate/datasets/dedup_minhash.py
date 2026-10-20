"""MinHash + LSH near-duplicate detection vs ALL eval sets.

Standard pretrain dedup practice (Pile / RefinedWeb / FineWeb): per-document
MinHash signature over k-gram shingles, Locality-Sensitive Hashing for fast
near-neighbour queries, Jaccard threshold ~0.5 for "near-dup" decisions.
Documented in Lee et al. 2022 (Deduplicating Training Data Makes Language
Models Better) and used as-is in subsequent pretrain corpora.

We use word 13-grams (~standard for natural language) and Jaccard threshold
0.5 (slightly more permissive than the 0.8 used by Pile; we are deduping a
candidate against a much smaller eval set, so we want recall, not precision).

Usage (programmatic):

    idx = build_eval_signature_index([
        "...all eval val.pkl paths...",      # training-eval val sets
        "...heldout caches...",
        "...LongBench cache...",
    ])
    decision = check_candidate(idx, candidate_iterable)   # yields per-row decisions

Or via CLI (see scripts/dedup_candidate.py):

    python -m tokengate.datasets.dedup_minhash \\
        --eval-roots ... \\
        --candidate-hf hotpot_qa:distractor \\
        --out-csv reports/dedup_hotpot.csv

Decision schema (CSV columns):
    candidate_idx, decision, nearest_eval_dataset, nearest_eval_idx, jaccard

decision ∈ {keep, near-dup}. The fastest dedup uses LSH (sub-linear); we
also report the exact Jaccard to the nearest neighbour so analyses can apply
different thresholds.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from datasketch import MinHash, MinHashLSH


# ── shingling ─────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """lower → strip non-word/space punctuation → collapse whitespace.

    Matches the convention used in standard pretrain dedup pipelines (FineWeb,
    Pile). Strong enough to canonicalise minor formatting differences without
    being so aggressive that it conflates distinct samples.
    """
    if text is None:
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def shingles(text: str, k: int = 13) -> List[bytes]:
    """Word k-grams as byte strings (ready to feed into MinHash.update)."""
    toks = normalize(text).split()
    if len(toks) < k:
        # Pad small docs with their full token list so they still get a signature.
        return [(" ".join(toks)).encode("utf-8")] if toks else []
    return [(" ".join(toks[i:i + k])).encode("utf-8") for i in range(len(toks) - k + 1)]


def minhash_for(text: str, k: int = 13, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles(text, k):
        m.update(s)
    return m


# ── eval signature index ──────────────────────────────────────────────────────

@dataclass
class EvalRow:
    dataset: str
    row_idx: int
    text: str
    minhash: MinHash


@dataclass
class EvalIndex:
    lsh: MinHashLSH
    rows: List[EvalRow]
    k: int
    num_perm: int
    threshold: float


def build_eval_signature_index(
    eval_sources: List[Tuple[str, Iterable[str]]],
    k: int = 13,
    num_perm: int = 128,
    threshold: float = 0.5,
) -> EvalIndex:
    """Build a MinHashLSH over a collection of eval document texts.

    Args:
        eval_sources: list of (dataset_tag, iterable-of-strings). Each
            iterable yields the article/context text of one eval sample.
            Pass dataset_tag = whatever you want shown in dedup reports
            (e.g. "v3_train_qa", "heldout_biodex", "longbench_hotpotqa").
        k: shingle size in words (default 13).
        num_perm: MinHash permutations (default 128, standard).
        threshold: Jaccard threshold for the LSH retrieval (default 0.5).

    Returns:
        EvalIndex with .lsh, .rows (parallel to LSH keys), and metadata.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    rows: List[EvalRow] = []
    next_idx = 0
    for dataset_tag, iterable in eval_sources:
        per_ds_count = 0
        for text in iterable:
            text = text or ""
            if not text.strip():
                continue
            mh = minhash_for(text, k=k, num_perm=num_perm)
            key = f"e{next_idx}"
            lsh.insert(key, mh)
            rows.append(EvalRow(dataset=dataset_tag, row_idx=per_ds_count,
                                text=text, minhash=mh))
            next_idx += 1
            per_ds_count += 1
    return EvalIndex(lsh=lsh, rows=rows, k=k, num_perm=num_perm, threshold=threshold)


# ── candidate scanning ────────────────────────────────────────────────────────

@dataclass
class Decision:
    candidate_idx: int
    decision: str               # "keep" | "near-dup"
    nearest_eval_dataset: Optional[str]
    nearest_eval_idx: Optional[int]
    jaccard: float              # to the nearest eval row (0 if none)


def check_candidate(
    idx: EvalIndex,
    candidate_iterable: Iterable[str],
) -> List[Decision]:
    """Per-row dedup decisions for a candidate stream against the eval index.

    `candidate_iterable` yields candidate article texts in order; the returned
    list has one Decision per yielded item, in input order.

    Jaccard is computed as the exact pairwise Jaccard of the **MinHash
    estimates** (not the true Jaccard of shingle sets) — sufficient at
    num_perm=128 (standard error ~1/sqrt(128) ≈ 9%).
    """
    out: List[Decision] = []
    for cidx, text in enumerate(candidate_iterable):
        text = text or ""
        if not text.strip():
            out.append(Decision(cidx, "keep", None, None, 0.0))
            continue
        mh = minhash_for(text, k=idx.k, num_perm=idx.num_perm)
        # LSH approximate near-neighbour candidates → exact MinHash jaccard.
        hits = idx.lsh.query(mh)
        if not hits:
            out.append(Decision(cidx, "keep", None, None, 0.0))
            continue
        best = max(
            ((int(h[1:]), idx.rows[int(h[1:])].minhash.jaccard(mh)) for h in hits),
            key=lambda x: x[1],
        )
        best_idx, best_j = best
        if best_j >= idx.threshold:
            er = idx.rows[best_idx]
            out.append(Decision(cidx, "near-dup", er.dataset, er.row_idx, best_j))
        else:
            er = idx.rows[best_idx]
            out.append(Decision(cidx, "keep", er.dataset, er.row_idx, best_j))
    return out


# ── helpers for reading our eval sources ──────────────────────────────────────

def _read_pkl_articles(pkl_path: str) -> Iterable[str]:
    """Yield 'article' strings from a tokenized val.pkl (list of sample dicts)."""
    if not os.path.exists(pkl_path):
        return
    with open(pkl_path, "rb") as f:
        rows = pickle.load(f)
    if not isinstance(rows, list):
        rows = rows.get("eval_samples", [])
    for r in rows:
        a = r.get("article") if isinstance(r, dict) else None
        if a:
            yield a


def collect_default_eval_sources(
    pretrain_data_root: str,
    model_name: str = "Qwen3-8B",
) -> List[Tuple[str, Iterable[str]]]:
    """Default eval-source list: every (dataset, task) in the manifest's val.pkl.

    Held-out OOD + LongBench eval sets aren't materialised on disk; for those we
    re-load on demand at index-build time. This helper covers only the
    training-eval val.pkls. Add held-out + LongBench separately by passing
    additional (tag, iterable) pairs to build_eval_signature_index.
    """
    sources: List[Tuple[str, Iterable[str]]] = []
    if not os.path.isdir(pretrain_data_root):
        return sources
    for ds in sorted(os.listdir(pretrain_data_root)):
        ds_root = os.path.join(pretrain_data_root, ds)
        if not os.path.isdir(ds_root):
            continue
        for task in sorted(os.listdir(ds_root)):
            val_pkl = os.path.join(ds_root, task, "tokenized", model_name, "val.pkl")
            if os.path.exists(val_pkl):
                sources.append((f"v3_val:{ds}/{task}", _read_pkl_articles(val_pkl)))
    return sources


# ── CLI ───────────────────────────────────────────────────────────────────────

def _read_jsonl_articles(path: str, article_field: str = "article") -> Iterable[str]:
    import json
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            t = row.get(article_field)
            if t:
                yield t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-roots", nargs="+", required=True,
                    help="space-separated list of dirs containing val.pkl trees "
                         "(typically just /.../pretraining_data).")
    ap.add_argument("--candidate-jsonl", required=True,
                    help="path to a JSONL where each row has an 'article' field "
                         "(or pass --article-field to point elsewhere).")
    ap.add_argument("--article-field", default="article")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--k", type=int, default=13)
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model-name", default="Qwen3-8B")
    args = ap.parse_args()

    eval_sources: List[Tuple[str, Iterable[str]]] = []
    for root in args.eval_roots:
        eval_sources.extend(collect_default_eval_sources(root, args.model_name))
    print(f"[dedup] indexing {len(eval_sources)} eval sources...", flush=True)
    idx = build_eval_signature_index(eval_sources, k=args.k,
                                     num_perm=args.num_perm,
                                     threshold=args.threshold)
    print(f"[dedup] indexed {len(idx.rows)} eval rows", flush=True)

    cand_iter = _read_jsonl_articles(args.candidate_jsonl, args.article_field)
    decisions = check_candidate(idx, cand_iter)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_idx", "decision", "nearest_eval_dataset",
                    "nearest_eval_idx", "jaccard"])
        for d in decisions:
            w.writerow([d.candidate_idx, d.decision,
                        d.nearest_eval_dataset or "",
                        d.nearest_eval_idx if d.nearest_eval_idx is not None else "",
                        f"{d.jaccard:.4f}"])
    n_dup = sum(1 for d in decisions if d.decision == "near-dup")
    print(f"[dedup] wrote {args.out_csv}: {n_dup}/{len(decisions)} near-dup "
          f"(threshold j>={args.threshold})", flush=True)


if __name__ == "__main__":
    main()
