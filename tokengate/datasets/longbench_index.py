"""Per-subset LongBench eval signature index (exact + MinHash near-dup).

Used by the `longbench-datasets` matched pretraining datasets: every training row drawn
from an upstream *original* source dataset is deduplicated against the LongBench
eval rows of the **same** subset, so the gate never trains on a row it is later
evaluated on.

Why per-subset (not one combined index): we want to measure and remove *direct*
contamination — a HotpotQA train row is checked against LongBench `hotpotqa`
rows only, never against (say) 2WikiMQA rows. A combined index would both blur
the per-task contamination number and risk dropping a legitimate HotpotQA row
because it superficially resembles a different subset.

Two complementary signals per subset:
  * exact   — normalized-text (blake2b) hash equality. Catches verbatim reuse.
  * neardup — MinHash-LSH Jaccard >= 0.5 over word 13-grams (the standard
              pretrain-dedup recipe already in `dedup_minhash.py`).

Both are computed over the LongBench `context` field (the bulk article the gate
prunes) — NOT context+question, since the upstream originals won't share the
LongBench-specific question phrasing but DO share the underlying document.

CLI:
    python -m tokengate.datasets.longbench_index build
    python -m tokengate.datasets.longbench_index counts
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from tokengate.datasets.dedup_minhash import (
    Decision,
    EvalIndex,
    build_eval_signature_index,
    minhash_for,
    normalize,
)

# ── the 16 English LongBench v1 subsets ───────────────────────────────────────
# (the 3 with no external original — multifieldqa_en, passage_count,
#  passage_retrieval_en — are eval-only here, but we still index them so the
#  index is the complete official eval set.)
LONGBENCH_EN_SUBSETS: Tuple[str, ...] = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)

# Subsets that have a clean external original we train on (see notes §3).
TRAINED_SUBSETS: Tuple[str, ...] = tuple(
    s
    for s in LONGBENCH_EN_SUBSETS
    if s not in ("multifieldqa_en", "passage_count", "passage_retrieval_en")
)

_REPO_LIB = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX_PATH = os.path.join(_REPO_LIB, "indexes", "longbench_eval_index.pkl")


def _exact_hash(text: str) -> str:
    return hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class SubsetIndex:
    subset: str
    n_rows: int
    exact: set                      # set[str] normalized-text hashes
    mh_index: EvalIndex             # MinHash-LSH over the same contexts
    k: int = 13
    num_perm: int = 128
    threshold: float = 0.5


@dataclass
class LongBenchEvalIndex:
    subsets: Dict[str, SubsetIndex] = field(default_factory=dict)
    split: str = "test"

    def total_rows(self) -> int:
        return sum(si.n_rows for si in self.subsets.values())


# ── building ──────────────────────────────────────────────────────────────────

def _iter_contexts(subset: str, split: str) -> Iterable[str]:
    """Yield the LongBench `context` field for one subset.

    Loads from THUDM/LongBench (cached locally). Fails loudly if a subset is
    missing or empty — a silent empty index would give a false 0% contamination.
    """
    from datasets import load_dataset

    raw = load_dataset("THUDM/LongBench", subset, split=split)
    if len(raw) == 0:
        raise RuntimeError(f"LongBench subset '{subset}' split '{split}' is empty")
    for r in raw:
        ctx = r.get("context", "") or ""
        if ctx.strip():
            yield ctx


def build_index(
    subsets: Tuple[str, ...] = LONGBENCH_EN_SUBSETS,
    split: str = "test",
    k: int = 13,
    num_perm: int = 128,
    threshold: float = 0.5,
) -> LongBenchEvalIndex:
    out = LongBenchEvalIndex(split=split)
    for subset in subsets:
        contexts = list(_iter_contexts(subset, split))
        exact = {_exact_hash(c) for c in contexts}
        mh_index = build_eval_signature_index(
            [(f"longbench:{subset}", iter(contexts))],
            k=k, num_perm=num_perm, threshold=threshold,
        )
        # Drop the stored eval text — dedup only needs the MinHash sigs + exact
        # hashes at query time, and the raw contexts (whole novels) dominate the
        # pickle size (~170MB → ~15MB on the small NFS partition).
        for er in mh_index.rows:
            er.text = ""
        out.subsets[subset] = SubsetIndex(
            subset=subset, n_rows=len(contexts), exact=exact,
            mh_index=mh_index, k=k, num_perm=num_perm, threshold=threshold,
        )
        print(f"[longbench-index] {subset:24s} {len(contexts):6d} rows "
              f"({len(exact)} unique-exact)", flush=True)
    print(f"[longbench-index] total {out.total_rows()} rows across "
          f"{len(out.subsets)} subsets", flush=True)
    return out


def save_index(idx: LongBenchEvalIndex, path: str = DEFAULT_INDEX_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    print(f"[longbench-index] wrote {path}", flush=True)


def load_index(path: str = DEFAULT_INDEX_PATH) -> LongBenchEvalIndex:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LongBench eval index not found at {path}; build it via "
            "`python -m tokengate.datasets.longbench_index build`"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ── deduping a candidate stream against ONE subset ────────────────────────────

@dataclass
class DedupResult:
    n_total: int
    n_exact: int
    n_neardup: int
    keep_mask: List[bool]           # parallel to input; True = keep
    decisions: List[Decision]       # MinHash decisions (for jaccard inspection)

    @property
    def n_dropped(self) -> int:
        return self.n_exact + self.n_neardup

    @property
    def dup_fraction(self) -> float:
        return self.n_dropped / max(self.n_total, 1)


def dedup_against_subset(
    idx: LongBenchEvalIndex,
    subset: str,
    candidate_texts: Iterable[str],
) -> DedupResult:
    """Exact + near-dup dedup of a candidate article stream vs one subset's evals.

    A row is dropped if it is an exact normalized-text match OR a MinHash near-dup
    (Jaccard >= threshold) of any LongBench eval row in `subset`.
    """
    if subset not in idx.subsets:
        raise KeyError(
            f"subset '{subset}' not in LongBench index "
            f"(have: {sorted(idx.subsets)})"
        )
    si = idx.subsets[subset]
    texts = list(candidate_texts)

    keep: List[bool] = []
    decisions: List[Decision] = []
    n_exact = 0
    n_near = 0
    for cidx, text in enumerate(texts):
        text = text or ""
        # exact
        if text.strip() and _exact_hash(text) in si.exact:
            n_exact += 1
            keep.append(False)
            decisions.append(Decision(cidx, "exact-dup", f"longbench:{subset}", None, 1.0))
            continue
        # near-dup via this subset's LSH
        if not text.strip():
            keep.append(True)
            decisions.append(Decision(cidx, "keep", None, None, 0.0))
            continue
        mh = minhash_for(text, k=si.k, num_perm=si.num_perm)
        hits = si.mh_index.lsh.query(mh)
        if not hits:
            keep.append(True)
            decisions.append(Decision(cidx, "keep", None, None, 0.0))
            continue
        best_idx, best_j = max(
            ((int(h[1:]), si.mh_index.rows[int(h[1:])].minhash.jaccard(mh)) for h in hits),
            key=lambda x: x[1],
        )
        if best_j >= si.threshold:
            n_near += 1
            keep.append(False)
            decisions.append(Decision(cidx, "near-dup", f"longbench:{subset}", best_idx, best_j))
        else:
            keep.append(True)
            decisions.append(Decision(cidx, "keep", f"longbench:{subset}", best_idx, best_j))

    return DedupResult(
        n_total=len(texts), n_exact=n_exact, n_neardup=n_near,
        keep_mask=keep, decisions=decisions,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    # Import via the canonical module path so pickled dataclasses are qualified
    # as src...longbench_index.* (NOT __main__.*, which `python -m`
    # would otherwise stamp them with, breaking load from any other process).
    from tokengate.datasets import longbench_index as M

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "counts"])
    ap.add_argument("--path", default=DEFAULT_INDEX_PATH)
    ap.add_argument("--split", default="test")
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="restrict to these subsets (default: all 16 English)")
    args = ap.parse_args()

    if args.cmd == "build":
        subsets = tuple(args.subsets) if args.subsets else M.LONGBENCH_EN_SUBSETS
        idx = M.build_index(subsets, split=args.split)
        M.save_index(idx, args.path)
    elif args.cmd == "counts":
        idx = M.load_index(args.path)
        print(f"split={idx.split}  subsets={len(idx.subsets)}  total_rows={idx.total_rows()}")
        for s in LONGBENCH_EN_SUBSETS:
            if s in idx.subsets:
                si = idx.subsets[s]
                print(f"  {s:24s} {si.n_rows:6d} rows  ({len(si.exact)} unique-exact)")


if __name__ == "__main__":
    main()
