"""Eval-faithful multi-hop twin reconstruction.

The short-snippet twins don't transfer: LongBench multi-hop evals use FULL
Wikipedia-article passages, `Passage N:\\n<title>\\n<body>`, ~10 passages, the
question's OWN gold + distractors, gold de-centered.  This renders that structure
from a prebuilt title->full-Wikipedia-text dict (build_wiki_dict.py) — matching the
eval's passage GRANULARITY (~10 long articles), the axis the format+length lbfmt
twin never matched (it used ~125 short snippets).

DECONTAM (QA best practice: dedup the eval INSTANCE, not the shared
background Wikipedia corpus).  Two layers, both against LongBench musique `test`:
  1. PRIMARY — question-level exact: drop train questions whose normalized text
     matches an eval question (question ⟺ instance ⟺ gold-context for MuSiQue, so
     this removes same-instance context).  (q,a) residual is verified/reported.
  2. SECONDARY near-dup net — GPT-3-style 13-gram CONTAINMENT ≥ 0.3 on the GOLD
     passage bodies (the answer-bearing object) vs the LongBench eval union.  With
     FULL articles this regains teeth: it drops a row whose gold article appears
     verbatim in an eval context (containment ≈ 1.0), while a single shared
     *background* distractor article (≪ 30% of gold shingles) is left in — we do NOT
     purge shared background Wikipedia (that over-flags).
"""
import os
import pickle
import random
import re

from datasets import load_dataset

from tokengate.datasets.mfqa_pack import _sh, _lb_shingles

_LB_SUBSET = {"musique": "musique", "hotpot": "hotpotqa", "2wiki": "2wikimqa"}


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _eval_qa(subset):
    """Normalized eval questions + (question, answer) pairs for the LB subset."""
    qs, qa = set(), set()
    for r in load_dataset("THUDM/LongBench", subset, split="test"):
        q = _norm(r["input"])
        qs.add(q)
        for a in (r.get("answers") or []):
            qa.add((q, _norm(a)))
    return qs, qa


def build_wiki_multihop(source_key, part, seed, n_out, wiki_path, q_path,
                        char_budget=60000, max_passages=15, gold_thr=0.3):
    """Yield reconstructed multi-hop rows for `source_key` in {musique,hotpot,2wiki}."""
    wiki = pickle.load(open(wiki_path, "rb"))
    questions = pickle.load(open(q_path, "rb"))[source_key]
    eval_q, eval_qa = _eval_qa(_LB_SUBSET[source_key])

    rng = random.Random(1234)
    rng.shuffle(questions)
    nval = min(300, len(questions) // 10)
    pool = questions[:-nval] if part == "train" else questions[-nval:]

    # ── layer 1: question-level exact decontam (primary) ──────────
    n0 = len(pool)
    pool = [it for it in pool if _norm(it["q"]) not in eval_q]
    n_qdrop = n0 - len(pool)
    # (q,a) residual sanity (should be ~0 after question-level)
    n_qa_resid = sum(1 for it in pool
                     if (_norm(it["q"]), _norm(it["a"])) in eval_qa)

    # ── layer 2: gold-context 13-gram containment net (secondary) ─
    ev_sh = _lb_shingles()
    kept, n_golddrop = [], 0
    for it in pool:
        gold_body = "\n".join(wiki.get(t, snip) for (t, snip) in it["gold"])
        sh = _sh(gold_body)
        if sh and len(sh & ev_sh) / len(sh) >= gold_thr:
            n_golddrop += 1
            continue
        kept.append(it)
    pool = kept
    print(f"[wiki-reconstruct] {source_key}/{part} decontam: {n0} → -{n_qdrop} q-exact "
          f"→ -{n_golddrop} gold-13gram≥{gold_thr} → {len(pool)} clean "
          f"((q,a) residual after q-exact: {n_qa_resid})", flush=True)
    if len(pool) < 100:
        raise ValueError(f"{source_key}: too few clean questions ({len(pool)})")

    prng = random.Random(seed)
    matched = total = 0
    for i in range(n_out):
        it = pool[i % len(pool)]
        gold = list(it["gold"])            # [(title, snippet)]
        dist = list(it["dist"]); prng.shuffle(dist)
        chosen = gold + dist[: max(0, max_passages - len(gold))]
        prng.shuffle(chosen)               # gold DE-CENTERED
        blocks, L = [], 0
        for (t, snip) in chosen:
            body = wiki.get(t, snip)       # full Wikipedia passage, else snippet fallback
            total += 1
            if t in wiki:
                matched += 1
            blocks.append((t, body)); L += len(body) + len(t) + 16
            if L >= char_budget:
                break
        ctx = "\n".join(f"Passage {j + 1}:\n{t}\n{body}" for j, (t, body) in enumerate(blocks))
        q = it["q"].strip()
        if not q.endswith("?"):
            q += "?"
        a = it["a"].strip()
        yield {
            "id": f"{source_key}_wiki_{part}_{i}",
            "dataset": f"mfqa_{source_key}_wiki",
            "task": "multihop_style",
            "article": f"{ctx}\n\nQuestion: {q}",
            "answer": a,
            "metadata": {"answers": [a]},
        }
    if total:
        print(f"[wiki-reconstruct] {source_key}/{part}: {100 * matched / total:.0f}% passages from "
              f"full Wikipedia ({matched}/{total}); rest snippet-fallback", flush=True)
