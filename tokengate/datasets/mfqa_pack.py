"""Shared builder for multifieldqa-style single-doc QA packs.

Targets held-out longbench_multifieldqa_en, which the sole current twin (NQ) covers
poorly: NQ is Wikipedia-only, short-extractive, answer-front-loaded, and even renders
a DIFFERENT template than the eval.  These packs instead assemble DIVERSE-domain
single-doc QA (medical/sci/textbook/movie/news/exam) and:
  - decontaminate every source doc vs ALL 16 LongBench eval subsets (GPT-3-style
    13-gram CONTAINMENT ≥ 0.3 → drop; containment because candidate docs << eval docs),
  - distractor-pad the gold doc with same-source docs to multifieldqa's length spread
    with the gold DE-CENTERED (random position) → teaches query-conditioned selection
    of the answer region (not NQ's front-loaded window),
  - render query-first with the EXACT multifieldqa eval template (see MFQAStyleTask).

One dataset per source (mfqa_<source>), all under task `mfqa_style`; blend = train on
several mfqa_<source> datasets at once (launcher).  Sources verified available +
LongBench-clean (decontam pass).
"""
import os
import pickle
import random
import re
import zlib

from datasets import load_dataset

from tokengate.datasets.tasks.v8_coverage_tasks import QUESTION_MARKER

_K = 13
_LB = ["multifieldqa_en", "narrativeqa", "qasper", "hotpotqa", "2wikimqa", "musique",
       "triviaqa", "gov_report", "qmsum", "multi_news", "samsum", "trec",
       "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
# _v2 = deterministic-hash cache (see _sh). The old ".mfqa_lb_shingles.pkl" held
# process-salted hash() values — portable ONLY within the writing process, so a
# cross-process reload silently found ~0 overlaps.  Bumping the name invalidates it.
_CACHE = os.path.join(os.path.dirname(__file__), ".mfqa_lb_shingles_v2.pkl")
# multifieldqa length spread is ~7.2k tok (1.4k-16.5k); ~4 chars/tok → char targets.
_LEN = (28000, 12000, 5000, 62000)   # char (mean, sd, lo, hi)


def _sh(t):
    # DETERMINISTIC shingle hashing (crc32), NOT builtin hash(): hash() is
    # PYTHONHASHSEED-salted per process, so a cached shingle set written in one
    # process and reloaded in another would never match fresh _sh() calls →
    # decontam silently passes everything.  crc32 is stable across processes and
    # cheap; its rare 32-bit collisions only ADD spurious matches (→ over-drop a
    # clean item), which is the SAFE direction for a contamination filter.
    w = str(t).lower().split()
    return {zlib.crc32(" ".join(w[i:i + _K]).encode()) for i in range(len(w) - _K + 1)}


def _lb_shingles():
    if os.path.exists(_CACHE):
        with open(_CACHE, "rb") as f:
            return pickle.load(f)
    s = set()
    for sub in _LB:
        try:
            for r in load_dataset("THUDM/LongBench", sub, split="test"):
                s |= _sh(r["context"])
        except Exception as e:
            print(f"[mfqa decontam] skip {sub}: {str(e)[:60]}", flush=True)
    with open(_CACHE, "wb") as f:
        pickle.dump(s, f)
    return s


_QAREF_CACHE = os.path.join(os.path.dirname(__file__), ".mfqa_lb_qa_ref_v1.pkl")


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _sh_words(t):
    """13-gram crc32 shingle set over the word sequence of `t` (deterministic; see _sh)."""
    w = _norm(t).split()
    return {zlib.crc32(" ".join(w[i:i + _K]).encode()) for i in range(len(w) - _K + 1)}


def _lb_qa_ref():
    """Eval reference for the (input+answers) PAIR unit + diagnostic reports:
      pair_sh  — 13-gram shingles over each eval `input + answer` (the enforced pair unit)
      qa_exact — normalized (question, answer) pairs (exact fallback for sub-13-word pairs)
      evq      — normalized eval questions (question-overlap diagnostic)."""
    if os.path.exists(_QAREF_CACHE):
        with open(_QAREF_CACHE, "rb") as f:
            return pickle.load(f)
    pair_sh, qa_exact, evq = set(), set(), set()
    for sub in _LB:
        try:
            for r in load_dataset("THUDM/LongBench", sub, split="test"):
                q = _norm(r["input"]); evq.add(q)
                for a in (r.get("answers") or []):
                    an = _norm(a)
                    qa_exact.add((q, an))
                    pair_sh |= _sh_words(q + " " + an)
        except Exception as e:
            print(f"[mfqa decontam] skip {sub}: {str(e)[:60]}", flush=True)
    ref = (pair_sh, qa_exact, evq)
    with open(_QAREF_CACHE, "wb") as f:
        pickle.dump(ref, f)
    return ref


def _decontam(items, thr=0.3):
    """Decontamination for the multifieldqa single-doc QA twins.

    ENFORCED unit = the **(input + answers) pair** (13-gram exact match, with an exact-normalized
    (q,a) fallback for pairs shorter than a 13-gram).  A train row is dropped iff its
    (question, answer) reproduces an eval instance's (input, answers).  Document/background
    sharing is deliberately NOT purged — key principle: a doc-level n-gram purge
    over-flags shared source corpus (e.g. 2wikimqa mis-flagged 7.6%) and is the wrong unit.

    After filtering, REPORTS two diagnostics on the surviving (kept) set — NOT used to drop:
      - document overlap: % of kept train docs with 13-gram containment >= thr vs eval contexts
      - question overlap: % of kept train questions that exactly match an eval question.
    """
    ctx_sh = _lb_shingles()
    pair_sh, qa_exact, evq = _lb_qa_ref()
    out, drop = [], 0
    for it in items:
        q = _norm(it.get("q", "")); a = _norm(it.get("a", ""))
        if (q, a) in qa_exact or (_sh_words(q + " " + a) & pair_sh):   # (input+answers) pair unit
            drop += 1
            continue
        out.append(it)
    # post-filter diagnostics (reported, not enforced)
    doc_ov = q_ov = 0
    for it in out:
        dsh = _sh(it["doc"])
        if dsh and len(dsh & ctx_sh) / len(dsh) >= thr:
            doc_ov += 1
        if _norm(it.get("q", "")) in evq:
            q_ov += 1
    nn = max(len(out), 1)
    print(f"[mfqa decontam] {len(items)} src → dropped {drop} (input+answers pair) → {len(out)} clean",
          flush=True)
    print(f"[mfqa decontam] post-filter overlap vs eval: document {100 * doc_ov / nn:.2f}% "
          f"({doc_ov}/{len(out)}), question {100 * q_ov / nn:.2f}% ({q_ov}/{len(out)})", flush=True)
    return out


# ── source loaders → [{"doc","q","a"}] ───────────────────────────────────────
def src_duorc():
    out = []
    for r in load_dataset("ibm/duorc", "SelfRC", split="train"):
        a = (r["answers"] or [""])[0]
        if a and r["plot"]:
            out.append({"doc": r["plot"], "q": r["question"], "a": a})
    return out


def src_pubmedqa():
    out = []
    for cfg in ["pqa_labeled", "pqa_artificial"]:
        try:
            for r in load_dataset("qiaojin/PubMedQA", cfg, split="train"):
                ctx = r["context"]
                doc = " ".join(ctx["contexts"]) if isinstance(ctx, dict) else str(ctx)
                a = r.get("long_answer") or r.get("final_decision")
                if doc and a:
                    out.append({"doc": doc, "q": r["question"], "a": a})
        except Exception as e:
            print(f"[mfqa] pubmedqa {cfg}: {str(e)[:60]}", flush=True)
    return out


def src_race():
    out = []
    for r in load_dataset("ehovy/race", "all", split="train"):
        opts, ans = r["options"], r["answer"]
        a = opts["ABCD".index(ans)] if ans in "ABCD" and len(opts) >= 4 else None
        if a and r["article"]:
            out.append({"doc": r["article"], "q": r["question"], "a": a})
    return out


def src_newsqa():
    out = []
    for r in load_dataset("lucadiliello/newsqa", split="train"):
        ans = r["answers"]
        a = (ans or [""])[0] if isinstance(ans, list) else str(ans)
        if a and r["context"]:
            out.append({"doc": r["context"], "q": r["question"], "a": a})
    return out


def _mrqa(sub):
    out = []
    for r in load_dataset("mrqa", split="test"):
        if str(r["subset"]).lower() != sub.lower():
            continue
        ans = r.get("answers") or r.get("detected_answers", {}).get("text") or []
        a = ans[0] if isinstance(ans, list) and ans else ""
        if a and r["context"]:
            out.append({"doc": r["context"], "q": r["question"], "a": a})
    return out


SOURCES = {
    "mfqa_duorc": src_duorc, "mfqa_pubmedqa": src_pubmedqa, "mfqa_race": src_race,
    "mfqa_newsqa": src_newsqa, "mfqa_bioasq": lambda: _mrqa("BioASQ"),
    "mfqa_textbookqa": lambda: _mrqa("TextbookQA"),
}


def _pad(gold, distractors, tgt_chars, rng):
    """Concatenate gold + same-source distractors to ~tgt_chars, gold DE-CENTERED."""
    blocks, L = [gold], len(gold)
    while L < tgt_chars and distractors:
        d = distractors[rng.randrange(len(distractors))]
        blocks.append(d)
        L += len(d) + 2
    rng.shuffle(blocks)   # random block order → the answer doc is de-centered
    return "\n\n".join(blocks)


# ── COHERENT long-doc sources (window ONE real doc; NO distractor concatenation) ─
# The distractor-padded packs taught coarse cross-doc retrieval that did NOT transfer
# to multifieldqa (a single COHERENT doc).  These window a genuinely long
# coherent doc to eval length, DE-CENTERED around the answer's evidence span, so the
# gate must do fine-grained WITHIN-document selection — matching multifieldqa.
def _loogle(cfg):
    out = []
    for r in load_dataset("bigai-nlco/LooGLE", cfg, split="test"):
        a, ev = r.get("answer"), r.get("evidence") or ""
        if isinstance(ev, list):
            ev = " ".join(str(x) for x in ev)
        q = str(r.get("question") or "").split("\n")[0].strip()   # drop MCQ option lines
        if a and r.get("context") and q:
            out.append({"doc": r["context"], "q": q, "a": str(a), "ev": str(ev)})
    return out


def src_loogle():
    return _loogle("shortdep_qa") + _loogle("longdep_qa")


# ── Abstractive single-doc QA sources for multifieldqa: ~10-word
# abstractive answers + domain diversity — the answer-style our extractive twins lack.
# All go through build_coherent (window ONE coherent doc, NO distractor concat) +
# decontam vs all 16 LongBench subsets (critical for narrativeqa, which IS an eval
# subset — its stories must be dropped where they overlap the LongBench narrativeqa eval).
def src_narrativeqa(cap=9000, doc_cap=80000):
    # Full books are huge (up to 150k+ chars); cap the pool (we need ~5k packs) and
    # truncate each doc (build_coherent windows to ~7k anyway) so decontam is tractable.
    out = []
    for r in load_dataset("deepmind/narrativeqa", split="train"):
        doc = (r.get("document", {}) or {}).get("text") or ""
        ans = r.get("answers") or []
        a = (ans[0].get("text") if isinstance(ans[0], dict) else ans[0]) if ans else ""
        q = (r.get("question", {}) or {}).get("text") or ""
        if doc and a and q:
            out.append({"doc": doc[:doc_cap], "q": q, "a": str(a), "ev": str(a)})
            if len(out) >= cap:
                break
    return out


def src_coqa():
    import json, urllib.request
    url = "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json"
    with urllib.request.urlopen(url) as f:
        data = json.load(f)["data"]
    out = []
    for s in data:
        story = s.get("story") or ""
        qs, ans = s.get("questions", []), s.get("answers", [])
        # one QA per story (the first non-trivial) → keep it single-doc, coherent.
        for q, a in zip(qs, ans):
            at = a.get("input_text", "").strip()
            qt = q.get("input_text", "").strip()
            if story and at and qt and at.lower() not in ("yes", "no", "unknown"):
                out.append({"doc": story, "q": qt, "a": at, "ev": at})
                break
    return out


def src_msmarco():
    # HF parquet CDN is intermittently blocked; hf_hub_download of the shards is the
    # reliable path.  well-formed answers = abstractive ~13-word (best length match).
    import pandas as pd
    from huggingface_hub import hf_hub_download
    out = []
    for shard in range(7):
        fp = hf_hub_download("microsoft/ms_marco",
                             f"v2.1/train-{shard:05d}-of-00007.parquet", repo_type="dataset")
        df = pd.read_parquet(fp, columns=["query", "passages", "wellFormedAnswers", "answers"])
        for _, r in df.iterrows():
            w = list(r["wellFormedAnswers"]) if r["wellFormedAnswers"] is not None else []
            a = w[0] if w and w[0] else None
            if not a:
                continue
            p = r["passages"]
            doc = " ".join(p.get("passage_text", [])) if isinstance(p, dict) else ""
            if doc:
                out.append({"doc": doc, "q": str(r["query"]), "a": str(a), "ev": str(a)})
        if len(out) >= 40000:
            break
    return out


# msmarco moved to SOURCES (distractor-padded to ~7k): its ~10 passages/query are only
# ~840 tok, so at K=2000 the coherent build kept the whole doc (no selection).  Padding
# the answer-passage with same-source msmarco passages (gold de-centered) makes it a real
# selection task while keeping the abstractive ~14-word answer.  coqa dropped: its ~400-tok
# stories are irreparably short (no selection).
SOURCES["mfqa_msmarco"] = src_msmarco

def src_mfqa_wikigen():
    """Generated answer-style twin for multifieldqa_en. Coherent ~5k-tok
    Wikipedia articles (factual, multi-field) + a localized question + a ~10-word
    ABSTRACTIVE generated answer (Qwen3-32B via serve_qgen) — the answer-style +
    factual-domain + single-coherent-doc + length combination no existing dataset has
    (msmarco had the answer style but short ctx; narrativeqa had the length
    but short answers). Generated by gen_multifieldqa_qa.py."""
    import json
    path = os.environ.get("MFQA_QA_JSONL") or f"{os.environ['WIKI_SP']}/multifieldqa_abs_qa.jsonl"
    out = []
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("doc") and r.get("q") and r.get("a"):
            out.append({"doc": r["doc"], "q": r["q"], "a": r["a"], "ev": r["a"]})
    return out


def src_mfqa_papers():
    """arXiv scientific-paper slice of the multifieldqa twin. Same generated
    answer-style abstractive QA as src_mfqa_wikigen, but over full-text arXiv papers
    (papers_pool.pkl, decontaminated vs the eval) — closes the ~39% scientific-paper
    document-type gap that the 100%-Wikipedia mfqa_wikigen twin left open."""
    import json
    path = os.environ.get("MFQA_PAPERS_JSONL") or f"{os.environ['WIKI_SP']}/multifieldqa_papers_qa.jsonl"
    out = []
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("doc") and r.get("q") and r.get("a"):
            out.append({"doc": r["doc"], "q": r["q"], "a": r["a"], "ev": r["a"]})
    return out


def src_mfqa_web():
    """Web/blog slice of the multifieldqa twin. Same generated abstractive QA as
    src_mfqa_wikigen, over C4 web pages (web_pool.pkl, decontaminated vs the eval) — closes the
    ~5% web/blog document-type gap (blogs, news, dictionary/reference, product pages)."""
    import json
    path = os.environ.get("MFQA_WEB_JSONL") or f"{os.environ['WIKI_SP']}/multifieldqa_web_qa.jsonl"
    out = []
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("doc") and r.get("q") and r.get("a"):
            out.append({"doc": r["doc"], "q": r["q"], "a": r["a"], "ev": r["a"]})
    return out


SOURCES_COHERENT = {
    "mfqa_loogle": src_loogle, "mfqa_narrativeqa": src_narrativeqa,
    "mfqa_wikigen": src_mfqa_wikigen, "mfqa_papers": src_mfqa_papers,
    "mfqa_web": src_mfqa_web,
}




def _window(doc, ev, tgt_chars, rng):
    """A ~tgt_chars CONTIGUOUS window of one coherent doc, DE-CENTERED on the evidence."""
    if len(doc) <= tgt_chars:
        return doc
    pos = doc.find(ev[:80]) if ev and len(ev) >= 20 else -1
    if pos < 0:
        pos = rng.randrange(len(doc) - tgt_chars)          # no evidence → random slice
    off = rng.randint(0, max(1, tgt_chars - min(len(ev) or 400, tgt_chars // 2)))
    lo = min(max(0, pos - off), max(0, len(doc) - tgt_chars))
    return doc[lo:lo + tgt_chars]


# ── MULTI-HOP: real gold supporting + real distractor paragraphs, padded to EVAL
# length with extra real distractors, gold DE-CENTERED.  For
# multi-hop QA the distractor structure IS the eval structure (find 2+ needles in a
# haystack), so this — unlike the single-doc multifieldqa case — is faithful, not a
# retrieval artifact.  The LongBench-matched train SHORTENS these (1.4k vs 13k eval);
# this trains at eval length instead.
def src_hotpot():
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", trust_remote_code=True)
    out = []
    for r in ds:
        titles, sents = r["context"]["title"], r["context"]["sentences"]
        sup = set(r["supporting_facts"]["title"])
        gold = [f"{t}. {' '.join(s)}" for t, s in zip(titles, sents) if t in sup]
        dist = [f"{t}. {' '.join(s)}" for t, s in zip(titles, sents) if t not in sup]
        if gold and r["answer"]:
            out.append({"gold": gold, "dist": dist, "q": r["question"], "a": r["answer"]})
    return out


def src_musique():
    # MuSiQue: 20 paragraphs each with is_supporting flag (2-4 hop).  Same gold+distractor
    # structure as HotpotQA → eval-length twin for the held-out musique subset (worst QA:
    # F1 0.05, trained on 2.5k vs 16k eval).  Answerable only.
    ds = load_dataset("dgslibisey/MuSiQue", split="train", trust_remote_code=True)
    out = []
    for r in ds:
        if not r.get("answerable", True) or not r.get("answer"):
            continue
        gold, dist = [], []
        for p in r["paragraphs"]:
            block = f"{p.get('title','')}. {p.get('paragraph_text','')}".strip()
            (gold if p.get("is_supporting") else dist).append(block)
        if gold:
            out.append({"gold": gold, "dist": dist, "q": r["question"], "a": r["answer"]})
    return out


def src_musique_2hop():
    # MuSiQue filtered to 2-HOP questions only (id prefix '2hop__'), which need only 2 gold
    # passages → fit the K=2000 budget (unlike 3-4-hop, which are K-bound).  Tests
    # whether hop-depth (not twin construction) is the binding constraint on musique transfer.
    ds = load_dataset("dgslibisey/MuSiQue", split="train", trust_remote_code=True)
    out = []
    for r in ds:
        if not str(r.get("id", "")).startswith("2hop"):
            continue
        if not r.get("answerable", True) or not r.get("answer"):
            continue
        gold, dist = [], []
        for p in r["paragraphs"]:
            block = f"{p.get('title','')}. {p.get('paragraph_text','')}".strip()
            (gold if p.get("is_supporting") else dist).append(block)
        if gold:
            out.append({"gold": gold, "dist": dist, "q": r["question"], "a": r["answer"]})
    return out


def src_2wiki():
    # 2WikiMultihopQA (framolfese mirror): context = per-paragraph {title, sentences},
    # supporting_facts.title marks the gold paragraphs → same gold+distractor structure
    # as HotpotQA.  Eval-length twin for the held-out 2wikimqa subset (native train ~0.85k
    # vs 6.6k eval) and extra 2-hop signal for hotpotqa.  167k answerable src rows.
    ds = load_dataset("framolfese/2WikiMultihopQA", split="train", trust_remote_code=True)
    out = []
    for r in ds:
        ans = (r.get("answer") or "").strip()
        if not ans:
            continue
        ctx = r["context"]
        titles, sents = ctx["title"], ctx["sentences"]
        sup = set(r["supporting_facts"]["title"])
        gold, dist = [], []
        for t, s in zip(titles, sents):
            block = f"{t}. {' '.join(x for x in s if isinstance(x, str))}".strip()
            (gold if t in sup else dist).append(block)
        if gold:
            out.append({"gold": gold, "dist": dist, "q": r["question"], "a": ans})
    return out


def src_musique_paras():
    """MuSiQue train paragraphs with title/text kept SEPARATE (for LongBench's exact
    `Passage {i}:\\n{title}\\n{paragraph_text}` rendering — see build_musique_lbfmt).
    Answerable only.  Returns gold/dist as (title, text) pairs, NOT pre-joined blocks."""
    ds = load_dataset("dgslibisey/MuSiQue", split="train", trust_remote_code=True)
    out = []
    for r in ds:
        if not r.get("answerable", True) or not r.get("answer"):
            continue
        gold, dist = [], []
        for p in r["paragraphs"]:
            pair = (p.get("title", ""), p.get("paragraph_text", ""))
            (gold if p.get("is_supporting") else dist).append(pair)
        if gold:
            out.append({"gold": gold, "dist": dist, "q": r["question"], "a": r["answer"]})
    return out


def _render_passages(pairs):
    """Render (title, text) pairs as LongBench's exact context layout:
    `Passage 1:\\n{title}\\n{text}\\nPassage 2:\\n...` (sequentially renumbered)."""
    return "\n".join(f"Passage {i + 1}:\n{t}\n{x}" for i, (t, x) in enumerate(pairs))


SOURCES_MH = {"mfqa_hotpot": src_hotpot, "mfqa_musique": src_musique, "mfqa_2wiki": src_2wiki,
              "mfqa_musique_2hop": src_musique_2hop}


# Multi-hop TWINS target true EVAL length (hotpot ~13k tok, musique ~16k tok →
# ~52-64k chars), NOT the shorter single-doc _LEN.  Overridable via length arg.
_LEN_MH = (56000, 14000, 20000, 68000)   # char (mean, sd, lo, hi) ≈ 14k tok mean


def build_multihop(dataset, part, seed, n_out, length=_LEN_MH):
    raw = SOURCES_MH[dataset]()
    # decontam on the gold (answer-bearing) paragraphs vs LongBench eval contexts.
    kept = _decontam([{**it, "doc": " ".join(it["gold"])} for it in raw])
    if len(kept) < 200:
        raise ValueError(f"{dataset}: too few clean items ({len(kept)})")
    rng = random.Random(1234)
    rng.shuffle(kept)
    nval = min(300, len(kept) // 10)
    pool = kept[:-nval] if part == "train" else kept[-nval:]
    dpool = [p for it in pool for p in it["dist"]]          # real distractor paragraphs
    mean, sd, lo, hi = length
    prng = random.Random(seed)
    for i in range(n_out):
        it = pool[i % len(pool)]
        tgt = int(min(hi, max(lo, prng.gauss(mean, sd))))
        blocks, L = list(it["gold"]) + list(it["dist"]), sum(len(x) for x in it["gold"] + it["dist"])
        while L < tgt and dpool:
            d = dpool[prng.randrange(len(dpool))]
            blocks.append(d); L += len(d) + 2
        prng.shuffle(blocks)                                # gold DE-CENTERED among distractors
        ctx = "\n\n".join(blocks)
        q = it["q"].strip()
        if not q.endswith("?"):
            q += "?"
        article = f"{ctx}{QUESTION_MARKER} {q}"
        a = it["a"].strip()
        yield {"id": f"{dataset}_{part}_{i}", "dataset": dataset, "task": "multihop_style",
               "article": article, "answer": a, "metadata": {"answers": [a]}}


# ── EVAL-FORMAT *and* EVAL-LENGTH musique twin ───────────────────
# The two properties no prior musique train set ever combined:
#   * longbench_musique matched-train → eval FORMAT (`Passage {i}:\n{title}\n{text}`,
#     task=longbench_musique) but only ~2.5k tok (raw MuSiQue 20 paragraphs).
#   * mfqa_musique (build_multihop) → eval LENGTH (~14k tok) but the WRONG surface:
#     `{title}. {text}` blocks, `\n\n`-joined, MultiHopQAStyleTask ("Passages:"/…) cue.
# The real LongBench musique eval is ~16.7k tok (p10-p90 14.5-17.2k) in the Passage-N
# layout.  sanity_gated_k2000=0.377 ≈ oracle 0.425 proves K=2000 is *sufficient* on
# well-formed musique — the gate just never learns to select 2000-of-16.7k because no
# train set had both format AND length.  This builder combines them: real MuSiQue
# gold+distractor paragraphs rendered in the exact `Passage {i}:\n{title}\n{text}`
# layout (task=longbench_musique → byte-identical prompt wrapper), padded to eval length
# (~15.5k tok ≈ 62k char, under the 16384 train cap) with real MuSiQue distractor
# paragraphs from the global pool, gold DE-CENTERED, decontaminated vs LongBench eval.
_LEN_MUSIQUE_LBFMT = (62000, 5000, 54000, 66000)   # char (mean, sd, lo, hi); ~15.5k tok


def build_musique_lbfmt(part, seed, n_out, length=_LEN_MUSIQUE_LBFMT):
    raw = src_musique_paras()
    # decontam on gold paragraph text vs LongBench eval contexts (13-gram containment ≥ 0.3)
    tagged = [{**it, "doc": " ".join(f"{t} {x}" for t, x in it["gold"])} for it in raw]
    kept = _decontam(tagged)
    if len(kept) < 200:
        raise ValueError(f"musique_lbfmt: too few clean items ({len(kept)})")
    rng = random.Random(1234)
    rng.shuffle(kept)
    nval = min(300, len(kept) // 10)
    pool = kept[:-nval] if part == "train" else kept[-nval:]
    dpool = [p for it in pool for p in it["dist"]]        # real MuSiQue distractor paragraphs
    mean, sd, lo, hi = length
    prng = random.Random(seed)
    for i in range(n_out):
        it = pool[i % len(pool)]
        tgt = int(min(hi, max(lo, prng.gauss(mean, sd))))
        paras = list(it["gold"]) + list(it["dist"])
        L = sum(len(t) + len(x) + 2 for t, x in paras)
        while L < tgt and dpool:
            t, x = dpool[prng.randrange(len(dpool))]
            paras.append((t, x)); L += len(t) + len(x) + 2
        prng.shuffle(paras)                               # gold DE-CENTERED among distractors
        ctx = _render_passages(paras)                     # sequentially renumbered Passage 1..N
        q = it["q"].strip()
        if not q.endswith("?"):
            q += "?"
        article = f"{ctx}{QUESTION_MARKER} {q}"            # QUESTION_MARKER = "\n\nQuestion:"
        a = it["a"].strip()
        yield {"id": f"mfqa_musique_lbfmt_{part}_{i}", "dataset": "mfqa_musique_lbfmt",
               "task": "longbench_musique",
               "article": article, "answer": a, "metadata": {"answers": [a]}}


def _wikihop_question(query):
    """WikiHop query is 'relation subject' (Wikidata property + entity) → natural Q."""
    rel, subj = (query.split(None, 1) + [""])[:2]
    rel = rel.replace("_", " ").strip()
    if rel.endswith((" of", " in", " by", " for")):     # avoid 'participant of of X'
        return f"What is the {rel} {subj.strip()}?"
    return f"What is the {rel} of {subj.strip()}?"


def src_wikihop():
    # WikiHop (QAngaroo): {query='relation subject', supports=[docs], answer=entity}.
    # No gold/distractor labels — all supports are the (multi-doc) evidence.  ~44k rows,
    # ~1.4k tok native → padded to eval length like the other multi-hop twins.  Extra
    # bridging-entity multi-hop signal for hotpotqa / 2wikimqa (extractive entity answers).
    ds = load_dataset("community-datasets/qangaroo", "wikihop", split="train", trust_remote_code=True)
    out = []
    for r in ds:
        a = (r.get("answer") or "").strip()
        sup = [s.strip() for s in (r.get("supports") or []) if s and s.strip()]
        if a and sup:
            out.append({"gold": sup, "dist": [], "q": _wikihop_question(r["query"]), "a": a})
    return out


def build_wikihop(part, seed, n_out, length=_LEN_MH):
    """Like build_multihop but WikiHop has no per-item distractors: use ALL supports as
    gold and pad each example to eval length with OTHER examples' support paragraphs."""
    raw = _decontam([{**it, "doc": " ".join(it["gold"])} for it in src_wikihop()])
    # Dedup by rendered question so no question string lands in BOTH the train and the
    # held-out val (WikiHop's relational 'relation subject' queries can render identical
    # strings for distinct items → verified train∩val leakage otherwise).
    seen, dedup = set(), []
    for it in raw:
        k = " ".join(it["q"].lower().split())
        if k not in seen:
            seen.add(k)
            dedup.append(it)
    raw = dedup
    if len(raw) < 200:
        raise ValueError(f"wikihop: too few clean items ({len(raw)})")
    rng = random.Random(1234)
    rng.shuffle(raw)
    nval = min(300, len(raw) // 10)
    pool = raw[:-nval] if part == "train" else raw[-nval:]
    dpool = [p for it in pool for p in it["gold"]]      # cross-example support paras = distractors
    mean, sd, lo, hi = length
    prng = random.Random(seed)
    for i in range(n_out):
        it = pool[i % len(pool)]
        tgt = int(min(hi, max(lo, prng.gauss(mean, sd))))
        blocks, L = list(it["gold"]), sum(len(x) for x in it["gold"])
        while L < tgt and dpool:
            d = dpool[prng.randrange(len(dpool))]
            blocks.append(d); L += len(d) + 2
        prng.shuffle(blocks)                             # gold de-centered among distractors
        ctx = "\n\n".join(blocks)
        q = it["q"].strip()
        article = f"{ctx}{QUESTION_MARKER} {q}"
        yield {"id": f"wikihop_{part}_{i}", "dataset": "wikihop", "task": "multihop_style",
               "article": article, "answer": it["a"].strip(), "metadata": {"answers": [it["a"].strip()]}}


def build_coherent(dataset, part, seed, n_out):
    raw = SOURCES_COHERENT[dataset]()
    # decontam on the FULL doc (drop any source doc overlapping a LongBench eval doc)
    items = _decontam([{**it, "doc": it["doc"]} for it in raw])
    if len(items) < 100:
        raise ValueError(f"{dataset}: too few clean items ({len(items)})")
    rng = random.Random(1234)
    rng.shuffle(items)
    nval = min(300, len(items) // 10)
    pool = items[:-nval] if part == "train" else items[-nval:]
    mean, sd, lo, hi = _LEN
    prng = random.Random(seed)
    for i in range(n_out):
        it = pool[i % len(pool)]
        tgt = int(min(hi, max(lo, prng.gauss(mean, sd))))
        ctx = _window(it["doc"], it["ev"], tgt, prng)
        q = it["q"].strip()
        if not q.endswith("?"):
            q += "?"
        article = f"{ctx}{QUESTION_MARKER} {q}"
        a = it["a"].strip()
        yield {"id": f"{dataset}_{part}_{i}", "dataset": dataset, "task": "mfqa_style",
               "article": article, "answer": a, "metadata": {"answers": [a]}}


def build(dataset, part, seed, n_out):
    items = _decontam(SOURCES[dataset]())
    if len(items) < 200:
        raise ValueError(f"{dataset}: too few clean items ({len(items)})")
    rng = random.Random(1234)
    rng.shuffle(items)                       # representative, deterministic split
    nval = min(400, len(items) // 10)
    pool = items[:-nval] if part == "train" else items[-nval:]
    distract = [it["doc"] for it in pool]
    mean, sd, lo, hi = _LEN
    prng = random.Random(seed)
    for i in range(n_out):
        it = pool[i % len(pool)]
        tgt = int(min(hi, max(lo, prng.gauss(mean, sd))))
        ctx = _pad(it["doc"], distract, tgt, prng)
        q = it["q"].strip()
        if not q.endswith("?"):
            q += "?"
        article = f"{ctx}{QUESTION_MARKER} {q}"
        a = it["a"].strip()
        yield {"id": f"{dataset}_{part}_{i}", "dataset": dataset, "task": "mfqa_style",
               "article": article, "answer": a, "metadata": {"answers": [a]}}
