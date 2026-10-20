"""build_wiki_dict.py — prerequisite for the full-Wikipedia multi-hop reconstructions
(mfqa_{musique,hotpot,2wiki}_wiki).

Produces, under $WIKI_SP:
  • wiki_questions.pkl : {src -> [ {q, a, gold:[(title,snippet)], dist:[(title,snippet)]} ]}
        for each src in $SOURCES (default musique,hotpot,2wiki).  gold = supporting paras.
  • wiki_dict.pkl      : {title -> full Wikipedia body}  for the union of all referenced
        titles, fetched by streaming wikimedia/wikipedia 20231101.en (exact title match).

The render step (wiki_multihop_reconstruct.build_wiki_multihop) then assembles
~10 FULL-article `Passage N:` rows/question (gold de-centered) — matching each LongBench
multi-hop eval's passage granularity (~10 long articles), which the short-snippet twins
never did.  Decontam is applied at render time (question-level + gold 13-gram).

Usage:
    WIKI_SP=<dir> [SOURCES=musique,hotpot,2wiki] [SRC_CAP=25000] \
        python -m tokengate.datasets.build_wiki_dict
"""
import os
import pickle
import sys

from datasets import load_dataset

BODY_CAP = int(os.environ.get("WIKI_BODY_CAP", "20000"))   # chars/article (eval passage p90 ~16k)
WIKI_CFG = os.environ.get("WIKI_CFG", "20231101.en")
SRC_CAP = int(os.environ.get("SRC_CAP", "25000"))          # max questions per multi-hop source


def _pairs_from_context(titles, sents, sup):
    """(title, ' '.join(sentences)) tuples split into gold/dist by supporting-title set."""
    gold, dist = [], []
    for t, s in zip(titles, sents):
        body = " ".join(x for x in s if isinstance(x, str)).strip()
        if not t or not body:
            continue
        (gold if t in sup else dist).append((t.strip(), body))
    return gold, dist


def extract_musique():
    ms = load_dataset("dgslibisey/MuSiQue", split="train")
    out = []
    for r in ms:
        if not r.get("answerable", True) or not r.get("answer"):
            continue
        gold, dist = [], []
        for p in r["paragraphs"]:
            t = (p["title"] or "").strip(); b = (p["paragraph_text"] or "").strip()
            if t and b:
                (gold if p["is_supporting"] else dist).append((t, b))
        if gold:
            out.append({"q": r["question"].strip(), "a": r["answer"].strip(), "gold": gold, "dist": dist})
    return out


def extract_hotpot():
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", trust_remote_code=True)
    out = []
    for r in ds:
        if not r.get("answer"):
            continue
        sup = set(r["supporting_facts"]["title"])
        gold, dist = _pairs_from_context(r["context"]["title"], r["context"]["sentences"], sup)
        if gold:
            out.append({"q": r["question"].strip(), "a": r["answer"].strip(), "gold": gold, "dist": dist})
        if len(out) >= SRC_CAP:
            break
    return out


def extract_2wiki():
    ds = load_dataset("framolfese/2WikiMultihopQA", split="train", trust_remote_code=True)
    out = []
    for r in ds:
        ans = (r.get("answer") or "").strip()
        if not ans:
            continue
        sup = set(r["supporting_facts"]["title"])
        gold, dist = _pairs_from_context(r["context"]["title"], r["context"]["sentences"], sup)
        if gold:
            out.append({"q": r["question"].strip(), "a": ans, "gold": gold, "dist": dist})
        if len(out) >= SRC_CAP:
            break
    return out


_EXTRACT = {"musique": extract_musique, "hotpot": extract_hotpot, "2wiki": extract_2wiki}


def fetch_wiki_bodies(titles):
    need = set(titles)
    wiki = {}
    wk = load_dataset("wikimedia/wikipedia", WIKI_CFG, split="train", streaming=True)
    seen = 0
    for w in wk:
        seen += 1
        if seen % 500_000 == 0:
            print(f"[build_wiki_dict] scanned {seen:,} wiki articles, matched {len(wiki):,}/{len(need):,}", flush=True)
        t = w["title"]
        if t in need and t not in wiki:
            wiki[t] = w["text"][:BODY_CAP]
            if len(wiki) == len(need):
                break
    print(f"[build_wiki_dict] DONE scan: {seen:,} articles; matched {len(wiki):,}/{len(need):,} "
          f"({100 * len(wiki) / max(1, len(need)):.1f}% of needed titles)", flush=True)
    return wiki


def main():
    sp = os.environ.get("WIKI_SP")
    if not sp:
        sys.exit("set WIKI_SP=<output dir>")
    os.makedirs(sp, exist_ok=True)
    sources = [s.strip() for s in os.environ.get("SOURCES", "musique,hotpot,2wiki").split(",") if s.strip()]
    questions, titles = {}, set()
    for src in sources:
        qs = _EXTRACT[src]()
        questions[src] = qs
        for it in qs:
            for (t, _) in it["gold"] + it["dist"]:
                titles.add(t)
        print(f"[build_wiki_dict] {src}: {len(qs)} questions", flush=True)
    with open(os.path.join(sp, "wiki_questions.pkl"), "wb") as f:
        pickle.dump(questions, f)
    print(f"[build_wiki_dict] wrote wiki_questions.pkl ({sources}); {len(titles)} unique titles", flush=True)
    wiki = fetch_wiki_bodies(titles)
    with open(os.path.join(sp, "wiki_dict.pkl"), "wb") as f:
        pickle.dump(wiki, f)
    print(f"[build_wiki_dict] wrote wiki_dict.pkl ({len(wiki)} titles)", flush=True)


if __name__ == "__main__":
    main()
