#!/usr/bin/env python3
"""Generate the Needle-in-a-Haystack dataset (Kamradt, 2023) with a TRAIN split.

Canonical form, faithfully reproduced:
  * Haystack: Paul Graham essays (the corpus Kamradt's LLMTest_NeedleInAHaystack
    ships), whole essays in per-sample shuffled order, trimmed at a sentence
    boundary to the per-sample token target.
  * Needle: "The best thing to do in {city} is {activity}." inserted at the
    sentence boundary nearest depth*total_tokens (Kamradt's insertion rule).
  * Question: "What is the best thing to do in {city}?"  Answer: the activity.
  * The literal canonical needle (San Francisco / "eat a sandwich and sit in
    Dolores Park on a sunny day") is reserved for the TEST split.

Train/test hygiene (this is the entire decontam story — all facts are generated):
  * essays hash-split 80/20; train haystacks use only train essays, test only test essays
  * cities hash-split; test uses 20 cities from the held-out pool (SF forced in),
    train never uses a test city → the query key never leaks
  * test activity strings are reserved; train sampling rejects them

Splits: --n-train rows at depth ~U[0,1], body target ~U[14000,16350] tokens;
test = 20 cities x 10 depths (linspace 0..1) at body target 16000±150 — the
single 16k tier, with train mass ABOVE the test mean (train must bracket the
eval lengths).

Usage:
  python datasets/niah_kamradt/generate_niah_kamradt.py \
      --output-file datasets/niah_kamradt/niah_kamradt_all.jsonl \
      --n-train 12000 --seed 0
"""
import argparse
import json
import random

import numpy as np
from transformers import AutoTokenizer

# Sibling module — run this script by path (python datasets/niah_kamradt/...).
from niah_common import (
    ACT_CONDS, ACT_CORES, ACT_VENUES, CANON_ACTIVITY, CANON_CITY, CITIES,
    EssayPool, MODEL_TOK, insert_at_depth, is_test_key, load_pg_essays,
)

TRAIN_LO, TRAIN_HI = 14000, 16350   # body-token target (needle included; < 16384 window);
                                    # floor at 14k so train mass clusters at the 16k tier
                                    # (train must bracket the eval lengths)
TEST_TARGET, TEST_JIT = 16000, 150
N_DEPTHS = 10


def _needle(city, activity):
    return f"The best thing to do in {city} is {activity}."


def _question(city):
    return f"What is the best thing to do in {city}?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test-cities", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-target", type=int, default=TEST_TARGET,
                    help="test-tier body-token target.  Above TRAIN_HI the "
                         "train-window cap is skipped (longer-than-training "
                         "tiers, e.g. 32000 for a 2x length-generalization "
                         "benchmark on the same checkpoints; Qwen3-1.7B "
                         "window is 40960).  Default = the 16k tier.")
    ap.add_argument("--test-jit", type=int, default=TEST_JIT)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_TOK, use_fast=True)
    essays = load_pg_essays()
    tr_essays = [e for e in essays if not is_test_key(e["title"])]
    te_essays = [e for e in essays if is_test_key(e["title"])]
    print(f"[essays] train={len(tr_essays)} test={len(te_essays)}")
    tr_pool = EssayPool(tr_essays, tok)
    te_pool = EssayPool(te_essays, tok)

    cities = sorted(set(CITIES))
    if len(cities) != len(CITIES):
        raise ValueError("CITIES has duplicates")
    tr_cities = [c for c in cities if not is_test_key(c) and c != CANON_CITY]
    te_city_pool = [c for c in cities if is_test_key(c) and c != CANON_CITY]
    rng_te = random.Random(args.seed + 1)
    te_cities = [CANON_CITY] + rng_te.sample(te_city_pool, args.n_test_cities - 1)
    assert not (set(tr_cities) & set(te_cities))

    triples = [(c, v, d) for c in ACT_CORES for v in ACT_VENUES for d in ACT_CONDS]

    def _activity(rng):
        c, v, d = triples[rng.randrange(len(triples))]
        return f"{c} {v} {d}"

    # test rows: 20 cities x 10 depths, held-out activities (canonical for SF)
    te_act = {CANON_CITY: CANON_ACTIVITY}
    for city in te_cities[1:]:
        te_act[city] = _activity(rng_te)
    reserved = set(te_act.values())
    depths = np.linspace(0.0, 1.0, N_DEPTHS)
    te_rows = []
    for city in te_cities:
        for dep in depths:
            ndl = _needle(city, te_act[city])
            ntk = len(tok(" " + ndl, add_special_tokens=False)["input_ids"])
            tgt = args.test_target + rng_te.randint(-args.test_jit, args.test_jit)
            if args.test_target <= TRAIN_HI:
                # the 16k tier keeps its original cap (byte-identical output);
                # longer tiers deliberately exceed the training window
                tgt = min(tgt, TRAIN_HI)
            tgt -= ntk
            s, n = te_pool.assemble(rng_te, tgt)
            s, n, dact = insert_at_depth(s, n, ndl, ntk, float(dep))
            te_rows.append({
                "split": "test", "body": " ".join(s), "question": _question(city),
                "answer": te_act[city], "city": city,
                "depth_req": round(float(dep), 3), "depth_actual": round(dact, 4),
                "n_body_tokens": sum(n),
            })
    rng_te.shuffle(te_rows)
    for i, r in enumerate(te_rows):
        r["id"] = f"{i:04d}_niah_test"

    rng = random.Random(args.seed)
    tr_rows = []
    for i in range(args.n_train):
        city = tr_cities[rng.randrange(len(tr_cities))]
        act = _activity(rng)
        while act in reserved:
            act = _activity(rng)
        ndl = _needle(city, act)
        ntk = len(tok(" " + ndl, add_special_tokens=False)["input_ids"])
        dep = rng.random()
        tgt = rng.randint(TRAIN_LO, TRAIN_HI) - ntk
        s, n = tr_pool.assemble(rng, tgt)
        s, n, dact = insert_at_depth(s, n, ndl, ntk, dep)
        tr_rows.append({
            "id": f"niah_train_{i:05d}", "split": "train", "body": " ".join(s),
            "question": _question(city), "answer": act, "city": city,
            "depth_req": round(dep, 4), "depth_actual": round(dact, 4),
            "n_body_tokens": sum(n),
        })
        if (i + 1) % 2000 == 0:
            print(f"[train] {i+1}/{args.n_train}")

    # hygiene assertions
    for rows, pool in ((tr_rows, set(tr_cities)), (te_rows, set(te_cities))):
        assert all(r["city"] in pool for r in rows)
        assert all(_needle(r["city"], r["answer"]) in r["body"] for r in rows)
    assert not ({r["answer"] for r in tr_rows} & reserved)

    with open(args.output_file, "w") as f:
        for r in tr_rows + te_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for nm, rows in (("train", tr_rows), ("test", te_rows)):
        if not rows:               # test-only tiers run with --n-train 0
            print(f"[{nm}] n=0")
            continue
        arr = np.array([r["n_body_tokens"] for r in rows])
        print(f"[{nm}] n={len(rows)} body tokens mean={arr.mean():.0f} "
              f"min={arr.min()} p25={np.percentile(arr,25):.0f} p75={np.percentile(arr,75):.0f} max={arr.max()}")
    print(f"[done] wrote {len(tr_rows)+len(te_rows)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()
