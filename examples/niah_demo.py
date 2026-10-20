#!/usr/bin/env python3
"""Needle-in-a-haystack with a released TokenGate checkpoint, at 32k tokens.

Each committed sample is a ~32k-token haystack of Paul Graham essays with one
planted needle ("The best thing to do in {city} is {activity}...") — the
canonical Kamradt NIAH setup. The demo downloads the trained gate for
Qwen/Qwen3-1.7B (gate layer 1; trained on the 16k tier, evaluated here at 32k),
then answers each question at several keep-budgets and reports, per budget:

  * retrieval accuracy (normalized containment of the gold activity phrase),
  * mean prefill time-to-first-token (batch 1) and its speedup vs keeping all,
  * whether the needle's tokens themselves SURVIVED the pruning — the gate
    keeps the one relevant sentence and drops ~32k tokens of essay hay.

    python examples/niah_demo.py                 # 5 haystacks, K = all/2000/500/100
    python examples/niah_demo.py --k 500 --n 2   # one budget, fewer rows

Use --checkpoint to point at a local run dir instead of the Hub, and
--gate-attn flex if the FlashAttention-2 key-bias kernel isn't built.

This file is also the reference use of the public inference API:
load_model → apply_chat_template(state=...) → model.generate.
"""
import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

HUB_REPO = "liana313/tokengate-niah-qwen3-1.7b"
DEFAULT_SAMPLES = os.path.join(_REPO_ROOT, "examples", "data",
                               "niah_32k_samples.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=HUB_REPO,
                    help="HF Hub repo id or local checkpoint dir "
                         f"(default: {HUB_REPO})")
    ap.add_argument("--samples", default=DEFAULT_SAMPLES,
                    help="jsonl of {id, body, question, answer, ...} rows")
    ap.add_argument("--k", type=int, nargs="+", default=[2000, 500, 100],
                    help="keep-budget sweep (a no-pruning arm always runs first)")
    ap.add_argument("--n", type=int, default=None, help="rows to run (default all)")
    ap.add_argument("--gate-attn", default="fa2", choices=["fa2", "flash", "flex"],
                    help="inference kernels; 'flex' needs no custom build")
    args = ap.parse_args()

    from tokengate import apply_chat_template, load_model, split_state_span
    from tokengate.datasets.tasks import get_task
    from tokengate.datasets.tasks.longbench import _normalize
    from transformers import AutoTokenizer

    # base_model + gate_layer come from the checkpoint's own metadata
    model = load_model(args.checkpoint, gate_attn=args.gate_attn)
    tok = AutoTokenizer.from_pretrained(model.base_model)

    task = get_task("niah_kamradt")
    samples = [json.loads(l) for l in open(args.samples)][: args.n]
    # The task's own prompt with the haystack as the {state} placeholder —
    # training-identical wording; the question and instructions are outside
    # the state and never pruned.
    ids = [apply_chat_template(
               tok, task.format_prompt("{state}\n\nQuestion: " + s["question"]),
               state=s["body"])
           for s in samples]
    clean = [split_state_span(seq) for seq in ids]  # (clean_ids, s, e) per row

    def _run(k_label, k):
        accs, ttfts, needle_kept = [], [], []
        for s, seq, (cids, st, en) in zip(samples, ids, clean):
            out = model.generate([seq], tok, max_kept_prefill_tokens=k)
            accs.append(task.evaluate(out.answers[0], s["answer"]))
            ttfts.append(out.prefill_s)
            # fraction of the needle phrase's words that survived the pruning
            kept_state = [cids[p] for p in out.kept_pos[0] if st <= p < en]
            kept_words = set(_normalize(tok.decode(kept_state)).split())
            needle_words = _normalize(s["answer"]).split()
            needle_kept.append(
                sum(w in kept_words for w in needle_words) / len(needle_words))
        return (sum(accs) / len(accs), sum(ttfts) / len(ttfts),
                sum(needle_kept) / len(needle_kept), accs)

    depths = ", ".join(f"{s['depth_actual']:.0%}" for s in samples)
    print(f"\n[niah] {len(samples)} haystacks of ~32k tokens, one needle each "
          f"(depths {depths})")
    arms = [("all", None)] + [(str(k), k) for k in args.k]
    results = {}
    full_ttft = None
    print(f"\n{'kept tokens':>12} {'acc':>6} {'TTFT(ms)':>10} {'speedup':>8} "
          f"{'needle kept':>12}")
    print("-" * 54)
    for label, k in arms:
        acc, ttft, needle_frac, _ = _run(label, k)
        full_ttft = full_ttft or ttft
        results[label] = acc
        print(f"{label:>12} {acc:>6.2f} {ttft * 1e3:>10.1f} "
              f"{full_ttft / ttft:>7.2f}x {needle_frac:>11.0%}")

    sf = next((i for i, s in enumerate(samples) if s["city"] == "San Francisco"), None)
    if sf is not None:
        out = model.generate([ids[sf]], tok,
                             max_kept_prefill_tokens=min(args.k))
        print(f"\n[needle @ K={min(args.k)}] {samples[sf]['question']}")
        print(f"  model:  {out.answers[0].strip()}")
        print(f"  gold :  {samples[sf]['answer']}")
        print(f"  ({out.n_kept[0]} of {len(ids[sf]) - 2} prompt tokens kept "
              f"in the KV cache)")
    return results


if __name__ == "__main__":
    main()
