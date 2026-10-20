#!/usr/bin/env python3
"""Run a released TokenGate checkpoint on BioDEX-Reactions validation samples.

Downloads the trained gate for Qwen/Qwen3-1.7B (gate layer 1, snapshot-SWA —
the benchmark-selected deployable checkpoint) from the Hugging Face Hub, prunes
each article's KV cache down to K kept tokens, decodes the adverse-drug-reaction
terms, and scores them against the gold answer with the task's exact-match set
F1 — so you can see real predictions, kept-token counts and accuracy in one
command:

    python examples/biodexr_demo.py                # full demo (6 samples, K=2000)
    python examples/biodexr_demo.py --k 500 --n 2  # harsher pruning, fewer rows

Use --checkpoint to point at a local run directory instead of the Hub (any dir
holding a gate checkpoint + its name-paired adapter), and --gate-attn flex if
the FlashAttention-2 key-bias kernel isn't built.

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

HUB_REPO = "liana313/tokengate-biodexr-qwen3-1.7b"
DEFAULT_SAMPLES = os.path.join(_REPO_ROOT, "examples", "data",
                               "biodexr_val_samples.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=HUB_REPO,
                    help="HF Hub repo id or local checkpoint dir "
                         f"(default: {HUB_REPO})")
    ap.add_argument("--samples", default=DEFAULT_SAMPLES,
                    help="jsonl of {id, article, target} rows")
    ap.add_argument("--k", type=int, default=2000,
                    help="max state tokens kept in the KV cache")
    ap.add_argument("--n", type=int, default=None, help="rows to run (default all)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gate-attn", default="fa2", choices=["fa2", "flash", "flex"],
                    help="inference kernels; 'flex' needs no custom build")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from tokengate import apply_chat_template, load_model
    from tokengate.datasets.tasks import get_task

    # base_model + gate_layer come from the checkpoint's own metadata
    model = load_model(args.checkpoint, gate_attn=args.gate_attn)
    tok = AutoTokenizer.from_pretrained(model.base_model)

    # The task's own prompt with the article slot as the {state} placeholder —
    # training-identical wording, and everything outside the state is never
    # pruned.
    task = get_task("biodex_reactions")
    prompt_template = task.format_prompt("{state}")

    samples = [json.loads(l) for l in open(args.samples)][: args.n]
    ids = [apply_chat_template(tok, prompt_template, state=s["article"])
           for s in samples]

    out = model.generate(ids, tok, max_new_tokens=args.max_new_tokens,
                         max_kept_prefill_tokens=args.k)

    f1s = []
    print(f"\n{'='*76}\nBioDEX-Reactions @ K={args.k}  "
          f"(prefill {out.prefill_s:.2f}s + decode {out.decode_s:.2f}s "
          f"for {len(samples)} articles)\n{'='*76}")
    for s, seq, ans, kept, n_state in zip(samples, ids, out.answers,
                                          out.n_kept, out.n_state):
        f1 = task.evaluate(ans, s["target"])
        f1s.append(f1)
        print(f"\n[{s['id']}]  kept {kept}/{len(seq) - 2} prompt tokens "
              f"(state: {min(args.k, n_state)}/{n_state})  F1={f1:.3f}")
        print(f"  predicted: {ans.strip()[:160]}")
        print(f"  gold     : {s['target'][:160]}")

    mean_f1 = sum(f1s) / max(len(f1s), 1)
    print(f"\n{'='*76}\nmean exact-match set F1 over {len(f1s)} samples: "
          f"{mean_f1:.3f}\n{'='*76}")
    return mean_f1


if __name__ == "__main__":
    main()
