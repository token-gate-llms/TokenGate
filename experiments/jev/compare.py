"""Compare full JEV prefill with simple pruning and, optionally, a trained gate.

Run from the repository root after installing experiments/jev/requirements.txt.
Reported time includes prompt preparation and one decision on a single GPU;
it is not a paper-style kernel or TTFT benchmark.
"""

import argparse
import json
import time

import torch
from transformers import AutoTokenizer, JevForDecision

from tokengate.experimental.jev import gated_decide, input_pruned_decide, standard_decide


def timed(fn, repeats: int):
    fn()  # warmup
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return result, sorted(timings)[len(timings) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="autotrust/JEV-9B")
    parser.add_argument("--state-file", help="UTF-8 text file; defaults to a repeated synthetic state")
    parser.add_argument("--filler-repetitions", type=int, default=300,
                        help="Synthetic filler sentences on each side of the fact")
    parser.add_argument("--needle-position", choices=("start", "middle", "end"), default="middle")
    parser.add_argument("--prefix-question", action="store_true",
                        help="Repeat the decision question before the state")
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gate-checkpoint", help="Gate trained for this exact JEV model and adapter")
    parser.add_argument("--gate-layer", type=int, default=1)
    parser.add_argument("--output", help="Optional path for machine-readable comparison results")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("JEV comparison requires a CUDA GPU")
    if args.repeats < 1 or args.filler_repetitions < 1 or any(k < 1 for k in args.budgets):
        parser.error("repeats, filler repetitions, and budgets must be positive")

    model = JevForDecision.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if model.decision_config.adapter:
        model.load_adapter(args.model, adapter_kwargs={"subfolder": model.decision_config.adapter})
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.state_file:
        with open(args.state_file, encoding="utf-8") as handle:
            state = handle.read()
    else:
        filler = "The shipping desk checked the inventory and found no issue. "
        fact = "The customer was charged twice and asks for a refund. "
        n = args.filler_repetitions
        state = {
            "start": fact + filler * (2 * n),
            "middle": filler * n + fact + filler * n,
            "end": filler * (2 * n) + fact,
        }[args.needle_position]
    question_text = "Does the customer ask for a refund?"
    if args.prefix_question:
        state = f"Decision question: {question_text}\n\n" + state
    request = {"state": state, "questions": {
        "refund": {"type": "noul", "instructions": question_text}
    }}

    rows = []
    result, elapsed = timed(lambda: standard_decide(model, tokenizer, request), args.repeats)
    rows.append(("full", result, elapsed))
    for k in args.budgets:
        for method in ("prefix", "middle", "suffix", "random"):
            result, elapsed = timed(
                lambda k=k, method=method: input_pruned_decide(
                    model, tokenizer, request, token_budget=k, method=method, seed=0
                ), args.repeats,
            )
            rows.append((f"{method}-{k}", result, elapsed))

    if args.gate_checkpoint:
        from tokengate.e2e_gate.gate_module.e2e_gate_module import load_gate
        from tokengate.e2e_gate.attn.qwen3_flex_attention import patch_model_with_flex_attention

        gate, metadata = load_gate(args.gate_checkpoint)
        if metadata.get("decision_model") != args.model:
            raise ValueError("Gate checkpoint metadata must name this exact decision_model")
        if metadata.get("gate_layer") != args.gate_layer:
            raise ValueError("Gate checkpoint and comparison must use the same gate layer")
        if metadata.get("prefix_question", False) != args.prefix_question:
            raise ValueError("Gate checkpoint and comparison must use the same question-prefix setting")
        if gate.d_model != model.config.hidden_size:
            raise ValueError("Gate hidden width does not match the JEV model")
        gate = gate.to(model.device).eval()
        patch_model_with_flex_attention(model)
        flex_full, flex_elapsed = timed(
            lambda: standard_decide(model, tokenizer, request), args.repeats,
        )
        rows.append(("full-flex", flex_full, flex_elapsed))
        for k in args.budgets:
            result, elapsed = timed(
                lambda k=k: gated_decide(
                    model, gate, tokenizer, request,
                    gate_layer=args.gate_layer, token_budget=k,
                ), args.repeats,
            )
            rows.append((f"gate-{k}", result, elapsed))

    print("method\tstate kept\tmedian end-to-end ms\tP(refund)")
    for name, result, elapsed in rows:
        probability = result.answers["refund"]["noul"]
        print(f"{name}\t{result.kept_state_tokens}/{result.state_tokens}"
              f"\t{elapsed:.1f}\t{probability:.4f}")
    payload = {name: {"ms": elapsed, "answer": result.answers,
                      "state_kept": result.kept_state_tokens,
                      "state_tokens": result.state_tokens}
               for name, result, elapsed in rows}
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
