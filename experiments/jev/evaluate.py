"""Held-out JEV decision agreement for full, input-pruned, and gated prefill."""

import argparse
import json
import random
import statistics
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, JevForDecision

from experiments.jev.train_gate import request_from_row
from tokengate.e2e_gate.attn.qwen3_flex_attention import patch_model_with_flex_attention
from tokengate.e2e_gate.gate_module.e2e_gate_module import load_gate
from tokengate.experimental.jev import gated_decide, input_pruned_decide, standard_decide


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="autotrust/JEV-9B")
    parser.add_argument("--gate-checkpoint", required=True)
    parser.add_argument("--gate-layer", type=int, default=1)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--background-excerpts", type=int, default=24)
    parser.add_argument("--output", default="jev_evaluation.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("evaluation requires CUDA")

    model = JevForDecision.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if model.decision_config.adapter:
        model.load_adapter(args.model, adapter_kwargs={"subfolder": model.decision_config.adapter})
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    gate, metadata = load_gate(args.gate_checkpoint)
    if metadata.get("decision_model") != args.model or gate.d_model != model.config.hidden_size:
        raise ValueError("Gate checkpoint does not match this JEV model")
    gate = gate.to(model.device).eval()
    patch_model_with_flex_attention(model)
    training = load_dataset("SargeDev/jev-distill-corpus-v3", split="train")
    held_out = load_dataset("SargeDev/jev-distill-corpus-v3", split="validation")
    rng = random.Random(1234)
    metrics = {"full": []}
    for budget in args.budgets:
        for method in ("prefix", "random", "gate"):
            metrics[f"{method}-{budget}"] = []

    for i, row in enumerate(held_out.select(range(min(args.rows, len(held_out))))):
        background = [training[rng.randrange(len(training))]["state"]
                      for _ in range(args.background_excerpts)]
        request = request_from_row(row, background)
        full, ms = timed(lambda: standard_decide(model, tokenizer, request))
        teacher = full.scores["decision"].float()
        metrics["full"].append({"ms": ms, "state_tokens": full.state_tokens})
        for budget in args.budgets:
            for method in ("prefix", "random", "gate"):
                if method == "gate":
                    fn = lambda: gated_decide(
                        model, gate, tokenizer, request,
                        gate_layer=args.gate_layer, token_budget=budget,
                    )
                else:
                    fn = lambda: input_pruned_decide(
                        model, tokenizer, request, token_budget=budget,
                        method=method, seed=0,
                    )
                result, ms = timed(fn)
                student = result.scores["decision"].float()
                kl = torch.nn.functional.kl_div(
                    student.log_softmax(-1), teacher.softmax(-1), reduction="sum"
                )
                metrics[f"{method}-{budget}"].append({
                    "ms": ms, "KL": float(kl),
                    "agree": int(student.argmax() == teacher.argmax()),
                })
        if (i + 1) % 5 == 0:
            print(f"evaluated {i + 1}/{min(args.rows, len(held_out))}", flush=True)

    summary = {}
    for name, entries in metrics.items():
        summary[name] = {"median_ms": statistics.median(e["ms"] for e in entries)}
        if name == "full":
            summary[name]["median_state_tokens"] = statistics.median(e["state_tokens"] for e in entries)
        else:
            summary[name]["mean_KL"] = statistics.mean(e["KL"] for e in entries)
            summary[name]["top1_agreement"] = statistics.mean(e["agree"] for e in entries)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
