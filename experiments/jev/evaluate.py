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
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--split", choices=("validation", "test_set_30k"), default="test_set_30k")
    parser.add_argument("--kind", choices=("all", "noul", "choice", "score"), default="all")
    parser.add_argument("--min-target-margin", type=float, default=0.0,
                        help="Skip near-tied corpus labels when reporting top-choice agreement")
    parser.add_argument("--prefix-question", action="store_true",
                        help="Repeat the decision question before the prunable state")
    parser.add_argument("--random-target-position", action="store_true",
                        help="Place the marked case at random depths among background excerpts")
    parser.add_argument("--background-excerpts", type=int, default=24)
    parser.add_argument("--zero-gate-bias", action="store_true",
                        help="Ablation: keep gate-selected tokens but disable its Phase-2 attention bias")
    parser.add_argument("--output", default="jev_evaluation.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("evaluation requires CUDA")
    if args.rows < 1 or not 0 <= args.min_target_margin <= 1:
        parser.error("rows must be positive and target margin must be between 0 and 1")

    model = JevForDecision.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if model.decision_config.adapter:
        model.load_adapter(args.model, adapter_kwargs={"subfolder": model.decision_config.adapter})
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    gate, metadata = load_gate(args.gate_checkpoint)
    if metadata.get("decision_model") != args.model or gate.d_model != model.config.hidden_size:
        raise ValueError("Gate checkpoint does not match this JEV model")
    if metadata.get("gate_layer") != args.gate_layer:
        raise ValueError("Gate checkpoint and evaluation must use the same gate layer")
    if metadata.get("prefix_question", False) != args.prefix_question:
        raise ValueError("Gate checkpoint and evaluation must use the same question-prefix setting")
    if metadata.get("random_target_position", False) != args.random_target_position:
        raise ValueError("Gate checkpoint and evaluation must use the same target-position setting")
    gate = gate.to(model.device).eval()
    gate.zero_gate_bias = args.zero_gate_bias
    patch_model_with_flex_attention(model)
    training = load_dataset("SargeDev/jev-distill-corpus-v3", split="train")
    held_out = load_dataset("SargeDev/jev-distill-corpus-v3", split=args.split).shuffle(seed=args.eval_seed)
    rng = random.Random(args.eval_seed)
    metrics = {"full": []}
    for budget in args.budgets:
        for method in ("prefix", "middle", "suffix", "random", "gate"):
            metrics[f"{method}-{budget}"] = []

    selected = []
    for row in held_out:
        if args.kind != "all" and row["kind"] != args.kind:
            continue
        sorted_target = sorted(row["target"], reverse=True)
        if len(sorted_target) < 2 or sorted_target[0] - sorted_target[1] < args.min_target_margin:
            continue
        selected.append(row)
        if len(selected) == args.rows:
            break
    if not selected:
        raise ValueError("No held-out rows match the requested kind and target margin")

    for i, row in enumerate(selected):
        background = [training[rng.randrange(len(training))]["state"]
                      for _ in range(args.background_excerpts)]
        target_index = rng.randrange(len(background) + 1) if args.random_target_position else None
        request = request_from_row(row, background, prefix_question=args.prefix_question,
                                   target_index=target_index)
        target = torch.tensor(row["target"], dtype=torch.float32)
        target = target / target.sum()
        full, ms = timed(lambda: standard_decide(model, tokenizer, request))
        teacher = full.scores["decision"].float()
        target = target.to(teacher.device)
        if target.numel() != teacher.numel():
            raise ValueError(f"Target has {target.numel()} options, model has {teacher.numel()}")
        metrics["full"].append({
            "ms": ms, "state_tokens": full.state_tokens,
            "target_KL": float(torch.nn.functional.kl_div(
                teacher.log_softmax(-1), target, reduction="sum")),
            "target_agree": int(teacher.argmax() == target.argmax()),
        })
        for budget in args.budgets:
            for method in ("prefix", "middle", "suffix", "random", "gate"):
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
                    "target_KL": float(torch.nn.functional.kl_div(
                        student.log_softmax(-1), target, reduction="sum")),
                    "target_agree": int(student.argmax() == target.argmax()),
                })
        if (i + 1) % 5 == 0:
            print(f"evaluated {i + 1}/{len(selected)}", flush=True)

    summary = {}
    for name, entries in metrics.items():
        summary[name] = {"median_ms": statistics.median(e["ms"] for e in entries)}
        summary[name]["mean_target_KL"] = statistics.mean(e["target_KL"] for e in entries)
        summary[name]["top1_target"] = statistics.mean(e["target_agree"] for e in entries)
        if name == "full":
            summary[name]["median_state_tokens"] = statistics.median(e["state_tokens"] for e in entries)
        else:
            summary[name]["mean_KL"] = statistics.mean(e["KL"] for e in entries)
            summary[name]["top1_agreement"] = statistics.mean(e["agree"] for e in entries)
    summary["_metadata"] = {
        "model": args.model, "gate_checkpoint": args.gate_checkpoint,
        "gate_layer": args.gate_layer, "split": args.split,
        "eval_seed": args.eval_seed, "rows": len(selected),
        "kind": args.kind, "min_target_margin": args.min_target_margin,
        "background_excerpts": args.background_excerpts,
        "prefix_question": args.prefix_question,
        "random_target_position": args.random_target_position,
        "zero_gate_bias": args.zero_gate_bias,
        "budgets": args.budgets,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"split={args.split} seed={args.eval_seed} kind={args.kind} min_target_margin={args.min_target_margin} "
          f"rows={len(selected)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
