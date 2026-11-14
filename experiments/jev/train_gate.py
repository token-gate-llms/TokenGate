"""Small JEV gate-distillation experiment; run on one free A100.

The public JEV corpus has mostly short states. This script adds unrelated
background excerpts and marks the case to decide, creating a longer state for
an exploratory gate run. It is not a benchmark reproduction.
"""

import argparse
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, JevForDecision

from tokengate.e2e_gate.gate_module.e2e_gate_module import save_gate
from tokengate.experimental.jev import (
    create_jev_gate, gated_decide, standard_decide, train_jev_gate_step,
)


def request_from_row(row, background, *, prefix_question=False, target_index=None):
    kind = row["kind"]
    qualifier = "final case" if target_index is None else "case marked TARGET CASE"
    question = {"type": kind, "instructions": f"For the {qualifier} only: {row['question']}"}
    if kind == "choice":
        question["criteria"] = {option: option for option in row["options"]}
    elif kind == "score":
        question["criteria"] = row["options"]
    elif kind != "noul":
        raise ValueError(f"Unsupported decision kind: {kind}")
    if target_index is None:
        state = "Background excerpts (unrelated to the final case):\n" + "\n".join(background)
        state += "\nFinal case to decide:\n" + row["state"]
    else:
        if not 0 <= target_index <= len(background):
            raise ValueError("target_index must be within the background excerpts")
        cases = [f"Background excerpt {i + 1}:\n{text}" for i, text in enumerate(background)]
        cases.insert(target_index, "TARGET CASE:\n" + row["state"])
        state = "Cases:\n" + "\n\n".join(cases)
    if prefix_question:
        state = f"Decision question: {question['instructions']}\n\n" + state
    return {"state": state, "questions": {"decision": question}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="autotrust/JEV-9B")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--background-excerpts", type=int, default=24)
    parser.add_argument("--min-background-excerpts", type=int,
                        help="Sample a background count between this and --background-excerpts")
    parser.add_argument("--gate-layer", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", default="jev_gate.pt")
    parser.add_argument("--eval-rows", type=int, default=20)
    parser.add_argument("--prefix-question", action="store_true",
                        help="Repeat the decision question before the prunable state")
    parser.add_argument("--random-target-position", action="store_true",
                        help="Place the marked case at random depths among background excerpts")
    parser.add_argument("--kind", choices=("all", "noul", "choice", "score"), default="all")
    parser.add_argument("--min-target-margin", type=float, default=0.0,
                        help="Train on examples with a clear top corpus label")
    args = parser.parse_args()
    min_background = (args.background_excerpts if args.min_background_excerpts is None
                      else args.min_background_excerpts)
    if args.steps < 1 or args.budget < 1 or args.background_excerpts < 1 \
            or not 1 <= min_background <= args.background_excerpts \
            or not 0 <= args.min_target_margin <= 1:
        parser.error("steps, budget, and background counts must be positive; min background must not exceed max; margin must be in [0, 1]")
    if not torch.cuda.is_available():
        parser.error("gate training requires CUDA")

    model = JevForDecision.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if model.decision_config.adapter:
        model.load_adapter(args.model, adapter_kwargs={"subfolder": model.decision_config.adapter})
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    gate = create_jev_gate(model, gate_layer=args.gate_layer)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.learning_rate)
    rows = load_dataset("SargeDev/jev-distill-corpus-v3", split="train")
    rng = random.Random(0)
    completed = 0
    attempts = 0
    while completed < args.steps:
        attempts += 1
        if attempts > args.steps * 20:
            raise RuntimeError("Too few requests longer than the chosen token budget")
        row = rows[rng.randrange(len(rows))]
        if args.kind != "all" and row["kind"] != args.kind:
            continue
        target = sorted(row["target"], reverse=True)
        if len(target) < 2 or target[0] - target[1] < args.min_target_margin:
            continue
        background_count = rng.randint(min_background, args.background_excerpts)
        background = [rows[rng.randrange(len(rows))]["state"] for _ in range(background_count)]
        target_index = rng.randrange(len(background) + 1) if args.random_target_position else None
        request = request_from_row(row, background, prefix_question=args.prefix_question,
                                   target_index=target_index)
        try:
            loss = train_jev_gate_step(
                model, gate, tokenizer, request, optimizer,
                gate_layer=args.gate_layer, token_budget=args.budget,
            )
        except ValueError as error:
            if "must prune some state tokens" in str(error):
                continue
            raise
        completed += 1
        if completed == 1 or completed % 10 == 0:
            print(f"step={completed} KL={loss:.5f}", flush=True)
    save_gate(gate, args.output, metadata={
        "decision_model": args.model,
        "gate_layer": args.gate_layer,
        "token_budget": args.budget,
        "steps": completed,
        "training_corpus": "SargeDev/jev-distill-corpus-v3/train",
        "prefix_question": args.prefix_question,
        "random_target_position": args.random_target_position,
        "training_kind": args.kind,
        "min_target_margin": args.min_target_margin,
        "background_excerpts": args.background_excerpts,
        "min_background_excerpts": min_background,
    })
    print(f"Saved exploratory gate to {args.output}")

    if args.eval_rows:
        validation = load_dataset("SargeDev/jev-distill-corpus-v3", split="validation").shuffle(seed=1)
        gate.eval()
        divergences = []
        agreement = []
        for row in validation:
            if args.kind != "all" and row["kind"] != args.kind:
                continue
            target = sorted(row["target"], reverse=True)
            if len(target) < 2 or target[0] - target[1] < args.min_target_margin:
                continue
            background = [rows[rng.randrange(len(rows))]["state"] for _ in range(args.background_excerpts)]
            target_index = rng.randrange(len(background) + 1) if args.random_target_position else None
            request = request_from_row(row, background, prefix_question=args.prefix_question,
                                       target_index=target_index)
            full = standard_decide(model, tokenizer, request).scores["decision"].float()
            pruned = gated_decide(
                model, gate, tokenizer, request,
                gate_layer=args.gate_layer, token_budget=args.budget,
            ).scores["decision"].float()
            divergences.append(float(torch.nn.functional.kl_div(
                pruned.log_softmax(-1), full.softmax(-1), reduction="sum"
            )))
            agreement.append(int(pruned.argmax() == full.argmax()))
            if len(divergences) == args.eval_rows:
                break
        if not divergences:
            raise ValueError("No validation rows match the requested kind and target margin")
        print(f"validation n={len(divergences)} mean_KL={sum(divergences)/len(divergences):.5f} "
              f"top1_agreement={sum(agreement)/len(agreement):.3f}")


if __name__ == "__main__":
    main()
