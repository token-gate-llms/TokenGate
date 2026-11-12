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


def request_from_row(row, background):
    kind = row["kind"]
    question = {"type": kind, "instructions": f"For the final case only: {row['question']}"}
    if kind == "choice":
        question["criteria"] = {option: option for option in row["options"]}
    elif kind == "score":
        question["criteria"] = row["options"]
    elif kind != "noul":
        raise ValueError(f"Unsupported decision kind: {kind}")
    state = "Background excerpts (unrelated to the final case):\n" + "\n".join(background)
    state += "\nFinal case to decide:\n" + row["state"]
    return {"state": state, "questions": {"decision": question}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="autotrust/JEV-9B")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--background-excerpts", type=int, default=24)
    parser.add_argument("--gate-layer", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", default="jev_gate.pt")
    parser.add_argument("--eval-rows", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1 or args.budget < 1 or args.background_excerpts < 1:
        parser.error("steps, budget, and background-excerpts must be positive")
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
        background = [rows[rng.randrange(len(rows))]["state"] for _ in range(args.background_excerpts)]
        request = request_from_row(row, background)
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
    })
    print(f"Saved exploratory gate to {args.output}")

    if args.eval_rows:
        validation = load_dataset("SargeDev/jev-distill-corpus-v3", split="validation")
        gate.eval()
        divergences = []
        agreement = []
        for row in validation.select(range(min(args.eval_rows, len(validation)))):
            background = [rows[rng.randrange(len(rows))]["state"] for _ in range(args.background_excerpts)]
            request = request_from_row(row, background)
            full = standard_decide(model, tokenizer, request).scores["decision"].float()
            pruned = gated_decide(
                model, gate, tokenizer, request,
                gate_layer=args.gate_layer, token_budget=args.budget,
            ).scores["decision"].float()
            divergences.append(float(torch.nn.functional.kl_div(
                pruned.log_softmax(-1), full.softmax(-1), reduction="sum"
            )))
            agreement.append(int(pruned.argmax() == full.argmax()))
        print(f"validation n={len(divergences)} mean_KL={sum(divergences)/len(divergences):.5f} "
              f"top1_agreement={sum(agreement)/len(agreement):.3f}")


if __name__ == "__main__":
    main()
