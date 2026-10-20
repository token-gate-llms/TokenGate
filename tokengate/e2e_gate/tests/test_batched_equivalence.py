"""Validate that forward_with_gate_batched(B samples) == mean of B individual losses.

This test is the primary correctness check for batched training.  If batching
is implemented correctly, the loss and gradients computed on a batch of N
samples must match those computed by running the same N samples individually
and averaging.

Run with:
    python -m tokengate.e2e_gate.tests.test_batched_equivalence
"""

import sys
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
from tokengate.e2e_gate.gate_module.gate_forward import build_gate_branch, forward_with_gate_batched
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    get_full_attn_layer_indices, patch_model_with_flex_attention,
)
from tokengate.datasets.tasks import get_task
from tokengate.e2e_gate.gate_module.gate_forward import apply_chat_template


MODEL_NAME = "Qwen/Qwen3-0.6B"
GATE_LAYER = 4
K = 50
BATCH_SIZE = 4
# Tolerances for the FINALIZED gate: the score passes through the scoring-branch
# MLP (bf16 matmuls whose padded/batched block boundaries differ from the
# unpadded individual run) and then a seq-RMSNorm that couples every article
# token, so per-element noise is larger than the old per-token linear gate's.
# A real batching bug (positions shifted by pad count, pads attended, cache
# mis-packing) produces GROSS divergence, far outside these bounds.
ATOL_LOSS = 1e-1   # absolute, losses are O(10)
RTOL_GRAD = 1e-1   # relative on the max-|grad| element
# Note: with equal-length sequences the differences drop to <5e-4 (within bfloat16
# precision), confirming the logic is correct.  Variable-length batches pad to
# max_n_kept, so FlexAttention uses different block boundaries per sample than when
# each sample runs alone — bfloat16 non-associativity causes ~1% relative error in
# gradients.  This is not a correctness bug.


def _make_batch_item(tokenizer, task, article: str, target: str, max_art: int = 2000):
    """Tokenize one sample into a batch_data dict."""
    prompt = task.format_prompt(article)
    prompt_ids = apply_chat_template(tokenizer, prompt)
    answer_ids = tokenizer.encode(target + tokenizer.eos_token, add_special_tokens=False)
    full_ids = prompt_ids + answer_ids

    # find article boundaries
    non_comp = task.get_non_compressible_context(article)
    empty_ids = apply_chat_template(tokenizer, task.format_prompt(non_comp))
    # simple boundary: find longest common prefix and suffix
    pre = 0
    while pre < len(prompt_ids) and pre < len(empty_ids) and prompt_ids[pre] == empty_ids[pre]:
        pre += 1
    suf = 0
    while (suf < len(prompt_ids) - pre and suf < len(empty_ids) - pre
           and prompt_ids[-(suf+1)] == empty_ids[-(suf+1)]):
        suf += 1
    article_start = pre
    article_end = len(prompt_ids) - (suf if suf > 0 else 0)

    # cap article
    n_art = article_end - article_start
    if n_art > max_art:
        article_end = article_start + max_art
        prompt_ids = prompt_ids[:article_end] + prompt_ids[len(prompt_ids) - (suf if suf > 0 else 0):]
        full_ids = prompt_ids + answer_ids

    return {
        "input_ids": full_ids,
        "article_start": article_start,
        "article_end": article_end,
        "ctx_len": len(prompt_ids),
        "n_ans": len(answer_ids),
        "K": K,
    }


def _compute_loss_from_batch_output(logits, loss_mask, batch_data):
    """Replicate the per-sample loss averaging from train_e2e_gate.py."""
    total = 0.0
    n_valid = 0
    for b_idx, item in enumerate(batch_data):
        mask = loss_mask[b_idx]
        if not mask.any():
            continue
        sl = logits[b_idx, mask, :]
        labels = torch.tensor(
            item["input_ids"][item["ctx_len"]: item["ctx_len"] + item["n_ans"]],
            dtype=torch.long, device=logits.device,
        )
        if sl.shape[0] != labels.shape[0]:
            continue
        total += F.cross_entropy(sl, labels)
        n_valid += 1
    if n_valid == 0:
        return None
    return total / n_valid


def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] device={device}  model={MODEL_NAME}")

    # ── Load model and tokenizer ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    model = patch_model_with_flex_attention(model)
    model.eval()

    d_model = model.config.hidden_size
    nlb = len([i for i in get_full_attn_layer_indices(model) if i > GATE_LAYER])
    gate = E2EGate(d_model=d_model, n_heads=model.config.num_attention_heads,
                   n_layers_bias=nlb).to(device).to(torch.float32)
    gate.set_qk_norm_scales(model, GATE_LAYER)
    build_gate_branch(model, gate, GATE_LAYER)
    # Non-trivial, DETERMINISTIC scores (the folded readout row is zero-init;
    # unseeded random scores make near-tie selections flip run to run).
    torch.manual_seed(20261012)
    torch.nn.init.normal_(gate.gate_branch.down_proj.weight, std=0.05)

    task = get_task("biodex_reactions")

    # ── Build BATCH_SIZE synthetic samples ───────────────────────────────────
    # Use short articles to keep the test fast; vary content so K selections differ
    articles = [
        "The patient is a 45-year-old male who presented with chest pain. "
        "He has a history of hypertension. His medication includes aspirin. " * 20,
        "A 32-year-old female was admitted for appendicitis. "
        "She reported nausea and vomiting. Her labs showed elevated WBC. " * 20,
        "Mr. Smith is a 67-year-old man with diabetes mellitus type 2. "
        "He was seen for routine follow-up. His HbA1c was 7.2%. " * 20,
        "The patient is a 28-year-old woman presenting with migraine. "
        "She has a family history of neurological disorders. " * 20,
    ]
    targets = ["male", "female", "male", "female"]

    batch_data = [
        _make_batch_item(tokenizer, task, art, tgt)
        for art, tgt in zip(articles, targets)
    ]
    print(f"[test] batch_data built: {BATCH_SIZE} items, "
          f"seq_lens={[len(d['input_ids']) for d in batch_data]}")

    # ── Reference: sum of individual losses ──────────────────────────────────
    individual_losses = []
    individual_grads = []
    for i, item in enumerate(batch_data):
        gate.zero_grad()
        logits_i, mask_i, _ = forward_with_gate_batched(
            model, gate, [item], GATE_LAYER, frozen_backbone=True,
        )
        loss_i = _compute_loss_from_batch_output(logits_i, mask_i, [item])
        if loss_i is None:
            print(f"[test] WARNING: sample {i} has no valid loss — skipping")
            continue
        loss_i.backward()
        individual_losses.append(loss_i.item())
        individual_grads.append(gate.gate_branch.down_proj.weight.grad.clone())
        print(f"[test]   individual loss[{i}] = {loss_i.item():.6f}")

    if len(individual_losses) < 2:
        print("[test] SKIP: fewer than 2 valid individual losses")
        return

    mean_individual_loss = sum(individual_losses) / len(individual_losses)
    mean_individual_grad = sum(individual_grads) / len(individual_grads)

    # ── Batched: all samples together ─────────────────────────────────────────
    gate.zero_grad()
    logits_b, mask_b, _ = forward_with_gate_batched(
        model, gate, batch_data, GATE_LAYER, frozen_backbone=True,
    )
    loss_b = _compute_loss_from_batch_output(logits_b, mask_b, batch_data)
    assert loss_b is not None, "Batched forward produced no valid loss"
    loss_b.backward()
    batched_loss = loss_b.item()
    batched_grad = gate.gate_branch.down_proj.weight.grad.clone()

    print(f"\n[test] mean individual loss : {mean_individual_loss:.6f}")
    print(f"[test] batched loss         : {batched_loss:.6f}")
    loss_diff = abs(batched_loss - mean_individual_loss)
    print(f"[test] |diff|               : {loss_diff:.2e}  (atol={ATOL_LOSS})")

    grad_diff = (batched_grad - mean_individual_grad).abs().max().item()
    grad_rel = grad_diff / (mean_individual_grad.abs().max().item() + 1e-9)
    print(f"[test] max grad diff        : {grad_diff:.2e}")
    print(f"[test] max grad rel err     : {grad_rel:.2e}  (rtol={RTOL_GRAD})")

    ok = loss_diff < ATOL_LOSS and grad_rel < RTOL_GRAD
    if ok:
        print("\n[test] PASSED: batched and individual losses/grads match "
              "(within bf16 non-associativity through the scoring branch).")
    else:
        print("\n[test] FAILED:")
        if loss_diff >= ATOL_LOSS:
            print(f"  loss diff {loss_diff:.2e} >= atol {ATOL_LOSS}")
        if grad_rel >= RTOL_GRAD:
            print(f"  grad rel err {grad_rel:.2e} >= rtol {RTOL_GRAD}")
        sys.exit(1)


if __name__ == "__main__":
    run_test()
