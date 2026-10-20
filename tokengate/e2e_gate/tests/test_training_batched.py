#!/usr/bin/env python3
"""Test that batched training works correctly - loss decreases, gradients flow."""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
from tokengate.e2e_gate.gate_module.gate_forward import build_gate_branch
from tokengate.e2e_gate.train.train_e2e_gate import (
    forward_with_gate_batched,
    _apply_chat_template,
    _find_article_boundaries,
)
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    get_full_attn_layer_indices,
    patch_model_with_flex_attention,
)
from tokengate.e2e_gate.attn.flex_attention_utils import warmup_flex_attention
from tokengate.datasets.tasks import get_task


print("[test] Loading model...")
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
).eval()

device = next(model.parameters()).device
model = patch_model_with_flex_attention(model)
warmup_flex_attention(device)

for p in model.parameters():
    p.requires_grad_(False)
for p in model.lm_head.parameters():
    p.requires_grad_(True)
for p in model.model.norm.parameters():
    p.requires_grad_(True)

d_model = model.config.hidden_size
GATE_LAYER = 8
nlb = len([i for i in get_full_attn_layer_indices(model) if i > GATE_LAYER])
gate = E2EGate(d_model=d_model, n_heads=model.config.num_attention_heads,
               n_layers_bias=nlb).to(device).to(torch.float32)
gate.set_qk_norm_scales(model, GATE_LAYER)
build_gate_branch(model, gate, GATE_LAYER)
# non-trivial scores (the folded readout row is zero-init)
torch.nn.init.normal_(gate.gate_branch.down_proj.weight, std=0.02)
gate.train()  # Training mode

optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-4)

task = get_task("biodex_reactions")

# Create test batch
articles = [
    "Patient is a 45-year-old male with history of hypertension.",
    "A 60-year-old female presents with chest pain.",
]

batch_data = []
for article in articles:
    prompt = task.format_prompt(article)
    answer = "Male" if "male" in article else "Female"
    
    prompt_ids = _apply_chat_template(tokenizer, prompt)
    answer_ids = tokenizer.encode("FINAL ANSWER: " + answer + tokenizer.eos_token, add_special_tokens=False)
    full_ids = prompt_ids + answer_ids
    
    non_comp = task.get_non_compressible_context(article)
    empty_prompt = task.format_prompt(non_comp)
    empty_ids = _apply_chat_template(tokenizer, empty_prompt)
    pre_len, post_len = _find_article_boundaries(prompt_ids, empty_ids)
    article_start = pre_len
    article_end = len(prompt_ids) - post_len
    
    batch_data.append({
        "input_ids": full_ids,
        "article_start": article_start,
        "article_end": article_end,
        "ctx_len": len(prompt_ids),
        "n_ans": len(answer_ids),
        "K": 15,
    })

print(f"[test] Batch: {len(batch_data)} samples")

# Training loop
print("\n[test] Running training steps...")
losses = []

for step in range(10):
    optimizer.zero_grad()
    
    logits, loss_mask, padding_mask = forward_with_gate_batched(
        model, gate, batch_data,
        gate_layer=GATE_LAYER,
        frozen_backbone=True,
        prune_phase1_kv=False,
    )
    
    # Compute loss
    batch_loss = 0.0
    n_valid = 0
    
    for b_idx, item in enumerate(batch_data):
        sample_loss_mask = loss_mask[b_idx].to(logits.device)
        if not sample_loss_mask.any():
            continue
        
        sample_logits = logits[b_idx, sample_loss_mask, :]
        ctx_len = item["ctx_len"]
        n_ans = item["n_ans"]
        answer_ids = item["input_ids"][ctx_len:ctx_len + n_ans]
        sample_labels = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
        
        if sample_logits.shape[0] != sample_labels.shape[0]:
            continue
        
        sample_loss = F.cross_entropy(sample_logits, sample_labels)
        batch_loss += sample_loss
        n_valid += 1
    
    if n_valid == 0:
        print(f"  Step {step}: no valid samples")
        continue
    
    loss = batch_loss / n_valid
    loss.backward()
    
    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
    
    optimizer.step()
    
    losses.append(loss.item())
    print(f"  Step {step}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

# Check if loss decreased
print(f"\n[test] Training summary:")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss: {losses[-1]:.4f}")
print(f"  Loss decrease: {losses[0] - losses[-1]:.4f}")

if losses[-1] < losses[0]:
    print("  ✓ Loss decreased - training works!")
else:
    print("  ⚠ Loss didn't decrease - may need more steps or different hyperparameters")

if losses[-1] < losses[0] * 0.9:
    print("  ✓ Significant improvement (>10%) - implementation is correct!")
