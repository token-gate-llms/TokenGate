#!/usr/bin/env python3
"""Smoke test for the Qwen3.5 hybrid-attention gate path against a REAL checkpoint.

The hybrid support (commit c2bc0263) was validated against Qwen3-Next before
Qwen3.5 weights existed; this script re-runs the load-bearing checks against
Qwen/Qwen3.5-2B (24 layers, full-attn at {3,7,11,15,19,23}):

    1. load + patch     — only full-attn layers patched, DeltaNet untouched,
                          model._full_attn_layer_indices matches config.
    2. logits parity    — patched vs unpatched full-context forward.
    3. bias steering    — a large negative bias on a token span at full-attn
                          layers > L changes the last-position logits.
    4. LoRA "all"       — resolve_all_adapter_layers restricts to full-attn
                          layers; zero-init adapters leave logits unchanged.

Run (single GPU, ~7 GB):
    python -m tokengate.e2e_gate.tests.test_qwen3_5_hybrid
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokengate.e2e_gate.gate_module.lora import (
    apply_lora_to_layers,
    collect_lora_params,
    resolve_all_adapter_layers,
)
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    get_full_attn_layer_indices,
    patch_model_with_flex_attention,
    set_gate_bias_for_layers,
)
from tokengate.e2e_gate.attn.flex_attention_utils import (
    warmup_flex_attention,
)

MODEL_NAME = "Qwen/Qwen3.5-2B"
EXPECTED_FULL_ATTN = [3, 7, 11, 15, 19, 23]
GATE_LAYER = 3


def _load(device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16).to(device)
    model.eval()
    return model


def _prompt_ids(tokenizer, device, n_repeat=40):
    text = ("The quick brown fox jumps over the lazy dog. " * n_repeat
            + "\n\nQuestion: What animal jumps over the dog?")
    msgs = [{"role": "user", "content": text}]
    rendered = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    return torch.tensor([ids], device=device)


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    input_ids = _prompt_ids(tokenizer, device)
    seq_len = input_ids.shape[1]
    print(f"[setup] {MODEL_NAME}  seq_len={seq_len}")
    results = {}

    # ── 1+2: reference logits, then patch and compare ────────────────────────
    model = _load(device)
    full_attn = get_full_attn_layer_indices(model)
    print(f"[1] full-attn layers: {full_attn}")
    assert full_attn == EXPECTED_FULL_ATTN, full_attn
    with torch.no_grad():
        ref_logits = model(input_ids).logits[0, -1].float()

    model = patch_model_with_flex_attention(model)
    warmup_flex_attention(device)
    # Patching grafts onto the original module (class name is unchanged), so
    # detect it via the use_flex flag the graft installs.
    n_patched = sum(bool(getattr(getattr(l, "self_attn", None), "use_flex", False))
                    for l in model.model.layers)
    n_selfattn = sum(hasattr(l, "self_attn") for l in model.model.layers)
    print(f"[1] patched={n_patched}  layers-with-self_attn={n_selfattn}")
    results["patch"] = (n_patched == len(EXPECTED_FULL_ATTN)
                        and model._full_attn_layer_indices == EXPECTED_FULL_ATTN)

    with torch.no_grad():
        flex_logits = model(input_ids).logits[0, -1].float()
    max_diff = (ref_logits - flex_logits).abs().max().item()
    same_argmax = ref_logits.argmax().item() == flex_logits.argmax().item()
    print(f"[2] patched-vs-unpatched: max|Δlogit|={max_diff:.4f}  "
          f"argmax match={same_argmax}")
    results["parity"] = same_argmax and max_diff < 0.5  # bf16 sdpa vs flex

    # ── 3: bias steering at full-attn layers > GATE_LAYER ────────────────────
    bias = torch.zeros(1, seq_len, device=device, dtype=torch.bfloat16)
    bias[0, 10:seq_len // 2] = -1e9  # kill attention to the first half span
    set_gate_bias_for_layers(model, GATE_LAYER + 1, len(model.model.layers), bias)
    with torch.no_grad():
        biased_logits = model(input_ids).logits[0, -1].float()
    set_gate_bias_for_layers(model, GATE_LAYER + 1, len(model.model.layers), None)
    steer = (biased_logits - flex_logits).abs().max().item()
    print(f"[3] bias steering: max|Δlogit|={steer:.4f} (expect >0.1)")
    results["steering"] = steer > 0.1

    # ── 4: LoRA "all" on hybrid = full-attn layers only, zero-init no-op ─────
    lora_layers = resolve_all_adapter_layers(model)
    print(f"[4] resolve_all_adapter_layers → {lora_layers}")
    assert lora_layers == EXPECTED_FULL_ATTN, lora_layers
    apply_lora_to_layers(model, lora_layers, rank=16, alpha=32,
                         targets=["q_proj", "v_proj"])
    n_params = len(collect_lora_params(model))
    with torch.no_grad():
        lora_logits = model(input_ids).logits[0, -1].float()
    lora_diff = (lora_logits - flex_logits).abs().max().item()
    print(f"[4] lora params={n_params} (expect {len(EXPECTED_FULL_ATTN)*2*2})  "
          f"zero-init max|Δlogit|={lora_diff:.6f}")
    results["lora"] = (n_params == len(EXPECTED_FULL_ATTN) * 2 * 2
                       and lora_diff == 0.0)

    print("\n" + "=" * 60)
    ok = all(results.values())
    for k, v in results.items():
        print(f"  {'✓' if v else '✗'} {k}")
    print(f"{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
