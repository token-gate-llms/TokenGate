"""Adapters above a recurrent gate must not extend the pre-pruning prefix."""

import pytest
import torch
import torch._functorch.config
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from tokengate.e2e_gate.attn.qwen3_flex_attention import patch_model_with_flex_attention
from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
from tokengate.e2e_gate.gate_module.gate_forward import (
    build_gate_branch,
    forward_backward_with_gate_multi_K_detach,
    forward_with_gate_batched,
)
from tokengate.e2e_gate.gate_module.lora import apply_lora_to_layers
from tokengate.e2e_gate.train.selection import trainable_parameters


@pytest.mark.parametrize("gate_layer", [1, 2])
def test_adapter_above_gate_preserves_prefix_and_gradients(gate_layer):
    torch.manual_seed(5)
    torch._functorch.config.donated_buffer = False
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    model = Qwen3_5ForCausalLM(config).cuda().to(torch.bfloat16).eval()
    model.requires_grad_(False)
    apply_lora_to_layers(model, [3, 7], 4, 8, ["q_proj", "v_proj"])
    gate = E2EGate(d_model=32, n_heads=2, n_layers_bias=2).cuda().float()
    gate.set_qk_norm_scales(model, gate_layer)
    build_gate_branch(model, gate, gate_layer).cuda()
    patch_model_with_flex_attention(model)
    row = dict(
        input_ids=list(range(1, 25)),
        ctx_len=20,
        n_ans=4,
        article_start=2,
        article_end=18,
        K=8,
    )
    with torch.no_grad():
        expected, _, _ = forward_with_gate_batched(
            model,
            gate,
            [row],
            gate_layer,
            lora_start_layer=-1,
        )
        actual, _, _ = forward_with_gate_batched(
            model,
            gate,
            [row],
            gate_layer,
            lora_start_layer=3,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = trainable_parameters(model, gate)
    losses, gradients = [], []
    for start in (-1, 3):
        for parameter in parameters.values():
            parameter.grad = None
        loss, count, _, _ = forward_backward_with_gate_multi_K_detach(
            model,
            gate,
            [row],
            gate_layer,
            [8, 16],
            lora_start_layer=start,
        )
        assert count == 2
        losses.append(loss)
        gradients.append({name: p.grad.clone() for name, p in parameters.items()})
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for name in parameters:
        torch.testing.assert_close(
            gradients[0][name], gradients[1][name], rtol=0, atol=0
        )
