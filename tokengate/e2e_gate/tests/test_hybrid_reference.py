"""Regression tests for the hybrid split boundary and recurrent padding."""

import pytest
import torch
from transformers import Qwen3_5TextConfig


def hybrid_config():
    return Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
    )


@pytest.mark.parametrize("gate_layer", [1, 3])
@torch.no_grad()
def test_pruned_hybrid_generation_cuda(gate_layer):
    """Exercise all-recurrent lower history and KV/recurrent decode masks."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import Qwen3_5ForCausalLM, TokenizersBackend

    from tokengate.e2e_gate.attn.qwen3_flex_attention import (
        patch_model_with_flex_attention,
    )
    from tokengate.e2e_gate.decode.gate_decode import generate_batch
    from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
    from tokengate.e2e_gate.gate_module.gate_forward import build_gate_branch

    torch.manual_seed(42)
    config = hybrid_config()
    config.num_hidden_layers = 8
    config.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    config.layer_types *= 2
    config.rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
    }
    model = Qwen3_5ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    patch_model_with_flex_attention(model)
    gate = E2EGate(d_model=32, n_heads=2, n_layers_bias=2 if gate_layer == 1 else 1)
    build_gate_branch(model, gate, gate_layer)
    gate.to("cuda").eval()
    gate.gate_branch.down_proj.weight.normal_(std=0.05)
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(
            WordLevel(
                {"[UNK]": 0, **{f"token{i}": i for i in range(1, 32)}},
                unk_token="[UNK]",
            )
        ),
        unk_token="[UNK]",
    )
    prompts = [list(range(4, 28)), list(range(5, 22))]
    result = generate_batch(
        model,
        gate,
        tokenizer,
        prompts,
        [2, 2],
        [22, 15],
        gate_layer,
        [3, 5],
        4,
        decode_phase2_form=True,
        prune_below_gate=False,
    )
    assert result.n_kepts == [7, 9]
    assert result.K_clampeds == [3, 5]
    assert result.n_gen == [4, 4]
    assert result.steps == 3
    for row, prompt in enumerate(prompts):
        kept = result.kept_pos[row, : result.n_kepts[row]].tolist()
        assert kept[:2] == [0, 1]
        assert kept[-2:] == [len(prompt) - 2, len(prompt) - 1]
        assert kept == sorted(set(kept))
