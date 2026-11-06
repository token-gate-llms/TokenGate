"""Regression tests for the hybrid split boundary and recurrent padding."""

import torch
import torch.nn.functional as F
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

from tokengate.e2e_gate.gate_module.gate_forward import (
    _attn_only_forward_eager,
    _gate_layer_mlp_eager,
    call_layer_padsafe,
)


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


@torch.no_grad()
def test_recurrent_gate_split_matches_decoder():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = Qwen3_5DecoderLayer(hybrid_config(), 0).to(device).eval()
    hidden = torch.randn(1, 6, 32, device=device)
    expected = layer(hidden, position_embeddings=None)
    boundary = _attn_only_forward_eager(layer, hidden, None, None, None, None)
    torch.testing.assert_close(
        _gate_layer_mlp_eager(layer, boundary), expected, rtol=0, atol=0
    )


@torch.no_grad()
def test_padding_does_not_advance_recurrent_state():
    torch.manual_seed(42)
    config = hybrid_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = Qwen3_5DecoderLayer(config, 0).to(device).eval()
    hidden = torch.randn(1, 6, 32, device=device)
    cache = DynamicCache(config=config)
    expected = layer(hidden, position_embeddings=None, past_key_values=cache)
    padded_cache = DynamicCache(config=config)
    padded = torch.cat((hidden, hidden.new_zeros(1, 3, 32)), dim=1)
    actual = call_layer_padsafe(
        layer, padded, [3], position_embeddings=None, past_key_values=padded_cache
    )
    torch.testing.assert_close(actual[:, :6], expected)
    torch.testing.assert_close(
        padded_cache.layers[0].conv_states, cache.layers[0].conv_states
    )
    torch.testing.assert_close(
        padded_cache.layers[0].recurrent_states, cache.layers[0].recurrent_states
    )


def test_gate_scale_uses_effective_zero_centered_norm_gain():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate

    model = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(
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
    )
    gate = E2EGate(d_model=32, n_heads=2, n_layers_bias=1)
    # Zero-initialized Qwen3.5 norm weights represent unit gains.
    values = gate.set_qk_norm_scales(model, 0)
    assert values == [1.0]
    torch.testing.assert_close(
        F.softplus(gate.gate_scale), torch.ones_like(gate.gate_scale)
    )
