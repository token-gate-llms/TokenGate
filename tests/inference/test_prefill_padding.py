"""Check that padded prefill preserves recurrent state for subsequent decode."""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

from tokengate.e2e_gate.gate_module.gate_forward import call_layer_padsafe


def decoder_layer(layer_type):
    config = Qwen3_5TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=[layer_type],
    )
    return Qwen3_5DecoderLayer(config, layer_idx=0)


def recurrent_outputs(hidden_states):
    """Use a decaying state so trailing padding changes the decode state."""
    state = torch.zeros_like(hidden_states[:, 0])
    outputs = []
    for token in hidden_states.unbind(dim=1):
        state = 0.5 * state + token
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def test_padded_linear_layer_matches_unpadded_state_and_outputs():
    layer = decoder_layer("linear_attention")
    hidden = torch.arange(1, 1 + 2 * 5 * 16, dtype=torch.float32).reshape(2, 5, 16)
    hidden[0, 3:] = 0
    captured = []

    def forward(values):
        output = recurrent_outputs(values)
        captured.append(output[:, -1])
        return output

    output = call_layer_padsafe(layer, hidden, [2, 0], fn=forward)
    for row, length in enumerate([3, 5]):
        expected = recurrent_outputs(hidden[row : row + 1, :length])
        torch.testing.assert_close(output[row, :length], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(captured[0][row], expected[0, -1], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("layer_type", "padding"),
    [("full_attention", [2, 0]), ("linear_attention", [0, 0])],
)
def test_full_attention_and_unpadded_layers_preserve_input(layer_type, padding):
    layer = decoder_layer(layer_type)
    hidden = torch.ones(2, 5, 16)
    output = call_layer_padsafe(layer, hidden, padding, fn=lambda values: values)
    assert output is hidden
