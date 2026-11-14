"""CPU regression for batched prefill's full-context activation lifetimes.

Run: python -m tests.inference.test_prefill_memory
"""

import copy
import weakref
from unittest.mock import patch

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from tokengate.e2e_gate.gate_module import gate_forward as gf


class DeterministicGate(torch.nn.Module):
    """A deterministic scorer; the test exercises real decoder layers and KV."""

    n_layers_bias = 0
    selection_mode = "topk"

    def __init__(self, branch):
        super().__init__()
        self.gate_branch = branch
        self.readout = torch.nn.Linear(16, 1, bias=False)

    def forward(self, hidden):
        return self.readout(hidden).squeeze(-1)

    def aggregate(self, scores):
        return scores


def check_prefill_lifetimes(branched, budget):
    torch.manual_seed(7)
    config = Qwen3Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=32,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    gate = DeterministicGate(copy.deepcopy(model.model.layers[1].mlp) if branched else None)
    inputs = [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]

    def run(retain):
        refs, held, handles = [], [], []

        def track(tensor):
            refs.append(weakref.ref(tensor))
            if retain:
                held.append(tensor)

        def phase1_output(module, args, output):
            track(output)

        handles.append(model.model.layers[0].register_forward_hook(phase1_output))
        branch_hs = gf.gate_branch_hs

        def score(*args):
            track(args[-1])
            result = branch_hs(*args)
            track(result)
            return result

        def phase2_entry(module, args):
            assert refs
            if not retain:
                assert all(ref() is None for ref in refs), (
                    "Full-context activations survived into Phase 2"
                )

        handles.append(model.model.layers[2].register_forward_pre_hook(phase2_entry))
        try:
            with patch.object(gf, "gate_branch_hs", score):
                out = gf.prefill_batch_for_decode(
                    model, gate, inputs, [1, 1], [5, 7], 1, budget,
                )
            return (
                out[0], out[2], out[3:],
                [(layer.keys.clone(), layer.values.clone()) for layer in out[1].layers],
            )
        finally:
            for handle in handles:
                handle.remove()

    # Keeping every observed Phase-1 activation alive reproduces the old lifetime
    # without changing arithmetic. Releasing them must preserve tokens and all KV.
    reference = run(retain=True)
    actual = run(retain=False)
    assert torch.equal(actual[0], reference[0])
    assert torch.equal(actual[1], reference[1])
    assert actual[2] == reference[2]
    for actual_kv, reference_kv in zip(actual[3], reference[3], strict=True):
        for actual_tensor, reference_tensor in zip(actual_kv, reference_kv, strict=True):
            torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)


def check_bounded_layer(dtype):
    torch.manual_seed(9)
    config = Qwen3Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=32,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).to(dtype).eval()
    layer = model.model.layers[0]
    hidden = torch.randn(2, 7, 16, dtype=dtype)
    positions = torch.arange(7).expand(2, -1)
    shapes = []
    with torch.no_grad():
        pe = model.model.rotary_emb(hidden, positions)
        reference_cache, actual_cache = DynamicCache(config=config), DynamicCache(config=config)
        kwargs = dict(
            attention_mask=None, position_ids=positions, use_cache=True,
            cache_position=torch.arange(7), position_embeddings=pe,
        )
        reference = layer(hidden, past_key_values=reference_cache, **kwargs)
        handle = layer.mlp.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
        try:
            with patch.object(gf, "_PREFILL_MLP_TOKEN_LIMIT", 4):
                actual = gf.call_layer_padsafe(layer, hidden, [0, 0], past_key_values=actual_cache, **kwargs)
        finally:
            handle.remove()
    assert [shape[0] for shape in shapes] == [4, 4, 4, 2]
    # Match the existing split/gather MLP test tolerances: changing GEMM row
    # counts can change floating-point reduction order, but not the computation.
    atol = 5e-6 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(actual, reference, rtol=0, atol=atol)
    torch.testing.assert_close(actual_cache.layers[0].keys, reference_cache.layers[0].keys, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache.layers[0].values, reference_cache.layers[0].values, rtol=0, atol=0)

    # Training keeps the original differentiable operation and full tensor shape.
    hidden.requires_grad_(True)
    with patch.object(gf, "_PREFILL_MLP_TOKEN_LIMIT", 4):
        trained = gf._gate_layer_mlp_eager(layer, hidden)
        expected = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    torch.testing.assert_close(trained, expected, rtol=0, atol=0)
    trained.sum().backward()
    assert torch.isfinite(hidden.grad).all()


def main():
    for branched in (False, True):
        for budget in (2, 100):
            check_prefill_lifetimes(branched, budget)
            print(f"PASS branched={branched} K={budget}: lifetimes, tokens, positions, KV")
    for dtype in (torch.float32, torch.bfloat16):
        check_bounded_layer(dtype)
        print(f"PASS {dtype}: bounded MLP, decoder outputs, exact KV, training gradients")


if __name__ == "__main__":
    main()
