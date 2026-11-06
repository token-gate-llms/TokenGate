"""Learned key-bias gradients must repeat exactly in deterministic training."""

import pytest
import torch

from tokengate.e2e_gate.attn.flex_attention_utils import flex_attention_gate_bias


@pytest.mark.parametrize("per_head", [False, True])
def test_flex_keybias_backward_is_exactly_repeatable(per_head):
    torch.manual_seed(7)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        length, heads, head_size = 513, 4, 256
        q = torch.randn(
            1, heads, length, head_size, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(1, 2, length, head_size, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        shape = (length, heads) if per_head else (length,)
        bias = torch.randn(shape, device="cuda") * 0.1
        inputs = [value.requires_grad_() for value in (q, k, v, bias)]
        upstream = torch.randn_like(q)
        gradients = []
        for _ in range(3):
            output = flex_attention_gate_bias(*inputs, per_head=per_head)
            loss = (output.float() * upstream.float()).sum()
            gradients.append(torch.autograd.grad(loss, inputs))
        for repeat in gradients[1:]:
            for name, actual, expected in zip(
                ("query", "key", "value", "bias"), repeat, gradients[0], strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("batched_bias", [False, True])
def test_flex_bias_gradient_matches_dense_reference(per_head, batched_bias):
    """Check causal self-attention, GQA, broadcasting, and masking against FP64."""
    torch.manual_seed(11)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        batch, heads, query_length, key_length, dim = 2, 4, 137, 137, 32
        query = torch.randn(batch, query_length, heads, dim, device="cuda")
        key = torch.randn(batch, key_length, 2, dim, device="cuda")
        value = torch.randn_like(key)
        shape = (key_length, heads) if per_head else (key_length,)
        if batched_bias:
            shape = (batch, *shape)
        bias = torch.randn(shape, device="cuda") * 0.1
        if per_head:
            bias[..., -10:, :] = -1e9
        else:
            bias[..., -10:] = -1e9
        inputs = [
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            bias,
        ]
        inputs = [tensor.requires_grad_() for tensor in inputs]
        reference = [tensor.detach().double().requires_grad_() for tensor in inputs]
        q, k, v, b = reference
        k = k.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)
        b = b.movedim(-1, -2) if per_head else b.unsqueeze(-2)
        scores = q @ k.transpose(-1, -2) * dim**-0.5 + b.unsqueeze(-2)
        causal = (
            torch.arange(query_length, device="cuda")[:, None]
            >= torch.arange(key_length, device="cuda")[None, :]
        )
        expected = scores.masked_fill(~causal, -torch.inf).softmax(-1) @ v
        actual = flex_attention_gate_bias(*inputs, per_head=per_head)
        upstream = torch.randn_like(actual)
        gradients = torch.autograd.grad(actual, inputs, upstream)
        expected_gradients = torch.autograd.grad(expected, reference, upstream.double())
        torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-4)
        for actual_grad, expected_grad in zip(
            gradients, expected_gradients, strict=True
        ):
            torch.testing.assert_close(
                actual_grad.double(), expected_grad, atol=2e-5, rtol=2e-4
            )
    finally:
        torch.use_deterministic_algorithms(previous)
