"""The key-bias extension must honor deterministic training for every gradient."""

import pytest
import torch

from kernels.fa2bias.keybias import (
    flash_attn_key_bias,
    flash_attn_key_bias_varlen,
)


@pytest.mark.parametrize("packed", [False, True])
def test_keybias_backward_is_exactly_repeatable(packed):
    torch.manual_seed(7)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        lengths = [513, 385] if packed else [513]
        length = sum(lengths)
        shape = (length,) if packed else (1, length)
        q = torch.randn(*shape, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(*shape, 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        bias = torch.randn(len(lengths), 4, max(lengths), device="cuda") * 0.1
        inputs = [value.requires_grad_() for value in (q, k, v, bias)]
        cumulative = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32)
        cumulative = cumulative.cumsum(0, dtype=torch.int32)
        outputs = []
        for _ in range(3):
            if packed:
                result = flash_attn_key_bias_varlen(
                    *inputs, cumulative, max(lengths), 128**-0.5
                )
            else:
                result = flash_attn_key_bias(*inputs, 128**-0.5)
            outputs.append(torch.autograd.grad(result.float().square().sum(), inputs))
        for actual in outputs[1:]:
            for gradient, expected in zip(actual, outputs[0]):
                torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous)
