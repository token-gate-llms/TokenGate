"""Fixed-order gradients for causal FlexAttention's learned key bias.

FlexAttention scatters captured-tensor gradients with atomic additions. For
deterministic training, each program instead owns a tile of keys and sums its
query contributions in order. Storage is linear in sequence length; the
attention matrix is never materialized.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit(do_not_specialize=["QUERY_LENGTH", "KEY_LENGTH"])
def _bias_gradient_kernel(
    Q,
    K,
    V,
    BIAS,
    DO,
    DELTA,
    LSE,
    DBIAS,
    QUERY_LENGTH,
    KEY_LENGTH,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_head = tl.program_id(0)
    key_block = tl.program_id(1)
    kv_batch_head = batch_head // HEADS * KV_HEADS
    kv_batch_head += (batch_head % HEADS) // (HEADS // KV_HEADS)
    keys = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, BLOCK_D)
    value_dims = tl.arange(0, BLOCK_V)
    key = tl.load(
        K + (kv_batch_head * KEY_LENGTH + keys[:, None]) * HEAD_DIM + dims[None, :],
        (keys[:, None] < KEY_LENGTH) & (dims[None, :] < HEAD_DIM),
        0,
    )
    value = tl.load(
        V
        + (kv_batch_head * KEY_LENGTH + keys[:, None]) * VALUE_DIM
        + value_dims[None, :],
        (keys[:, None] < KEY_LENGTH) & (value_dims[None, :] < VALUE_DIM),
        0,
    )
    bias = tl.load(BIAS + batch_head * KEY_LENGTH + keys, keys < KEY_LENGTH, 0)
    gradient = tl.full((BLOCK_K,), 0, tl.float32)
    for start in range(key_block * BLOCK_K // BLOCK_Q * BLOCK_Q, QUERY_LENGTH, BLOCK_Q):
        queries = start + tl.arange(0, BLOCK_Q)
        query = tl.load(
            Q
            + (batch_head * QUERY_LENGTH + queries[None, :]) * HEAD_DIM
            + dims[:, None],
            (queries[None, :] < QUERY_LENGTH) & (dims[:, None] < HEAD_DIM),
            0,
        )
        dout = tl.load(
            DO
            + (batch_head * QUERY_LENGTH + queries[None, :]) * VALUE_DIM
            + value_dims[:, None],
            (queries[None, :] < QUERY_LENGTH) & (value_dims[:, None] < VALUE_DIM),
            0,
        )
        normalizer = tl.load(
            LSE + batch_head * QUERY_LENGTH + queries, queries < QUERY_LENGTH, 0
        )
        delta = tl.load(
            DELTA + batch_head * QUERY_LENGTH + queries, queries < QUERY_LENGTH, 0
        )
        scores = tl.dot(key, query, input_precision="ieee") * HEAD_DIM**-0.5
        scores += bias[:, None]
        probability = tl.exp(scores - normalizer[None, :])
        probability = tl.where(
            (keys[:, None] <= queries[None, :])
            & (queries[None, :] < QUERY_LENGTH)
            & (keys[:, None] < KEY_LENGTH)
            & (normalizer[None, :] != -float("inf")),
            probability,
            0,
        )
        dp = tl.dot(value, dout, input_precision="ieee")
        gradient += tl.sum(probability * (dp - delta[None, :]), axis=1)
    tl.store(DBIAS + batch_head * KEY_LENGTH + keys, gradient, keys < KEY_LENGTH)


@torch.library.custom_op("tokengate::flex_bias_backward", mutates_args=())
def _bias_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
) -> torch.Tensor:
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    delta = (output.float() * grad_output.float()).sum(-1).contiguous()
    result = torch.empty(
        (batch, heads, key_length), device=query.device, dtype=torch.float32
    )
    _bias_gradient_kernel[(batch * heads, triton.cdiv(key_length, 32))](
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        bias.contiguous(),
        grad_output.contiguous(),
        delta,
        lse.contiguous(),
        result,
        query_length,
        key_length,
        heads,
        key.shape[1],
        head_dim,
        value.shape[-1],
        triton.next_power_of_2(head_dim),
        triton.next_power_of_2(value.shape[-1]),
        64,
        32,
        num_warps=4,
    )
    return result


@_bias_backward.register_fake
def _bias_backward_fake(query, key, value, bias, output, grad_output, lse):
    return query.new_empty(
        (query.shape[0], query.shape[1], key.shape[2]), dtype=torch.float32
    )


class _BiasGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, query, key, value, bias, lse, per_head):
        ctx.save_for_backward(output, query, key, value, bias, lse)
        ctx.per_head = per_head
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        output, query, key, value, bias, lse = ctx.saved_tensors
        if ctx.per_head:
            expanded = bias.movedim(-1, -2)
        else:
            expanded = bias.unsqueeze(-2)
        expanded = expanded.expand(query.shape[0], query.shape[1], key.shape[2])
        gradient = _bias_backward(query, key, value, expanded, output, grad_output, lse)
        if ctx.per_head:
            gradient = gradient.movedim(-2, -1)
        else:
            gradient = gradient.sum(-2)
        gradient = gradient.sum_to_size(bias.shape).to(bias.dtype)
        return grad_output, None, None, None, gradient, None, None


def attach_bias_gradient(output, query, key, value, bias, lse, per_head):
    """Attach only dBias; the existing output graph supplies dQ, dK, and dV.

    Args:
        output: Causal FlexAttention output computed with a detached key bias.
        query: Queries in (batch, query heads, query length, head dimension).
        key: Keys in (batch, KV heads, key length, head dimension).
        value: Values in (batch, KV heads, key length, value dimension).
        bias: Scalar or per-query-head key bias, optionally batched.
        lse: Natural-log normalizers returned by the same FlexAttention call.
        per_head: Whether bias has a trailing query-head dimension.

    Returns:
        The unchanged attention output with a deterministic bias gradient.
    """
    return _BiasGradient.apply(output, query, key, value, bias, lse, per_head)
