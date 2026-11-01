"""Short-context inference fallback without compiled/custom attention kernels."""
import torch
import torch.nn.functional as F


def sdpa_causal(Q, K, V, gate_bias=None, per_head=False, chunk_size=128):
    """Exact causal attention with optional additive key bias, chunked over Q.

    Chunking bounds temporary attention storage even when SDPA selects its
    math implementation. This is a portability path, not a fast-kernel claim.
    Inputs use (batch, heads, sequence, head_dim); prefill Q and K must align.
    """
    B, H, S, _ = Q.shape
    if S != K.shape[2] or chunk_size < 1:
        raise ValueError("sdpa compatibility prefill needs aligned Q/K and chunk_size >= 1")
    if H % K.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    groups = H // K.shape[1]
    K = K.repeat_interleave(groups, dim=1) if groups > 1 else K
    V = V.repeat_interleave(groups, dim=1) if groups > 1 else V
    bias = None
    if gate_bias is not None:
        from .flex_attention_utils import _bias_bhs
        bias = _bias_bhs(gate_bias, per_head, B, H, S).unsqueeze(-2)
    outputs = []
    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        allowed = (torch.arange(end, device=Q.device)[None, :]
                   <= torch.arange(start, end, device=Q.device)[:, None])
        mask = torch.zeros((end - start, end), device=Q.device, dtype=Q.dtype)
        if bias is not None:
            mask = mask + bias[..., :end].to(Q.dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
        outputs.append(F.scaled_dot_product_attention(
            Q[:, :, start:end], K[:, :, :end], V[:, :, :end],
            attn_mask=mask, dropout_p=0.0))
    return torch.cat(outputs, dim=2)
