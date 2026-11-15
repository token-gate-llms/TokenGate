"""CPU-safe reference checks; run with python -m tests.attention.test_sdpa_compat.

These establish attention semantics, not T4 end-to-end compatibility.
"""
import torch

from tokengate.e2e_gate.attn.sdpa_compat import sdpa_causal


def main():
    torch.manual_seed(7)
    B, H, S, D = 2, 4, 19, 8
    for dtype in (torch.float32, torch.float16):
        for kv_heads in (1, 2, H):
            q = torch.randn(B, H, S, D, dtype=dtype)
            k = torch.randn(B, kv_heads, S, D, dtype=dtype)
            v = torch.randn_like(k)
            for shape, per_head in ((None, False), ((S,), False), ((B, S), False),
                                    ((S, H), True), ((B, S, H), True)):
                bias = torch.randn(*shape) if shape else None
                logits = q.float() @ k.float().repeat_interleave(H // kv_heads, 1).transpose(-2, -1) / D**0.5
                if bias is not None:
                    if per_head:
                        expanded = bias.T[None, :, None, :] if bias.ndim == 2 else bias.permute(0, 2, 1)[:, :, None, :]
                    else:
                        expanded = bias[None, None, None, :] if bias.ndim == 1 else bias[:, None, None, :]
                    # The compatibility mask follows the backbone compute dtype.
                    logits += expanded.to(dtype).float()
                logits.masked_fill_(~torch.ones(S, S, dtype=torch.bool).tril(), float('-inf'))
                reference = logits.softmax(-1) @ v.float().repeat_interleave(H // kv_heads, 1)
                for chunk in (1, 7, 128):
                    out = sdpa_causal(q, k, v, bias, per_head, chunk)
                    torch.testing.assert_close(out.float(), reference, atol=2e-3 if dtype == torch.float16 else 1e-6,
                                               rtol=2e-3 if dtype == torch.float16 else 1e-5)
                    assert torch.isfinite(out).all()
    print('PASS: chunk boundaries, causality, GQA, scalar/per-head bias, FP32 and FP16')


if __name__ == '__main__':
    main()
