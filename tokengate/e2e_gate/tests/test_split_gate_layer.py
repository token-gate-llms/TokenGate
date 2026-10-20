"""Correctness of the a_j phase-boundary split (split_gate_layer_mlp).

Validates, on a REAL Qwen3DecoderLayer (small config, CPU, no GPU needed):

  1. attn_only_forward + gate_layer_mlp compose to EXACTLY the layer's own forward
     (bit-identical, including the KV-cache write) — the split is a refactor of the
     HF layer, not a reimplementation.
  2. gather-then-MLP == MLP-then-gather (position-wise), fp32 and bf16 — pruning
     before the gate layer's MLP is mathematically exact. Numerically it is equal
     up to GEMM blocking: the row count of the matmul changes, so MKL/cuBLAS may
     reassociate the K-dim reduction differently (measured 2.4e-07 abs / 7e-08 rel
     in fp32 on CPU; LN2 alone is bitwise identical). This is the same
     nondeterminism class as changing the batch size, which the codebase already
     treats as equivalent — asserted here with tight tolerances, not bitwise.
  3. Zero (padding) rows map to zero through gate_layer_mlp — batched callers
     right-pad with zeros.
  4. The gradient path the split restores: scoring loss on a branch fed the
     DETACHED-then-RECONNECTED boundary (the hs_d pattern) produces nonzero grads
     on the layer's ATTENTION weights. Under the old severed hook capture this
     grad was identically zero (also asserted).

Run:  python -m tokengate.e2e_gate.tests.test_split_gate_layer
"""
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tokengate.e2e_gate.gate_module.gate_forward import (  # noqa: E402
    attn_only_forward, gate_layer_mlp,
)

from transformers import AutoConfig  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import (  # noqa: E402
    Qwen3DecoderLayer, Qwen3RotaryEmbedding,
)


def _small_cfg():
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
    cfg.hidden_size = 64
    cfg.intermediate_size = 160
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.num_hidden_layers = 1
    cfg._attn_implementation = "eager"
    return cfg


def _setup(dtype=torch.float32, seq=48):
    torch.manual_seed(0)
    cfg = _small_cfg()
    layer = Qwen3DecoderLayer(cfg, layer_idx=0).to(dtype).eval()
    hs = torch.randn(1, seq, cfg.hidden_size, dtype=dtype)
    position_ids = torch.arange(seq).unsqueeze(0)
    pe = Qwen3RotaryEmbedding(cfg)(hs, position_ids)
    cache_pos = torch.arange(seq)
    return cfg, layer, hs, position_ids, pe, cache_pos


def test_split_matches_full_layer():
    """(1) full layer forward == attn_only_forward + gate_layer_mlp, incl. KV write."""
    ok = True
    for dtype in (torch.float32, torch.bfloat16):
        cfg, layer, hs, position_ids, pe, cache_pos = _setup(dtype)
        with torch.no_grad():
            cache_a = DynamicCache(config=cfg)
            full = layer(hs, attention_mask=None, position_ids=position_ids,
                         past_key_values=cache_a, use_cache=True,
                         cache_position=cache_pos, position_embeddings=pe)
            cache_b = DynamicCache(config=cfg)
            a = attn_only_forward(layer, hs, position_ids, cache_b, cache_pos, pe)
            split = gate_layer_mlp(layer, a)
        same_out = torch.equal(full, split)
        same_k = torch.equal(cache_a.layers[0].keys, cache_b.layers[0].keys)
        same_v = torch.equal(cache_a.layers[0].values, cache_b.layers[0].values)
        print(f"  [{dtype}] out identical={same_out}  kv identical={same_k and same_v}")
        ok &= same_out and same_k and same_v
    return ok


def test_gather_mlp_commute():
    """(2)+(3) MLP-then-gather == gather-then-MLP (up to GEMM blocking); zero rows
    map exactly to zero (a zero matmul row is exact under any blocking)."""
    ok = True
    # abs tolerance per dtype: ~10x the measured GEMM-blocking discrepancy
    tol = {torch.float32: 5e-6, torch.bfloat16: 5e-2}
    for dtype in (torch.float32, torch.bfloat16):
        cfg, layer, hs, position_ids, pe, cache_pos = _setup(dtype, seq=100)
        with torch.no_grad():
            a = attn_only_forward(layer, hs, position_ids,
                                  DynamicCache(config=cfg), cache_pos, pe)
            keep = torch.randperm(100)[:30].sort().values
            full_then_gather = gate_layer_mlp(layer, a)[:, keep, :]
            gather_then_mlp = gate_layer_mlp(layer, a[:, keep, :])
            # padding: right-pad the gathered rows with zeros, as batched callers do
            padded = torch.cat([a[:, keep, :],
                                torch.zeros(1, 7, cfg.hidden_size, dtype=dtype)], dim=1)
            pad_out = gate_layer_mlp(layer, padded)
        d_commute = (full_then_gather - gather_then_mlp).abs().max().item()
        d_body = (pad_out[:, :30, :] - gather_then_mlp).abs().max().item()
        pad_zero = torch.equal(pad_out[:, 30:, :],
                               torch.zeros(1, 7, cfg.hidden_size, dtype=dtype))
        commute = d_commute <= tol[dtype]
        body = d_body <= tol[dtype]
        print(f"  [{dtype}] commute max|diff|={d_commute:.2e} (tol {tol[dtype]:.0e}) "
              f"ok={commute}  pad rows exactly zero={pad_zero}  "
              f"pad body max|diff|={d_body:.2e} ok={body}")
        ok &= commute and pad_zero and body
    return ok


def test_scoring_grad_reaches_attention():
    """(4) branch scoring grad reaches the layer's ATTENTION weights via the
    reconnected boundary; the old severed capture gives exactly zero."""
    import copy
    cfg, layer, hs, position_ids, pe, cache_pos = _setup(torch.float32)
    layer.train()
    for p in layer.parameters():
        p.requires_grad_(True)
    branch = copy.deepcopy(layer.mlp)          # MLP_gate clone, as build_gate_branch does

    def run(sever: bool):
        layer.zero_grad(set_to_none=True)
        a = attn_only_forward(layer, hs, position_ids,
                              DynamicCache(config=cfg), cache_pos, pe)
        if sever:
            boundary = a.detach()              # old hook behaviour (f696ad4c)
        else:
            boundary = a.detach().requires_grad_(True)   # hs_d pattern
        h_gate = boundary + branch(layer.post_attention_layernorm(boundary))
        h_gate.pow(2).sum().backward()         # scoring-loss stand-in
        if not sever and boundary.grad is not None:
            a.backward(boundary.grad)          # the ONE reconnection
        g = layer.self_attn.q_proj.weight.grad
        return 0.0 if g is None else g.abs().max().item()

    g_reconnected = run(sever=False)
    g_severed = run(sever=True)
    print(f"  reconnected: max|dL/dW_q| = {g_reconnected:.3e}  (must be > 0)")
    print(f"  severed:     max|dL/dW_q| = {g_severed:.3e}  (must be exactly 0)")
    return g_reconnected > 0.0 and g_severed == 0.0


def main():
    results = {}
    print("[1] split == full layer (fwd + KV cache)")
    results["split==full"] = test_split_matches_full_layer()
    print("[2] gather/MLP commute + padding")
    results["commute+pad"] = test_gather_mlp_commute()
    print("[3] scoring grad reaches attention weights")
    results["grad path"] = test_scoring_grad_reaches_attention()
    print()
    n_pass = sum(results.values())
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{n_pass}/{len(results)} passed")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
