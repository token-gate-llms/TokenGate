#!/usr/bin/env python3
"""Phase A: FlexAttention Correctness Tests

Verify that FlexAttention produces numerically equivalent results to
the current dense mask implementation for:
1. Causal attention (phase 1)
2. Causal + gate bias (phase 2)
3. Gradient flow through score_mod

Run:
    python -m tokengate.e2e_gate.test_flex_attention
"""

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Compile flex_attention for better performance and correctness
flex_attention = torch.compile(flex_attention)


def test_causal_correctness():
    """Compare FlexAttention causal vs manual SDPA + dense mask."""
    print("\n" + "="*60)
    print("TEST 1: Causal Attention Correctness")
    print("="*60)
    
    B, H, S, D = 1, 32, 2048, 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    Q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    
    print(f"Input shape: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Device: {device}, dtype: {Q.dtype}")
    
    # Reference: SDPA with dense causal mask
    print("\nComputing reference (SDPA with dense mask)...")
    causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
    mask_4d = causal_mask.view(1, 1, S, S).expand(B, H, S, S)
    
    with torch.no_grad():
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, attn_mask=~mask_4d
        )
    
    # FlexAttention
    print("Computing FlexAttention...")
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    
    block_mask = create_block_mask(
        causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device
    )
    
    with torch.no_grad():
        flex_out = flex_attention(Q, K, V, block_mask=block_mask)
    
    # Numerical check
    max_diff = (ref_out - flex_out).abs().max().item()
    mean_diff = (ref_out - flex_out).abs().mean().item()
    
    print(f"\nNumerical comparison:")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-2:
        print(f"✓ PASS: Max diff {max_diff:.6f} < 1e-2")
        return True
    else:
        print(f"✗ FAIL: Max diff {max_diff:.6f} >= 1e-2")
        return False


def test_gate_bias_correctness():
    """Compare FlexAttention gate bias vs manual broadcast."""
    print("\n" + "="*60)
    print("TEST 2: Gate Bias Correctness")
    print("="*60)
    
    B, H, S, D = 1, 32, 512, 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    Q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    b = torch.randn(S, device=device, dtype=torch.bfloat16)  # per-key bias
    
    print(f"Input shape: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Bias shape: {b.shape}")
    print(f"Device: {device}, dtype: {Q.dtype}")
    
    # Reference: manual score + bias + causal mask
    print("\nComputing reference (manual broadcast)...")
    with torch.no_grad():
        scores = (Q @ K.transpose(-2, -1)) / (D ** 0.5)
        bias_4d = b.view(1, 1, 1, S).expand(B, H, S, S)
        causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        scores = scores + bias_4d
        scores = scores.masked_fill(causal_mask.view(1, 1, S, S), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        ref_out = attn @ V
    
    # FlexAttention
    print("Computing FlexAttention...")
    def causal(b_idx, h_idx, q_idx, kv_idx):
        return q_idx >= kv_idx
    
    def gate_bias(score, b_idx, h_idx, q_idx, kv_idx):
        return score + b[kv_idx]
    
    block_mask = create_block_mask(
        causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device
    )
    
    with torch.no_grad():
        flex_out = flex_attention(Q, K, V, score_mod=gate_bias, block_mask=block_mask)
    
    # Numerical check
    max_diff = (ref_out - flex_out).abs().max().item()
    mean_diff = (ref_out - flex_out).abs().mean().item()
    rel_diff = ((ref_out - flex_out).abs() / (ref_out.abs() + 1e-8)).mean().item()
    
    print(f"\nNumerical comparison:")
    print(f"  Max diff:  {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Rel diff:  {rel_diff:.6f}")
    
    # For bfloat16 with bias, slightly higher threshold
    # Gate bias can have larger absolute values, so max diff can be higher
    # But mean diff and relative diff should still be small
    max_threshold = 0.05  # ~1.5% of typical attention output range
    mean_threshold = 0.002  # Mean should be much smaller
    
    if max_diff < max_threshold and mean_diff < mean_threshold:
        print(f"✓ PASS: Max diff {max_diff:.6f} < {max_threshold}, Mean diff {mean_diff:.6f} < {mean_threshold}")
        return True
    else:
        print(f"✗ FAIL: Max diff {max_diff:.6f} >= {max_threshold} or Mean diff {mean_diff:.6f} >= {mean_threshold}")
        print(f"  Note: Relative diff {rel_diff:.4f} indicates {'good' if rel_diff < 0.1 else 'poor'} agreement")
        return False


def test_gate_bias_gradients():
    """Verify gradients flow through score_mod to bias tensor."""
    print("\n" + "="*60)
    print("TEST 3: Gradient Flow Through score_mod")
    print("="*60)
    
    B, H, S, D = 1, 8, 256, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    Q = torch.randn(B, H, S, D, device=device, requires_grad=True)
    K = torch.randn(B, H, S, D, device=device, requires_grad=True)
    V = torch.randn(B, H, S, D, device=device, requires_grad=True)
    b = torch.randn(S, device=device, requires_grad=True, dtype=torch.float32)
    
    print(f"Input shape: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Bias shape: {b.shape}, dtype: {b.dtype}")
    print(f"Device: {device}")
    
    def causal(b_idx, h_idx, q_idx, kv_idx):
        return q_idx >= kv_idx
    
    def gate_bias(score, b_idx, h_idx, q_idx, kv_idx):
        return score + b[kv_idx]
    
    print("\nComputing forward + backward...")
    block_mask = create_block_mask(
        causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device
    )
    out = flex_attention(Q, K, V, score_mod=gate_bias, block_mask=block_mask)
    loss = out.sum()
    loss.backward()
    
    # Check gradients
    print(f"\nGradient check:")
    print(f"  Q.grad: {'✓' if Q.grad is not None else '✗'} "
          f"(norm={Q.grad.norm().item():.4f})" if Q.grad is not None else "")
    print(f"  K.grad: {'✓' if K.grad is not None else '✗'} "
          f"(norm={K.grad.norm().item():.4f})" if K.grad is not None else "")
    print(f"  V.grad: {'✓' if V.grad is not None else '✗'} "
          f"(norm={V.grad.norm().item():.4f})" if V.grad is not None else "")
    print(f"  b.grad: {'✓' if b.grad is not None else '✗'} "
          f"(norm={b.grad.norm().item():.4f})" if b.grad is not None else "")
    
    if b.grad is None:
        print("✗ FAIL: No gradient for bias tensor")
        return False
    
    if b.grad.abs().sum() == 0:
        print("✗ FAIL: Zero gradient for bias tensor")
        return False
    
    print(f"✓ PASS: Bias gradient norm = {b.grad.norm().item():.4f}")
    return True


def test_sequence_lengths():
    """Test FlexAttention on biodex-relevant sequence lengths."""
    print("\n" + "="*60)
    print("TEST 4: Biodex Sequence Lengths")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, D = 1, 32, 128
    
    # Biodex distribution: median=5336, P95=9902, max=13244
    test_lengths = [2048, 5336, 8192, 9902, 13244]
    
    for S in test_lengths:
        print(f"\n  Testing S={S}...")
        Q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
        K = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
        V = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
        
        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        
        try:
            block_mask = create_block_mask(
                causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device
            )
            
            with torch.no_grad():
                out = flex_attention(Q, K, V, block_mask=block_mask)
            
            print(f"    ✓ S={S}: output shape {out.shape}")
        except Exception as e:
            print(f"    ✗ S={S}: FAILED with {type(e).__name__}: {e}")
            return False
    
    print(f"\n✓ PASS: All sequence lengths work")
    return True


def main():
    print("\n" + "="*70)
    print("  FlexAttention Correctness Tests (Phase A)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA not available, tests will run on CPU (slower)")
    
    results = []
    
    # Run all tests
    results.append(("Causal Correctness", test_causal_correctness()))
    results.append(("Gate Bias Correctness", test_gate_bias_correctness()))
    results.append(("Gradient Flow", test_gate_bias_gradients()))
    results.append(("Sequence Lengths", test_sequence_lengths()))
    
    # Summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n" + "="*70)
        print("  ✓ ALL TESTS PASSED")
        print("  Ready to proceed to Phase B (benchmarks)")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("  ✗ SOME TESTS FAILED")
        print("  Fix issues before proceeding to Phase B")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(main())
