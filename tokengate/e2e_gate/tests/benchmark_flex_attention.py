#!/usr/bin/env python3
"""Phase B: FlexAttention Performance Benchmarks

Measure actual speedup and memory savings for:
1. Phase 1: Causal attention on full sequence (biodex distribution)
2. Phase 2: Causal + gate bias on pruned sequence

Run:
    python -m tokengate.e2e_gate.tests.benchmark_flex_attention
"""

import json
import time
from typing import List, Dict

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Compile for performance
flex_attention = torch.compile(flex_attention)


def _make_causal_mask(
    q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Dense causal mask (baseline implementation)."""
    min_val = torch.finfo(dtype).min
    mask = torch.full((1, 1, q_len, kv_len), min_val, device=device, dtype=dtype)
    causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril()
    mask.masked_fill_(causal.unsqueeze(0).unsqueeze(0), 0.0)
    return mask


def benchmark_phase1_causal(
    seq_lengths: List[int] = [2048, 4096, 5336, 8192, 9902, 13244],
    n_trials: int = 10,
    warmup: int = 2,
) -> List[Dict]:
    """Benchmark Phase 1: causal attention on full sequence.
    
    Seq lengths chosen based on biodex distribution:
    - 5336: median
    - 9902: P95
    - 13244: max observed
    """
    print("\n" + "="*70)
    print("  PHASE 1: Causal Attention Benchmark")
    print("  (Full sequence, layers 0→L)")
    print("="*70)
    
    results = []
    B, H, D = 1, 32, 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    
    if device == 'cpu':
        print("\n⚠ WARNING: Running on CPU (CUDA not available). Results may not be representative.")
    
    for S in seq_lengths:
        print(f"\n{'─'*70}")
        print(f"Sequence Length: {S}")
        print(f"{'─'*70}")
        
        # Generate random inputs
        Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        K = torch.randn(B, H, S, D, device=device, dtype=dtype)
        V = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # ── Baseline: Dense mask + SDPA ──────────────────────────────────────
        print("\n[Baseline] Dense causal mask + SDPA")
        
        # Warmup
        for _ in range(warmup):
            mask = _make_causal_mask(S, S, device, dtype)
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
            if device == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
        
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        
        for _ in range(n_trials):
            mask = _make_causal_mask(S, S, device, dtype)
            out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        baseline_time = (time.time() - t0) / n_trials
        baseline_mem = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
        
        print(f"  Time:   {baseline_time*1000:.2f} ms")
        print(f"  Memory: {baseline_mem:.3f} GB")
        
        # ── FlexAttention ────────────────────────────────────────────────────
        print("\n[FlexAttention] BlockMask + compiled kernel")
        
        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        
        # Create BlockMask (do this once, cache in real usage)
        block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device)
        
        # Warmup
        for _ in range(warmup):
            _ = flex_attention(Q, K, V, block_mask=block_mask)
            if device == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
        
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        
        for _ in range(n_trials):
            out = flex_attention(Q, K, V, block_mask=block_mask)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        flex_time = (time.time() - t0) / n_trials
        flex_mem = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
        
        print(f"  Time:   {flex_time*1000:.2f} ms")
        print(f"  Memory: {flex_mem:.3f} GB")
        
        # ── Comparison ───────────────────────────────────────────────────────
        speedup = baseline_time / flex_time if flex_time > 0 else 0
        mem_saved = baseline_mem - flex_mem
        
        print(f"\n[Improvement]")
        print(f"  Speedup:     {speedup:.2f}×")
        print(f"  Memory saved: {mem_saved:.3f} GB ({mem_saved*1000:.0f} MB)")
        
        if speedup >= 1.5:
            print(f"  ✓ Target met: {speedup:.2f}× ≥ 1.5×")
        else:
            print(f"  ✗ Below target: {speedup:.2f}× < 1.5×")
        
        results.append({
            'seq_len': S,
            'baseline_time_ms': baseline_time * 1000,
            'flex_time_ms': flex_time * 1000,
            'baseline_mem_gb': baseline_mem,
            'flex_mem_gb': flex_mem,
            'speedup': speedup,
            'mem_saved_gb': mem_saved,
            'mem_saved_mb': mem_saved * 1000,
        })
    
    return results


def benchmark_phase2_gate_bias(
    pruned_lengths: List[int] = [256, 512, 1024, 1500, 2000],
    n_trials: int = 10,
    warmup: int = 2,
) -> List[Dict]:
    """Benchmark Phase 2: causal + gate bias on pruned sequence.
    
    Pruned lengths represent K + overhead after article pruning.
    Typical: K ∈ [50, 1000], overhead ~200 → n ∈ [250, 1200]
    """
    print("\n" + "="*70)
    print("  PHASE 2: Gate Bias Attention Benchmark")
    print("  (Pruned sequence, layers L+1→end)")
    print("="*70)
    
    results = []
    B, H, D = 1, 32, 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    
    for n in pruned_lengths:
        print(f"\n{'─'*70}")
        print(f"Pruned Length: {n}")
        print(f"{'─'*70}")
        
        # Generate random inputs
        Q = torch.randn(B, H, n, D, device=device, dtype=dtype)
        K = torch.randn(B, H, n, D, device=device, dtype=dtype)
        V = torch.randn(B, H, n, D, device=device, dtype=dtype)
        b = torch.randn(n, device=device, dtype=dtype)  # gate bias
        
        # ── Baseline: Dense mask + bias broadcast ────────────────────────────
        print("\n[Baseline] Dense causal mask + bias broadcast")
        
        # Warmup
        for _ in range(warmup):
            causal_mask = _make_causal_mask(n, n, device, dtype)
            bias_mask = b.view(1, 1, 1, n).expand(1, 1, n, n)
            combined = causal_mask + bias_mask
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=combined)
            if device == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
        
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        
        for _ in range(n_trials):
            causal_mask = _make_causal_mask(n, n, device, dtype)
            bias_mask = b.view(1, 1, 1, n).expand(1, 1, n, n)
            combined = causal_mask + bias_mask
            out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=combined)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        baseline_time = (time.time() - t0) / n_trials
        baseline_mem = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
        
        print(f"  Time:   {baseline_time*1000:.2f} ms")
        print(f"  Memory: {baseline_mem:.3f} GB")
        
        # ── FlexAttention ────────────────────────────────────────────────────
        print("\n[FlexAttention] BlockMask + score_mod")
        
        def causal(b_idx, h_idx, q_idx, kv_idx):
            return q_idx >= kv_idx
        
        def gate_bias_score_mod(score, b_idx, h_idx, q_idx, kv_idx):
            return score + b[kv_idx]
        
        block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=n, KV_LEN=n, device=device)
        
        # Warmup
        for _ in range(warmup):
            _ = flex_attention(Q, K, V, score_mod=gate_bias_score_mod, block_mask=block_mask)
            if device == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
        
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        
        for _ in range(n_trials):
            out = flex_attention(Q, K, V, score_mod=gate_bias_score_mod, block_mask=block_mask)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        flex_time = (time.time() - t0) / n_trials
        flex_mem = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
        
        print(f"  Time:   {flex_time*1000:.2f} ms")
        print(f"  Memory: {flex_mem:.3f} GB")
        
        # ── Comparison ───────────────────────────────────────────────────────
        speedup = baseline_time / flex_time if flex_time > 0 else 0
        mem_saved = baseline_mem - flex_mem
        
        print(f"\n[Improvement]")
        print(f"  Speedup:      {speedup:.2f}×")
        print(f"  Memory saved: {mem_saved:.3f} GB ({mem_saved*1000:.0f} MB)")
        
        if speedup >= 1.2:
            print(f"  ✓ Target met: {speedup:.2f}× ≥ 1.2×")
        else:
            print(f"  ⚠ Below target: {speedup:.2f}× < 1.2×")
        
        results.append({
            'pruned_len': n,
            'baseline_time_ms': baseline_time * 1000,
            'flex_time_ms': flex_time * 1000,
            'baseline_mem_gb': baseline_mem,
            'flex_mem_gb': flex_mem,
            'speedup': speedup,
            'mem_saved_gb': mem_saved,
            'mem_saved_mb': mem_saved * 1000,
        })
    
    return results


def print_summary(phase1_results: List[Dict], phase2_results: List[Dict]):
    """Print benchmark summary and decision recommendation."""
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY")
    print("="*70)
    
    # Phase 1 summary
    print("\n[Phase 1: Causal Attention]")
    print(f"{'Seq Len':<10} {'Baseline':<12} {'Flex':<12} {'Speedup':<10} {'Mem Saved':<12}")
    print("─" * 70)
    
    for r in phase1_results:
        print(f"{r['seq_len']:<10} "
              f"{r['baseline_time_ms']:>8.2f} ms  "
              f"{r['flex_time_ms']:>8.2f} ms  "
              f"{r['speedup']:>6.2f}×    "
              f"{r['mem_saved_mb']:>8.0f} MB")
    
    # Phase 2 summary
    print("\n[Phase 2: Gate Bias Attention]")
    print(f"{'Pruned Len':<12} {'Baseline':<12} {'Flex':<12} {'Speedup':<10} {'Mem Saved':<12}")
    print("─" * 70)
    
    for r in phase2_results:
        print(f"{r['pruned_len']:<12} "
              f"{r['baseline_time_ms']:>8.2f} ms  "
              f"{r['flex_time_ms']:>8.2f} ms  "
              f"{r['speedup']:>6.2f}×    "
              f"{r['mem_saved_mb']:>8.0f} MB")
    
    # Decision criteria
    print("\n" + "="*70)
    print("  DECISION CRITERIA")
    print("="*70)
    
    # Check median biodex case (S=5336)
    median_result = next((r for r in phase1_results if r['seq_len'] == 5336), None)
    p95_result = next((r for r in phase1_results if r['seq_len'] == 9902), None)
    
    criteria_met = []
    
    if median_result:
        print(f"\n✓ Biodex median (S=5336):")
        print(f"    Speedup: {median_result['speedup']:.2f}× {'✓' if median_result['speedup'] >= 1.5 else '✗'} (target: ≥1.5×)")
        print(f"    Memory:  {median_result['mem_saved_mb']:.0f} MB saved {'✓' if median_result['mem_saved_mb'] >= 500 else '✗'} (target: ≥500 MB)")
        criteria_met.append(median_result['speedup'] >= 1.5 and median_result['mem_saved_mb'] >= 500)
    
    if p95_result:
        print(f"\n✓ Biodex P95 (S=9902):")
        print(f"    Speedup: {p95_result['speedup']:.2f}× {'✓' if p95_result['speedup'] >= 1.8 else '✗'} (target: ≥1.8×)")
        print(f"    Memory:  {p95_result['mem_saved_gb']:.2f} GB saved")
        criteria_met.append(p95_result['speedup'] >= 1.8)
    
    # Overall recommendation
    print("\n" + "="*70)
    if all(criteria_met):
        print("  ✓ RECOMMENDATION: PROCEED TO PHASE C")
        print("  All performance targets met. FlexAttention shows clear wins.")
    elif any(criteria_met):
        print("  ⚠ RECOMMENDATION: REVIEW RESULTS")
        print("  Some targets met. Consider proceeding with caution.")
    else:
        print("  ✗ RECOMMENDATION: INVESTIGATE OR RECONSIDER")
        print("  Performance targets not met. Debug or reconsider approach.")
    print("="*70)


def main():
    print("\n" + "="*70)
    print("  FlexAttention Performance Benchmarks (Phase B)")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cpu':
        print("⚠ WARNING: CUDA not available. Benchmarks will run on CPU.")
        print("Results may not be representative of GPU performance.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run benchmarks
    print("\nRunning Phase 1 benchmarks (causal attention)...")
    phase1_results = benchmark_phase1_causal()
    
    print("\nRunning Phase 2 benchmarks (gate bias)...")
    phase2_results = benchmark_phase2_gate_bias()
    
    # Print summary
    print_summary(phase1_results, phase2_results)
    
    # Save results
    output_file = 'flex_attention_benchmarks.json'
    with open(output_file, 'w') as f:
        json.dump({
            'device': device,
            'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else None,
            'phase1_causal': phase1_results,
            'phase2_gate_bias': phase2_results,
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Exit code based on criteria
    median_result = next((r for r in phase1_results if r['seq_len'] == 5336), None)
    if median_result and median_result['speedup'] >= 1.5 and median_result['mem_saved_mb'] >= 500:
        return 0  # Success
    else:
        return 1  # Criteria not met


if __name__ == "__main__":
    exit(main())
