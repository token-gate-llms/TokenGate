#!/usr/bin/env python3
"""Test FlexAttention integration with E2E gate training.

Quick sanity check that the patched model works correctly.

Run:
    python -m tokengate.e2e_gate.test_integration
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    patch_model_with_flex_attention,
    set_gate_bias_for_layers,
)
from tokengate.e2e_gate.attn.flex_attention_utils import (
    warmup_flex_attention,
    get_block_mask_cache_stats,
)


def test_model_loading():
    """Test that model loads and patches correctly."""
    print("\n" + "="*70)
    print("TEST 1: Model Loading & Patching")
    print("="*70)
    
    model_name = "Qwen/Qwen3-8B"
    print(f"Loading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Model loaded: {len(model.model.layers)} layers")
    print(f"Original attention class: {type(model.model.layers[0].self_attn).__name__}")
    
    # Patch model
    print("\nPatching model with FlexAttention...")
    model = patch_model_with_flex_attention(model)
    
    print(f"Patched attention class: {type(model.model.layers[0].self_attn).__name__}")
    
    # Verify all layers patched
    from tokengate.e2e_gate.attn.qwen3_flex_attention import Qwen3FlexAttention
    all_patched = all(
        isinstance(layer.self_attn, Qwen3FlexAttention)
        for layer in model.model.layers
    )
    
    if all_patched:
        print("✓ PASS: All layers successfully patched")
        return True
    else:
        print("✗ FAIL: Some layers not patched")
        return False


def test_forward_pass():
    """Test that forward pass works with FlexAttention."""
    print("\n" + "="*70)
    print("TEST 2: Forward Pass")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen3-8B"
    
    print(f"Loading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    model = patch_model_with_flex_attention(model)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Warmup
    print("\nWarming up FlexAttention...")
    warmup_flex_attention(torch.device(device))
    
    # Test input
    text = "The quick brown fox jumps over the lazy dog. " * 20  # ~200 tokens
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    print(f"\nRunning forward pass (seq_len={inputs.input_ids.shape[1]})...")
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {outputs.logits.shape}")
        
        # Check BlockMask cache
        stats = get_block_mask_cache_stats()
        print(f"\nBlockMask cache stats:")
        print(f"  Cached masks: {stats['num_cached']}")
        print(f"  Cached lengths: {stats['cached_lengths']}")
        print(f"  Memory: ~{stats['total_memory_mb']} MB")
        
        return True
    except Exception as e:
        print(f"✗ FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gate_bias():
    """Test that gate bias can be set and used."""
    print("\n" + "="*70)
    print("TEST 3: Gate Bias Setting")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen3-8B"
    
    print(f"Loading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    model = patch_model_with_flex_attention(model)
    model.eval()
    
    # Create dummy gate bias
    seq_len = 512
    gate_bias = torch.randn(seq_len, device=device, dtype=torch.bfloat16)
    
    print(f"\nSetting gate bias for layers 10-20...")
    set_gate_bias_for_layers(model, 10, 20, gate_bias)
    
    # Verify bias is set
    has_bias = model.model.layers[15].self_attn.gate_bias is not None
    no_bias = model.model.layers[5].self_attn.gate_bias is None
    
    if has_bias and no_bias:
        print("✓ PASS: Gate bias set correctly")
        print(f"  Layer 15 has bias: {has_bias}")
        print(f"  Layer 5 has no bias: {no_bias}")
        return True
    else:
        print("✗ FAIL: Gate bias not set correctly")
        return False


def test_generation():
    """Test that generation works with FlexAttention."""
    print("\n" + "="*70)
    print("TEST 4: Text Generation")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen3-8B"
    
    print(f"Loading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    model = patch_model_with_flex_attention(model)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Warmup
    warmup_flex_attention(torch.device(device))
    
    # Test generation
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"\nGenerating from prompt: '{prompt}'...")
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation successful")
        print(f"  Generated: {generated_text}")
        return True
    except Exception as e:
        print(f"✗ FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("  FlexAttention Integration Tests")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA not available, tests will run on CPU (slower)")
    
    results = []
    
    # Run tests
    results.append(("Model Loading & Patching", test_model_loading()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Gate Bias Setting", test_gate_bias()))
    results.append(("Text Generation", test_generation()))
    
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
        print("  FlexAttention integration is working correctly")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("  ✗ SOME TESTS FAILED")
        print("  Fix issues before proceeding to training")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(main())
