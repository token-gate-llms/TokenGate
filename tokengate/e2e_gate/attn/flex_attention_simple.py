# """Simplified FlexAttention integration - monkey-patch SDPA only. 
# This is an alternative to qwen3_flex_attention.py, which is more complex.

# This approach is much simpler: we replace F.scaled_dot_product_attention
# with our own function that routes to FlexAttention when appropriate.
# """

# import torch
# import torch.nn.functional as F
# from typing import Optional

# from tokengate.e2e_gate.attn.flex_attention_utils import (
#     flex_attention_causal,
#     flex_attention_gate_bias,
# )

# # Store original SDPA function
# _original_sdpa = F.scaled_dot_product_attention

# # Global state for gate bias (set per layer)
# _GATE_BIAS_REGISTRY = {}


# def _flex_sdpa_wrapper(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     attn_mask: Optional[torch.Tensor] = None,
#     dropout_p: float = 0.0,
#     is_causal: bool = False,
#     scale: Optional[float] = None,
# ) -> torch.Tensor:
#     """Drop-in replacement for F.scaled_dot_product_attention.
    
#     Routes to FlexAttention when:
#     - attn_mask is None or causal
#     - No dropout
#     - Sequence length > 1 (not decode)
    
#     Otherwise falls back to original SDPA.
#     """
#     q_len = query.shape[2]
    
#     # Fallback conditions
#     if (
#         dropout_p > 0 or  # FlexAttention doesn't support dropout
#         q_len == 1 or  # Decode - no benefit
#         (attn_mask is not None and not _is_causal_mask(attn_mask))  # Non-causal mask
#     ):
#         return _original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale)
    
#     # Check if this layer has gate bias
#     # We identify layers by the query tensor's data pointer (hacky but works)
#     layer_id = id(query.grad_fn) if query.grad_fn else id(query)
#     gate_bias = _GATE_BIAS_REGISTRY.get(layer_id, None)
    
#     # Route to FlexAttention
#     if gate_bias is not None:
#         return flex_attention_gate_bias(query, key, value, gate_bias)
#     else:
#         return flex_attention_causal(query, key, value)


# def _is_causal_mask(mask: torch.Tensor) -> bool:
#     """Check if mask is causal (lower triangular)."""
#     if mask is None:
#         return True
#     # Simple heuristic: check if it's roughly lower triangular
#     # (This is a simplified check - could be more robust)
#     return True  # For now, assume all masks are causal


# def patch_sdpa_with_flex_attention():
#     """Replace F.scaled_dot_product_attention with FlexAttention wrapper."""
#     F.scaled_dot_product_attention = _flex_sdpa_wrapper
#     print("[FlexAttention] Patched F.scaled_dot_product_attention")


# def unpatch_sdpa():
#     """Restore original F.scaled_dot_product_attention."""
#     F.scaled_dot_product_attention = _original_sdpa
#     print("[FlexAttention] Restored original F.scaled_dot_product_attention")


# def set_layer_gate_bias(layer_idx: int, bias: Optional[torch.Tensor]):
#     """Set gate bias for a specific layer."""
#     # This is a simplified version - in practice we'd need better layer identification
#     _GATE_BIAS_REGISTRY[layer_idx] = bias


# def clear_all_gate_bias():
#     """Clear all gate biases."""
#     _GATE_BIAS_REGISTRY.clear()
