"""FlexAttention utilities for E2E gate.

Provides:
- BlockMask caching with sequence bucketing
- Helper functions for causal and gate bias attention
- Cache management and statistics
"""

import os

import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Optional, Callable, Dict, Tuple
from collections import OrderedDict

# Increase recompile limit.  The default is 8, which is exhausted by
# 4 score_mod variants × 2 grad modes (train vs eval/no_grad).  Set to 64
# to leave plenty of headroom.  Without this, dynamo falls back to eager
# mode after 8 recompiles, materialising the full S×S attention matrix and
# causing OOM on long sequences.
torch._dynamo.config.recompile_limit = 64

# Disable device specialization to prevent recompiles due to device changes
# This is safe because we manually ensure Q/K/V are on the same device
torch._dynamo.config.specialize_int = False

# Compile with dynamic=True to handle varying sequence lengths (n_kept changes each
# step during training due to random K) without hitting the recompile limit.
# fullgraph=False allows partial compilation even if some parts can't be compiled
flex_attention = torch.compile(flex_attention, dynamic=True, fullgraph=False)

# Cache BlockMasks to avoid recomputation.  LRU-BOUNDED: non-bucketed
# (exact-length) keys — e.g. decode / gated kept-lengths that vary per sample —
# would otherwise grow this dict unboundedly (~8MB per BlockMask), causing a
# slow ~80GB GPU-memory creep over training that empty_cache CANNOT free (the
# masks are referenced here, not in the allocator's free pool).  This was the
# root cause of run4/6's OOM regression.
_BLOCK_MASK_CACHE: "OrderedDict[Tuple[int, torch.device], 'BlockMask']" = OrderedDict()
_BLOCK_MASK_CACHE_MAX = 128  # ~1GB cap; hot bucketed masks stay, cold one-offs evict
_BUCKET_SIZE = 512  # Round seq_len to nearest multiple for cache efficiency
# Only build the block mask via the compiled (O(n)-memory) path above this
# length.  Below it the eager path materializes an n²·8-byte index grid that is
# small (16384²·8 ≈ 2 GiB) and is the battle-tested code path; the compiled
# path triggered a cudaErrorIllegalAddress on short, highly-varied Phase-2
# lengths (biodex K_LOW=10).  See get_causal_block_mask for the full rationale.
_COMPILE_BLOCK_MASK_MIN_LEN = 16384


def _round_up_seq_len(seq_len: int, bucket_size: int = _BUCKET_SIZE) -> int:
    """Round sequence length up to nearest bucket for cache efficiency.
    
    Example: seq_len=5336 → 5632 (bucket_size=512)
    
    This reduces BlockMask cache misses for variable-length biodex articles.
    Memory overhead is minimal (~8MB per BlockMask at S=32k).
    """
    return ((seq_len + bucket_size - 1) // bucket_size) * bucket_size


def get_causal_block_mask(
    seq_len: int,
    device: torch.device,
    use_bucketing: bool = True,
) -> 'BlockMask':
    """Get or create cached causal BlockMask.
    
    Args:
        seq_len: Actual sequence length
        device: CUDA device
        use_bucketing: If True, round seq_len up to nearest bucket for caching
                      (disabled for very small sequences < 128)
    
    Returns:
        BlockMask for causal attention (may be slightly larger than seq_len if bucketed)
    """
    # Disable bucketing for very small sequences (e.g., during decode)
    # to avoid cache mismatches
    if use_bucketing and seq_len >= 128:
        bucketed_len = _round_up_seq_len(seq_len)
    else:
        bucketed_len = seq_len
    
    key = (bucketed_len, device)
    _cached = _BLOCK_MASK_CACHE.get(key)
    if _cached is not None:
        _BLOCK_MASK_CACHE.move_to_end(key)  # LRU: mark most-recently used
        return _cached

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # Broadcast over batch and heads (FAQ: saves memory).
    # _compile=True is CRITICAL for long sequences: without it,
    # create_block_mask materializes the full (Q_LEN, KV_LEN) index grid to
    # compute block sparsity — an int64 [n, n] tensor = n²·8 bytes.  At
    # 120k that is 120000²·8 ≈ 107 GiB and OOMs a single H100).  With _compile=True the mask_mod
    # is evaluated block-by-block (O(n) memory), so 128k Phase-1 attention
    # fits.
    #
    # BUT _compile=True is only used above _COMPILE_BLOCK_MASK_MIN_LEN.  The
    # compiled block-mask path interacts badly with the rest of the Phase-2
    # forward on the short, highly-varied sequence lengths produced by small
    # training K (biodex K_LOW=10 → n_kept ~ 10–5000): it triggered a
    # reproducible cudaErrorIllegalAddress around step ~120–190 (surfacing at
    # the create_block_mask kernel launch — the sticky-error catch point).
    # The eager (_compile=False) path is the battle-tested one used by every
    # pre-mixed-K run and never crashed.  Below the threshold the materialized
    # n²·8 grid is small (16384²·8 ≈ 2 GiB), so eager is both safe and cheap;
    # only genuinely long sequences pay for / need the compiled path.
    _use_compile = bucketed_len > _COMPILE_BLOCK_MASK_MIN_LEN
    _BLOCK_MASK_CACHE[key] = create_block_mask(
        causal, B=None, H=None, Q_LEN=bucketed_len, KV_LEN=bucketed_len,
        device=device, _compile=_use_compile,
    )
    if len(_BLOCK_MASK_CACHE) > _BLOCK_MASK_CACHE_MAX:
        _BLOCK_MASK_CACHE.popitem(last=False)  # evict oldest → bounds GPU memory
    return _BLOCK_MASK_CACHE[key]


# ── Stable score_mod for gate bias ────────────────────────────────────────────
# We use module-level function objects that read from a shared state rather than
# creating new closures each call.  torch.compile with dynamic=True compiles each
# function once and reuses the kernel across varying n_kept shapes.
#
# Two score_mod variants:
#   _gate_bias_score_mod         — scalar bias: (S,) or (B, S)
#   _gate_bias_per_head_score_mod — per-head bias: (S, H) or (B, S, H)
#
# flex_attention_gate_bias selects the appropriate variant via per_head=True/False.
# Each variant is compiled into its own kernel; both are warmed up at startup.

class _GateBiasState:
    """Holds the current gate_bias tensor for the score_mod closures.

    Attributes:
        bias: Gate bias tensor — shape depends on mode:
            Scalar:   (S,) single-sample  |  (B, S) batched
            Per-head: (S, H) single-sample  |  (B, S, H) batched

    Note: attn_scale is applied by the caller before this call — both Q and
    gate_bias are pre-scaled so that the score_mod remains a stable compiled
    function with no captured grad tensors.
    """
    bias: Optional[torch.Tensor] = None

_gate_bias_state = _GateBiasState()


# ── Segment-aware state for packed-sequence training ─────────────────────────
# When training packed sequences (multiple logical samples concatenated into
# one batch element), attention must be blocked across segments as well as
# causal within a segment.  We use the same module-level-state pattern so the
# score_mod functions stay compile-stable.

class _SegmentIdsState:
    """Holds the current segment-id tensor for segment-aware score / mask mods.

    Attributes:
        sids: (S,) int32 tensor (single-sample) OR (B, S) int32 (batched).
              Tokens with sids[i] == sids[j] are in the same segment.
              The convention used by packed_sequences.py is: 0 = padding,
              1..N = segment indices (so sids == 0 correctly fails the
              same_segment test against any other position).
    """
    sids: Optional[torch.Tensor] = None

_segment_ids_state = _SegmentIdsState()


def _gate_bias_score_mod(score, b, h, q_idx, kv_idx):
    """Scalar bias score_mod: (S,) or (B, S).

    Single stable function object reused across all scalar-bias calls.
    torch.compile specialises on bias.dim() (a Python int) and emits two
    kernels (1-D and 2-D) that are each reused without recompilation.
    """
    if _gate_bias_state.bias.dim() == 1:
        return score + _gate_bias_state.bias[kv_idx]
    else:
        return score + _gate_bias_state.bias[b, kv_idx]


def _gate_bias_per_head_score_mod(score, b, h, q_idx, kv_idx):
    """Per-head bias score_mod: (S, H) or (B, S, H).

    Single stable function object reused across all per-head calls.
    The head index ``h`` maps directly to query heads (FlexAttention always
    passes the query-head index, and K/V are already repeated for GQA before
    flex_attention is called).
    """
    if _gate_bias_state.bias.dim() == 2:
        return score + _gate_bias_state.bias[kv_idx, h]
    else:
        return score + _gate_bias_state.bias[b, kv_idx, h]


def _gqa(Q: torch.Tensor, K: torch.Tensor) -> bool:
    """Grouped-query attention is done inside the kernel: callers pass K/V at
    their own head count and flex_attention broadcasts them over the query
    heads (enable_gqa), instead of materialising repeat_interleave copies."""
    if K.shape[1] == Q.shape[1]:
        return False
    if Q.shape[1] % K.shape[1] != 0:
        raise ValueError(f"query heads {Q.shape[1]} not a multiple of kv heads {K.shape[1]}")
    return True


def flex_attention_causal(
    Q: torch.Tensor,  # (B, H, S, D)
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """FlexAttention with causal masking only (Phase 1)."""
    # Ensure Q/K/V are on same device (critical for torch.compile)
    if K.device != Q.device or V.device != Q.device:
        raise RuntimeError(
            f"Q/K/V device mismatch in flex_attention_causal: "
            f"Q={Q.device}, K={K.device}, V={V.device}. "
            f"This will cause torch.compile fallback and OOM!"
        )
    
    q_len = Q.shape[2]
    kv_len = K.shape[2]
    device = Q.device
    block_mask = get_causal_block_mask(max(q_len, kv_len), device, use_bucketing=False)
    return flex_attention(Q, K, V, block_mask=block_mask, enable_gqa=_gqa(Q, K))


def flex_attention_gate_bias(
    Q: torch.Tensor,  # (B, H, S, D)
    K: torch.Tensor,
    V: torch.Tensor,
    gate_bias: torch.Tensor,
    per_head: bool = False,
    padding_mask: Optional[torch.Tensor] = None,  # Deprecated, use gate_bias=-1e9 for padding
) -> torch.Tensor:
    """FlexAttention with causal + gate bias (Phase 2).

    Supports two bias modes selected by ``per_head``:

    Scalar mode (per_head=False):
        gate_bias shape: (S,) single-sample  |  (B, S) batched
        The same bias value is added to every head's attention logits for
        token j: score += gate_bias[j]

    Per-head mode (per_head=True):
        gate_bias shape: (S, H) single-sample  |  (B, S, H) batched
        Each head h gets its own bias for token j: score += gate_bias[j, h]
        H must equal the number of query heads seen by flex_attention
        (= model.config.num_attention_heads after GQA expansion).

    Uses stable module-level score_mod functions to avoid recompilation.
    The bias is passed via a module-level state object so torch.compile sees
    the same function identity on every call.

    To implement attn_scale * (QK^T/sqrt(d) + b), callers should pre-scale
    both Q and gate_bias before this call.  This keeps learnable parameters
    in the regular autograd graph, avoiding FlexAttention inductor backward
    constraints on captured grad tensors.

    Args:
        Q, K, V: Attention tensors (B, H, S, D)
        gate_bias: Gate bias tensor, pre-scaled by attn_scale if applicable.
        per_head: If True, use the per-head score_mod (gate_bias has an H dim).
        padding_mask: Deprecated — encode padding as gate_bias=-1e9 instead.
    """
    if K.device != Q.device or V.device != Q.device:
        raise RuntimeError(
            f"Q/K/V device mismatch in flex_attention_gate_bias: "
            f"Q={Q.device}, K={K.device}, V={V.device}. "
            f"This will cause torch.compile fallback and OOM!"
        )

    q_len = Q.shape[2]
    kv_len = K.shape[2]
    device = Q.device

    _gate_bias_state.bias = gate_bias.to(device)
    score_mod = _gate_bias_per_head_score_mod if per_head else _gate_bias_score_mod

    block_mask = get_causal_block_mask(max(q_len, kv_len), device, use_bucketing=False)
    return flex_attention(Q, K, V, score_mod=score_mod, block_mask=block_mask,
                          enable_gqa=_gqa(Q, K))


# ── Segment-causal BlockMask + helpers (packed-sequence training) ──────────
# Same caching contract as the causal BlockMask, but the mask_mod is
# (causal AND same-segment).  The captured `_segment_ids_state.sids` tensor
# defines the segment topology; the BlockMask is keyed on (bucketed_seq_len,
# device, id(sids)) so we get fresh masks when the topology changes.

def _segment_causal_mask_mod_2d(b, h, q_idx, kv_idx):
    """mask_mod for (S,) segment_ids (single-sample / broadcast over batch)."""
    sids = _segment_ids_state.sids
    return (sids[q_idx] == sids[kv_idx]) & (q_idx >= kv_idx)


def _segment_causal_mask_mod_3d(b, h, q_idx, kv_idx):
    """mask_mod for (B, S) segment_ids (per-batch-element topology)."""
    sids = _segment_ids_state.sids
    return (sids[b, q_idx] == sids[b, kv_idx]) & (q_idx >= kv_idx)


def get_segment_causal_block_mask(
    seq_len: int,
    device: torch.device,
    segment_ids: torch.Tensor,
) -> 'BlockMask':
    """Build a segment-causal BlockMask.

    Args
    ----
    seq_len:      length of the sequence
    device:       CUDA device
    segment_ids:  (S,) or (B, S) int32 tensor identifying segments.  Must be
                  set on `_segment_ids_state.sids` before calling.

    Returns
    -------
    A BlockMask that allows attention only when (q and kv are in the same
    segment) AND (q >= kv).  No bucketing — each segment topology demands
    its own mask, and BlockMask construction is cheap (< 1 ms typically).
    """
    _segment_ids_state.sids = segment_ids.to(device)
    if segment_ids.dim() == 1:
        mask_mod = _segment_causal_mask_mod_2d
        B = None
    else:
        mask_mod = _segment_causal_mask_mod_3d
        B = segment_ids.shape[0]
    # _compile=True avoids the O(n²) index-grid materialization at long
    # seq_len (see get_causal_block_mask).  Packed-sequence training would
    # otherwise OOM at the same lengths as the causal path.
    return create_block_mask(
        mask_mod, B=B, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
        _compile=True,
    )


def flex_attention_segment_causal(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """FlexAttention with segment-causal masking (Phase 1 for packed seqs)."""
    if K.device != Q.device or V.device != Q.device:
        raise RuntimeError(
            f"Q/K/V device mismatch in flex_attention_segment_causal: "
            f"Q={Q.device}, K={K.device}, V={V.device}."
        )
    block_mask = get_segment_causal_block_mask(Q.shape[2], Q.device, segment_ids)
    return flex_attention(Q, K, V, block_mask=block_mask, enable_gqa=_gqa(Q, K))


def flex_attention_segment_gate_bias(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    gate_bias: torch.Tensor,
    segment_ids: torch.Tensor,
    per_head: bool = False,
) -> torch.Tensor:
    """FlexAttention with segment-causal + gate bias (Phase 2 for packed seqs).

    The block-mask enforces the segment-causal structure (cross-segment
    attention blocked, causal within segment).  The score_mod adds the
    per-token gate bias on top; reuses the existing stable score_mod
    functions in `_gate_bias_state` so the compile cache is not duplicated.
    """
    if K.device != Q.device or V.device != Q.device:
        raise RuntimeError(
            f"Q/K/V device mismatch in flex_attention_segment_gate_bias: "
            f"Q={Q.device}, K={K.device}, V={V.device}."
        )
    block_mask = get_segment_causal_block_mask(Q.shape[2], Q.device, segment_ids)
    _gate_bias_state.bias = gate_bias.to(Q.device)
    score_mod = _gate_bias_per_head_score_mod if per_head else _gate_bias_score_mod
    return flex_attention(Q, K, V, score_mod=score_mod, block_mask=block_mask,
                          enable_gqa=_gqa(Q, K))


# ── Bias-free kernels for the two attention forms the gate needs ─────────────
# Phase 1 is plain causal attention; Phase 2 is causal attention with an
# additive per-(token, head) bias.  FlexAttention carries the bias as a
# score_mod, but a score_mod that loads from a tensor per kv tile costs its
# kernel 1.7-2.8x, and on a b200 cuDNN's fused
# attention runs the bias-free problem 3-4x faster than flex or FA2 to begin
# with.  cuDNN takes no bias, so Phase 2 uses the identity
#
#   softmax_j(s_j + b_j) . v_j  =  (sum_j p_j w_j v_j) / (sum_j p_j w_j),
#   p = softmax(s),  w_j = exp(b_j - max_j b_j)
#
# one bias-free attention over q, k whose values are v' = [w*v | w | 0-pad]:
# the first D output columns are the numerator, column D the denominator.
# cuDNN takes d_v = 136 with d_qk = 128 through the aten op directly (the
# public dispatcher refuses d > 128): 0.49 ms at 8k tokens against 0.24 for
# plain causal, 0.66 for two d_v = 128 calls and 1.67 for flex+bias
# (probe_cudnn_bias.py, probe_vweight.py).  Padded slots (bias -1e9) get
# w = 0 and drop out exactly; the per-head max shift keeps w in (0, 1].  The
# division is the extra rounding this form pays: numerator and denominator
# are bf16 kernel outputs, so the result carries ~2 bf16 ulps of per-row
# scale noise where the score_mod path carries one.

CAUSAL_BACKENDS = ("flex", "cudnn", "flash", "fa4", "auto")
BIAS_BACKENDS = ("flex", "vweight-cudnn", "aug-flash", "aug-fa4", "fa2-bias")

# "auto" picks the plain causal kernel by context length: on a b200 cuDNN's
# fused attention wins below ~12k tokens and FlashAttention's CuTe kernel
# above (probe_fa4.py: 0.239 vs 0.265 ms at 8k, 0.889 vs 0.828 at 16k, 1.77 vs
# 1.43 at 22k).  The crossover is measured, not assumed: set_attn_crossover
# must be called before the first "auto" call.
_ATTN_CROSSOVER = None


def set_attn_crossover(n_tokens: int) -> None:
    global _ATTN_CROSSOVER
    _ATTN_CROSSOVER = int(n_tokens)


_fa4_available = None
_fallback_warned = set()


def fa4_available() -> bool:
    global _fa4_available
    if _fa4_available is None:
        _fa4_available = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
    return _fa4_available


def resolve_backends(causal: str, bias: str) -> tuple[str, str]:
    from kernels.fa2bias.keybias import available as fa2_bias_available
    swaps = {}  # name -> (replacement, reason)
    if not fa4_available():
        why = (f"needs FlashAttention's Blackwell kernel; "
               f"{torch.cuda.get_device_name() if torch.cuda.is_available() else 'this machine'} has none")
        swaps["auto"] = ("cudnn", why)
        swaps["fa4"] = ("cudnn", why)
        swaps["aug-fa4"] = ("fa2-bias" if fa2_bias_available() else "aug-flash", why)
    if not fa2_bias_available():
        # Inference-only fallback: the fa2 training step fails fast instead
        # (pack_phase2 has no flex equivalent; a silent swap would change the run).
        swaps["fa2-bias"] = ("aug-flash", "needs the key-bias kernel, which is not built "
                             "on this machine (python kernels/fa2bias/build.py)")
    out = []
    for name in (causal, bias):
        to, why = swaps.get(name, (None, None))
        if to is not None and name not in _fallback_warned:
            _fallback_warned.add(name)
            warnings.warn(f"attention backend {name!r} {why}, using {to!r}", stacklevel=3)
        out.append(to or name)
    return out[0], out[1]


def _cudnn_causal(Q, K, V):
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        return F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=_gqa(Q, K))


def _flash_causal(Q, K, V):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=_gqa(Q, K))


def _fa4_causal(Q, K, V):
    # (B, heads, S, d) views of the kernel's (B, S, heads, d) layout, no copies.
    o = torch.ops.recursive_kvs.fa4_causal(Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                                           Q.shape[-1] ** -0.5)
    return o.transpose(1, 2)


def attention_causal(Q, K, V, backend: str) -> torch.Tensor:
    """Causal attention (Phase 1, or any layer with no gate bias set)."""
    if backend == "auto":
        if _ATTN_CROSSOVER is None:
            raise RuntimeError("causal backend 'auto' needs set_attn_crossover(n_tokens) first "
                               "(--attn-crossover in the bench scripts)")
        backend = "fa4" if Q.shape[2] >= _ATTN_CROSSOVER else "cudnn"
    if backend == "flex":
        return flex_attention_causal(Q, K, V)
    if backend == "cudnn":
        return _cudnn_causal(Q, K, V)
    if backend == "flash":
        return _flash_causal(Q, K, V)
    if backend == "fa4":
        return _fa4_causal(Q, K, V)
    raise ValueError(f"causal attention backend {backend!r} not in {CAUSAL_BACKENDS}")


def attention_gate_bias(Q, K, V, gate_bias, per_head: bool, backend: str) -> torch.Tensor:
    """Causal attention plus the gate's additive bias (Phase 2).

    gate_bias: (S,) / (B, S) scalar-per-token, or (S, H) / (B, S, H) per head,
    already scaled by whatever attn_scale / ssmax factor the caller applies.
    """
    if backend == "flex":
        return flex_attention_gate_bias(Q, K, V, gate_bias, per_head=per_head)
    if backend == "vweight-cudnn":
        return _vweight_attention(Q, K, V, gate_bias, per_head, _cudnn_causal_vw)
    if backend == "aug-flash":
        return _aug_flash_attention(Q, K, V, gate_bias, per_head)
    if backend == "aug-fa4":
        return _aug_fa4_attention(Q, K, V, gate_bias, per_head)
    if backend == "fa2-bias":
        return _fa2_bias_attention(Q, K, V, gate_bias, per_head)
    raise ValueError(f"gate-bias attention backend {backend!r} not in {BIAS_BACKENDS}")


def _fa2_bias_attention(Q, K, V, gate_bias, per_head: bool) -> torch.Tensor:
    from kernels.fa2bias.keybias import flash_attn_key_bias
    B, H, S, D = Q.shape
    b = _bias_bhs(gate_bias, per_head, B, H, S)
    o = flash_attn_key_bias(Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), b, D ** -0.5, causal=True)
    return o.transpose(1, 2)


# (cu_seqlens int32, max_len, lens) of the packed targets. Read here, inside the
# compiler-disabled call, and not passed through the compiled layer: the lengths change every
# step and as arguments they would be guarded constants, one recompile per step.
PACKED_CU = None


@torch.compiler.disable
def fa2_bias_attention_packed(Q, K, V, gate_bias, per_head: bool) -> torch.Tensor:
    """Several Phase-2 targets concatenated along the sequence axis of ONE batch row, one
    variable-length launch: causal within each target, each key's bias from its own target."""
    from kernels.fa2bias.keybias import flash_attn_key_bias_varlen
    cu_seqlens, max_len, lens = PACKED_CU
    B, H, S, D = Q.shape
    if B != 1 or S != sum(lens):
        raise ValueError(f"packed targets expect one batch row of {sum(lens)} tokens, got {tuple(Q.shape)}")
    b = _bias_bhs(gate_bias, per_head, 1, H, S)[0]
    pb = torch.stack([torch.nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in b.split(lens, dim=-1)])
    o = flash_attn_key_bias_varlen(Q[0].transpose(0, 1), K[0].transpose(0, 1), V[0].transpose(0, 1),
                                   pb, cu_seqlens, max_len, D ** -0.5, causal=True)
    return o.transpose(0, 1).unsqueeze(0)


# ── FlashAttention's CuTe-DSL Blackwell forward ("FA4") with the bias in the MMA ──
# flash_attn/cute/flash_fwd_sm100.py (flash-attn 2.8.3 sdist) runs the plain causal
# problem within 10% of cuDNN on a b200 and pads head dims to a multiple of 16, so
# the bias can ride in extra q/k columns: q' = [q | e_g | 0], k' = [k | b_g sqrt(d) | 0]
# with g the query head's index in its GQA group.  head_dim 192 is a supported
# tile (136/160 fail to launch): the QK MMA grows 1.5x, the softmax stage and PV
# are untouched, and the kernel runs at 0.288 ms vs 0.238 for cuDNN's plain causal
# at 8k tokens -- against 0.49 for the d_v=136 cuDNN
# form.  One rounding, like flex+bias.  Loading the bias in the softmax stage
# instead (a patched kernel) cost 1.9x: those warps are the pipeline's bottleneck.
# Staged by kernels/bench/fa4_setup.sh under ~/.cache/tokengate/fa4/site41
# (override with TOKENGATE_FA4_SITE); imported on first use.
_FA4_SITE = os.environ.get("TOKENGATE_FA4_SITE",
                           os.path.expanduser("~/.cache/tokengate/fa4/site41"))
# 160: the kernel's own tiles are 64/96/128/192/256; 144 and 160 launch once its
# head_dim-192 shared-memory scheme is enabled for them (two lines, applied by
# fa4_setup.sh).  Per layer at 22k tokens: plain 1.42 ms, 160 1.59, 192 1.64,
# 144 1.65 (a 32-byte swizzle atom); at 4k: 0.127 / 0.129 / 0.134 / 0.130.
_FA4_QK_DIM = 160
_FA4_STOCK_DIMS = (64, 96, 128, 192, 256)
_FA4_PATCH_MARK = "self.head_dim_padded in (144, 160, 192)"
_fa4_fwd = None


def _fa4():
    global _fa4_fwd
    if _fa4_fwd is None:
        import site
        if not os.path.isdir(_FA4_SITE):
            raise RuntimeError(f"aug-fa4 needs FlashAttention's CuTe kernels staged at {_FA4_SITE} "
                               f"(run kernels/bench/fa4_setup.sh)")
        site.addsitedir(_FA4_SITE)
        from flash_attn.cute.interface import _flash_attn_fwd
        if _FA4_QK_DIM not in _FA4_STOCK_DIMS:
            import flash_attn.cute.flash_fwd_sm100 as _k
            with open(_k.__file__) as f:
                if _FA4_PATCH_MARK not in f.read():
                    raise RuntimeError(f"aug-fa4 at head_dim {_FA4_QK_DIM} needs the staged kernel's "
                                       f"shared-memory scheme enabled for it ({_k.__file__} lacks "
                                       f"{_FA4_PATCH_MARK!r}); rerun kernels/bench/fa4_setup.sh")
        _fa4_fwd = _flash_attn_fwd
    return _fa4_fwd


_fa4_kernel = {}


class _NoSyncDLPack:
    """A tensor's DLPack face with the stream synchronisation left out.  cutlass
    calls __dlpack__() with no stream, which torch reads as the legacy default
    stream and answers by making that stream wait on the current one: an
    event and a wait per tensor per call, and under CUDA-graph capture an
    error (the legacy stream may not depend on a capturing stream).  The
    kernel is launched on the current stream anyway, so stream=-1 (no sync)
    is the right contract."""
    __slots__ = ("_t",)

    def __init__(self, t):
        self._t = t

    def __dlpack__(self, stream=None, **kwargs):
        return self._t.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._t.__dlpack_device__()


def _fa4_fast(q, k, v, scale):
    """The interface's per-call work without its asserts and re-derivations: the
    wrapper costs ~60 us of host time per call (probe_fa4_cpu.py) against ~9 for
    the baseline's SDPA call, and at short contexts the GPU waits for it. After
    one full call has compiled and cached the kernel for this configuration, the
    compiled callable is invoked directly with the same conversions."""
    G = q.shape[2] // k.shape[2]
    key = (q.dtype, q.shape[-1], v.shape[-1], G)
    fn = _fa4_kernel.get(key)
    if fn is None:
        fwd = _fa4()
        out, _ = fwd(q, k, v, softmax_scale=scale, causal=True)
        # The interface's compile_key for exactly this call: causal (which sets
        # window_size_right = 0), no softcap / lse / varlen / paging / sink /
        # bias, its default tile sizes, pack_gqa for a GQA group on sm100.
        from flash_attn.cute.interface import torch2cute_dtype_map
        cc = torch.cuda.get_device_capability()[0]
        pack_gqa = G > 1 and (cc != 10 or 128 % G == 0)
        ckey = (torch2cute_dtype_map[q.dtype], q.shape[-1], v.shape[-1], G, True, False,
                True, True, True, True, True, False, False, True, False, False,
                128, 128, 384, pack_gqa, cc)
        if ckey not in fwd.compile_cache:
            raise RuntimeError(f"fa4 fast path: no compiled kernel under {ckey}; cache has {list(fwd.compile_cache)}")
        _fa4_kernel[key] = fwd.compile_cache[ckey]
        return out
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda
    out = torch.empty(q.shape[0], q.shape[1], q.shape[2], v.shape[-1], dtype=q.dtype, device=q.device)
    qt, kt, vt, ot = [from_dlpack(_NoSyncDLPack(t), assumed_align=16).mark_layout_dynamic(leading_dim=3)
                      for t in (q, k, v, out)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # Exactly the interface's call: no lse / varlen / paging / softcap / sink,
    # window_size_left None and window_size_right 0 (what causal=True sets; the
    # compiled function is specialised on that, None in its place faults).
    fn(qt, kt, vt, ot, None, scale, stream, None, None, None, None, None, None, None, 0, None)
    return out


@torch.library.custom_op("recursive_kvs::fa4_causal", mutates_args=())
def _fa4_causal_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """q (B, S, H, Dq), k (B, S, Hkv, Dq), v (B, S, Hkv, Dv), causal; returns (B, S, H, Dv)."""
    return _fa4_fast(q, k, v, scale)


@_fa4_causal_op.register_fake
def _fa4_causal_fake(q, k, v, scale):
    return q.new_empty(q.shape[0], q.shape[1], q.shape[2], v.shape[-1])


def _aug_fa4_attention(Q, K, V, gate_bias, per_head: bool) -> torch.Tensor:
    B, H, S, D = Q.shape
    Hkv = K.shape[1]
    G = H // Hkv
    E = _FA4_QK_DIM - D
    if E < G or E <= 0:
        raise ValueError(f"head_dim {D}: no room for a {G}-wide GQA group in {_FA4_QK_DIM} columns")
    b = _bias_bhs(gate_bias, per_head, B, H, S)                       # (B, H, S) fp32
    scale = D ** -0.5
    # FA layout (B, S, heads, dim); Q/K/V arrive as (B, heads, S, dim) views of it.
    qt, kt, vt = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
    onehot = torch.zeros(H, E, dtype=Q.dtype, device=Q.device)
    onehot[torch.arange(H, device=Q.device), torch.arange(H, device=Q.device) % G] = 1
    qa = torch.cat([qt, onehot.view(1, 1, H, E).expand(B, S, H, E)], dim=-1)
    kb = (b.view(B, Hkv, G, S).permute(0, 3, 1, 2) / scale).to(K.dtype)   # (B, S, Hkv, G)
    ka = torch.cat([kt, kb, kb.new_zeros(B, S, Hkv, E - G)], dim=-1)
    o = torch.ops.recursive_kvs.fa4_causal(qa, ka, vt, scale)         # (B, S, H, D)
    return o.transpose(1, 2)


def _bias_bhs(gate_bias, per_head, B, H, S):
    """gate_bias in any of its layouts -> (B, H, S) fp32."""
    b = gate_bias
    if b.dim() == 1 or (b.dim() == 2 and not per_head):
        b = b.reshape(1, -1, 1) if b.dim() == 1 else b.unsqueeze(-1)   # (B|1, S, 1)
    elif b.dim() == 2:
        b = b.unsqueeze(0)                                              # (1, S, H)
    if b.shape[-2] != S:
        raise ValueError(f"gate_bias covers {b.shape[-2]} tokens, K/V {S}")
    return b.float().permute(0, 2, 1).expand(B, H, S)


def _aug_flash_attention(Q, K, V, gate_bias, per_head: bool) -> torch.Tensor:
    """The bias in the logits themselves, for kernels that need d_q = d_k = d_v:
    q' = [q | e_g], k' = [k | b_g * sqrt(d)] with g the query head's index in its
    GQA group, so q'.k' * (1/sqrt(d)) = q.k/sqrt(d) + b.  Eight extra columns:
    the bias split into a bf16 high part and its bf16 residual (G columns each,
    summed in the kernel's fp32 accumulator, so the bias enters with ~16 bits
    instead of 8), zeros in the rest and in v', output sliced back to d.
    torch's flash kernel: 1.32 ms at 8k tokens vs flex+bias 1.67 on a b200
    (probe_attn.py aug/sdpa/flash); it keeps K/V at their own head count.
    Meant for GPUs where cuDNN's fused attention is unavailable or slow."""
    B, H, S, D = Q.shape
    Hkv = K.shape[1]
    G = H // Hkv
    if 2 * G > _VW_PAD:
        raise ValueError(f"GQA group {G}: hi+lo bias columns need {2 * G} of the {_VW_PAD} spare columns")
    b = _bias_bhs(gate_bias, per_head, B, H, S)                       # (B, H, S)
    scale = D ** -0.5
    t = (b.view(B, Hkv, G, S) / scale).permute(0, 1, 3, 2)             # (B, Hkv, S, G) fp32
    t_hi = t.to(K.dtype)
    t_lo = (t - t_hi).to(K.dtype)
    kx = torch.zeros(B, Hkv, S, D + _VW_PAD, dtype=K.dtype, device=K.device)
    kx[..., :D] = K
    kx[..., D:D + G] = t_hi
    kx[..., D + G:D + 2 * G] = t_lo
    qx = torch.zeros(B, H, S, D + _VW_PAD, dtype=Q.dtype, device=Q.device)
    qx[..., :D] = Q
    onehot = torch.zeros(H, _VW_PAD, dtype=Q.dtype, device=Q.device)
    _h = torch.arange(H, device=Q.device)
    onehot[_h, _h % G] = 1
    onehot[_h, G + _h % G] = 1
    qx[..., D:] = onehot.view(1, H, 1, _VW_PAD)
    vx = F.pad(V, (0, _VW_PAD))
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        o = F.scaled_dot_product_attention(qx, kx, vx, is_causal=True, scale=scale,
                                           enable_gqa=_gqa(qx, kx))
    return o[..., :D]


_VW_PAD = 8   # v' columns past D: w in the first, zeros after (d_v must be a multiple of 8)


# A custom op around the aten cuDNN kernel so torch.compile can trace it: the
# aten op's meta lays its output out like the query, but for d_v != d_qk the
# real kernel returns a contiguous (B, H, S, d_v) tensor, and inductor's stride
# assertion on the fallback fails.  This op's contract is "contiguous output",
# in eager and in the fake kernel alike.
@torch.library.custom_op("recursive_kvs::cudnn_causal_vw", mutates_args=())
def _cudnn_causal_vw_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = torch.ops.aten._scaled_dot_product_cudnn_attention(
        q, k, v, None, False, 0.0, True, False, scale=q.shape[-1] ** -0.5)[0]
    return out.contiguous()


@_cudnn_causal_vw_op.register_fake
def _cudnn_causal_vw_fake(q, k, v):
    return q.new_empty(q.shape[0], q.shape[1], q.shape[2], v.shape[-1])


def _vweight_attention(Q, K, V, gate_bias, per_head: bool, attn) -> torch.Tensor:
    B, H, S, D = Q.shape
    Hkv = K.shape[1]
    if V.shape[1] != Hkv:
        raise ValueError(f"K has {Hkv} heads, V {V.shape[1]}")
    b = _bias_bhs(gate_bias, per_head, B, H, S)                         # (B, H, S)
    w = torch.exp(b - b.amax(dim=-1, keepdim=True)).to(V.dtype)         # (B, H, S)
    G = H // Hkv
    # Built token-major, (B, S, H, D+pad) in memory, then viewed as (B, H, S, .):
    # V arrives as a (B, S, Hkv, D)-strided transpose view, so the kernel that
    # writes v' reads and writes in one order, and that is cuDNN's native layout.
    # Functional (no out= / in-place) so torch.compile fuses it into one kernel.
    vt = V.transpose(1, 2).unsqueeze(3)                                  # (B, S, Hkv, 1, D)
    wt = w.permute(0, 2, 1).reshape(B, S, Hkv, G)                        # (B, S, Hkv, G)
    wv = (vt * wt.unsqueeze(-1)).reshape(B, S, H, D)
    vw = torch.cat([wv, wt.reshape(B, S, H, 1), w.new_zeros(B, S, H, _VW_PAD - 1)], dim=-1)
    o = attn(Q, K, vw.transpose(1, 2))
    return o[..., :D] / o[..., D:D + 1]


def _cudnn_causal_vw(Q, K, V):
    """Direct cuDNN op: is_causal, no bias, K/V may carry fewer heads than Q
    (GQA) and V a wider head dim than Q/K."""
    return torch.ops.recursive_kvs.cudnn_causal_vw(Q, K, V)


def clear_block_mask_cache():
    """Clear cached BlockMasks (call when switching to different seq lengths)."""
    global _BLOCK_MASK_CACHE
    _BLOCK_MASK_CACHE.clear()


def get_block_mask_cache_stats() -> dict:
    """Return BlockMask cache statistics for monitoring."""
    return {
        'num_cached': len(_BLOCK_MASK_CACHE),
        'cached_lengths': sorted(set(k[0] for k in _BLOCK_MASK_CACHE.keys())),
        'total_memory_mb': len(_BLOCK_MASK_CACHE) * 8,  # ~8MB per mask estimate
    }


def warmup_flex_attention(device: torch.device = torch.device('cuda')):
    """Warmup flex_attention compilation with dummy inputs.

    Compiles all four kernels in BOTH gradient modes (no_grad and with_grad)
    so that switching between eval (no_grad) and training (grad enabled) never
    triggers a recompilation at runtime.  Without this, the first training step
    after a no_grad warmup recompiles 4 kernels, quickly exhausting the
    recompile_limit and causing a fallback to eager mode that materialises the
    full S×S attention matrix (OOM for long sequences).

    Variants compiled (×2 for each grad mode, ×2 for equal-heads / GQA = 16):
      1. Causal only (Phase 1)
      2. Scalar bias, single-sample  (S,)
      3. Scalar bias, batched        (B, S)
      4. Per-head bias, batched      (B, S, H)

    Call this once at the start of training to avoid first-step compile delay.
    """
    print("[FlexAttention] Warming up compilation...")
    H = 8

    def _run_warmup_passes():
        for H_kv in (H, H // 2):
            Q = torch.randn(1, H, 512, 64, device=device, dtype=torch.bfloat16)
            K = torch.randn(1, H_kv, 512, 64, device=device, dtype=torch.bfloat16)
            V = torch.randn(1, H_kv, 512, 64, device=device, dtype=torch.bfloat16)
            # 1. Causal
            _ = flex_attention_causal(Q, K, V)
            # 2. Scalar bias, single-sample (S,)
            _ = flex_attention_gate_bias(Q, K, V, torch.randn(512, device=device, dtype=torch.bfloat16))
            # 3. Scalar bias, batched (B, S)
            _ = flex_attention_gate_bias(Q, K, V, torch.randn(1, 512, device=device, dtype=torch.bfloat16))
            # 4. Per-head bias, batched (B, S, H)
            _ = flex_attention_gate_bias(Q, K, V, torch.randn(1, 512, H, device=device, dtype=torch.bfloat16), per_head=True)

    # Compile eval-mode (no_grad) kernels first
    with torch.no_grad():
        _run_warmup_passes()

    # Compile training-mode (grad enabled) kernels so train steps never recompile
    _run_warmup_passes()

    print("[FlexAttention] Warmup complete")
