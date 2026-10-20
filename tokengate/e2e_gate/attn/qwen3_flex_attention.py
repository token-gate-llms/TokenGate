"""Patched full-attention layer with FlexAttention support.

This module provides drop-in replacements for the full-softmax attention
layers in Qwen3 (homogeneous) and Qwen3.5 (hybrid 3:1 with DeltaNet) that
use FlexAttention instead of dense attention masks.

Two patched classes are exposed:
- ``Qwen3FlexAttention``    — wraps ``Qwen3Attention`` (Qwen3, all-full models).
- ``Qwen3_5FlexAttention``  — wraps ``Qwen3_5Attention`` / ``Qwen3NextAttention``
  full-attention layers in hybrid Qwen3.5 models.  Mirrors the Qwen3 port but
  also carries the chunked q_proj into a (query, output_gate) split and applies
  ``sigmoid(gate)`` to the attention output before ``o_proj``.

``patch_model_with_flex_attention`` walks ``model.model.layers`` and patches
only the full-attention layers, leaving any DeltaNet / linear-attention layers
untouched.  In Qwen3.5 hybrid models, only the gate biases applied at full-attn
layers actually steer attention logits; DeltaNet layers still process the
top-K-pruned residual stream because Phase 2 narrows the residual stream
upstream of every layer above the gate.
"""

from typing import List, Optional, Tuple

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache

from tokengate.e2e_gate.attn.flex_attention_utils import (
    resolve_backends,
    BIAS_BACKENDS,
    CAUSAL_BACKENDS,
    attention_causal,
    attention_gate_bias,
    fa2_bias_attention_packed,
    flex_attention_segment_causal,
    flex_attention_segment_gate_bias,
)


# Qwen3.5 full-attention class lives in transformers ≥ 4.57.  Import lazily so
# that environments running pure Qwen3 don't have to have a recent transformers.
def _try_import_qwen3_5_attention():
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
        return Qwen3_5Attention
    except (ImportError, ModuleNotFoundError):
        return None


def _try_import_qwen3_next_attention():
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention
        return Qwen3NextAttention
    except (ImportError, ModuleNotFoundError):
        return None


def _import_qwen3_next_rotary():
    """Return Qwen3-Next's partial-rotation aware apply_rotary_pos_emb.

    Qwen3-Next / Qwen3.5 set ``partial_rotary_factor < 1`` so cos/sin are
    narrower than head_dim: the function splits q/k into the rotary prefix
    and a pass-through suffix.  Qwen3's apply_rotary_pos_emb assumes full
    rotation and would shape-mismatch on partial rotary configs.
    """
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        apply_rotary_pos_emb as qwen3_next_apply_rotary_pos_emb,
    )
    return qwen3_next_apply_rotary_pos_emb

# Sentinel used to distinguish "attribute not set" from None (which means "disabled").
_UNSET = object()

# Phase-2 attention statistics buffer for W&B logging.
# Each entry is a dict with per-head mean-abs tensors (H,) for:
#   'b'  — gate bias as it enters the softmax (after gate_scale + attn_scale)
#   'qk' — sampled QK logits (after 1/sqrt(d_head) and attn_scale)
#   'si' — softmax input = qk + bias
# Accumulated across all Phase-2 layer forward passes; read and cleared by the trainer.
_phase2_attn_stats_buf: list = []

# Only the trainer's rank 0 consumes the buffer (for W&B logging), and it turns
# accumulation on itself (set_phase2_stats_enabled(is_main) at startup).  Off by
# default: every other caller is inference, where the stats are never read, the
# buffer grows without bound, and each Phase-2 layer would pay ~10 host syncs
# (.item() / boolean-mask indexing) plus extra QK matmuls -- measured at ~20 ms
# of a 118 ms prefill (kernels/bench/profile_ttft.py).
_phase2_stats_enabled: bool = False


def set_phase2_stats_enabled(enabled: bool) -> None:
    """Enable/disable Phase-2 attention-stats accumulation (see above)."""
    global _phase2_stats_enabled
    _phase2_stats_enabled = bool(enabled)


@torch.compiler.disable
def _accumulate_phase2_attn_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    bias: torch.Tensor,
    scaling: float,
    scaled_bias: Optional[torch.Tensor] = None,
    scaled_q: Optional[torch.Tensor] = None,
    s_eff_stats: Optional[torch.Tensor] = None,
) -> None:
    """Accumulate per-head mean-abs stats for b_normalized, qk_normalized, softmax_input.

    Args:
        q:           (B, H, S_q, D) raw query states, BEFORE attn_scale.  Used for
                     qk_normalized.
        k:           (B, H_kv, S_k, D) key states (may or may not be GQA-expanded)
        bias:        (S_k, H) or (B, S_k, H) raw gate bias — after gate_scale and noise
                     but BEFORE attn_scale.  Used for b_normalized.
        scaling:     1/sqrt(head_dim)
        scaled_bias: same shape as bias, multiplied by attn_scale.  Used for
                     softmax_input (= QK_scaled + scaled_bias).  Defaults to bias.
        scaled_q:    same shape as q, multiplied by attn_scale.  Used for
                     softmax_input.  Defaults to q.
    """
    if not _phase2_stats_enabled:
        return
    with torch.no_grad():
        b  = bias.detach().float()
        sb = (scaled_bias if scaled_bias is not None else bias).detach().float()
        sq = (scaled_q   if scaled_q   is not None else q).detach()
        # Padding sentinel: positions beyond n_kept are filled with -1e9 in the
        # batched forward.  Exclude them with a threshold so only real gate scores
        # (< 1e8) contribute to the stats.  Also exclude zero non-article positions.
        _PAD_SENTINEL = 1e8

        # Use the first batch item as the reference for all per-position masking,
        # normalised to (S_k, H_or_1). Callers pass every bias layout the forward
        # paths produce: (S_k,) single scalar, (B, S_k) batched scalar, (S_k, H)
        # single per-head, (B, S_k, H) batched per-head. The old expression
        # `b if b.dim() == 2 else b[0]` misread the BATCHED SCALAR case as
        # (S_k, H) — masks came out over the batch dim and the qk_sc indexing
        # below raised IndexError (mask [B] vs kv-dim S_k). Disambiguate dim-2
        # via S_k from the key tensor; the (B, S_k) reading is preferred only
        # when shape[0] cannot itself be S_k.
        S_k = k.shape[2]

        def _ref(x):
            if x.dim() == 1:
                return x.unsqueeze(-1)                     # (S_k,) -> (S_k, 1)
            if x.dim() == 2:
                if x.shape[1] == S_k and x.shape[0] != S_k:
                    return x[0].unsqueeze(-1)              # (B, S_k) -> (S_k, 1)
                return x                                   # (S_k, H)
            return x[0].reshape(S_k, -1)                   # (B, S_k, ...) -> (S_k, H')

        b_ref = _ref(b)
        max_abs = b_ref.abs().max(dim=-1).values  # (S_k,)
        art_mask   = (max_abs > 0) & (max_abs < _PAD_SENTINEL)  # kept article tokens
        valid_mask = max_abs < _PAD_SENTINEL                     # all non-padding tokens

        b_art = b_ref[art_mask]   # (n_art, H)
        if b_art.shape[0] == 0:
            return

        b_head = b_art.abs().mean(dim=0)   # (H,) per-head mean-abs over article tokens

        # Sample a subset of query positions to estimate QK logit scale cheaply.
        S_q = q.shape[2]
        n_sample = min(16, S_q)
        step = max(1, S_q // n_sample)
        q_s    = q.detach()[:, :, ::step, :][:, :, :n_sample, :]   # (B, H, n_sample, D) raw
        sq_s   = sq[:, :, ::step, :][:, :, :n_sample, :]            # (B, H, n_sample, D) scaled

        # Expand KV heads for GQA if not already expanded.
        # In some code paths (e.g. prefill_for_decode) key_states is already at
        # full head count; repeat_interleave only when H_kv < H_q.
        H_q = q_s.shape[1]
        k_det = k.detach()
        k_exp = k_det.repeat_interleave(H_q // k_det.shape[1], dim=1) if k_det.shape[1] < H_q else k_det

        # qk_normalized: raw QK/sqrt(d), no attn_scale.
        qk     = torch.matmul(q_s,  k_exp.transpose(-2, -1)) * scaling  # (B, H, n_sample, S_k)
        qk_head = qk.abs().mean(dim=(0, 2, 3))   # (H,)

        # Softmax input = QK_scaled + scaled_bias, over non-padding KV positions only.
        # Both scaled_q and scaled_bias include attn_scale — this is the actual softmax input.
        # valid_mask is derived from unscaled bias so padding exclusion is attn_scale-independent.
        qk_sc  = torch.matmul(sq_s, k_exp.transpose(-2, -1)) * scaling  # (B, H, n_sample, S_k)
        sb_ref = _ref(sb)  # (S_k, H_or_1) for reference batch item (see _ref above)
        if valid_mask.any():
            sb_valid  = sb_ref[valid_mask].t().unsqueeze(0).unsqueeze(2)    # (1, H, 1, n_valid)
            qk_valid  = qk_sc[:1, :, :, valid_mask]                         # (1, H, n_sample, n_valid)
            si_head   = (qk_valid + sb_valid).abs().mean(dim=(0, 2, 3))     # (H,)
        else:
            si_head = qk_head

        # Signed mean/variance scalars of the attention-formula terms:
        #   (a) logit term  = raw qk/sqrt(d)          -> qk_m / qk_v
        #   (b) gate term   = bias as built (sp(G)*tanh(b) in tanh mode,
        #                     BEFORE attn_scale)      -> b_m / b_v
        #   softmax input   = attn-scaled (a) + (b)   -> si_m / si_v
        # The abs-based per-head entries above stay unchanged (older runs'
        # W&B key continuity); these are flat scalars over sampled entries.
        b_m, b_v = b_art.mean().item(), b_art.var(unbiased=False).item()
        qk_m, qk_v = qk.mean().item(), qk.var(unbiased=False).item()
        if valid_mask.any():
            _si_flat = (qk_valid + sb_valid)
            si_m, si_v = _si_flat.mean().item(), _si_flat.var(unbiased=False).item()
        else:
            si_m, si_v = qk_m, qk_v
        se_m = se_v = None
        if s_eff_stats is not None:
            _se = s_eff_stats.detach().float()
            se_m, se_v = _se.mean().item(), _se.var(unbiased=False).item()
        _phase2_attn_stats_buf.append({
            'se_m': se_m, 'se_v': se_v,
            'b':  b_head.cpu(),
            'qk': qk_head.cpu(),
            'si': si_head.cpu(),
            'b_m': b_m, 'b_v': b_v,
            'qk_m': qk_m, 'qk_v': qk_v,
            'si_m': si_m, 'si_v': si_v,
        })


class Qwen3FlexAttention(Qwen3Attention):
    """Qwen3Attention with FlexAttention support for E2E gate.
    
    
    This class extends Qwen3Attention to use FlexAttention instead of
    dense attention masks, providing:
    - 3× speedup for causal attention
    - O(1) memory for attention masks instead of O(S²)
    - Support for gate bias in Phase 2
    
    Usage:
        # Replace attention layer in-place
        layer.self_attn.__class__ = Qwen3FlexAttention
        layer.self_attn.gate_bias = None
        layer.self_attn.use_flex = True
        
        # Set gate bias for Phase 2 (None for Phase 1)
        layer.self_attn.set_gate_bias(bias_tensor)
    """
    
    def __init__(self, *args, **kwargs):
        """
            - Inherits from Qwen3Attention and overrides the forward method to use FlexAttention.
            - Inherits config, layer_idx, head_dim, num_key_value_gropus, scaling, attention_dropout, 
                is_causal, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm sliding_window from Qwen3Attention.
            - Adds gate_bias attribute for Phase 2 (None for Phase 1) and sets use_flex=True.
        """
        super().__init__(*args, **kwargs)
        self.gate_bias: Optional[torch.Tensor] = None
        self.use_flex = True
        self._static_inputs = False
        self._static_bufs = {}
    
    @staticmethod
    def from_qwen3_attention(original: Qwen3Attention) -> Qwen3Attention:
        """Convert Qwen3Attention to FlexAttention by monkey-patching.
        
        We can't change __class__ because accelerate hooks cache the old forward.
        Instead, we directly replace the forward method and add attributes.
        """
        # add new states to original
        original.gate_bias = None
        original._gate_per_head = False  # True when gate_bias has an H dim
        original.attn_scale = None
        original.ssmax_ratio = None
        original._static_inputs = False
        original._static_bufs = {}
        original._static = Qwen3FlexAttention._static.__get__(original, type(original))
        # segment_ids enables packed-sequence training (multiple logical
        # samples concatenated in one batch element).  When set, attention
        # is segment-causal: causal within a segment AND blocked across
        # segments.  Shape (S,) or (B, S), int32, with 0 = padding.
        original.segment_ids = None
        original.packed = False
        original.use_flex = True
        # Which kernel serves each attention form (flex_attention_utils):
        # causal for layers with no gate bias set, bias for Phase-2 layers.
        # "flex" is what training runs; set_attention_backends_for_layers picks
        # the inference kernels.
        original.attn_backend_causal = "flex"
        original.attn_backend_bias = "flex"

        # reference the new forward
        new_forward = Qwen3FlexAttention.forward

        # wrapper -- when called invoke the new forward and pass all args/kwargs through
        def patched_forward(self, *args, **kwargs):
            return new_forward(self, *args, **kwargs)

        # LP: we need to bind the patched_forward to the original instance so that it receives self correctly
        # a bound method is a function that is "linked" to a specific instance of a class
        # bound methods automatically pass self as the first arg, allowing it to access instance attributes and other methods
        # if we do original.forward = patched_forward, self never gets passed and we get: TypeError: patched_forward() missing 1 required positional argument: 'self'
        # __get__ injects the instance (original) as self into patched_forward
        original.forward = patched_forward.__get__(original, type(original))

        # while we need to overwrite the old forward, set_gate_bias/set_attn_scale are easy because they're new methods
        original.set_gate_bias = Qwen3FlexAttention.set_gate_bias.__get__(original, type(original))
        original.set_attn_scale = Qwen3FlexAttention.set_attn_scale.__get__(original, type(original))
        original.set_ssmax_ratio = Qwen3FlexAttention.set_ssmax_ratio.__get__(original, type(original))
        original.set_segment_ids = Qwen3FlexAttention.set_segment_ids.__get__(original, type(original))

        return original

    def _static(self, name: str, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Under CUDA graphs (set_static_inputs_for_layers) the per-call tensors
        this module reads as attributes -- the gate bias slice, the ssmax ratio,
        the attention scale -- are copied into a buffer of this module's own, so
        their address is the same on every call: torch.compile treats module
        attribute tensors as static inputs and re-records the graph whenever one
        arrives at a new address (a fresh allocation every prefill did exactly
        that, ~85 ms per re-record).  The buffer grows in powers of two."""
        if t is None or not self._static_inputs:
            return t
        # One buffer per (name, batch, dtype, rank), its sequence axis sized
        # STATIC_SEQ_MAX once: a captured graph reads these addresses forever,
        # so they must never move.  A longer sequence is an error, not a regrow.
        key = (name, t.shape[0] if t.dim() >= 2 else None, t.dtype, t.dim())
        buf = self._static_bufs.get(key)
        if buf is None:
            shape = list(t.shape)
            if t.dim() >= 2:
                shape[1] = STATIC_SEQ_MAX
            buf = torch.zeros(shape, dtype=t.dtype, device=t.device)
            self._static_bufs[key] = buf
        if any(n > b for n, b in zip(t.shape, buf.shape)):
            raise ValueError(f"{name}: {tuple(t.shape)} exceeds the static buffer {tuple(buf.shape)} "
                             f"(STATIC_SEQ_MAX={STATIC_SEQ_MAX})")
        view = buf[tuple(slice(0, n) for n in t.shape)]
        view.copy_(t)
        return view

    def set_gate_bias(self, bias: Optional[torch.Tensor], per_head: bool = False):
        """Set gate bias for Phase 2 (None for Phase 1).

        Args:
            bias: Gate bias tensor.
                Scalar mode  (per_head=False): (S,) or (B, S)
                Per-head mode (per_head=True): (S, H) or (B, S, H)
            per_head: If True, treat the last dimension of bias as the head
                dimension and apply a separate bias per query head in score_mod.
        """
        self.gate_bias = self._static("gate_bias", bias)
        self._gate_per_head = per_head

    def set_ssmax_ratio(self, ratio: Optional[torch.Tensor]):
        """per-(B, S_q) log-ratio r_q = log(m_q)/log(p_q). When set,
        attn_scale is interpreted as gamma and the applied per-query multiplier is
        s_eff = 1 + gamma*(r - 1)  (== 1 at full context where r == 1)."""
        self.ssmax_ratio = self._static("ssmax_ratio", ratio)

    def set_attn_scale(self, scale: Optional[torch.Tensor]):
        """Set learnable attention scale (None = disabled)."""
        self.attn_scale = self._static("attn_scale", scale)

    def set_segment_ids(self, sids: Optional[torch.Tensor]):
        """Set segment IDs for packed-sequence training (None = unpacked).

        Args:
            sids: (S,) or (B, S) int32 tensor.  Tokens with the same value
                  share a segment.  0 = padding; 1+ = segment indices.
                  When set, attention becomes segment-causal: causal within
                  a segment AND blocked across segments.
        """
        self.segment_ids = sids
    
    @torch.no_grad()
    def _decode_scaled_forward(self, hidden_states, attention_mask, past_key_value,
                               cache_position, position_embeddings, q_scale):
        """One decode step (q_len == 1) of the EXACT phase-2 softmax form:
        softmax((s_q * q) @ K^T / sqrt(d) + mask), where mask ALREADY holds the
        per-key term mean_h s_eff(r_j) * b_j (constant through decode -- the key
        ratios are fixed at prefill, so gate_generate_batch folds it into the
        per-layer attention_mask ONCE per batch) and q_scale (B, H|1, 1, 1) is
        the decode token's own s_eff, recomputed per step by the harness.

        Every decode-varying tensor arrives as an ARGUMENT -- dynamo treats
        arguments as graph inputs, so this stays compiled.  The first version
        read per-step-swapped module attributes instead, and compiled layer
        code guards attr tensors by identity: all 26 layers recompiled every
        step (70 min inside one batch at GPU 0%), then an eager fallback cost
        2.4x per decode token."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position})
        q_for_attn = query_states * q_scale.to(query_states.dtype)
        mask = attention_mask[..., : key_states.shape[2]] if attention_mask is not None else None
        # repeat_kv + NO enable_gqa, mirroring stock sdpa_attention_forward:
        # under torch.compile, enable_gqa=True makes inductor DECOMPOSE SDPA
        # into the math path (per-step KV expand+clone + two bmms — 55 ms/step
        # at K=full, the whole 1.86x overhead); the explicit repeat keeps the
        # fused efficient-attention lowering the plain decode gets.
        n_rep = q_for_attn.shape[1] // key_states.shape[1]
        if n_rep > 1:
            key_states = repeat_kv(key_states, n_rep)
            value_states = repeat_kv(value_states, n_rep)
        attn = F.scaled_dot_product_attention(
            q_for_attn.contiguous(), key_states, value_states,
            attn_mask=mask, scale=self.scaling)
        attn = attn.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn), None

    def _ssmax_s_eff(self, r, dtype, batch):
        '''Scaled-softmax s_eff (B, H|1, S) from the log-ratio r (B, S) or (S,) under
        the active variant flags.  Shared by the prefill transform (r = each
        kept slot's own ratio) and decode_phase2_layer_tensors (the prefill r
        again for the per-key bias term).'''
        gamma = self.attn_scale.to(dtype)                       # (H,) or scalar
        r = r.to(dtype)
        if r.dim() == 1:
            r = r.unsqueeze(0).expand(batch, -1)
        _g = gamma.view(1, -1, 1) if gamma.numel() > 1 else gamma.reshape(1, 1, 1)
        # s_eff = r * S_eff. attn_scale here holds the EFFECTIVE
        # softplus(sigma) per head (attn_scale_eff slice); ==r at init
        # (S_eff==1) so step 0 is exactly the base model.
        s_eff = _g * r.unsqueeze(1)                         # (B, H|1, S)
        return s_eff

    def _phase2_q_bias(self, query_states):
        '''The EXACT phase-2 softmax-input transform: returns (q_for_attn,
        bias_for_attn, s_eff_stats) under the learned attn_scale /
        scaled-softmax machinery.  Shared by the prefill kernel path and
        the decode-phase2 branch (ssmax_ratio is (B, S_q); decode uses S_q=1).'''
        per_head = getattr(self, '_gate_per_head', False)
        q_for_attn = query_states
        bias_for_attn = self.gate_bias
        _se_for_stats = None
        if self.attn_scale is not None and getattr(self, "ssmax_ratio", None) is not None:
            # Scaled-softmax mode: attn_scale IS gamma.
            # s_eff(b,h,q) = 1 + gamma_h * (r_bq - 1); r==1 at full context so
            # s_eff==1 there for any gamma. Q is scaled per (b,h,q); the additive
            # bias is scaled per KEY at the key's own ratio (exact under STE where
            # the bias is zero; see the note's implementation section).
            s_eff = Qwen3FlexAttention._ssmax_s_eff(
                self, self.ssmax_ratio, query_states.dtype, query_states.shape[0])
            _se_for_stats = s_eff
            q_for_attn = query_states * s_eff.unsqueeze(-1)
            if self.gate_bias is not None:
                _sb = s_eff.mean(dim=1) if s_eff.shape[1] > 1 else s_eff[:, 0]   # (B, S)
                _gb = self.gate_bias
                bias_for_attn = _gb * (_sb.unsqueeze(-1) if _gb.dim() == 3 else _sb[0].unsqueeze(-1) if _gb.dim() == 2 else _sb[0])
        elif self.attn_scale is not None:
            scale = self.attn_scale.to(query_states.dtype)
            if scale.numel() == 1:
                # Scalar attn_scale — original behaviour
                q_for_attn = query_states * scale
                bias_for_attn = self.gate_bias * scale
            else:
                # Per-head attn_scale: scale ∈ R^H
                _se_for_stats = scale
                q_for_attn = query_states * scale.view(1, -1, 1, 1)
                if per_head:
                    # gate_bias: (S, H) or (B, S, H) — H is the last dim
                    bias_for_attn = self.gate_bias * scale.view(*([1] * (self.gate_bias.dim() - 1)), -1)
                else:
                    # Scalar bias with per-head scale: apply mean scale to keep
                    # behaviour well-defined (should not normally occur).
                    bias_for_attn = self.gate_bias * scale.mean()
        return q_for_attn, bias_for_attn, _se_for_stats

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        if 'past_key_values' in kwargs:
            past_key_value = kwargs.pop('past_key_values')
        
        # Fall back to original implementation if:
        # - FlexAttention disabled
        # - output_attentions requested (FlexAttention doesn't support this)
        # - During decode (seq_len=1, no benefit from FlexAttention)
        
        # DEBUG: Check hidden_states shape
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states should be a Tensor, got {type(hidden_states)}")
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states should be 3D (batch, seq, hidden), got shape {hidden_states.shape}")
        
        bsz, q_len, _ = hidden_states.size()
        
        # decode under the exact phase-2 softmax form: the per-key bias term
        # rides in attention_mask; the per-step query scale lives in a
        # PROCESS-LIFETIME buffer view (_dp2_qsbuf) the harness writes
        # in-place each step.  A per-step kwarg instead cost ~6 ms/step of
        # host dispatch at small L (26 extra-input guard evaluations); a
        # stable attr object adds none.
        if q_len == 1 and getattr(self, "_dp2_active", False):
            return Qwen3FlexAttention._decode_scaled_forward(
                self, hidden_states, attention_mask, past_key_value,
                cache_position, position_embeddings, self._dp2_qsbuf)

        if not self.use_flex or output_attentions or q_len == 1:
            return Qwen3Attention.forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
            )
        
        # ── FlexAttention path ────────────────────────────────────────────────
        
        # QKV projection + normalization 
        # Qwen3 uses q_norm and k_norm 
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Capture Q/K norms if enabled (training diagnostics).
        # Sampled before RoPE — RoPE is an isometry so norms are unchanged by it.
        # Apply rotary embeddings
        if position_embeddings is None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            cos, sin = position_embeddings
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # KV cache handling
        if past_key_value is not None:
            # Retrieve or initialize cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        # Ensure Q/K/V are on same device (critical for torch.compile stability)
        # With device_map="auto", KV cache may be on different GPU than query
        target_device = query_states.device
        if key_states.device != target_device:
            key_states = key_states.to(target_device)
        if value_states.device != target_device:
            value_states = value_states.to(target_device)
        
        # GQA is handled inside the kernel (flex_attention(enable_gqa=True)
        # broadcasts the kv heads over the query heads); K/V stay at
        # num_key_value_heads, so no repeat_interleave copies and half the K/V
        # traffic in the attention loop.  The score_mod's ``h`` is still the
        # query-head index.

        # FlexAttention wants contiguous operands (compile stability); cuDNN
        # takes the transposed (B, S, H, D)-strided views as they are -- its
        # native layout -- and the copies were ~4 ms of a 67 ms prefill.
        _backend = self.attn_backend_bias if self.gate_bias is not None else self.attn_backend_causal
        if _backend == "flex":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if self.gate_bias is not None:
            # Phase 2: causal + gate bias (+ optional learned attn_scale).
            #
            # To implement attn_scale * (QK^T/sqrt(d) + b) without placing a
            # grad-tracked tensor inside the compiled score_mod (which breaks
            # FlexAttention's inductor backward), we factor the scaling out:
            #   attn_scale * (Q @ K^T / sqrt(d) + b)
            #   = (attn_scale * Q) @ K^T / sqrt(d) + attn_scale * b
            # Both multiplications stay in regular PyTorch autograd.
            #
            # Per-head attn_scale (n_heads > 1):
            #   scale shape: (H,)
            #   Q scaling:    query_states (B, H, S, D) * scale.view(1, H, 1, 1)
            #   bias scaling: gate_bias (S, H) or (B, S, H) * scale broadcast on H dim
            per_head = getattr(self, '_gate_per_head', False)
            q_for_attn, bias_for_attn, _se_for_stats = Qwen3FlexAttention._phase2_q_bias(self, query_states)
            if _phase2_stats_enabled:
                _accumulate_phase2_attn_stats(
                    query_states, key_states, self.gate_bias,
                    self.scaling, scaled_bias=bias_for_attn, scaled_q=q_for_attn,
                    s_eff_stats=_se_for_stats,
                )
            if self.packed:
                # Phase-2 targets packed into one sequence (gate_forward, PACK_PHASE2):
                # causal within each target, the bias per key of its own target.
                if self.attn_backend_bias != "fa2-bias":
                    raise ValueError("packed Phase-2 targets need the fa2-bias kernel, "
                                     f"got {self.attn_backend_bias!r}")
                attn_output = fa2_bias_attention_packed(
                    q_for_attn, key_states, value_states, bias_for_attn, per_head)
            elif self.segment_ids is not None:
                # Packed-sequence training: segment-causal + gate bias.
                if self.attn_backend_bias != "flex":
                    raise ValueError("packed segments need the flex gate-bias kernel, "
                                     f"got {self.attn_backend_bias!r}")
                attn_output = flex_attention_segment_gate_bias(
                    q_for_attn, key_states, value_states, bias_for_attn,
                    segment_ids=self.segment_ids,
                    per_head=per_head,
                )
            else:
                attn_output = attention_gate_bias(
                    q_for_attn, key_states, value_states, bias_for_attn,
                    per_head=per_head, backend=self.attn_backend_bias,
                )
        else:
            # Phase 1: causal only (or segment-causal in packed mode)
            if self.segment_ids is not None:
                if self.attn_backend_causal != "flex":
                    raise ValueError("packed segments need the flex causal kernel, "
                                     f"got {self.attn_backend_causal!r}")
                attn_output = flex_attention_segment_causal(
                    query_states, key_states, value_states,
                    segment_ids=self.segment_ids,
                )
            else:
                attn_output = attention_causal(
                    query_states, key_states, value_states,
                    backend=self.attn_backend_causal,
                )
        
        # Reshape and project output (same as original)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        # Qwen3Attention returns (attn_output, attn_weights)
        # We don't compute attn_weights with FlexAttention, so return None
        return attn_output, None


class Qwen3_5FlexAttention(nn.Module):
    """Qwen3.5 / Qwen3-Next full-attention layer patched to use FlexAttention.

    Wraps ``Qwen3_5Attention`` (subclass of ``Qwen3NextAttention``).  The wrapped
    layer's ``q_proj`` is twice as wide as a vanilla attention layer: it returns
    ``[query_states || gate]``.  After the attention output is computed we
    multiply elementwise by ``sigmoid(gate)`` before the final ``o_proj`` — this
    is the Qwen3.5 output-gating mechanism (``attn_output_gate`` in the config).

    Gate-bias and learnable attn-scale machinery (``set_gate_bias`` /
    ``set_attn_scale``) is identical to ``Qwen3FlexAttention``.  Phase-2
    statistics are accumulated into the same ``_phase2_attn_stats_buf``.
    """

    @staticmethod
    def from_attention(original: nn.Module) -> nn.Module:
        """Monkey-patch ``original`` (a Qwen3_5Attention/Qwen3NextAttention layer).

        We don't change ``__class__`` because accelerate hooks cache the forward
        bound at hook-install time.  Instead we add the new state and rebind
        ``forward``.
        """
        original.gate_bias = None
        original._gate_per_head = False
        original.attn_scale = None
        original.ssmax_ratio = None
        original.use_flex = True
        # set_gate_bias/set_attn_scale are borrowed from Qwen3FlexAttention and
        # route tensors through the CUDA-graph static-buffer helper, so the
        # same _static state must exist here (missing it crashed the first
        # real-checkpoint smoke test with AttributeError: '_static').
        original._static_inputs = False
        original._static_bufs = {}
        original._static = Qwen3FlexAttention._static.__get__(original, type(original))
        # Kernel-backend attrs read by the decode-phase2 (_decode_scaled_forward)
        # path and set_attention_backends_for_layers — same defaults as the
        # Qwen3 graft ("flex" = the training configuration).
        original.attn_backend_causal = "flex"
        original.attn_backend_bias = "flex"

        new_forward = Qwen3_5FlexAttention.forward

        def patched_forward(self, *args, **kwargs):
            return new_forward(self, *args, **kwargs)

        original.forward = patched_forward.__get__(original, type(original))
        original.set_gate_bias = Qwen3FlexAttention.set_gate_bias.__get__(original, type(original))
        original.set_attn_scale = Qwen3FlexAttention.set_attn_scale.__get__(original, type(original))
        original.set_ssmax_ratio = Qwen3FlexAttention.set_ssmax_ratio.__get__(original, type(original))
        original._original_forward = type(original).forward
        return original

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states should be a Tensor, got {type(hidden_states)}")
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states should be 3D (batch, seq, hidden), got shape {hidden_states.shape}"
            )

        bsz, q_len, _ = hidden_states.size()
        output_attentions = kwargs.get("output_attentions", False)

        # decode under the exact phase-2 softmax form — the Qwen3.5 port of
        # Qwen3FlexAttention's branch (see that class for the argument-based
        # design rationale: per-key bias rides in attention_mask, the per-step
        # query scale lives in the process-lifetime _dp2_qsbuf view)
        if q_len == 1 and getattr(self, "_dp2_active", False):
            return Qwen3_5FlexAttention._decode_scaled_forward(
                self, hidden_states, attention_mask, past_key_values,
                position_embeddings, self._dp2_qsbuf,
                kwargs.get("cache_position"))

        # Fall back to the original Qwen3.5 forward when FlexAttention isn't
        # applicable (decode step or output_attentions requested).
        if not self.use_flex or output_attentions or q_len == 1:
            return self._original_forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        # ── QKV + gate projection ─────────────────────────────────────────────
        # Qwen3.5: q_proj outputs (num_heads * head_dim * 2) and splits into
        # (query_states, gate).  q_norm/k_norm are applied per-head; head_dim is
        # taken from the wrapped layer.
        head_dim = self.head_dim
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)
        qg = self.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2)
        query_states, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # Qwen3-Next / Qwen3.5 use partial_rotary_factor < 1, so we must use
        # the partial-rotation-aware function (Qwen3's would shape-mismatch).
        qwen3_next_apply_rotary = _import_qwen3_next_rotary()
        query_states, key_states = qwen3_next_apply_rotary(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        target_device = query_states.device
        if key_states.device != target_device:
            key_states = key_states.to(target_device)
        if value_states.device != target_device:
            value_states = value_states.to(target_device)

        # GQA expansion — Qwen3.5 full-attn uses 16 Q heads, 4 KV heads.
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        if num_kv_heads != num_heads:
            key_states = key_states.repeat_interleave(num_heads // num_kv_heads, dim=1)
            value_states = value_states.repeat_interleave(num_heads // num_kv_heads, dim=1)

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        if self.gate_bias is not None:
            # Same logic as Qwen3FlexAttention.forward: factor attn_scale out of
            # the score_mod so no grad-tracked tensors are captured inside the
            # compiled FlexAttention kernel.
            per_head = getattr(self, '_gate_per_head', False)
            q_for_attn = query_states
            bias_for_attn = self.gate_bias
            _se_for_stats = None
            if self.attn_scale is not None and getattr(self, "ssmax_ratio", None) is not None:
                # Scaled-softmax mode (see Qwen3FlexAttention.forward).
                gamma = self.attn_scale.to(query_states.dtype)
                r = self.ssmax_ratio.to(query_states.dtype)
                _B = query_states.shape[0]
                if r.dim() == 1:
                    r = r.unsqueeze(0).expand(_B, -1)
                _g = gamma.view(1, -1, 1) if gamma.numel() > 1 else gamma.reshape(1, 1, 1)
                s_eff = _g * r.unsqueeze(1)     # r * S_eff
                _se_for_stats = s_eff
                q_for_attn = query_states * s_eff.unsqueeze(-1)
                _sb = s_eff.mean(dim=1) if s_eff.shape[1] > 1 else s_eff[:, 0]
                _gb = self.gate_bias
                bias_for_attn = _gb * (_sb.unsqueeze(-1) if _gb.dim() == 3 else _sb[0].unsqueeze(-1) if _gb.dim() == 2 else _sb[0])
            elif self.attn_scale is not None:
                scale = self.attn_scale.to(query_states.dtype)
                if scale.numel() == 1:
                    q_for_attn = query_states * scale
                    bias_for_attn = self.gate_bias * scale
                else:
                    _se_for_stats = scale
                    q_for_attn = query_states * scale.view(1, -1, 1, 1)
                    if per_head:
                        bias_for_attn = self.gate_bias * scale.view(
                            *([1] * (self.gate_bias.dim() - 1)), -1
                        )
                    else:
                        bias_for_attn = self.gate_bias * scale.mean()
            if _phase2_stats_enabled:
                _accumulate_phase2_attn_stats(
                    query_states, key_states, self.gate_bias,
                    self.scaling, scaled_bias=bias_for_attn, scaled_q=q_for_attn,
                    s_eff_stats=_se_for_stats,
                )
            # Use the CONFIGURED phase-2 kernel (aug-fa4 etc.), not hardcoded
            # flex: the biased flex kernel over a full 16k context is ~17x
            # slower than aug-fa4 (the gate_full anchor read 6-7s vs ~360ms),
            # while the dense Qwen3 path already dispatches on attn_backend_bias.
            # The kernel operates on projected q/k/v/bias, so it is model-
            # agnostic (numerically ~identical to flex, verified on niah).
            attn_output = attention_gate_bias(
                q_for_attn, key_states, value_states, bias_for_attn,
                per_head=per_head, backend=self.attn_backend_bias,
            )
        else:
            attn_output = attention_causal(query_states, key_states, value_states,
                                           backend=self.attn_backend_causal)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        # Qwen3.5 output gate — applied between attention output and o_proj.
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    @torch.no_grad()
    def _decode_scaled_forward(self, hidden_states, attention_mask, past_key_values,
                               position_embeddings, q_scale, cache_position):
        """Qwen3.5 twin of Qwen3FlexAttention._decode_scaled_forward: one decode
        step of softmax((s_q * q) @ K^T / sqrt(d) + mask), where the mask
        already carries the per-key phase-2 bias term and q_scale is the decode
        token's own s_eff.  Differences from the Qwen3 version, mirroring this
        class's prefill path exactly: q_proj splits into (query, output-gate),
        rotary is partial-rotation-aware, and sigmoid(gate) multiplies the
        attention output before o_proj."""
        head_dim = self.head_dim
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)
        qg = self.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2)
        query_states, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.reshape(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        qwen3_next_apply_rotary = _import_qwen3_next_rotary()
        query_states, key_states = qwen3_next_apply_rotary(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position})
        q_for_attn = query_states * q_scale.to(query_states.dtype)
        mask = attention_mask[..., : key_states.shape[2]] if attention_mask is not None else None
        # explicit repeat_kv (not enable_gqa): under torch.compile enable_gqa
        # decomposes SDPA into the math path — see the Qwen3 twin
        n_rep = q_for_attn.shape[1] // key_states.shape[1]
        if n_rep > 1:
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)
        attn = F.scaled_dot_product_attention(
            q_for_attn.contiguous(), key_states, value_states,
            attn_mask=mask, scale=self.scaling)
        attn = attn.transpose(1, 2).reshape(*input_shape, -1)
        attn = attn * torch.sigmoid(gate)
        return self.o_proj(attn), None


def get_full_attn_layer_indices(model: torch.nn.Module) -> List[int]:
    """Return the layer indices in ``model.model.layers`` that use full softmax attention.

    For homogeneous Qwen3 models (no ``layer_types`` attribute on the config),
    every layer is full-attention and the full range is returned.

    For Qwen3.5 hybrid models, ``config.layer_types`` (or ``config.text_config.
    layer_types`` if the wrapper is multimodal) lists per-layer mixer types and
    only the entries equal to ``"full_attention"`` are returned.

    Raises:
        ValueError: when ``layer_types`` is present but its length does not
            match the number of decoder layers in the model.
    """
    cfg = model.config
    text_cfg = getattr(cfg, 'text_config', cfg)
    layer_types = getattr(text_cfg, 'layer_types', None)
    n_layers = len(model.model.layers)
    if layer_types is None:
        return list(range(n_layers))
    if len(layer_types) != n_layers:
        raise ValueError(
            f"config.layer_types has {len(layer_types)} entries but model has "
            f"{n_layers} layers — refusing to guess which is right."
        )
    return [i for i, lt in enumerate(layer_types) if lt == "full_attention"]


def patch_model_with_flex_attention(model: torch.nn.Module) -> torch.nn.Module:
    """Patch full-attention layers in ``model`` to use FlexAttention.

    Dispatches per-layer by attention class:
    - ``Qwen3Attention``            → ``Qwen3FlexAttention``
    - ``Qwen3_5Attention`` / ``Qwen3NextAttention`` → ``Qwen3_5FlexAttention``

    DeltaNet / linear-attention layers in hybrid Qwen3.5 are skipped — they have
    no per-token KV cache to bias and they do not expose ``self_attn``.  We rely
    on ``get_full_attn_layer_indices`` to enumerate the eligible layer indices.

    The list of patched indices is stashed on ``model._full_attn_layer_indices``
    for downstream consumers (gate bias application, baseline compressor,
    layer-wise gate-scale indexing).

    Raises:
        ValueError: if no layer was eligible for patching.
        NotImplementedError: if a full-attn layer uses an attention class we
            haven't ported yet.
    """
    full_attn_indices = get_full_attn_layer_indices(model)
    if not full_attn_indices:
        raise ValueError("patch_model_with_flex_attention: no full-attention layers found in model")

    qwen3_5_attn_cls = _try_import_qwen3_5_attention()
    qwen3_next_attn_cls = _try_import_qwen3_next_attention()

    patched = 0
    for i in full_attn_indices:
        layer = model.model.layers[i]
        attn = getattr(layer, 'self_attn', None)
        if attn is None:
            raise AttributeError(
                f"Layer {i} is listed as full_attention but has no .self_attn attribute"
            )
        if isinstance(attn, Qwen3Attention):
            layer.self_attn = Qwen3FlexAttention.from_qwen3_attention(attn)
        elif (qwen3_5_attn_cls is not None and isinstance(attn, qwen3_5_attn_cls)) or (
            qwen3_next_attn_cls is not None and isinstance(attn, qwen3_next_attn_cls)
        ):
            layer.self_attn = Qwen3_5FlexAttention.from_attention(attn)
        else:
            raise NotImplementedError(
                f"patch_model_with_flex_attention: layer {i} has attention class "
                f"{type(attn).__name__!r} — no FlexAttention port available. "
                f"Supported: Qwen3Attention, Qwen3_5Attention, Qwen3NextAttention."
            )
        patched += 1

    model._full_attn_layer_indices = full_attn_indices
    print(f"[FlexAttention] patched {patched} full-attention layers: {full_attn_indices}")
    return model


def decode_phase2_layer_tensors(model, gate_layer: int, prefill_len: int,
                                total_len: int, base_mask: torch.Tensor):
    """Fold the exact phase-2 softmax form's DECODE constants into per-layer
    argument tensors, read from the bias/scale attrs the prefill left live.

    Returns (masks, gammas): masks[layer_idx] is an additive attention mask
    (B, H, 1, total_len) = base_mask + mean_h s_eff(r_j) * b_j on the first
    prefill_len keys (the bias term is CONSTANT through decode -- the key
    ratios were fixed at prefill; the bucket-padded bias tail past prefill_len
    is dropped, decode-appended keys stay 0); gammas is the (n_layers, H)
    attn_scale stack for the per-step query scale, or None when the gate has
    no scaled-softmax.  The caller must CLEAR the layer attrs before decoding:
    reading swapped attr tensors from compiled layers recompiles every layer
    every step (see _decode_scaled_forward)."""
    masks, gammas, idx, ss_layers = {}, [], {}, []
    for li in range(gate_layer + 1, len(model.model.layers)):
        attn = getattr(model.model.layers[li], "self_attn", None)  # DeltaNet layers: skip
        gb = getattr(attn, "gate_bias", None)
        if gb is None:
            continue
        if gb.dim() == 2:
            # Single-sample decode (prefill_for_decode) leaves a per-head
            # (S, H) bias; the batched path leaves (B, S, H). Promote to
            # B=1 so the one fold covers both — base_mask is B=1 here too.
            gb = gb.unsqueeze(0)
        elif gb.dim() != 3:
            raise NotImplementedError(
                f"decode-phase2 expects per-head (B, S, H) or (S, H) bias, "
                f"got {tuple(gb.shape)} at layer {li}")
        if gb.shape[1] < prefill_len:
            raise ValueError(f"layer {li}: bias covers {gb.shape[1]} keys < prefill_len {prefill_len}")
        bias = gb[:, :prefill_len]
        ssmax = attn.attn_scale is not None and getattr(attn, "ssmax_ratio", None) is not None
        if ssmax:
            s_k = Qwen3FlexAttention._ssmax_s_eff(
                attn, attn.ssmax_ratio[..., :prefill_len], bias.dtype, gb.shape[0])
            sb = s_k.mean(dim=1) if s_k.shape[1] > 1 else s_k[:, 0]     # (B, prefill_len)
            bias = bias * sb.unsqueeze(-1)
            idx[li] = len(gammas)
            gammas.append(attn.attn_scale.detach().reshape(-1).clone())
            ss_layers.append(li)
        elif attn.attn_scale is not None:
            if bool(torch.all(attn.attn_scale == 1)):
                # 2026-09-22 (noS ablation): frozen all-ones scale — s_eff == 1
                # exactly, so the fold is the UNSCALED bias mask and no per-step
                # query scale. Exact, not an approximation.
                pass
            else:
                raise NotImplementedError(
                    "decode-phase2: plain attn_scale (no ssmax) is wired only "
                    "for the all-ones (identity) buffer; the general constant "
                    "q-scale fold has no user yet -- add it deliberately")
        m = base_mask.expand(-1, gb.shape[2], -1, -1).clone()           # (B, H, 1, total)
        m[..., :prefill_len] += bias.permute(0, 2, 1).unsqueeze(2).to(m.dtype)
        masks[li] = m
    if gammas and len(gammas) != len(masks):
        raise RuntimeError("decode-phase2: some phase-2 layers have ssmax and some do not")
    if not gammas:
        return masks, None
    # The per-step query scale lives in ONE stacked process-lifetime buffer per
    # batch size; each ssmax layer holds a VIEW of it as _dp2_qsbuf.  The
    # harness writes the buffer in place every step, so the attr object the
    # compiled layer reads never changes identity (a per-step kwarg instead
    # cost ~6 ms/step of extra-input guard dispatch at small L; a per-step
    # attr SWAP recompiles).  Views are (re)assigned only when the batch size
    # changes (adaptive halving) — one recompile per size, then stable.
    B, nH = base_mask.shape[0], gammas[0].numel()
    bufs = model.__dict__.setdefault("_dp2_qs_bufs", {})
    key = (B, len(gammas), nH)
    if key not in bufs:
        bufs[key] = torch.ones(len(gammas), B, nH, 1, 1, dtype=torch.float32,
                               device=base_mask.device)
    qs_buf = bufs[key]
    if model.__dict__.get("_dp2_qs_installed") != key:
        for li in ss_layers:
            model.model.layers[li].self_attn._dp2_qsbuf = qs_buf[idx[li]]
        model._dp2_qs_installed = key
    for li in ss_layers:
        model.model.layers[li].self_attn._dp2_active = True
    return masks, (torch.stack(gammas), idx, qs_buf)


class FormDecodeState:
    """Decode with the SAME softmax form as prefill — the default for every
    gate decode path since 2026-09-13 (in-training eval, compressor, releval;
    the benchmark already defaulted to it via --decode-phase2-form).

    Wraps decode_phase2_layer_tensors: construct IMMEDIATELY after the gated
    prefill (the fold reads the live bias/ssmax/attn_scale attrs), which also
    clears those attrs; then per decode step call ``step(next_pos, steps)``
    and pass ``mask_for(li, cache_len)`` as each layer's attention_mask; call
    ``finish()`` after the loop.  Works for homogeneous Qwen3 and hybrid
    Qwen3.5 (DeltaNet layers are skipped by the fold and take the base mask).
    """

    def __init__(self, model, gate_layer: int, n_kepts, cache_next: int,
                 max_gen: int, base_mask: torch.Tensor):
        self.model = model
        self.gate_layer = gate_layer
        self.base_mask = base_mask
        self.masks, self.gam = decode_phase2_layer_tensors(
            model, gate_layer, cache_next, cache_next + max_gen, base_mask)
        if self.gam is not None:
            self.gammas, self.gidx, self.qs_buf = self.gam
            _a0 = model.model.layers[next(iter(self.gidx))].self_attn
            self.n_kept_t = torch.as_tensor(
                n_kepts, device=base_mask.device, dtype=torch.float32)
        n_layers = len(model.model.layers)
        set_gate_bias_for_layers(model, 0, n_layers, None)
        set_ssmax_ratio_for_layers(model, gate_layer + 1, n_layers, None)
        set_attn_scale_for_layers(model, gate_layer + 1, n_layers, None)

    def step(self, next_pos: torch.Tensor, steps: int) -> None:
        """Write this decode step's per-layer query scale s_eff(r_dec) into the
        process-lifetime buffer (see decode_phase2_layer_tensors)."""
        if self.gam is None:
            return
        _m = (self.n_kept_t + 1 + steps).clamp(min=2.0)
        _p = (next_pos[:, 0].to(torch.float32) + 1.0).clamp(min=2.0)
        _r = (torch.log(_m) / torch.log(_p)).clamp(max=1.0)          # (B,)
        _gr = self.gammas[:, None, :] if self.gammas.dim() == 2 \
            else self.gammas[:, None, None]
        _sq = _gr * _r[None, :, None]
        self.qs_buf.copy_(_sq.unsqueeze(-1).unsqueeze(-1))

    def mask_for(self, li: int, cache_len: int) -> torch.Tensor:
        """Additive attention mask for layer ``li`` at current cache length.
        dp2(ssmax) layers take their mask FULL length (the scaled forward
        slices to key length itself); every other path gets a pre-sliced mask."""
        _pm = self.masks.get(li)
        if _pm is not None and self.gam is not None:
            return _pm
        if _pm is not None:
            return _pm[..., :cache_len]
        return self.base_mask[..., :cache_len]

    def finish(self) -> None:
        set_dp2_active(self.model, self.gate_layer + 1,
                       len(self.model.model.layers), False)


def set_dp2_active(model, start_layer: int, end_layer: int, on: bool) -> None:
    """Toggle the decode-phase2 branch (see decode_phase2_layer_tensors; the
    harness turns it off after each decode loop)."""
    for layer in model.model.layers[start_layer:end_layer]:
        attn = getattr(layer, "self_attn", None)   # DeltaNet layers: no self_attn
        if attn is not None and hasattr(attn, "gate_bias"):
            attn._dp2_active = bool(on)


def set_gate_bias_for_layers(
    model: torch.nn.Module,
    start_layer: int,
    end_layer: int,
    gate_bias: Optional[torch.Tensor],
    per_head: bool = False,
    padding_mask: Optional[torch.Tensor] = None,  # Deprecated
):
    """Set gate bias for a range of layers.

    Args:
        model: Qwen3ForCausalLM model with patched attention
        start_layer: First layer to set bias (inclusive)
        end_layer: Last layer to set bias (exclusive)
        gate_bias: Gate bias tensor or None to clear.
            Scalar:   (S,) or (B, S)
            Per-head: (S, H) or (B, S, H)
        per_head: If True, treat the last dim of gate_bias as the head dim
            and dispatch to the per-head score_mod kernel.
        padding_mask: Deprecated — encode padding as gate_bias=-1e9 instead.
    """
    for i in range(start_layer, end_layer):
        # Hybrid models (Qwen3.5) attach `linear_attn` rather than `self_attn`
        # on DeltaNet layers — skip them silently here.  Full-attn layers always
        # expose `self_attn`, so the bias only lands where it's meaningful.
        attn = getattr(model.model.layers[i], 'self_attn', None)
        if attn is not None and hasattr(attn, 'set_gate_bias'):
            attn.set_gate_bias(gate_bias, per_head=per_head)


GATE_ATTN = {
    # name -> (causal backend, gate-bias backend); see flex_attention_utils.
    "flex": ("flex", "flex"),
    "cudnn-p2": ("flex", "vweight-cudnn"),
    "cudnn": ("cudnn", "vweight-cudnn"),
    "flash-p2": ("flex", "aug-flash"),
    "flash": ("flash", "aug-flash"),
    "fa2": ("flash", "fa2-bias"),
    "fa2-cudnn": ("cudnn", "fa2-bias"),
    "fa4-p2": ("flex", "aug-fa4"),
    "fa4": ("cudnn", "aug-fa4"),
    "fa4-auto": ("auto", "aug-fa4"),
}


def set_attention_backends_for_layers(model: torch.nn.Module, gate_attn: str) -> None:
    """Pick the inference attention kernels on every patched layer: one of
    GATE_ATTN's named pairs. "flex" is the training configuration; "fa4-auto"
    is the inference configuration (bias folded into the QK product on
    FlashAttention's Blackwell kernel); "cudnn-p2" / "cudnn" are the fallback
    for GPUs without that kernel (bias folded into the values on cuDNN, about
    1.85x a plain attention call)."""
    causal, bias = resolve_backends(*GATE_ATTN[gate_attn])
    if causal not in CAUSAL_BACKENDS or bias not in BIAS_BACKENDS:
        raise ValueError(f"{gate_attn}: {causal!r}/{bias!r}")
    if bias == "vweight-cudnn":
        warnings.warn(
            f"GATE_ATTN={gate_attn!r}: Phase-2 bias through the cuDNN value-weight fallback "
            f"(about 1.85x a plain attention call). Use 'fa4-auto' wherever FlashAttention's "
            f"Blackwell kernel is available (see kernels/bench/fa4_setup.sh).",
            stacklevel=2)
    for i in model._full_attn_layer_indices:
        attn = model.model.layers[i].self_attn
        attn.attn_backend_causal = causal
        attn.attn_backend_bias = bias


def set_packed_cu_for_layers(model: torch.nn.Module, start_layer: int, end_layer: int, packed_cu) -> None:
    """(cu_seqlens int32, max_len, lens) of the Phase-2 targets packed into one sequence, or None."""
    from tokengate.e2e_gate.attn import flex_attention_utils
    flex_attention_utils.PACKED_CU = packed_cu
    for i in range(start_layer, end_layer):
        attn = getattr(model.model.layers[i], 'self_attn', None)
        if attn is not None and hasattr(attn, "packed"):
            attn.packed = packed_cu is not None


def set_segment_ids_for_layers(
    model: torch.nn.Module,
    start_layer: int,
    end_layer: int,
    segment_ids: Optional[torch.Tensor],
):
    """Set segment IDs on every attention module in [start_layer, end_layer).

    When `segment_ids` is non-None, each layer's attention switches from
    plain-causal (or causal+bias) to segment-causal (causal within segment,
    blocked across segments).  Pass None to clear and restore the unpacked
    behaviour.
    """
    for i in range(start_layer, end_layer):
        # DeltaNet layers in hybrid models expose no self_attn — skip (same
        # convention as set_gate_bias_for_layers).
        attn = getattr(model.model.layers[i], 'self_attn', None)
        if attn is not None and hasattr(attn, "set_segment_ids"):
            attn.set_segment_ids(segment_ids)




# longest pruned sequence the static attribute buffers hold (CUDA-graph capture
# only).  Sized to the run's max prompt via PHASE2_SEQ_MAX (exported by the
# bench/train drivers = (max_article_tokens + 2048) rounded to 64), so a 128k
# tier gets a big-enough buffer; falls back to 65536 when unset (unchanged for
# every prior <=64k run, whose PHASE2_SEQ_MAX was ~68k anyway).  Explicit
# STATIC_SEQ_MAX env overrides both.
STATIC_SEQ_MAX = int(os.environ.get("STATIC_SEQ_MAX")
                     or os.environ.get("PHASE2_SEQ_MAX")
                     or 65536)


def set_static_inputs_for_layers(model: torch.nn.Module, start_layer: int, end_layer: int, on: bool) -> None:
    """Route the per-call attribute tensors of layers [start, end) through
    per-module buffers with stable addresses (Qwen3FlexAttention._static) --
    required when those layers run under CUDA graphs."""
    for i in range(start_layer, end_layer):
        attn = getattr(model.model.layers[i], 'self_attn', None)
        if attn is not None and hasattr(attn, '_static'):
            attn._static_inputs = bool(on)
            if not on:
                attn._static_bufs = {}


def set_ssmax_ratio_for_layers(
    model: torch.nn.Module,
    start_layer: int,
    end_layer: int,
    ratio: "Optional[torch.Tensor]",
):
    """store the per-(B, S_q) scaled-softmax log-ratio r on a range of
    Phase-2 layers (None disables the mode and restores plain attn_scale)."""
    for i in range(start_layer, end_layer):
        attn = getattr(model.model.layers[i], 'self_attn', None)
        if attn is not None and hasattr(attn, 'set_ssmax_ratio'):
            attn.set_ssmax_ratio(ratio)


def set_attn_scale_for_layers(
    model: torch.nn.Module,
    start_layer: int,
    end_layer: int,
    attn_scale: Optional[torch.Tensor],
):
    """Set the learnable attention scale for a range of Phase 2 layers.

    Args:
        model: Qwen3ForCausalLM model with patched attention
        start_layer: First layer to set scale (inclusive)
        end_layer: Last layer to set scale (exclusive)
        attn_scale: Scalar tensor (0-d or (1,)) from E2EGate.attn_scale, or None
                   to disable scaling.  The tensor is stored by reference so
                   gradients flow back through it during training.
    """
    for i in range(start_layer, end_layer):
        attn = getattr(model.model.layers[i], 'self_attn', None)
        if attn is not None and hasattr(attn, 'set_attn_scale'):
            attn.set_attn_scale(attn_scale)
