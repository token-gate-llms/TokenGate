"""Packed-sequence training utilities for the E2E gate.

Status: **Phase A scaffold + segment-aware FlexAttention score_mod.**  Phase B
(full integration into `forward_with_gate_batched`) is intentionally NOT
landed in this PR — see the design plan at the bottom of this file.

Motivation
----------
With cross-dataset multitask / pretrain training, individual samples vary in
length by 10-30 ×.  Even with length-bucketed sampling, within a bucket the
DDP all-reduce still waits
for the longest sample at each step.  Packed sequences eliminate that by
concatenating N short samples into one batch element so every rank processes
the same total token count.

The trade-off: the gate's Phase-1 → top-K → Phase-2 forward currently assumes
*one* (article, question, answer) per batch element.  Packing requires
per-segment article boundaries, per-segment kept-token indices, and
per-segment loss masks.

What this file provides today (Phase A)
---------------------------------------
1. `make_packed_causal_gate_bias_score_mod(segment_ids, gate_bias, per_head)`:
   a FlexAttention `score_mod` that enforces *segment-causal* attention
   (causal within a segment, fully blocked across segments) AND adds the
   gate's per-token bias inside the segment boundary.  This is the piece
   the rest of the gate code can call once the larger integration lands.
2. `pack_samples(samples, *, max_total_len)`:
   First-Fit-Decreasing packer.  Takes a list of sample dicts (with `full_ids`,
   `article_start`, `article_end`, `ctx_len`, `n_ans`) and packs them into
   batch elements whose total token count ≤ max_total_len.  Returns the
   packed batch + a `segments` metadata list describing each segment's
   offsets, article range, and answer range within the packed element.
3. `make_segment_ids(segments, seq_len)`: builds a `(seq_len,)` int tensor
   tagging every token with its segment index, for the score_mod.
4. `Phase-A equivalence test`: runs the unpacked gate forward on two
   samples and the (scaffolded) packed forward, asserting logits at answer
   positions agree to ≤ 1e-3 in bf16.  The packed forward currently
   `raise NotImplementedError`; the test is wired but skipped pending Phase B.

What's deferred to Phase B
--------------------------
- `forward_with_gate_packed()` itself — the per-segment gate firing,
  per-segment top-K selection, per-segment KV-cache pruning, and per-segment
  loss-mask construction.  Estimated 400-600 LOC + a numerical equivalence
  test pass against `forward_with_gate_batched`.
- Integration with the dataset sampler so packed elements are produced
  during training.
- BlockMask construction that knows about segment boundaries (the existing
  bucketed cache key needs an extra dimension).

See the bottom of this file for the full Phase B integration plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# 1. Segment-aware FlexAttention score_mod
# ─────────────────────────────────────────────────────────────────────────────

def make_packed_causal_gate_bias_score_mod(
    segment_ids: torch.Tensor,
    gate_bias: Optional[torch.Tensor] = None,
    *,
    per_head: bool = False,
) -> Callable:
    """Build a FlexAttention `score_mod` for packed-causal + gate-bias attention.

    Semantics
    ---------
    - Attention is causal *within* a segment (q_idx >= kv_idx).
    - Attention is fully *blocked* across segments (segment_ids[q] != segment_ids[kv] → -inf).
    - If `gate_bias` is provided, its per-token value is added to the score
      for kv_idx positions inside the bias's coverage.  `per_head=True` selects
      a separate bias per query head.

    Args
    ----
    segment_ids:  (S_k,) int tensor on the same device as q/k.  Each token
                  carries its segment index (0, 1, 2, …).
    gate_bias:    None or (S_k,) or (B, S_k) or (S_k, H) or (B, S_k, H) —
                  same shape semantics as the existing
                  `flex_attention_gate_bias` API.  Passing None reduces to
                  pure segment-causal (no gate).
    per_head:     interpret the last dim of gate_bias as the query-head dim.

    Returns
    -------
    A `score_mod` function compatible with `torch.nn.attention.flex_attention`.
    """
    # Capture-by-closure pattern: FlexAttention treats `segment_ids` and
    # `gate_bias` as captured tensors and will avoid re-compiling when their
    # shapes match a previously-seen pattern.
    if gate_bias is None:
        def score_mod(score, b, h, q_idx, kv_idx):  # type: ignore[unused-argument]
            same_seg = segment_ids[q_idx] == segment_ids[kv_idx]
            causal = q_idx >= kv_idx
            return torch.where(same_seg & causal, score, torch.full_like(score, -float("inf")))
        return score_mod

    if per_head:
        # gate_bias is (S_k, H) or (B, S_k, H)
        if gate_bias.dim() == 2:
            _gb = gate_bias

            def score_mod_ph2(score, b, h, q_idx, kv_idx):  # type: ignore[unused-argument]
                same_seg = segment_ids[q_idx] == segment_ids[kv_idx]
                causal = q_idx >= kv_idx
                contribution = _gb[kv_idx, h]
                return torch.where(same_seg & causal, score + contribution,
                                   torch.full_like(score, -float("inf")))
            return score_mod_ph2

        _gb = gate_bias  # (B, S_k, H)

        def score_mod_ph3(score, b, h, q_idx, kv_idx):
            same_seg = segment_ids[q_idx] == segment_ids[kv_idx]
            causal = q_idx >= kv_idx
            contribution = _gb[b, kv_idx, h]
            return torch.where(same_seg & causal, score + contribution,
                               torch.full_like(score, -float("inf")))
        return score_mod_ph3

    # Scalar bias path: gate_bias is (S_k,) or (B, S_k)
    if gate_bias.dim() == 1:
        _gb = gate_bias

        def score_mod_s1(score, b, h, q_idx, kv_idx):  # type: ignore[unused-argument]
            same_seg = segment_ids[q_idx] == segment_ids[kv_idx]
            causal = q_idx >= kv_idx
            contribution = _gb[kv_idx]
            return torch.where(same_seg & causal, score + contribution,
                               torch.full_like(score, -float("inf")))
        return score_mod_s1

    _gb = gate_bias  # (B, S_k)

    def score_mod_s2(score, b, h, q_idx, kv_idx):  # type: ignore[unused-argument]
        same_seg = segment_ids[q_idx] == segment_ids[kv_idx]
        causal = q_idx >= kv_idx
        contribution = _gb[b, kv_idx]
        return torch.where(same_seg & causal, score + contribution,
                           torch.full_like(score, -float("inf")))
    return score_mod_s2


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pack metadata and the FFD packer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PackedSegment:
    """Where one logical sample lives inside a packed batch element.

    All offsets are within the packed sequence (NOT into the original sample).
    """
    seg_idx: int          # 0-based segment index within the packed element
    token_start: int      # first token belonging to this segment
    token_end: int        # one past the last token (so range = [start, end))
    article_start: int    # first article token (inside [token_start, token_end))
    article_end: int      # one past the last article token
    ctx_end: int          # token index where the context ends and answer begins
    n_ans: int            # answer token count
    K: int                # top-K budget for THIS segment


@dataclass
class PackedElement:
    """One batch element containing multiple packed segments."""
    input_ids: List[int]
    segments: List[PackedSegment]

    @property
    def total_len(self) -> int:
        return len(self.input_ids)


def pack_samples(
    samples: Sequence[dict],
    *,
    max_total_len: int,
    K_per_sample: Optional[Sequence[int]] = None,
) -> List[PackedElement]:
    """First-Fit-Decreasing pack samples into batch elements ≤ max_total_len.

    Each input sample dict must have:
        - "full_ids":      List[int]   the full token sequence (article + Q + A)
        - "article_start": int          start offset of the article portion
        - "article_end":   int          one-past-last offset of the article
        - "ctx_len":       int          end of context (start of answer)
        - "n_ans":         int          number of answer tokens
    Optional:
        - "K":             int          top-K budget; falls back to K_per_sample[i]

    Returns a list of `PackedElement`s.  Samples longer than max_total_len are
    placed in their own element (no truncation).
    """
    # Sort by length descending (the "decreasing" part of FFD).
    enumerated = sorted(
        enumerate(samples),
        key=lambda iv: -len(iv[1]["full_ids"]),
    )
    elements: List[PackedElement] = []

    for orig_i, s in enumerated:
        n = len(s["full_ids"])
        K_i = s.get("K") if "K" in s else (
            K_per_sample[orig_i] if K_per_sample is not None else 0
        )

        # Try to fit into an existing element.
        placed = False
        for elem in elements:
            if elem.total_len + n <= max_total_len:
                _append_segment(elem, s, K_i)
                placed = True
                break
        if not placed:
            elem = PackedElement(input_ids=[], segments=[])
            _append_segment(elem, s, K_i)
            elements.append(elem)
    return elements


def _append_segment(elem: PackedElement, sample: dict, K: int) -> None:
    """Mutate `elem` to add `sample` as a new segment at the tail."""
    base = elem.total_len
    n = len(sample["full_ids"])
    seg = PackedSegment(
        seg_idx=len(elem.segments),
        token_start=base,
        token_end=base + n,
        article_start=base + sample["article_start"],
        article_end=base + sample["article_end"],
        ctx_end=base + sample["ctx_len"],
        n_ans=sample["n_ans"],
        K=int(K),
    )
    elem.input_ids.extend(sample["full_ids"])
    elem.segments.append(seg)


def make_segment_ids(
    segments: Sequence[PackedSegment],
    seq_len: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the (S_k,) int tensor tagging every token with its segment index.

    Used as the `segment_ids` captured tensor for the score_mod above.
    """
    out = torch.zeros(seq_len, dtype=torch.int32, device=device)
    for seg in segments:
        out[seg.token_start:seg.token_end] = seg.seg_idx
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase-A scaffold for forward_with_gate_packed
# ─────────────────────────────────────────────────────────────────────────────

def forward_with_gate_packed(
    model: torch.nn.Module,
    gate: torch.nn.Module,
    packed_batch: List[PackedElement],
    gate_layer: int,
    *,
    frozen_backbone: bool = True,
    prune_phase1_kv: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed analogue of `forward_with_gate_batched`.

    Supported configuration (Phase-B v1)
    ------------------------------------
    - `frozen_backbone=True` (production default)
    - No LoRA (`lora_start_layer < 0`)
    - Scalar gate bias (`per_head=False`)
    - `input_all_prev_layers=False`
    - `prune_phase1_kv=True` (Phase 2 cache is rebuilt from kept tokens)

    These match `run_e2e_gate_multitask.sh` and the pretrain config; broader
    coverage is layered on in follow-up patches once this path is validated
    end-to-end against `forward_with_gate_batched`.

    Inputs / outputs match the batched API:
      Inputs:
        packed_batch: List[PackedElement] — one entry per packed sample.
                      Each carries `input_ids` + per-segment metadata
                      (`article_start`, `article_end`, `ctx_end`, `n_ans`, `K`).
      Returns:
        logits:      (B, max_n_kept, vocab_size)
        loss_mask:   (B, max_n_kept) bool — True at answer positions across
                     all segments.
        padding_mask:(B, max_n_kept) bool — True at non-padding positions.
    """
    # Late imports keep `packed_sequences.py` import-cheap when this path is
    # not used by the rest of the trainer.
    from transformers.cache_utils import DynamicCache  # type: ignore
    import torch.nn.functional as F  # noqa: F401
    from tokengate.e2e_gate.attn.qwen3_flex_attention import (
        set_gate_bias_for_layers,
        set_segment_ids_for_layers,
        set_attn_scale_for_layers,
    )
    # `_apply_gate_bias` handles the n_layers_bias > 0 case (per-(layer,head)
    # gate_scale + attn_scale) exactly as the batched path does; reuse it so
    # packed and unpacked share identical Phase-2 layer setup.
    # `build_gate_bias_vectorized` builds the pruned-sequence bias while keeping
    # the autograd graph from the gate scores intact.
    from tokengate.e2e_gate.gate_module.gate_forward import (
        _apply_gate_bias,
        build_gate_bias_vectorized,
    )

    inner = model.model
    device = next(model.parameters()).device
    B = len(packed_batch)
    max_total_len = max(elem.total_len for elem in packed_batch)
    n_layers = len(inner.layers)

    # ─── (1) Build padded input_ids, segment_ids, position_ids ────────────
    # segment_ids convention: 0 = padding, 1..N = segment index (so two
    # padded positions across different batch elements are NOT considered
    # in-segment with each other — see _segment_causal_mask_mod_*).
    input_ids = torch.zeros(B, max_total_len, dtype=torch.long, device=device)
    segment_ids = torch.zeros(B, max_total_len, dtype=torch.int32, device=device)
    position_ids = torch.zeros(B, max_total_len, dtype=torch.long, device=device)
    for b, elem in enumerate(packed_batch):
        input_ids[b, :elem.total_len] = torch.tensor(elem.input_ids, dtype=torch.long, device=device)
        for seg in elem.segments:
            seg_len = seg.token_end - seg.token_start
            segment_ids[b, seg.token_start:seg.token_end] = seg.seg_idx + 1
            position_ids[b, seg.token_start:seg.token_end] = torch.arange(
                seg_len, dtype=torch.long, device=device,
            )

    # ─── (2) Phase 1: full packed forward with segment-causal mask ────────
    set_gate_bias_for_layers(model, 0, gate_layer + 1, None)
    set_segment_ids_for_layers(model, 0, gate_layer + 1, segment_ids)
    with torch.no_grad():
        hidden_states = inner.embed_tokens(input_ids)
        dtype = hidden_states.dtype
        pe = inner.rotary_emb(hidden_states, position_ids)
        cache = DynamicCache()
        cache_pos = torch.arange(max_total_len, device=device)
    if frozen_backbone:
        with torch.no_grad():
            for i in range(gate_layer + 1):
                hidden_states = inner.layers[i](
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_pos,
                    position_embeddings=pe,
                )
    else:
        for i in range(gate_layer + 1):
            hidden_states = inner.layers[i](
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_pos,
                position_embeddings=pe,
            )
    hs_device = hidden_states.device

    # ─── (3) Per-segment gate firing + top-K ──────────────────────────────
    # We collect, per segment, the absolute kept indices in the packed
    # phase-1 sequence and the (possibly per-head) gate scores at the kept
    # ARTICLE positions.  Non-article kept positions get bias 0.
    gate_param = next(gate.parameters(), None)
    gdevice = gate_param.device if gate_param is not None else hs_device
    gdtype = gate_param.dtype if gate_param is not None else hidden_states.dtype

    # Per-element list of segments with their kept-in-packed indices, scores,
    # K_clamped, and per-segment offsets within the eventual phase-2 sequence.
    per_elem_segments: List[List[dict]] = [[] for _ in range(B)]
    max_n_kept = 0

    for b, elem in enumerate(packed_batch):
        cursor = 0  # offset in the phase-2 sequence for this element
        for seg in elem.segments:
            n_art = seg.article_end - seg.article_start
            if n_art <= 0:
                continue
            h_article = hidden_states[b, seg.article_start:seg.article_end, :].to(
                device=gdevice, dtype=gdtype,
            )
            b_scores = gate(h_article)  # (n_art,) scalar or (n_art, H)
            K_clamped = max(1, min(seg.K, n_art))
            with torch.no_grad():
                b_agg = gate.aggregate(b_scores).detach()
                topk_local = torch.topk(b_agg, K_clamped).indices  # local to article range

            # Build kept indices in the packed phase-1 sequence:
            #   [token_start..article_start)   pre-article tokens
            #   article_start + topk_local      kept article tokens
            #   [article_end..ctx_end + n_ans)  post-article + answer
            n_pre = seg.article_start - seg.token_start
            n_post = seg.token_end - seg.article_end
            kept_mask_local = torch.zeros(
                seg.token_end - seg.token_start, dtype=torch.bool, device=hs_device,
            )
            kept_mask_local[:n_pre] = True
            kept_mask_local[n_pre + (seg.article_end - seg.article_start):] = True
            kept_mask_local[n_pre + topk_local.to(hs_device)] = True
            kept_local_idx = kept_mask_local.nonzero(as_tuple=True)[0]
            kept_abs_idx = kept_local_idx + seg.token_start  # in packed phase-1 sequence

            n_kept_seg = kept_local_idx.shape[0]
            per_elem_segments[b].append({
                "seg": seg,
                "kept_abs_idx": kept_abs_idx,
                "n_pre": n_pre,
                "n_post": n_post,
                "K_clamped": K_clamped,
                "topk_local": topk_local,
                "b_scores": b_scores,
                "phase2_start": cursor,
                "phase2_end": cursor + n_kept_seg,
            })
            cursor += n_kept_seg
        max_n_kept = max(max_n_kept, cursor)

    if max_n_kept == 0:
        raise RuntimeError("forward_with_gate_packed: all segments had zero kept tokens")

    # ─── (4) Build phase-2 input + new segment_ids + positions + gate_bias
    hs_pruned = torch.zeros(B, max_n_kept, hidden_states.shape[2], dtype=dtype, device=hs_device)
    segment_ids_p2 = torch.zeros(B, max_n_kept, dtype=torch.int32, device=hs_device)
    position_ids_p2 = torch.zeros(B, max_n_kept, dtype=torch.long, device=hs_device)
    padding_mask = torch.zeros(B, max_n_kept, dtype=torch.bool, device=hs_device)
    loss_mask = torch.zeros(B, max_n_kept, dtype=torch.bool, device=hs_device)

    # Detect per-head from the first segment's gate output (must be consistent
    # across all segments because they share gate parameters).
    first_score = per_elem_segments[0][0]["b_scores"]
    per_head = (first_score.dim() == 2)
    n_gate_heads = first_score.shape[-1] if per_head else 1
    if per_head:
        gate_bias_p2 = torch.zeros(B, max_n_kept, n_gate_heads, dtype=dtype, device=hs_device)
    else:
        gate_bias_p2 = torch.zeros(B, max_n_kept, dtype=dtype, device=hs_device)

    # Per-segment KV gather buffers — concat across segments after this loop.
    kv_kept_per_layer: List[List[torch.Tensor]] = []  # outer index = layer, inner = batch
    for _ in range(gate_layer + 1):
        kv_kept_per_layer.append([])

    for b in range(B):
        # Gather and concatenate per-segment kept hidden states for this elem.
        if not per_elem_segments[b]:
            continue
        idxs_for_elem = [s["kept_abs_idx"] for s in per_elem_segments[b]]
        cat_idx = torch.cat(idxs_for_elem)
        n_kept_b = cat_idx.shape[0]
        hs_pruned[b, :n_kept_b, :] = hidden_states[b, cat_idx, :]
        padding_mask[b, :n_kept_b] = True
        # Segment ids + position ids for phase 2.  The position id of a kept
        # token must equal its PHASE-1 position (= segment-local index in the
        # packed phase-1 layout), so rotary embeddings see the same position
        # for the same logical token in unpacked vs packed.  We compute that
        # from the kept_abs_idx by subtracting the segment's token_start.
        for s in per_elem_segments[b]:
            seg = s["seg"]
            sp, ep = s["phase2_start"], s["phase2_end"]
            segment_ids_p2[b, sp:ep] = seg.seg_idx + 1
            position_ids_p2[b, sp:ep] = (s["kept_abs_idx"] - seg.token_start).to(torch.long)
            # Loss mask: the n_ans tokens just before the segment's phase-2 end
            # are the answer.  In phase-2 layout, [phase2_start..phase2_start+n_kept_seg)
            # ends at phase2_end; the last n_ans tokens are the answer's gold
            # targets minus the last (since loss is shifted).
            n_kept_seg = ep - sp
            # Answer-token positions in phase-2 layout: the post-article block
            # of length n_post (= ctx_end-article_end + n_ans) is at the tail.
            # Per the unpacked code, ans_start_logit = pruned_ctx_len - 1
            # where pruned_ctx_len = article_start_in_pruned + K_clamped + post_article_len.
            # The post_article_len = ctx_end - article_end (no n_ans).
            post_article_len = (seg.ctx_end - seg.article_end)
            pruned_ctx_len_seg = s["n_pre"] + s["K_clamped"] + post_article_len
            ans_start_in_seg = pruned_ctx_len_seg - 1
            ans_end_in_seg = ans_start_in_seg + seg.n_ans
            if ans_end_in_seg <= n_kept_seg:
                loss_mask[b, sp + ans_start_in_seg:sp + ans_end_in_seg] = True

            # Gate bias for this segment's phase-2 slice, built with the
            # gradient-preserving vectorized helper (same as the batched path).
            # We work in segment-local coordinates: article at
            # [n_pre, n_pre+n_art), kept indices = kept_abs_idx - token_start,
            # seq_len = segment length.
            seg_keep_local = (s["kept_abs_idx"] - seg.token_start)
            b_pruned_seg = build_gate_bias_vectorized(
                s["b_scores"], s["topk_local"],
                s["n_pre"], s["n_pre"] + (seg.article_end - seg.article_start),
                seg_keep_local, seg.token_end - seg.token_start, dtype,
            )  # (n_kept_seg,) or (n_kept_seg, H)
            if getattr(gate, "zero_gate_bias", False):
                b_pruned_seg = torch.zeros_like(b_pruned_seg)
            if per_head:
                gate_bias_p2[b, sp:ep, :] = b_pruned_seg.to(hs_device)
            else:
                gate_bias_p2[b, sp:ep] = b_pruned_seg.to(hs_device)
        # Padding positions beyond n_kept_b → -1e9 so phase-2 attention never
        # attends to them (matches the batched path's padding convention).
        if n_kept_b < max_n_kept:
            gate_bias_p2[b, n_kept_b:] = -1e9

    # Build pruned KV cache from phase-1 cache, gathered at kept indices.
    if prune_phase1_kv:
        legacy_entries = []
        for layer_idx in range(gate_layer + 1):
            layer = cache.layers[layer_idx]
            layer_device = layer.keys.device
            keys_b, vals_b = [], []
            for b in range(B):
                idxs_for_elem = [s["kept_abs_idx"] for s in per_elem_segments[b]]
                if not idxs_for_elem:
                    keys_b.append(torch.zeros(1, layer.keys.shape[1], 0, layer.keys.shape[3],
                                              dtype=layer.keys.dtype, device=layer_device))
                    vals_b.append(torch.zeros_like(keys_b[-1]))
                    continue
                cat_idx = torch.cat(idxs_for_elem).to(layer_device)
                n_kept_b = cat_idx.shape[0]
                k_i = layer.keys[b:b + 1, :, cat_idx, :]
                v_i = layer.values[b:b + 1, :, cat_idx, :]
                pad = max_n_kept - n_kept_b
                if pad > 0:
                    k_i = torch.nn.functional.pad(k_i, (0, 0, 0, pad))
                    v_i = torch.nn.functional.pad(v_i, (0, 0, 0, pad))
                keys_b.append(k_i)
                vals_b.append(v_i)
            legacy_entries.append((torch.cat(keys_b, dim=0), torch.cat(vals_b, dim=0)))
        cache = DynamicCache(ddp_cache_data=tuple(legacy_entries), config=model.config)

    # ─── (5) Phase 2 forward (segment-causal + gate-bias) ─────────────────
    set_segment_ids_for_layers(model, gate_layer + 1, n_layers, segment_ids_p2)
    # _apply_gate_bias sets the per-layer gate bias AND, when
    # gate.n_layers_bias > 0, the per-(layer,head) gate_scale / attn_scale —
    # mirroring the batched path exactly.  For n_layers_bias == 0 it just
    # broadcasts one bias to all layers and the caller sets the scalar
    # attn_scale separately (below).
    _apply_gate_bias(model, gate, gate_layer, n_layers, gate_bias_p2, per_head)
    if gate.n_layers_bias == 0:
        _attn_scale = gate.attn_scale if hasattr(gate, "attn_scale") else None
        set_attn_scale_for_layers(model, gate_layer + 1, n_layers, _attn_scale)
    pe_pruned = inner.rotary_emb(hs_pruned, position_ids_p2)
    cache_pos_pruned = torch.arange(max_n_kept, device=hs_device)
    for i in range(gate_layer + 1, n_layers):
        hs_pruned = inner.layers[i](
            hs_pruned,
            attention_mask=None,
            position_ids=position_ids_p2,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_pos_pruned,
            position_embeddings=pe_pruned,
        )

    hs_pruned = inner.norm(hs_pruned)
    logits = model.lm_head(hs_pruned)

    # Clear segment_ids on attention modules so subsequent unpacked calls
    # (eval, etc.) use the regular causal path.  Caller must NOT rely on
    # segment_ids being still set after this call.
    set_segment_ids_for_layers(model, 0, n_layers, None)

    return logits, loss_mask, padding_mask


# ─────────────────────────────────────────────────────────────────────────────
# 4. Phase B integration plan
# ─────────────────────────────────────────────────────────────────────────────
# Order of operations for the full integration, in roughly the order they
# should be implemented and tested:
#
#  a. Pack data structure       — `PackedElement` + `pack_samples`  ✅ here
#  b. Segment-aware score_mod   — `make_packed_causal_gate_bias_score_mod`  ✅ here
#  c. Phase-1 forward on packed element
#     - Same as today but with a `segment_ids`-aware causal mask via the
#       score_mod above (BlockMask cached by (total_len, max_seg) tuple).
#     - Output: hidden states at `gate_layer + 1` for all tokens.
#  d. Per-segment gate firing
#     - For each segment, slice the hidden states to its article range and
#       call `gate(article_hs_per_segment)`.  Concatenate per-segment scores
#       into one `(S_k,)` or `(S_k, H)` tensor padded with -1e9 at non-article
#       positions.
#  e. Per-segment top-K
#     - For each segment, take top-K among its article tokens only.
#     - Build per-segment kept_indices: List[Tensor[K_seg]].
#     - Concatenate kept tokens (article-kept) + post-article (Q + A) per
#       segment into the phase-2 input.
#  f. KV-cache rebuild
#     - For each segment, gather K/V at the segment's kept indices, then
#       concatenate per-segment slices in segment order.
#     - The packed phase-2 sequence has the same `segment_ids` topology, so
#       the same score_mod is reused.
#  g. Phase-2 forward
#     - Apply `set_gate_bias_for_layers` with the SEGMENT-AWARE score_mod
#       (instead of the current scalar-or-per-head one).  This needs a small
#       extension in `qwen3_flex_attention.py` to choose the score_mod factory
#       based on whether `segment_ids` is set on the attention module.
#  h. Loss mask construction
#     - For each segment, mark its answer-token positions in the packed
#       phase-2 output.  Aggregate per-segment losses.
#
# Validation strategy for Phase B:
#  1. Numerical equivalence test: run two samples through
#     `forward_with_gate_batched` (unpacked) and through
#     `forward_with_gate_packed` (1 packed element of 2 segments).  Logits at
#     each segment's answer positions should agree to < 1e-3 in bf16.
#  2. DDP straggler test: time a 100-step training run on 8 GPUs with packed
#     vs unpacked length-bucketed sampling.  Expected ≥ 30 % wall-time win on
#     the multitask config.
#  3. Loss-curve regression: train 1k steps with seeds matched and confirm
#     the loss curve overlaps within noise band.
