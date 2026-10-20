"""Shared forward pass logic for E2E Gate (training and inference).

This module contains the core forward pass logic used by both training
and inference, eliminating code duplication.
"""

from typing import List, Optional, Tuple

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin

# ── Opt-in phase-timing instrumentation (training_bench/homogenization) ────────
# When PROFILE_PHASES=1, forward_backward_with_gate_multi_K_detach records CUDA
# events around Phase-1 fwd, the gate, each per-K Phase-2 fwd+bwd, and the single
# Phase-1 bwd, and stashes them (plus per-K context lengths) in PHASE_TIMING so
# the trainer's rank-0 [profile] path can read elapsed_time once per step.  OFF by
# default → the production forward is byte-for-byte unchanged (guard skipped, no
# events, no sync).  Records events only; the ONE sync/step happens in the trainer
# (rank 0), so non-master ranks pay nothing.
PROFILE_PHASES = os.environ.get("PROFILE_PHASES", "0") == "1"
# Packed Phase 2 (forward_backward_with_gate_multi_K_detach): set by the trainer from
# --pack-phase2-max-tokens (0 = off), never from the environment, so params.json records it.
PACK_PHASE2 = False
PACK_PHASE2_MAX_TOKENS = 0
# Populated per step by the detach path when PROFILE_PHASES; consumed + reset by
# the trainer.  Holds raw torch.cuda.Event objects (timing read on rank 0) plus
# scalar Phase-1/Phase-2 context lengths and the per-K clamped K values.
PHASE_TIMING: Optional[dict] = None
# Per-K training loss from the last multi-K detach step (rank-local): parallel to K_list.
# {"K": [...], "loss_sum": [...], "n": [...]}; the trainer maps K index → phase-2 length and
# logs mean = loss_sum/n to wandb.  Refreshed every step.
LAST_PER_K_TRAIN_LOSS: Optional[dict] = None
# prefill_batch_for_decode phase timing (CUDA events on the current stream, read
# after a sync): enabled by set_prefill_phase_timing(True); LAST_PREFILL_PHASES then
# holds [(name, start_event, end_event), ...] of the most recent call.
_PREFILL_PHASE_TIMING = False
LAST_PREFILL_PHASES: Optional[list] = None


def set_prefill_phase_timing(enabled: bool) -> None:
    global _PREFILL_PHASE_TIMING
    _PREFILL_PHASE_TIMING = bool(enabled)


def prefill_phase_ms() -> dict:
    """Elapsed ms per phase of the last prefill_batch_for_decode (synchronizes)."""
    torch.cuda.synchronize()
    return {n: s.elapsed_time(e) for n, s, e in (LAST_PREFILL_PHASES or [])}

from tokengate.e2e_gate.gate_module.e2e_gate_module import (
    E2EGate, CastLinear, assert_fp32_params,
)
from tokengate.e2e_gate.attn.kv_cache import PrefillCache
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    set_gate_bias_for_layers,
    set_attn_scale_for_layers,
    set_packed_cu_for_layers,
    set_ssmax_ratio_for_layers,
)


_PRUNE_STREAMS: dict = {}


def _prune_stream(device) -> torch.cuda.Stream:
    key = torch.device(device).index
    if key not in _PRUNE_STREAMS:
        _PRUNE_STREAMS[key] = torch.cuda.Stream(device=device)
    return _PRUNE_STREAMS[key]


def _keep_indices(topk_local, article_start, article_end, seq_len, device):
    parts = [torch.arange(article_start, device=device),
             article_start + topk_local.to(device),
             torch.arange(article_end, seq_len, device=device)]
    return torch.sort(torch.cat(parts)).values


def _select_topk_or_random(
    b_for_topk: torch.Tensor,
    K_clamped: int,
    random_topk: bool,
) -> torch.Tensor:
    """Return K_clamped article-relative indices.

    When ``random_topk`` is True, picks a fresh uniform random K-subset of the
    article positions (without replacement) instead of running torch.topk on the
    gate scores.  The gate's score tensor is ignored except to take its length /
    device — used by the random-K baseline (see --random-topk in train_e2e_gate.py).
    """
    if random_topk:
        n = b_for_topk.shape[0]
        return torch.randperm(n, device=b_for_topk.device)[:K_clamped]
    return torch.topk(b_for_topk, K_clamped).indices


def _select_threshold(
    b_for_topk: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Return article-relative indices of tokens with sigmoid(b) >= tau.

    Non-strict comparison: at init W_B=0 → b=0 → σ(b)=0.5 exactly.  Strict
    `σ > τ` would reject EVERY token at τ=0.5 and fall through to argmax-1,
    making the starting state degenerate (1 article token kept regardless of
    article length).  With `>=`, init keeps every article token at τ=0.5
    (σ=0.5 >= 0.5 passes), the model starts from no-compression behaviour,
    and the gate learns to slim the kept set down over training.

    Number of kept indices is sample-dependent.  If the threshold rejects
    every token (sparsity collapse on a single sample, post-training) we
    keep the single argmax so Phase 2 always has at least one article token
    to attend to and downstream shape invariants hold.
    """
    keep_mask = torch.sigmoid(b_for_topk) >= tau
    indices = keep_mask.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return b_for_topk.argmax(dim=0, keepdim=True)
    return indices


def _select_indices(
    b_for_topk: torch.Tensor,
    K_clamped: int,
    gate,
) -> torch.Tensor:
    """Dispatch to the active selection mode (topk or threshold).

    ``gate.selection_mode == "topk"`` (default) preserves the prior behaviour
    exactly (random-K baseline still applies when ``gate.random_topk`` is set).
    ``"threshold"`` ignores ``K_clamped`` and keeps every article token whose
    sigmoid(b) exceeds ``gate.tau``.
    """
    if getattr(gate, "selection_mode", "topk") == "threshold":
        return _select_threshold(b_for_topk, getattr(gate, "tau", 0.5))
    return _select_topk_or_random(
        b_for_topk, K_clamped, getattr(gate, "random_topk", False),
    )








def _accumulate_gate_sparsity(gate, b_agg: torch.Tensor, n_kept: int) -> None:
    """Accumulate a differentiable sparsity term + per-sample kept-count on the gate.

    The trainer initialises ``gate._sparsity_term`` to a zero scalar (with
    ``requires_grad=True`` because b_agg is grad-tracked here) before each
    forward and reads it afterwards, adding ``lambda * term / n_valid`` to the
    loss.

    Two penalty formulations, switched by ``gate.sparsity_mode`` (defaults
    to ``"soft_mean"`` for back-compat with existing runs):

      "soft_mean": mean(sigmoid(b)) over article tokens. Original formulation.
        Per-token gradient ∝ σ'(b_i)/n, which (for n ~ 6000) is essentially
        negligible per token but uniformly pulls every score down across many
        steps — produces gates whose sigmoid(b) is below tau everywhere, so
        _select_threshold's argmax-fallback fires on every article and the
        threshold mechanism is effectively bypassed.

      "soft_count": sum(sigmoid((b - logit(tau)) * sharpness)) over article
        tokens. Soft, differentiable approximation of "number of tokens
        with sigmoid(b) > tau". Per-token gradient ∝ σ'((b-logit_tau)*s) * s
        — peaks AT the decision boundary and saturates far from it. Drives
        a bimodal distribution: tokens settle either definitively above tau
        (gradient saturates) or definitively below (gradient saturates), with
        the count of above-tau tokens controlled directly by lambda.

      "soft_count_frac": same shape as soft_count but normalised by n_art
        (mean instead of sum). The term is then a soft estimate of the
        FRACTION of article tokens above tau, bounded in [0, 1] regardless
        of article length. Per-token gradient ∝ σ'(...)·sharpness / n_art,
        so the total magnitude of the sparsity loss no longer scales with
        article size — useful when n_art varies a lot across datasets/batches
        and the count form would swamp the CE signal on long articles.
        Empirically calibrated: λ_frac sets a fixed "pressure" decoupled
        from article length (~λ_count·E[n_art] for absolute parity, but the
        whole point is *not* to match — start with λ_frac that puts
        sparsity at ~0.3–0.5×CE).

    sharpness controls how sharply the soft sigmoid approximates a step:
    larger → more like a hard count, harder to optimise; smaller → smoother
    gradient signal. 5-10 is a reasonable starting range.

    Kept-count is appended for per-sample logging in threshold mode (variable K).

    Eval-time capture: set ``gate._sparsity_eval_capture = True`` before the
    eval loss loop and ``False`` after, so eval can compute the same penalty
    component for W&B without polluting the training step's accumulation.
    """
    if not gate.training and not getattr(gate, "_sparsity_eval_capture", False):
        return
    mode = getattr(gate, "sparsity_mode", "soft_mean")
    if mode in ("soft_count", "soft_count_frac", "soft_count_frac_l2"):
        tau = float(getattr(gate, "tau", 0.5))
        sharpness = float(getattr(gate, "sparsity_sharpness", 5.0))
        # logit(tau) is the b-value at which sigmoid(b) == tau.
        # math.log(tau / (1-tau)); compute inline as a tensor scalar for grad-safety.
        logit_tau = math.log(tau / (1.0 - tau))
        # Soft count above tau: sigmoid((b - logit_tau) * sharpness) ≈ 1[sigmoid(b)>tau]
        # soft_count uses .sum() (scales with n_art); soft_count_frac uses
        # .mean() (bounded in [0,1]; magnitude decoupled from article length).
        # soft_count_frac_l2 squares the mean — gradient on b is proportional
        # to p (not constant), so sparsity pressure DECREASES as the kept
        # fraction shrinks.  Doesn't add a restoring force vs argmax-1, but
        # delays the collapse trajectory because near p=0 the gradient is
        # ~0 instead of ~constant.
        per_tok = torch.sigmoid((b_agg - logit_tau) * sharpness)
        if mode == "soft_count":
            soft = per_tok.sum()
        elif mode == "soft_count_frac":
            soft = per_tok.mean()
        else:  # soft_count_frac_l2
            p = per_tok.mean()
            soft = p * p
    elif mode == "soft_mean":
        soft = torch.sigmoid(b_agg).mean()
    else:
        raise ValueError(
            f"Unknown gate.sparsity_mode={mode!r}; expected one of "
            "'soft_mean', 'soft_count', 'soft_count_frac', 'soft_count_frac_l2'."
        )
    if not hasattr(gate, "_sparsity_term") or gate._sparsity_term is None:
        gate._sparsity_term = soft
    else:
        gate._sparsity_term = gate._sparsity_term + soft
    if not hasattr(gate, "_kept_counts") or gate._kept_counts is None:
        gate._kept_counts = []
    gate._kept_counts.append(int(n_kept))
    if not hasattr(gate, "_n_art_counts") or gate._n_art_counts is None:
        gate._n_art_counts = []
    gate._n_art_counts.append(int(b_agg.shape[0]))


def _full_attn_layers_above(model: torch.nn.Module, gate_layer: int) -> list:
    """Return full-attention layer indices strictly above ``gate_layer``.

    Reads ``model._full_attn_layer_indices`` (populated by
    ``patch_model_with_flex_attention``).  For homogeneous Qwen3 this is just
    every layer above ``gate_layer``; for Qwen3.5 hybrid only the full-attn
    subset is returned.  Falls back to a dense range when the attribute is
    missing (e.g. unit tests that don't run the patcher), which preserves
    legacy behaviour.
    """
    full_attn = getattr(model, '_full_attn_layer_indices', None)
    n_layers = len(model.model.layers)
    if full_attn is None:
        return list(range(gate_layer + 1, n_layers))
    return [i for i in full_attn if i > gate_layer]


def build_gate_branch(model, gate, gate_layer: int):
    """Create the dedicated SCORING branch as a trainable clone of `gate_layer`.

    The branch is a copy of the block's MLP: h_gate = a_j + MLP_gate(LN2(a_j)),
    fed the a_j phase boundary (split_gate_layer_mlp) — no hooks, no second
    attention pass.

    Initialized from the CURRENT layer-`gate_layer` weights, so at step 0
    h_gate == h_fwd exactly and the model matches the tied architecture bit-for-bit.
    Attached by the caller as `gate.gate_branch`.

    Parameter STORAGE is always float32 (CastLinear): the optimizer update and the
    SWA running mean both need it -- a bf16 param rounds away every per-step Muon
    update below its ULP (at |w|~0.02: ~80% of elements at lr=1e-4, ~97% at 1.4e-5),
    and a bf16 running mean decays as n0/n (tests/test_swa_precision.py).
    COMPUTE runs in the dtype of the branch input (the backbone's a_j boundary: bf16
    for a bf16 backbone) -- CastLinear casts its fp32 weights to the input dtype per
    forward, so the matmuls stay at backbone precision and cost (the branch is
    150.99M; there is no autocast in the training path, and fp32 compute measured
    ~+1.08 s/it).  There is no dtype knob: fp32 storage + backbone-dtype compute is
    the one behaviour, identical for training, SWA copies, bench and compressor.
    """
    import copy
    src = model.model.layers[gate_layer]
    br = copy.deepcopy(src.mlp)

    # MATERIALIZE every linear into a fp32-storage CastLinear.  The source layer is
    # usually adapter-wrapped (DeltaLinear = frozen .linear + trainable .delta, or
    # LoRALinear): fold W_eff = W + delta (resp. W + scaling * (A @ B)^T) in fp32 so the
    # branch is an independent copy of the CURRENT effective weights with no adapter
    # plumbing and none of the frozen base unfrozen.
    def _effective(child):
        if child.__class__.__name__ in ("DeltaLinear", "LoRALinear"):
            base = child.linear
            w = base.weight.detach().to(torch.float32)
            if child.__class__.__name__ == "DeltaLinear":
                w = w + child.delta.detach().to(torch.float32)
            else:
                A = child.lora_A.detach().to(torch.float32)
                B = child.lora_B.detach().to(torch.float32)
                w = w + child.scaling * (A @ B).t()
            b = None
            if base.bias is not None:
                b = base.bias.detach().to(torch.float32)
                db = getattr(child, "delta_bias", None)
                if db is not None:
                    b = b + db.detach().to(torch.float32)
            return w, b
        if isinstance(child, (nn.Linear, CastLinear)):
            return (child.weight.detach().to(torch.float32),
                    None if child.bias is None else child.bias.detach().to(torch.float32))
        return None

    def _materialize(mod):
        for name, child in list(mod.named_children()):
            eff = _effective(child)
            if eff is not None:
                setattr(mod, name, CastLinear.from_tensors(*eff))
            else:
                _materialize(child)
    _materialize(br)

    # Absorbed readout: the branch's job is no longer to produce a d_model hidden
    # state but to produce this half of the SCORE, so `down_proj` (4096, 12288) is
    # replaced by the composed row (W_g W_down), shape (1, 12288). Zero-init, which
    # matches the readout it replaces -- W_g is zero-init, so the product it stands
    # for is zero anyway and the gate still emits b = 0 at step 0.
    if getattr(gate, "absorbed_readout", False):
        _dp = br.down_proj
        _row = CastLinear(_dp.in_features, 1, bias=_dp.bias is not None,
                          device=_dp.weight.device)
        with torch.no_grad():
            nn.init.zeros_(_row.weight)
            if _row.bias is not None:
                nn.init.zeros_(_row.bias)
        br.down_proj = _row

    for prm in br.parameters():
        prm.requires_grad_(True)
    assert_fp32_params(br, "gate scoring branch")
    gate.gate_branch = br
    return gate.gate_branch


def split_gate_layer_mlp(gate) -> bool:
    """True when Phase 1 ends at the gate layer's attention residual a_j — i.e.
    whenever a scoring branch is attached (mode 'mlp' is the only branch mode).

    The split hands the branch the boundary tensor directly (no hooks; in training
    the branch reads hs_d, so the scoring gradient reaches Phase 1 through the same
    single reconnection as the LM gradient) and runs the BACKBONE gate-layer MLP on
    kept tokens only at Phase-2 entry (`gate_layer_mlp`) — exact, because the MLP is
    position-wise (tests/test_split_gate_layer.py).
    """
    return getattr(gate, "gate_branch", None) is not None


# Compiled twins of the gate layer's two halves (attn_only_forward /
# gate_layer_mlp), installed by set_split_layer_compile(True) for inference
# configurations that compile Phase 1 (kernel parity); off, the eager functions
# run exactly as trained.
_SPLIT_COMPILED: dict = {}


def set_split_layer_compile(enabled: bool) -> None:
    _SPLIT_COMPILED.clear()
    if enabled:
        _SPLIT_COMPILED["attn"] = torch.compile(_attn_only_forward_eager, dynamic=True, fullgraph=False)
        _SPLIT_COMPILED["mlp"] = torch.compile(_gate_layer_mlp_eager, dynamic=True, fullgraph=False)


def attn_only_forward(layer, hidden_states, position_ids, cache, cache_pos, pe):
    fn = _SPLIT_COMPILED.get("attn", _attn_only_forward_eager)
    return fn(layer, hidden_states, position_ids, cache, cache_pos, pe)


def gate_layer_mlp(layer, a):
    fn = _SPLIT_COMPILED.get("mlp", _gate_layer_mlp_eager)
    return fn(layer, a)


def _attn_only_forward_eager(layer, hidden_states, position_ids, cache, cache_pos, pe):
    """The gate layer's forward UP TO the token-mixer residual: returns a_j.

    Mirrors the decoder layer's forward exactly (input_layernorm → mixer →
    residual add), stopping before post_attention_layernorm + MLP, which run at
    Phase-2 entry on kept tokens only (`gate_layer_mlp`). The layer's cache write
    (KV for full attention, conv/recurrent state for DeltaNet) happens inside the
    mixer call as usual, so cache handling downstream is unchanged.

    Hybrid Qwen3.5: a linear-attention gate layer dispatches to `linear_attn`
    with its own signature (no RoPE, no cache_position — mirrors
    Qwen3_5DecoderLayer.forward); homogeneous Qwen3 layers have no `layer_type`
    attribute and take the full-attention path unchanged.
    """
    residual = hidden_states
    hs = layer.input_layernorm(hidden_states)
    if getattr(layer, "layer_type", "full_attention") == "linear_attention":
        hs = layer.linear_attn(
            hidden_states=hs,
            cache_params=cache,
            attention_mask=None,
        )
    else:
        hs, _ = layer.self_attn(
            hidden_states=hs,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_pos,
            position_embeddings=pe,
        )
    return residual + hs


def _gate_layer_mlp_eager(layer, a):
    """The gate layer's MLP sublayer: a + MLP(LN2(a)). Position-wise.

    Applied to the PRUNED hidden states at Phase-2 entry when the split is active
    (`split_gate_layer_mlp`). Uses the backbone layer's own (possibly delta-adapted)
    modules. Batched callers pass right-zero-padded rows: RMSNorm and the Qwen3 MLP
    linears are bias-free, so zero rows map to zero and padding is preserved.
    """
    return a + layer.mlp(layer.post_attention_layernorm(a))


def call_layer_padsafe(layer, hs, pad_lens, **kwargs):
    """Decoder-layer call that stays exact under RIGHT-zero-padded rows for
    hybrid linear-attention (DeltaNet) layers.

    DeltaNet's recurrent/conv state is mutated by every position it sees:
    trailing pad tokens decay the state and shift the conv window, poisoning
    the cache that DECODE continues from (measured 2026-09-10: keep-all
    boundary-3 heldout 0.679 vs 1.0, misses exactly the padded rows).
    Rolling each row's padding to the FRONT makes the pad prefix a state
    no-op — a zero state stays exactly zero (k=v=0 writes nothing, decay
    multiplies 0, conv window flushes with the real tail) — so the real
    tokens run on a fresh state and the layer's final cache state is exact.
    The output is rolled back so the surrounding right-padded layout
    (full-attn KV slots, last-valid indexing) is untouched.  Full-attention
    layers (and pad-free batches) pass through unchanged.  Loss/logit
    positions were never affected (causality), so the TRAIN forward does not
    route through this helper."""
    call = kwargs.pop("fn", layer)
    if getattr(layer, "layer_type", "full_attention") != "linear_attention" \
            or not any(pad_lens):
        return call(hs, **kwargs)
    B, S, H = hs.shape
    dev = hs.device
    ar = torch.arange(S, device=dev).unsqueeze(0).expand(B, S)
    pads = torch.tensor(pad_lens, device=dev, dtype=torch.long).unsqueeze(1)
    fwd = (ar - pads) % S                       # rolled[j] = hs[(j - p) mod S]
    rolled = hs.gather(1, fwd.unsqueeze(-1).expand(-1, -1, H))
    out = call(rolled.masked_fill((ar < pads).unsqueeze(-1), 0), **kwargs)
    back = (ar + pads) % S
    return out.gather(1, back.unsqueeze(-1).expand(-1, -1, out.shape[2]))


def gate_branch_hs(gate, model, gate_layer, boundary):
    """h^gate for the scoring branch: boundary + MLP_gate(LN_2(boundary)).

    Returns `boundary` unchanged when no branch is attached (legacy tied gate).
    `boundary` is the a_j phase boundary — hs_d in the detach trainer (so the scoring
    gradient reaches Phase 1 through the same single reconnection as the LM gradient),
    the plain forwarded boundary in no-grad eval paths. No hooks, no capture: the
    branch's input IS the Phase-1 output (see split_gate_layer_mlp).
    """
    br = getattr(gate, "gate_branch", None)
    if br is None:
        if getattr(gate, "absorbed_readout", False):
            raise ValueError("absorbed_readout has no meaning without a scoring "
                             "branch: the fold is of the readout INTO the branch's "
                             "last layer.")
        return boundary
    compiled = getattr(gate, "_compiled_branch_hs", None)
    if compiled is not None:
        return compiled(boundary)
    layer = model.model.layers[gate_layer]
    z = layer.post_attention_layernorm(boundary)
    # The branch computes in z's dtype (the backbone's): its params are fp32 storage
    # and CastLinear casts them to the input dtype per forward.
    if getattr(gate, "absorbed_readout", False):
        # The score itself (width 1) rather than a d_model hidden state: the branch,
        # whose down_proj is now the composed row (W_g W_down), is the MLP half, and
        # `proj` is the residual half W_g.a_j when there is one. E2EGate.forward takes
        # this as its raw score.
        mlp = br(z)
        if not getattr(gate, "readout_residual", True):
            return mlp.to(boundary.dtype)   # the branch is the whole readout
        res = gate.proj(boundary.to(dtype=gate.proj.weight.dtype))
        return (res + mlp.to(res.dtype)).to(boundary.dtype)
    return boundary + br(z).to(boundary.dtype)


_PHASE2_BUCKET = None
_PHASE2_GRAPHS = 0


def set_phase2_bucket(n, cuda_graphs: int = 0) -> None:
    """Pad the pruned (Phase-2) sequence to a multiple of n tokens.  Exact for
    every real row: the pads sit after the last kept token, causal attention
    never lets a real query see them, their bias is -1e9 besides, their own
    rows are discarded and the caches are cut back before returning.  Bounds
    the number of distinct Phase-2 shapes, which is what lets CUDA graphs
    (compile mode reduce-overhead, one recording per shape) cover them.
    cuda_graphs: 1 -- the Phase-2 layers are compiled in inductor's
    reduce-overhead mode (a graph per layer), so every prefill opens a new
    cudagraph-trees step; 2 -- the whole Phase-2 loop is one graph per shape
    (phase2_graph.replay_layers).  Either way the previous call's Phase-2 K/V
    may be overwritten by the next prefill: a caller that keeps its cache
    across the next prefill must clone it."""
    global _PHASE2_BUCKET, _PHASE2_GRAPHS
    _PHASE2_BUCKET = None if n is None else int(n)
    _PHASE2_GRAPHS = int(cuda_graphs)


def compile_gate_branch(gate, model, gate_layer) -> None:
    """torch.compile the scoring branch's forward (gate_branch_hs's body: the
    layer norm, the branch MLP, the readout residual and the casts) for the
    kernel-parity configuration.  Fusing the norm, the activation and the casts
    changes their roundings, so the selection is no longer training's: never
    for the bit-exact configuration (load_gate_model wires it for compile "all")."""
    br = getattr(gate, "gate_branch", None)
    if br is None:
        raise ValueError("compile_gate_branch: the gate has no scoring branch")
    if not getattr(gate, "absorbed_readout", False):
        raise ValueError("compile_gate_branch: only the absorbed-readout branch is wired")
    layer = model.model.layers[gate_layer]
    residual = getattr(gate, "readout_residual", True)

    def _hs(boundary):
        z = layer.post_attention_layernorm(boundary)
        mlp = br(z)
        if not residual:
            return mlp.to(boundary.dtype)
        res = gate.proj(boundary.to(dtype=gate.proj.weight.dtype))
        return (res + mlp.to(res.dtype)).to(boundary.dtype)

    gate._compiled_branch_hs = torch.compile(_hs, dynamic=True, fullgraph=False)


def _ssmax_ratio(kept_positions: torch.Tensor) -> torch.Tensor:
    """r_q = log(m_q)/log(p_q) per phase-2 slot. kept_positions holds the
    ORIGINAL 0-based position ids of the kept tokens, ordered; m_q is the 1-based
    slot index. Clamped so r==1 for the first tokens (p<2) and always r<=1."""
    if kept_positions.dim() == 1:
        kept_positions = kept_positions.unsqueeze(0)
    m = torch.arange(1, kept_positions.shape[1] + 1,
                     device=kept_positions.device, dtype=torch.float32)
    m = m.unsqueeze(0).clamp(min=2.0)
    pos = (kept_positions.to(torch.float32) + 1.0).clamp(min=2.0)
    r = torch.log(m) / torch.log(pos)
    return r.clamp(max=1.0)


def _apply_gate_bias(
    model: torch.nn.Module,
    gate: E2EGate,
    gate_layer: int,
    n_layers: int,
    gate_bias: torch.Tensor,
    per_head: bool,
    ssmax_ratio: "Optional[torch.Tensor]" = None,
) -> None:
    """Set gate bias and attn_scale for Phase 2 layers.

    When ``gate.n_layers_bias > 0``, each Phase 2 *full-attention* layer gets its
    own:
      - ``gate_scale[l_idx]``: per-(layer, head) RMS output scale applied to gate_bias
      - ``attn_scale[l_idx]``: per-(layer, head) attention temperature

    ``l_idx`` indexes over the full-attention layers above ``gate_layer`` only.
    For Qwen3 this is identical to the previous behaviour.  For Qwen3.5 hybrid,
    DeltaNet layers above ``gate_layer`` are skipped — they have no per-token
    KV cache to bias.

    ``gate_bias`` entering here is already RMS-normalized (but unscaled) by
    ``E2EGate.forward()`` when ``normed_gate=True`` and ``n_layers_bias > 0``.

    When ``gate.n_layers_bias == 0``, applies the same gate_bias uniformly (the
    range still spans ``[gate_layer+1, n_layers)``; ``set_gate_bias_for_layers``
    silently skips DeltaNet layers via the ``hasattr`` check).
    """
    full_attn_above = _full_attn_layers_above(model, gate_layer)

    # BASELINE-EVAL mode: the in-training randomk diagnostic swaps only the
    # SELECTION rule but otherwise runs this same gate path, so without this it would
    # inherit the gate's learned bias AND attn_scale — i.e. it would drift as the gate
    # trains. Clearing both here keeps the diagnostic selection-only.
    if getattr(gate, "_baseline_no_bias_scale", False):
        set_gate_bias_for_layers(model, gate_layer + 1, n_layers, None)
        set_attn_scale_for_layers(model, gate_layer + 1, n_layers, None)
        set_ssmax_ratio_for_layers(model, gate_layer + 1, n_layers, None)
        return

    if gate.n_layers_bias > 0:
        if len(full_attn_above) != gate.n_layers_bias:
            raise ValueError(
                f"n_layers_bias mismatch: gate has {gate.n_layers_bias} layer slots "
                f"but model has {len(full_attn_above)} full-attention layers above "
                f"gate_layer={gate_layer}: {full_attn_above}"
            )
        # per_layer_gate: gate_bias carries its own layer axis (..., L', H) and was ALREADY
        # scaled by gate_scale inside E2EGate.forward(), so slice the layer and do NOT
        # re-apply gate_scale (that would square it, as the rank-1 path effectively does).
        # Scaled-softmax mode: store/clear the per-query ratio for the
        # whole Phase-2 range (layer-independent; gamma stays per-layer via
        # attn_scale slices below). Always set, so no stale tensor survives.
        set_ssmax_ratio_for_layers(
            model, gate_layer + 1, n_layers,
            ssmax_ratio if getattr(gate, "scaled_softmax_scale", False) else None)
        _per_layer = getattr(gate, "per_layer_gate", False) and gate_bias.dim() >= 3
        has_gate_scale = (not _per_layer) and hasattr(gate, 'gate_scale') and gate.gate_scale.dim() == 2
        has_attn_scale = hasattr(gate, 'attn_scale') and gate.attn_scale.dim() == 2
        # One softplus over the (L', H) table, sliced per layer below -- not one
        # kernel per layer.
        _attn_scale_eff = gate.attn_scale_eff() if has_attn_scale else None
        for l_idx, l in enumerate(full_attn_above):
            if _per_layer:
                scaled = gate_bias[..., l_idx, :]                      # (..., n_kept, H)
            elif has_gate_scale:
                scale = gate.gate_scale[l_idx].to(dtype=gate_bias.dtype, device=gate_bias.device)
                scaled = gate_bias * scale if per_head else gate_bias * scale.mean()
            else:
                scaled = gate_bias
            set_gate_bias_for_layers(model, l, l + 1, scaled, per_head=per_head)
            if has_attn_scale:
                set_attn_scale_for_layers(model, l, l + 1, _attn_scale_eff[l_idx])
    else:
        set_gate_bias_for_layers(model, gate_layer + 1, n_layers, gate_bias, per_head=per_head)


# Sentinel ids demarcating the state span inside the ids returned by
# apply_chat_template(state=...).  Negative, so they can never collide with a
# real token id; split_state_span strips them before anything reaches the
# model (a sentinel leaking into a forward fails loudly at the embedding).
STATE_START_ID = -1001
STATE_END_ID = -1002


def split_state_span(ids: List[int]) -> Tuple[List[int], int, int]:
    """Strip the state sentinels from ``ids`` → ``(clean_ids, start, end)``.

    ``ids`` must come from ``apply_chat_template(state=...)``: exactly one
    STATE_START_ID followed by exactly one STATE_END_ID. Raises ValueError
    otherwise — never guess a span.
    """
    starts = [i for i, t in enumerate(ids) if t == STATE_START_ID]
    ends = [i for i, t in enumerate(ids) if t == STATE_END_ID]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError(
            "ids carry no valid state span: expected exactly one STATE_START_ID "
            "followed by one STATE_END_ID (produce ids with "
            f"apply_chat_template(state=...)); found starts={starts} ends={ends}")
    s, e = starts[0], ends[0]
    clean = ids[:s] + ids[s + 1:e] + ids[e + 1:]
    return clean, s, e - 1


def apply_chat_template(tokenizer, prompt: str, use_chat_template: bool = True,
                        state: str = None) -> List[int]:
    """Apply chat template to prompt (shared by train and inference).

    When ``use_chat_template=False``, tokenize the prompt as raw text — no
    user/assistant wrapping, no reasoning preamble. Use this for tasks where
    the prompt is a literal continuation (e.g. next-line code completion).

    state : the prunable document.  When given, ``prompt`` is a template that
    must contain the placeholder ``{state}`` exactly once; everything else in
    the prompt (instructions, question, few-shot examples) is never pruned.
    The prompt is rendered with and without the state and the state's token
    span is derived by diffing the two (``compute_prunable_span``), so the
    span is exact under any chat template.  The span is demarcated IN the
    returned ids by the STATE_START_ID / STATE_END_ID sentinels, so the
    return type is a plain ``List[int]`` either way; ``TokenGateModel.generate``
    (or ``split_state_span``) strips the sentinels and recovers the span.
    """
    if state is not None:
        if prompt.count("{state}") != 1:
            raise ValueError(
                "apply_chat_template(state=...) requires the prompt to contain "
                f"the placeholder '{{state}}' exactly once; found "
                f"{prompt.count('{state}')} occurrences.")
        full_ids = apply_chat_template(
            tokenizer, prompt.replace("{state}", state), use_chat_template)
        empty_ids = apply_chat_template(
            tokenizer, prompt.replace("{state}", ""), use_chat_template)
        state_start, state_end = compute_prunable_span(full_ids, empty_ids)
        return (full_ids[:state_start] + [STATE_START_ID]
                + full_ids[state_start:state_end] + [STATE_END_ID]
                + full_ids[state_end:])
    if not use_chat_template:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def find_article_boundaries(
    full_ids: List[int], empty_ids: List[int],
) -> Tuple[int, int]:
    """Find article token boundaries by comparing full and empty prompts.
    
    Returns (pre_len, post_len) where:
    - pre_len: number of tokens before article
    - post_len: number of tokens after article
    """
    pre_len = 0
    for i in range(min(len(full_ids), len(empty_ids))):
        if full_ids[i] == empty_ids[i]:
            pre_len = i + 1
        else:
            break
    post_len = 0
    max_suf = min(len(full_ids) - pre_len, len(empty_ids) - pre_len)
    for i in range(1, max_suf + 1):
        if full_ids[-i] == empty_ids[-i]:
            post_len = i
        else:
            break
    return pre_len, post_len


def compute_prunable_span(
    prompt_ids,
    empty_ids,
):
    """Single source of truth for ``(article_start, article_end)``.

    Every caller -- trainer tokenization, in-training eval, and the
    inference-time compressor -- must go through this, so the prunable region
    cannot silently differ between training and inference.

    The prunable span is the article (the document/state), found by diffing
    the full prompt against the task's non-compressible baseline.
    """
    pre_len, post_len = find_article_boundaries(prompt_ids, empty_ids)
    return pre_len, len(prompt_ids) - post_len


def run_phase1_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    gate_layer: int,
    enable_grad: bool = False,
    lora_start_layer: int = -1,
) -> Tuple[torch.Tensor, DynamicCache, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run layers 0..gate_layer on full sequence (Phase 1).

    Args:
        model: The language model
        input_ids: (1, seq_len) input token IDs
        gate_layer: Layer index where gate is applied
        enable_grad: Whether to enable gradients (for training)
        lora_start_layer: When enable_grad=True and lora_start_layer>=0, layers
            0..lora_start_layer-1 run under no_grad (frozen) and layers
            lora_start_layer..gate_layer run under enable_grad (LoRA adapters
            receive gradients).  -1 means no LoRA; ignored when enable_grad=False.

    Returns:
        hidden_states: (1, seq_len, d_model) after layer gate_layer
        cache: DynamicCache with KV for layers 0..gate_layer
        position_ids: (1, seq_len) position indices
        pe: Position embeddings
        cache_pos: (seq_len,) cache position indices
    """
    inner = model.model
    seq_len = input_ids.shape[1]

    # Setup: always no_grad (embed_tokens and rotary_emb are frozen)
    with torch.no_grad():
        hidden_states = inner.embed_tokens(input_ids)
        device = hidden_states.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pe = inner.rotary_emb(hidden_states, position_ids)
        cache = DynamicCache(config=model.config)
        cache_pos = torch.arange(seq_len, device=device)
        # Phase 1: no gate bias, causal attention only
        set_gate_bias_for_layers(model, 0, gate_layer + 1, None)

    if enable_grad and lora_start_layer >= 0:
        # Frozen prefix: layers 0..lora_start_layer-1 (no grad needed)
        # When lora_start_layer==0 this loop is empty (range(0)=[]).
        with torch.no_grad():
            for i in range(lora_start_layer):
                hidden_states = inner.layers[i](
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_pos,
                    position_embeddings=pe,
                )
        # LoRA-enabled suffix: layers lora_start_layer..gate_layer
        with torch.enable_grad():
            for i in range(lora_start_layer, gate_layer + 1):
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
        ctx = torch.enable_grad() if enable_grad else torch.no_grad()
        with ctx:
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

    return hidden_states, cache, position_ids, pe, cache_pos


def compute_gate_scores_and_topk(
    gate: E2EGate,
    hidden_states: torch.Tensor,
    article_start: int,
    article_end: int,
    K: int,
    detach_topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Compute gate scores for article tokens and select top-K.

    Args:
        gate: The E2EGate module
        hidden_states: (1, seq_len, d_model) gate-layer hidden states
        article_start: Start index of article tokens
        article_end: End index of article tokens
        K: Number of tokens to keep
        detach_topk: Whether to detach before topk (for training)

    Returns:
        b: (n_article,) scalar scores  OR  (n_article, n_heads) per-head scores
        topk_local: (n_kept_art,) indices of selected tokens (article-relative).
            In top-K mode this has length K_clamped; in threshold mode the
            length is sample-dependent.
        n_kept_art: Number of selected article tokens (== K_clamped in top-K
            mode; topk_local.shape[0] in threshold mode).  Use this — not
            ``K_clamped`` — for downstream pruned_ctx_len calculations.
    """
    h_article = hidden_states[:, article_start:article_end, :]
    n_article = article_end - article_start

    gate_param = next(gate.parameters(), None)
    device = gate_param.device if gate_param is not None else h_article.device
    dtype = gate_param.dtype if gate_param is not None else h_article.dtype

    h_article_cast = h_article[0].to(device=device, dtype=dtype)
    b = gate(h_article_cast)

    K_clamped = max(1, min(K, n_article))

    # Aggregate per-head scores to a scalar for top-K selection.
    # For n_heads=1, gate.aggregate() is an identity.
    b_scores = gate.aggregate(b)  # always (n_article,)

    b_for_topk = b_scores.detach() if detach_topk else b_scores
    topk_local = _select_indices(b_for_topk, K_clamped, gate)
    # Accumulate sparsity term on the *grad-tracked* b_scores (not the detached
    # b_for_topk), so the soft-sigmoid penalty can backprop into the gate even
    # though selection itself is non-differentiable.
    _accumulate_gate_sparsity(gate, b_scores, int(topk_local.shape[0]))
    # Effective kept-article count.  Equals K_clamped in top-K mode; sample-
    # dependent in threshold mode where `K_clamped` is an unused upper bound.
    n_kept_art = int(topk_local.shape[0])

    return b, topk_local, n_kept_art


def create_pruning_mask(
    seq_len: int,
    article_start: int,
    article_end: int,
    topk_local: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Create boolean mask for tokens to keep after pruning.
    
    Keeps:
    - All pre-article tokens [0:article_start]
    - Top-K article tokens (selected by gate)
    - All post-article tokens [article_end:]
    
    Args:
        seq_len: Total sequence length
        article_start: Start index of article
        article_end: End index of article
        topk_local: (K,) article-relative indices of tokens to keep
        device: Device to create mask on
    
    Returns:
        keep_mask: (seq_len,) boolean mask
        keep_indices: (n_kept,) indices of kept tokens
        n_kept: Number of kept tokens
    """
    keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    keep_mask[:article_start] = True
    keep_mask[article_end:] = True
    keep_mask[article_start + topk_local.to(device)] = True
    keep_indices = _keep_indices(topk_local, article_start, article_end, seq_len, device)
    n_kept = keep_indices.shape[0]
    return keep_mask, keep_indices, n_kept



def build_gate_bias_vectorized(
    b: torch.Tensor,
    topk_local: torch.Tensor,
    article_start: int,
    article_end: int,
    keep_indices: torch.Tensor,
    seq_len: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Build gate bias for pruned sequence (vectorized, gradient-preserving).

    Supports both scalar and per-head gate outputs:
      Scalar:   b shape (n_article,)         → returns (n_kept,)
      Per-head: b shape (n_article, n_heads)  → returns (n_kept, n_heads)

    Args:
        b: Gate scores — (n_article,) or (n_article, n_heads)
        topk_local: (K,) article-relative indices of kept tokens
        article_start: Start index of article in full sequence
        article_end: End index of article in full sequence
        keep_indices: (n_kept,) indices of kept tokens in full sequence
        seq_len: Length of full sequence
        target_dtype: Target dtype for gate bias

    Returns:
        gate_bias: (n_kept,) or (n_kept, n_heads) bias for pruned sequence
    """
    n_article = article_end - article_start

    if b.dim() == 3:
        # per_layer_gate: (n_article, n_layers_bias, n_heads). Same construction as the
        # per-head path but with an extra layer axis carried through untouched:
        # zero out unselected article tokens, scatter into full-seq positions, gather kept.
        L2, n_heads = b.shape[-2], b.shape[-1]
        keep_mask_art = torch.zeros(n_article, 1, 1, device=b.device, dtype=b.dtype)
        keep_mask_art[topk_local.to(b.device)] = 1.0
        b_kept_art = b * keep_mask_art                       # (n_article, L', H)
        b_full = torch.zeros(seq_len, L2, n_heads, device=b.device, dtype=b.dtype)
        b_full[article_start:article_end] = b_kept_art       # non-article rows stay 0
        b_pruned = b_full[keep_indices.to(b.device)]         # (n_kept, L', H)
        return b_pruned.to(target_dtype)

    # dim-3 (per_layer_gate) early-returned above; here b is (n_art,) or (n_art, H)
    per_head = b.dim() == 2

    if per_head:
        n_heads = b.shape[-1]
        # keep_mask_art: (n_article, 1) — broadcast over heads
        keep_mask_art = torch.zeros(n_article, 1, device=b.device, dtype=b.dtype)
        keep_mask_art[topk_local.to(b.device)] = 1.0
        b_kept_art = b * keep_mask_art  # (n_article, n_heads)

        b_full = torch.zeros(seq_len, n_heads, device=b.device, dtype=b.dtype)
        article_indices = torch.arange(article_start, article_end, device=b.device)
        # scatter: index (n_article, n_heads), source (n_article, n_heads)
        b_full = b_full.scatter(0, article_indices.unsqueeze(-1).expand(-1, n_heads), b_kept_art)
        b_pruned = b_full[keep_indices.to(b.device)]  # (n_kept, n_heads)
    else:
        # Scalar path — original logic
        keep_mask_art = torch.zeros(n_article, device=b.device, dtype=b.dtype)
        keep_mask_art[topk_local.to(b.device)] = 1.0
        b_kept_art = b * keep_mask_art  # (n_article,)

        b_full = torch.zeros(seq_len, device=b.device, dtype=b.dtype)
        article_indices = torch.arange(article_start, article_end, device=b.device)
        b_full = b_full.scatter(0, article_indices, b_kept_art)
        b_pruned = b_full[keep_indices.to(b.device)]  # (n_kept,)

    return b_pruned.to(target_dtype)


def build_gate_bias_simple(
    b: torch.Tensor,
    article_start: int,
    article_end: int,
    keep_indices: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Build gate bias for pruned sequence (simple, for inference).

    No gradient preservation — faster than the vectorized version.

    Supports both scalar and per-head gate outputs:
      Scalar:   b shape (n_article,)         → returns (n_kept,)
      Per-head: b shape (n_article, n_heads)  → returns (n_kept, n_heads)

    Args:
        b: Gate scores — (n_article,) or (n_article, n_heads)
        article_start: Start index of article in full sequence
        article_end: End index of article in full sequence
        keep_indices: (n_kept,) indices of kept tokens in full sequence
        target_dtype: Target dtype for gate bias

    Returns:
        gate_bias: (n_kept,) or (n_kept, n_heads) bias for pruned sequence
    """
    n_kept = keep_indices.shape[0]
    device = keep_indices.device

    is_article = (keep_indices >= article_start) & (keep_indices < article_end)
    art_local_indices = keep_indices[is_article] - article_start

    # Trailing shape from the gate output: () scalar, (H,) per-head, or (L',H) for
    # per_layer_gate — one allocation covers all three (kept non-article rows stay 0).
    gate_bias = torch.zeros(n_kept, *b.shape[1:], device=device, dtype=target_dtype)
    gate_bias[is_article] = b[art_local_indices].to(target_dtype)

    return gate_bias


def prune_kv_cache(
    cache: DynamicCache,
    keep_indices: torch.Tensor,
    num_layers: int,
) -> None:
    """Prune KV cache to keep only selected tokens (in-place).
    
    Args:
        cache: DynamicCache to prune
        keep_indices: (n_kept,) indices of tokens to keep
        num_layers: Number of layers to prune
    """
    for li in range(num_layers):
        lc = cache.layers[li]
        if isinstance(lc, LinearAttentionCacheLayerMixin):
            continue
        ld = lc.keys.device
        ki = keep_indices.to(ld)
        lc.keys = lc.keys[:, :, ki, :]
        lc.values = lc.values[:, :, ki, :]




@torch.no_grad()
def prefill_for_decode(
    model: torch.nn.Module,
    gate: E2EGate,
    input_ids: torch.Tensor,
    article_start: int,
    article_end: int,
    gate_layer: int,
    K: int,
    prune_phase1_kv: bool = False,
) -> Tuple[torch.Tensor, DynamicCache, torch.Tensor, int, int]:
    """Phase 1 → gate → prune → Phase 2 → first token, returning KV cache for decoding.

    Shared implementation used by both E2EGateCompressor (inference) and
    evaluate() in train_e2e_gate.py.  Always runs under torch.no_grad().

    Args:
        model: The language model (FlexAttention-patched).
        gate: The E2EGate module.
        input_ids: (1, seq_len) prompt token IDs (no answer tokens).
        article_start: Start index of article tokens in input_ids.
        article_end: End index of article tokens in input_ids.
        gate_layer: Layer index where the gate is applied.
        K: Number of article tokens to keep.
        prune_phase1_kv: If True, prune Phase 1 KV cache before Phase 2 runs.
            False (default) matches training behaviour (Phase 2 attends to all
            Phase 1 KV entries).

    Returns:
        first_token: (1, 1) argmax token at the last kept position.
        cache: DynamicCache with all layers pruned to n_kept entries, ready
            for autoregressive decoding.
        kept_pos: (1, n_kept) position IDs for kept tokens.  Since
            position_ids = arange(seq_len), kept_pos[0] == keep_indices.
        n_kept: Total number of kept tokens.
        n_kept_art: Number of kept article tokens (= min(K, n_art)).

    Note:
        The gate bias is left set for Phase 2 layers after this call.
        Callers that use manual layer-by-layer decoding should clear it with
        set_gate_bias_for_layers(model, 0, num_layers, None) before decoding.
    """
    inner = model.model
    device = next(model.parameters()).device
    n_art = article_end - article_start

    seq_len = input_ids.shape[1]
    hidden_states = inner.embed_tokens(input_ids)
    dtype = hidden_states.dtype
    hs_device = hidden_states.device
    position_ids = torch.arange(seq_len, device=hs_device).unsqueeze(0)
    pe = inner.rotary_emb(hidden_states, position_ids)
    cache = DynamicCache(config=model.config)
    cache_pos = torch.arange(seq_len, device=hs_device)

    # Scoring-branch support: 'mlp' via the a_j phase boundary (split); any OTHER
    # branch mode would silently score from h^fwd here (this single-sample path has
    # no capture hooks) — the branch would be ignored entirely. Fail loudly instead.
    _split = split_gate_layer_mlp(gate)
    if getattr(gate, "gate_branch", None) is not None and not _split:
        raise NotImplementedError(
            f"prefill_for_decode does not support gate_branch_mode="
            f"{getattr(gate, 'gate_branch_mode', None)!r} (only 'mlp' via the a_j "
            f"split); scoring would silently ignore the branch. Use "
            f"prefill_batch_for_decode.")

    # ── Phase 1: layers 0..gate_layer on full sequence (no gate bias) ─────────
    set_gate_bias_for_layers(model, 0, gate_layer + 1, None)
    for i in range(gate_layer + 1):
        if _split and i == gate_layer:
            hidden_states = attn_only_forward(inner.layers[i], hidden_states,
                                              position_ids, cache, cache_pos, pe)
            continue
        hidden_states = inner.layers[i](
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_pos,
            position_embeddings=pe,
        )
    hs_device = hidden_states.device  # re-capture: may change with device_map="auto"

    # ── Gate: score article tokens, select top-K ──────────────────────────────
    # Branch attached: score from h^gate = boundary + MLP_gate(LN2(boundary));
    # no branch: gate_branch_hs returns the forwarded hidden state (legacy tied gate).
    gate_hs = gate_branch_hs(gate, model, gate_layer, hidden_states)
    h_art = gate_hs[:, article_start:article_end, :]
    gate_param = next(gate.parameters(), None)
    gdevice = gate_param.device if gate_param is not None else h_art.device
    gdtype = gate_param.dtype if gate_param is not None else h_art.dtype
    h_art_cast = h_art[0].to(device=gdevice, dtype=gdtype)
    b = gate(h_art_cast)
    # (n_art,) or (n_art, H)
    K_clamped = max(1, min(K, n_art))
    b_scores = gate.aggregate(b)  # (n_art,) — scalar for top-K selection
    topk_local = _select_indices(b_scores, K_clamped, gate)  # article-relative indices

    # Build keep mask over full sequence
    keep_indices = _keep_indices(topk_local, article_start, article_end, seq_len, hs_device)
    n_kept = keep_indices.shape[0]
    # Threshold mode produces a sample-dependent kept count; pruned_ctx_len
    # downstream must use the actual size, not K_clamped (which is ignored
    # in threshold mode).
    n_kept_art = int(topk_local.shape[0])

    # Gate bias for kept tokens — (n_kept,) scalar or (n_kept, H) per-head
    b_kept = build_gate_bias_simple(b, article_start, article_end, keep_indices, dtype)
    if getattr(gate, "zero_gate_bias", False):
        # Eval-only ablation (bench --zero-gate-bias): zero the additive Phase-2
        # bias while keeping SELECTION unchanged (it already used the raw scores).
        b_kept = torch.zeros_like(b_kept)
    # >= 2, matching the batched paths: 2=(n_art,H) per-head, 3=(n_art,L',H)
    # per_layer_gate. `== 2` here mis-flagged the per-layer gate as scalar —
    # the module then read its (n_kept, H) slice with flat addressing and
    # collapsed per-head attn_scale to its mean (silent; compiled kernel).
    per_head = b.dim() >= 2

    # Prune hidden states
    hs_pruned = hidden_states[:, keep_indices, :]
    if _split:
        # Phase-2 entry: gate layer's own MLP on kept tokens.
        hs_pruned = gate_layer_mlp(inner.layers[gate_layer], hs_pruned)

    # Optionally prune Phase 1 KV before Phase 2
    if prune_phase1_kv:
        prune_kv_cache(cache, keep_indices, gate_layer + 1)

    # Recompute position embeddings for pruned sequence
    kept_pos = position_ids[:, keep_indices.to(position_ids.device)]
    pe_pruned = inner.rotary_emb(hs_pruned, kept_pos.to(hs_device))
    cache_pos_pruned = torch.arange(n_kept, device=hs_device)

    # ── Phase 2: pruned sequence with gate bias ────────────────────────────────
    gate_bias = b_kept.to(hs_pruned.dtype)
    _apply_gate_bias(model, gate, gate_layer, len(inner.layers), gate_bias, per_head,
                     ssmax_ratio=_ssmax_ratio(kept_pos) if getattr(gate, 'scaled_softmax_scale', False) else None)
    if gate.n_layers_bias == 0:
        # Per-layer attn_scale handled inside _apply_gate_bias when n_layers_bias > 0.
        _attn_scale = (None if getattr(gate, "_baseline_no_bias_scale", False)
                       else (gate.attn_scale_eff() if hasattr(gate, 'attn_scale') else None))
        set_attn_scale_for_layers(model, gate_layer + 1, len(inner.layers), _attn_scale)
    for i in range(gate_layer + 1, len(inner.layers)):
        hs_pruned = inner.layers[i](
            hs_pruned,
            attention_mask=None,
            position_ids=kept_pos.to(hs_device),
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_pos_pruned,
            position_embeddings=pe_pruned,
        )

    # Final norm + lm_head → first token
    hs_pruned = inner.norm(hs_pruned)
    first_token = model.lm_head(hs_pruned)[:, -1].argmax(-1, keepdim=True).to(device)

    # Always prune Phase 1 KV after Phase 2 so all layers have n_kept entries.
    # When prune_phase1_kv=True the Phase 1 cache is already pruned; skip to
    # avoid double-indexing with stale original keep_indices.
    if not prune_phase1_kv:
        prune_kv_cache(cache, keep_indices, gate_layer + 1)

    return first_token, cache, kept_pos.to(device), n_kept, n_kept_art


@torch.no_grad()
def prefill_batch_for_decode(
    model: torch.nn.Module,
    gate: E2EGate,
    prompt_ids_list: List[List[int]],
    article_starts: List[int],
    article_ends: List[int],
    gate_layer: int,
    K: int,
    prune_below_gate: bool = False,
) -> Tuple[torch.Tensor, DynamicCache, torch.Tensor, List[int], List[int]]:
    """Batched Phase 1 → gate → prune → Phase 2 → first token for B prompts.

    prune_below_gate=False (the DEFAULT) skips the Phase-1 KV gather:
    layers 0..gate_layer keep their FULL padded-context cache (width = max
    phase-1 seq len, pad slots included) so decode at those layers attends
    over the whole prompt.  The caller's decode loop must then route a
    full-width pad mask + cache_position to layers <= gate_layer
    (gate_generate_batch does).  prune_below_gate=True additionally slices
    the below-gate KV down to the kept tokens (smaller decode cache; the
    noprunebelowL ablation showed the two settings are accuracy-identical).
    Phase 2 is unaffected — it only ever touches layers above the gate.

    Mirrors prefill_for_decode but processes a batch of B variable-length
    prompts in parallel.  Shorter sequences are right-padded with zeros to
    max_seq_len; because the attention is causal, padding tokens at the end
    never corrupt the hidden states of real tokens.

    All prompts should share the same article_start offset and post_len so
    that n_kept = article_start + K_clamped + post_len is identical across
    samples.  When n_kepts differ (e.g. n_art < K for a short sample), the
    shorter caches are zero-padded to max_n_kept with gate_bias=-1e9 masking
    out the dummy entries during Phase 2.

    Args:
        model: Language model (FlexAttention-patched).
        gate: E2EGate module (in eval mode, no grad).
        prompt_ids_list: List of B token-ID lists, variable length.
        article_starts: List of B ints — article start position in each prompt.
        article_ends: List of B ints — article end position in each prompt.
        gate_layer: Layer index where the gate is applied.
        K: Number of article tokens to keep.

    Returns:
        first_tokens : (B, 1) int64 — argmax first token per sample.
        cache        : DynamicCache with max_n_kept entries for all layers,
                       ready for autoregressive decoding.
        kept_positions: (B, max_n_kept) int64 position IDs of kept tokens.
        n_kepts      : List[int] — actual kept-token count per sample.
        K_clampeds   : List[int] — effective K used per sample.
    """
    if _PHASE2_GRAPHS == 1:
        torch.compiler.cudagraph_mark_step_begin()
    B = len(prompt_ids_list)
    inner = model.model
    device = next(model.parameters()).device
    _rf = torch.profiler.record_function
    global LAST_PREFILL_PHASES
    _ev = [] if _PREFILL_PHASE_TIMING else None

    def _mark(name):
        if _ev is None:
            return
        if _ev:
            _ev[-1][2].record()
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        _ev.append([name, e0, torch.cuda.Event(enable_timing=True)])

    # ── Phase 1: right-pad sequences to max_seq_len, run batched ─────────────
    _p = _rf("prefill/phase1"); _p.__enter__()
    _mark("phase1")
    max_seq_len = max(len(ids) for ids in prompt_ids_list)
    # Prompts may arrive as device tensors (a serving loop's tokens are already
    # on the GPU): a Python list costs a list->tensor conversion and a pageable
    # host->device copy, both synchronous, per call.
    if B == 1 and torch.is_tensor(prompt_ids_list[0]):
        input_ids_padded = prompt_ids_list[0].to(device=device, dtype=torch.long).reshape(1, -1)
    else:
        input_ids_padded = torch.zeros(B, max_seq_len, dtype=torch.long, device=device)
        for b, ids in enumerate(prompt_ids_list):
            input_ids_padded[b, :len(ids)] = (ids.to(device=device, dtype=torch.long) if torch.is_tensor(ids)
                                              else torch.tensor(ids, dtype=torch.long, device=device))

    set_gate_bias_for_layers(model, 0, gate_layer + 1, None)
    hidden_states = inner.embed_tokens(input_ids_padded)
    dtype = hidden_states.dtype
    hs_device = hidden_states.device
    position_ids = torch.arange(max_seq_len, device=hs_device).unsqueeze(0).expand(B, -1)
    pe = inner.rotary_emb(hidden_states, position_ids)
    cache = PrefillCache(model.config)
    cache_pos = torch.arange(max_seq_len, device=hs_device)

    # a_j phase boundary for the 'mlp' scoring branch (split_gate_layer_mlp).
    _split = split_gate_layer_mlp(gate)
    _gate_layer_input = None
    _p1_pads = [max_seq_len - len(ids) for ids in prompt_ids_list]
    for i in range(gate_layer + 1):
        if i == gate_layer:
            _gate_layer_input = hidden_states  # input to the gate/TSP layer (for the FastKV rule)
            if _split:
                hidden_states = call_layer_padsafe(
                    inner.layers[i], hidden_states, _p1_pads,
                    fn=lambda h, _l=inner.layers[i]: attn_only_forward(
                        _l, h, position_ids, cache, cache_pos, pe))
                continue
        hidden_states = call_layer_padsafe(
            inner.layers[i], hidden_states, _p1_pads,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_pos,
            position_embeddings=pe,
        )
    hs_device = hidden_states.device  # re-capture (may change with device_map)
    _p.__exit__(None, None, None)
    _p = _rf("prefill/score+select"); _p.__enter__()
    _mark("score+select")

    # SCORING representation: the branch fed the a_j boundary when attached,
    # else the forwarded hidden state (legacy tied gate).
    gate_hs = gate_branch_hs(gate, model, gate_layer, hidden_states)

    # ── Gate: score article tokens per sample ────────────────────────────────
    all_keep_indices: List[tuple] = []
    max_n_kept = 0
    gate_param = next(gate.parameters(), None)

    for b_idx in range(B):
        article_start = article_starts[b_idx]
        article_end   = article_ends[b_idx]
        n_art = article_end - article_start
        seq_len = len(prompt_ids_list[b_idx])  # actual length (not padded)

        h_art = gate_hs[b_idx, article_start:article_end, :]
        gdevice = gate_param.device if gate_param is not None else h_art.device
        gdtype  = gate_param.dtype  if gate_param is not None else h_art.dtype
        h_art_cast = h_art.to(device=gdevice, dtype=gdtype)

        b_scores = gate(h_art_cast)

        _Kb = K[b_idx] if isinstance(K, (list, tuple)) else K  # per-sample K (fractional-K eval)
        K_clamped = max(1, min(_Kb, n_art))
        b_agg = gate.aggregate(b_scores)  # (n_art,)
        topk_local = _select_indices(b_agg, K_clamped, gate)

        # Kept positions in ascending order, without the boolean-mask nonzero
        # (a host sync) -- the same tensor: prefix, the selected article
        # positions sorted, suffix.
        keep_indices = torch.cat([
            torch.arange(article_start, device=hs_device),
            article_start + torch.sort(topk_local.to(hs_device)).values,
            torch.arange(article_end, seq_len, device=hs_device),
        ])
        n_kept = keep_indices.shape[0]
        # Effective kept article count — equals K_clamped in top-K mode but
        # is sample-dependent in threshold mode; downstream uses this for
        # pruned_ctx_len.
        n_kept_art = int(topk_local.shape[0])

        all_keep_indices.append((b_idx, keep_indices, topk_local, article_start, article_end, b_scores, n_kept_art))
        max_n_kept = max(max_n_kept, n_kept)

    # ── Prune: gather kept tokens, build batched gate bias ───────────────────
    _p.__exit__(None, None, None)
    _p = _rf("prefill/prune+bias"); _p.__enter__()
    _mark("prune+bias")
    _b_sample = all_keep_indices[0][5]
    per_head = _b_sample.dim() >= 2  # 2=(n_art,H) per-head; 3=(n_art,L',H) per_layer_gate
    n_gate_heads = _b_sample.shape[-1] if per_head else 1

    n_alloc = max_n_kept if _PHASE2_BUCKET is None else -(-max_n_kept // _PHASE2_BUCKET) * _PHASE2_BUCKET
    hs_pruned     = torch.zeros(B, n_alloc, hidden_states.shape[2], dtype=dtype, device=hs_device)
    kept_positions = torch.zeros(B, n_alloc, dtype=torch.long, device=hs_device)

    # Trailing shape comes from the gate output: () scalar, (H,) per-head, or
    # (L', H) per_layer_gate — so the same allocation covers all three.
    _tail = tuple(_b_sample.shape[1:])
    gate_bias_batched = torch.zeros((B, n_alloc, *_tail), dtype=dtype, device=hs_device)

    n_kepts: List[int]    = []
    K_clampeds: List[int] = []
    for b_idx, keep_indices, topk_local, art_start, art_end, b_scores, K_clamped in all_keep_indices:
        n_kept  = keep_indices.shape[0]
        seq_len = len(prompt_ids_list[b_idx])
        n_kepts.append(n_kept)
        K_clampeds.append(K_clamped)

        hs_pruned[b_idx, :n_kept, :]      = hidden_states[b_idx, keep_indices.to(hs_device), :]
        kept_positions[b_idx, :n_kept]    = keep_indices.to(hs_device)

        b_pruned = build_gate_bias_vectorized(
            b_scores, topk_local, art_start, art_end,
            keep_indices.to(b_scores.device), seq_len, dtype,
        )
        gate_bias_batched[b_idx, :n_kept] = b_pruned.to(hs_device)
        if n_kept < n_alloc:
            gate_bias_batched[b_idx, n_kept:] = -1e9

    if _split:
        # Phase-2 entry: gate layer's own MLP on kept tokens (padding rows stay zero).
        hs_pruned = gate_layer_mlp(inner.layers[gate_layer], hs_pruned)

    # ── Prune Phase 1 KV cache before Phase 2 ────────────────────────────────
    # In-place mutation of the existing cache so per-layer slot types
    # (DynamicLayer vs LinearAttentionLayer for Qwen3.5 hybrid) are preserved.
    # Linear-attn layers carry a fixed-size state, not per-token KV, so there
    # is nothing to prune for them — they are skipped.
    #
    # Nothing in Phase 2 reads these layers' caches (they are consumed at
    # decode), so the gathers run on a side stream, concurrently with Phase 2;
    # the main stream waits for them before returning the cache.  The source
    # tensors are handed to the side stream (record_stream) so the allocator
    # does not recycle them under the gather when this function drops them.
    _prune_done = None
    if prune_below_gate:
        padding_lens = [max_n_kept - n for n in n_kepts]
        _side = _prune_stream(hs_device)
        _side.wait_stream(torch.cuda.current_stream(hs_device))
        with torch.cuda.stream(_side):
            for layer_idx in range(gate_layer + 1):
                layer = cache.layers[layer_idx]
                if isinstance(layer, LinearAttentionCacheLayerMixin):
                    continue
                layer_device = layer.keys.device
                layer.keys.record_stream(_side)
                layer.values.record_stream(_side)
                keys_l, vals_l = [], []
                for b_idx, keep_indices, _, _, _, _, _ in all_keep_indices:
                    n_kept = keep_indices.shape[0]
                    p2pad  = padding_lens[b_idx]
                    ki  = keep_indices.to(layer_device)
                    # index_select, not advanced indexing: the latter ran a generic
                    # index kernel at ~0.3 TB/s (0.15 ms per tensor at 22k tokens).
                    k_i = layer.keys[b_idx:b_idx+1].index_select(2, ki)
                    v_i = layer.values[b_idx:b_idx+1].index_select(2, ki)
                    if p2pad > 0:
                        k_i = F.pad(k_i, (0, 0, 0, p2pad))
                        v_i = F.pad(v_i, (0, 0, 0, p2pad))
                    keys_l.append(k_i)
                    vals_l.append(v_i)
                # The gathers above are already copies; a one-element cat would copy again.
                layer.keys = keys_l[0] if len(keys_l) == 1 else torch.cat(keys_l, dim=0)
                layer.values = vals_l[0] if len(vals_l) == 1 else torch.cat(vals_l, dim=0)
        _prune_done = torch.cuda.Event()
        _prune_done.record(_side)

    # ── Phase 2: pruned sequence with gate bias ───────────────────────────────
    _p.__exit__(None, None, None)
    _p = _rf("prefill/phase2"); _p.__enter__()
    _mark("phase2")
    _apply_gate_bias(model, gate, gate_layer, len(inner.layers), gate_bias_batched, per_head,
                     ssmax_ratio=_ssmax_ratio(kept_positions) if getattr(gate, 'scaled_softmax_scale', False) else None)
    if gate.n_layers_bias == 0:
        _attn_scale = (None if getattr(gate, "_baseline_no_bias_scale", False)
                       else (gate.attn_scale_eff() if hasattr(gate, 'attn_scale') else None))
        set_attn_scale_for_layers(model, gate_layer + 1, len(inner.layers), _attn_scale)

    if _PHASE2_GRAPHS == 2:
        from tokengate.e2e_gate.gate_module.phase2_graph import replay_layers
        hs_pruned = replay_layers(model, gate_layer + 1, cache, hs_pruned, kept_positions.to(hs_device), n_alloc)
    else:
        pe_pruned      = inner.rotary_emb(hs_pruned, kept_positions.to(hs_pruned.device))
        cache_pos_pruned = torch.arange(n_alloc, device=hs_device)
        _la_pads = [n_alloc - n for n in n_kepts]   # per-row right-pad widths
        for i in range(gate_layer + 1, len(inner.layers)):
            hs_pruned = call_layer_padsafe(
                inner.layers[i],
                hs_pruned,
                _la_pads,
                attention_mask=None,
                position_ids=kept_positions.to(hs_device),
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_pos_pruned,
                position_embeddings=pe_pruned,
            )

    # ── Final norm → first token (last valid position per sample) ────────────
    _p.__exit__(None, None, None)
    _p = _rf("prefill/head"); _p.__enter__()
    _mark("head")
    # The final norm is per token: only the row that feeds lm_head needs it.
    last_valid = torch.tensor([n - 1 for n in n_kepts], device=hs_device)
    hs_last    = inner.norm(hs_pruned[torch.arange(B, device=hs_device), last_valid, :])  # (B, hidden)
    first_tokens = model.lm_head(hs_last).argmax(-1, keepdim=True).to(device)  # (B, 1)
    if n_alloc != max_n_kept:
        # The bucket's pad slots leave the caches and the positions here.
        for i in range(gate_layer + 1, len(inner.layers)):
            cl = cache.layers[i]
            if isinstance(cl, LinearAttentionCacheLayerMixin):
                continue
            cl.keys = cl.keys[:, :, :max_n_kept]
            cl.values = cl.values[:, :, :max_n_kept]
        kept_positions = kept_positions[:, :max_n_kept]
    if _prune_done is not None:
        torch.cuda.current_stream(hs_device).wait_event(_prune_done)
    _p.__exit__(None, None, None)
    if _ev is not None:
        _ev[-1][2].record()
        LAST_PREFILL_PHASES = _ev

    return first_tokens, cache, kept_positions, n_kepts, K_clampeds


def forward_with_gate_batched(
    model: torch.nn.Module,
    gate: E2EGate,
    batch_data: List[dict],
    gate_layer: int,
    frozen_backbone: bool = True,
    prune_phase1_kv: bool = True,
    lora_start_layer: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched forward with gate — shared implementation for training and eval loss.

    Args:
        model: Transformer model (FlexAttention-patched).
        gate: E2E gate module.
        batch_data: List of dicts with keys:
            input_ids (List[int]), article_start, article_end,
            ctx_len, n_ans, K.
        gate_layer: Layer index for gate.
        frozen_backbone: If True run Phase 1 under no_grad.
        prune_phase1_kv: If True prune Phase 1 KV to match Phase 2 lengths.
        lora_start_layer: When frozen_backbone=True and lora_start_layer>=0,
            layers 0..lora_start_layer-1 run under no_grad and layers
            lora_start_layer..gate_layer run under enable_grad so LoRA
            adapter gradients can flow.  -1 means no LoRA active.

    Returns:
        logits: (B, max_n_kept, vocab_size)
        loss_mask: (B, max_n_kept) bool — True at answer token positions.
        padding_mask: (B, max_n_kept) bool — True at valid (non-padding) positions.
    """
    inner = model.model
    device = next(model.parameters()).device
    batch_size = len(batch_data)

    # Pad input sequences to max length in batch
    max_seq_len = max(len(item["input_ids"]) for item in batch_data)
    input_ids_padded = []
    for item in batch_data:
        ids = item["input_ids"]
        padded = ids + [0] * (max_seq_len - len(ids))
        input_ids_padded.append(padded)
    input_ids = torch.tensor(input_ids_padded, dtype=torch.long, device=device)

    # ── Phase 1: full sequence through layer gate_layer ───────────────────────
    set_gate_bias_for_layers(model, 0, gate_layer + 1, None)
    # Setup under no_grad (embed_tokens and rotary_emb are frozen)
    with torch.no_grad():
        hidden_states = inner.embed_tokens(input_ids)
        dtype = hidden_states.dtype
        hs_device = hidden_states.device
        position_ids = torch.arange(max_seq_len, device=hs_device).unsqueeze(0)
        if batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1)
        pe = inner.rotary_emb(hidden_states, position_ids)
        cache = DynamicCache(config=model.config)
        cache_pos = torch.arange(max_seq_len, device=hs_device)
    # a_j phase boundary for the 'mlp' scoring branch (split_gate_layer_mlp): the
    # gate layer's MLP runs on kept tokens at Phase-2 entry instead.
    _split = split_gate_layer_mlp(gate)

    def _run_layer(i):
        if _split and i == gate_layer:
            return attn_only_forward(inner.layers[i], hidden_states,
                                     position_ids, cache, cache_pos, pe)
        return inner.layers[i](
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_pos,
            position_embeddings=pe,
        )

    if not frozen_backbone:
        with torch.enable_grad():
            for i in range(gate_layer + 1):
                hidden_states = _run_layer(i)
    elif lora_start_layer >= 0:
        # Frozen prefix: layers 0..lora_start_layer-1
        # When lora_start_layer==0 this loop is empty (range(0)=[]).
        with torch.no_grad():
            for i in range(lora_start_layer):
                hidden_states = _run_layer(i)
        # LoRA-enabled suffix: layers lora_start_layer..gate_layer
        with torch.enable_grad():
            for i in range(lora_start_layer, gate_layer + 1):
                hidden_states = _run_layer(i)
    else:
        with torch.no_grad():
            for i in range(gate_layer + 1):
                hidden_states = _run_layer(i)
    hs_device = hidden_states.device  # re-capture after Phase 1

    # SCORING representation: the branch fed the a_j boundary when attached,
    # else the forwarded hidden state (legacy tied gate).
    gate_hs = gate_branch_hs(gate, model, gate_layer, hidden_states)


    # ── Gate: score article tokens per sample ─────────────────────────────────
    all_keep_indices = []  # (b_idx, keep_indices, topk_local, art_start, art_end, b_scores, K_clamped)
    max_n_kept = 0
    gate_param = next(gate.parameters(), None)
    for b_idx, item in enumerate(batch_data):
        article_start = item["article_start"]
        article_end = item["article_end"]
        K = item["K"]
        n_art = article_end - article_start

        h_article = gate_hs[b_idx, article_start:article_end, :]
        gdevice = gate_param.device if gate_param is not None else h_article.device
        gdtype = gate_param.dtype if gate_param is not None else h_article.dtype
        h_article_cast = h_article.to(device=gdevice, dtype=gdtype)

        b = gate(h_article_cast)
        # (n_art,) scalar  OR  (n_art, n_heads) per-head

        K_clamped = max(1, min(K, n_art))
        # Grad-tracked aggregation for the sparsity penalty (mean of sigmoid).
        # Selection itself uses the detached copy so we never backprop through
        # the discrete top-K / threshold operator — gradient flows to the gate
        # only via (a) the sparsity term computed here, and (b) the gate_bias
        # additive in Phase 2.
        b_agg_grad = gate.aggregate(b)
        with torch.no_grad():
            b_agg = b_agg_grad.detach()
            topk_local = _select_indices(b_agg, K_clamped, gate)
        _accumulate_gate_sparsity(gate, b_agg_grad, int(topk_local.shape[0]))

        seq_len = len(item["input_ids"])
        keep_indices = _keep_indices(topk_local, article_start, article_end, seq_len, hs_device)

        # Effective kept article count — equals K_clamped in top-K mode, but
        # is sample-dependent in threshold mode.  Downstream loss-mask construction
        # uses this for pruned_ctx_len, not K_clamped.
        n_kept_art = int(topk_local.shape[0])
        all_keep_indices.append((b_idx, keep_indices, topk_local, article_start, article_end, b, n_kept_art))
        max_n_kept = max(max_n_kept, keep_indices.shape[0])

    # Detect per-head mode from the first sample's gate output
    _b_sample = all_keep_indices[0][5]
    per_head = _b_sample.dim() >= 2  # 2=(n_art,H) per-head; 3=(n_art,L',H) per_layer_gate
    n_gate_heads = _b_sample.shape[-1] if per_head else 1

    # ── Prune: gather kept tokens into padded batch ───────────────────────────
    hs_pruned = torch.zeros(batch_size, max_n_kept, hidden_states.shape[2],
                            dtype=hidden_states.dtype, device=hs_device)
    kept_positions = torch.zeros(batch_size, max_n_kept, dtype=torch.long, device=hs_device)
    padding_mask = torch.zeros(batch_size, max_n_kept, dtype=torch.bool, device=hs_device)
    # Gate bias shape:
    #   Scalar:   (B, max_n_kept)          — 0 at valid positions, -1e9 at padding
    #   Per-head: (B, max_n_kept, n_heads)  — same conventions, per head
    # Do NOT initialise to -1e9 globally — that would mask system-prompt tokens.
    # trailing shape from the gate output: () scalar, (H,) per-head, (L',H) per_layer_gate
    _tail = tuple(_b_sample.shape[1:])
    gate_bias_batched = torch.zeros((batch_size, max_n_kept, *_tail), dtype=dtype, device=hs_device)

    for b_idx, keep_indices, topk_local, art_start, art_end, b_scores, K_clamped in all_keep_indices:
        n_kept = keep_indices.shape[0]
        seq_len = len(batch_data[b_idx]["input_ids"])
        hs_pruned[b_idx, :n_kept, :] = hidden_states[b_idx, keep_indices.to(hs_device), :]
        kept_positions[b_idx, :n_kept] = keep_indices.to(hs_device)
        padding_mask[b_idx, :n_kept] = True

        # Vectorized bias builder handles both scalar and per-head b_scores.
        b_pruned = build_gate_bias_vectorized(
            b_scores, topk_local, art_start, art_end,
            keep_indices.to(b_scores.device), seq_len, dtype,
        )
        # b_pruned: (n_kept,) or (n_kept, n_heads)
        gate_bias_batched[b_idx, :n_kept] = b_pruned.to(hs_device)

        if n_kept < max_n_kept:
            gate_bias_batched[b_idx, n_kept:] = -1e9  # broadcasts over head dim if per-head

    if _split:
        # Phase-2 entry: gate layer's own MLP on kept tokens (padding rows stay zero).
        hs_pruned = gate_layer_mlp(inner.layers[gate_layer], hs_pruned)

    # Optionally prune Phase 1 KV to make cache symmetric across layers.
    # In-place mutation; skip linear-attn layers (no per-token KV to prune).
    if prune_phase1_kv:
        padding_lens = [max_n_kept - all_keep_indices[b_idx][1].shape[0]
                        for b_idx in range(batch_size)]
        for layer_idx in range(gate_layer + 1):
            layer = cache.layers[layer_idx]
            if isinstance(layer, LinearAttentionCacheLayerMixin):
                continue
            layer_device = layer.keys.device
            keys_l, vals_l = [], []
            for b_idx in range(batch_size):
                _, keep_idx, _, _, _, _, _ = all_keep_indices[b_idx]
                n_kept = keep_idx.shape[0]
                p2pad = padding_lens[b_idx]
                ki = keep_idx.to(layer_device)
                k_i = layer.keys[b_idx:b_idx+1, :, ki, :]
                v_i = layer.values[b_idx:b_idx+1, :, ki, :]
                if p2pad > 0:
                    k_i = F.pad(k_i, (0, 0, 0, p2pad))
                    v_i = F.pad(v_i, (0, 0, 0, p2pad))
                keys_l.append(k_i)
                vals_l.append(v_i)
            layer.keys = torch.cat(keys_l, dim=0)
            layer.values = torch.cat(vals_l, dim=0)

    # ── Phase 2: pruned sequence with gate bias ────────────────────────────────
    _apply_gate_bias(model, gate, gate_layer, len(inner.layers), gate_bias_batched, per_head,
                     ssmax_ratio=_ssmax_ratio(kept_positions) if getattr(gate, 'scaled_softmax_scale', False) else None)
    if gate.n_layers_bias == 0:
        # Per-layer attn_scale handled inside _apply_gate_bias when n_layers_bias > 0.
        _attn_scale = (None if getattr(gate, "_baseline_no_bias_scale", False)
                       else (gate.attn_scale_eff() if hasattr(gate, 'attn_scale') else None))
        set_attn_scale_for_layers(model, gate_layer + 1, len(inner.layers), _attn_scale)
    pe_pruned = inner.rotary_emb(hs_pruned, kept_positions.to(hs_pruned.device))
    cache_pos_pruned = torch.arange(max_n_kept, device=hs_device)
    for i in range(gate_layer + 1, len(inner.layers)):
        hs_pruned = inner.layers[i](
            hs_pruned,
            attention_mask=None,
            position_ids=kept_positions.to(hs_device),
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_pos_pruned,
            position_embeddings=pe_pruned,
        )

    # ── Final norm + lm_head ──────────────────────────────────────────────────
    hs_pruned = inner.norm(hs_pruned)
    logits = model.lm_head(hs_pruned)  # (B, max_n_kept, vocab_size)

    # ── Build loss mask for answer positions ──────────────────────────────────
    loss_mask = torch.zeros(batch_size, max_n_kept, dtype=torch.bool, device=logits.device)
    for b_idx, item in enumerate(batch_data):
        _, keep_indices, _, art_start, art_end, _, K_clamped = all_keep_indices[b_idx]
        ctx_len = item["ctx_len"]
        n_ans = item["n_ans"]
        post_article_len = ctx_len - art_end
        pruned_ctx_len = art_start + K_clamped + post_article_len
        n_kept = keep_indices.shape[0]
        ans_start_logit = pruned_ctx_len - 1
        ans_end_logit = ans_start_logit + n_ans
        if ans_end_logit <= n_kept:
            loss_mask[b_idx, ans_start_logit:ans_end_logit] = True

    return logits, loss_mask, padding_mask


def forward_lora_baseline_batched(
    model: torch.nn.Module,
    batch_data: List[dict],
    frozen_backbone: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vanilla full-model forward pass for the no-gate LoRA baseline.

    Runs the entire model end-to-end with FlexAttention causal masking (no
    Phase 1/Phase 2 split, no gate bias, no token pruning), then builds a
    loss mask covering the answer positions of each batch element.

    Pads variable-length sequences on the right with token id 0; causal
    masking guarantees real positions never attend to padding.  Padding
    positions are masked out of the loss.

    Args:
        model: Transformer model (FlexAttention-patched).
        batch_data: List of dicts with keys input_ids (List[int]), ctx_len, n_ans.
        frozen_backbone: If True, run all layers under no_grad EXCEPT LoRA-wrapped
            attention projections, which always allow gradients (LoRA wraps the
            frozen base linear with trainable lora_A/B and the wrapper's autograd
            graph is established at the parameter level).  When False, the whole
            model runs under enable_grad (standard fine-tuning).

    Returns:
        logits: (B, max_seq_len, vocab_size)
        loss_mask: (B, max_seq_len) bool — True at answer token positions.
        padding_mask: (B, max_seq_len) bool — True at valid (non-padding) positions.
    """
    inner = model.model
    device = next(model.parameters()).device
    batch_size = len(batch_data)

    # Pad input sequences to max length in batch (right-padding with id 0).
    max_seq_len = max(len(item["input_ids"]) for item in batch_data)
    input_ids_padded: List[List[int]] = []
    for item in batch_data:
        ids = item["input_ids"]
        input_ids_padded.append(ids + [0] * (max_seq_len - len(ids)))
    input_ids = torch.tensor(input_ids_padded, dtype=torch.long, device=device)

    # No gate, no attn_scale: clear both on every layer so the patched
    # FlexAttention falls back to its causal-only path.
    n_layers = len(inner.layers)
    set_gate_bias_for_layers(model, 0, n_layers, None)
    set_attn_scale_for_layers(model, 0, n_layers, None)

    # Forward.  LoRA adapter parameters keep requires_grad=True regardless of
    # frozen_backbone, so a single enable_grad context lets their gradients
    # flow even though the underlying nn.Linear weights are frozen.  When
    # frozen_backbone=False, this same context covers the unfrozen backbone.
    with torch.enable_grad():
        outputs = model(input_ids, use_cache=False)
    logits = outputs.logits  # (B, max_seq_len, vocab_size)

    loss_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=logits.device)
    padding_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=logits.device)
    for b_idx, item in enumerate(batch_data):
        seq_len = len(item["input_ids"])
        ctx_len = item["ctx_len"]
        n_ans = item["n_ans"]
        padding_mask[b_idx, :seq_len] = True
        # The logit at position ctx_len-1 predicts the first answer token.
        ans_start_logit = ctx_len - 1
        ans_end_logit = ans_start_logit + n_ans
        if ans_end_logit <= seq_len:
            loss_mask[b_idx, ans_start_logit:ans_end_logit] = True

    return logits, loss_mask, padding_mask


# ── Boundary-detach sequential backward (training) ────────────────────────────
# Self-contained (production forward_with_gate_batched_
# multi_K is left untouched as the golden reference). Validated end-to-end by a
# short detach-vs-amortized loss-match smoke run before any long training use.


def forward_backward_with_gate_multi_K_detach(
    model: torch.nn.Module,
    gate: E2EGate,
    batch_data: List[dict],
    gate_layer: int,
    K_list: List[int],
    frozen_backbone: bool = True,
    prune_phase1_kv: bool = False,
    lora_start_layer: int = -1,
    grad_accum: int = 1,
    phase2_grad_checkpoint: bool = False,
) -> Tuple[torch.Tensor, int, float, float]:
    """§1b boundary-detach sequential backward, fused with the training loss.

    Runs Phase 1 ONCE, detaches the gate-layer hidden_states (the only tensor
    crossing into Phase 2 — Phase-2 layers never read the 0..L KV), runs the
    gate + all M Phase-2 forwards from the detached copy, computes per-(K,sample)
    next-token CE and backwards each K immediately (freeing its Phase-2 graph),
    then ONE Phase-1 backward fed the accumulated boundary gradient.

    Gradient-identical to the (removed) amortized multi-K forward + summed
    single backward (validated in profiling/correctness), at flat-in-M peak
    memory.
    REQUIRES torch._functorch.config.donated_buffer = False (set by the caller).

    Returns (batch_loss_detached, n_valid, batch_loss_letter, batch_loss_eos) —
    the same per-step accumulators the trainer logs.
    """
    inner = model.model
    device = next(model.parameters()).device
    batch_size = len(batch_data)
    M = len(K_list)

    # ── Phase-timing instrumentation (opt-in; see PROFILE_PHASES) ─────────────
    # New timing events each call; consumed on rank 0 by the trainer's profile
    # path.  `_ev()` returns a fresh timing-enabled CUDA event or None.
    _prof = PROFILE_PHASES and device.type == "cuda"
    def _ev():
        return torch.cuda.Event(enable_timing=True) if _prof else None
    _e_start, _e_p1fwd, _e_gate = _ev(), _ev(), _ev()
    _e_k_fwd: List[Optional["torch.cuda.Event"]] = []      # per K: after fwd+loss, before bwd
    _e_k: List[Optional["torch.cuda.Event"]] = []          # per K: after that K's backward
    _e_p1bwd_s, _e_p1bwd_e = _ev(), _ev()
    _p2_ctx: List[int] = []                                 # per-K pruned Phase-2 seq len
    if _prof:
        _e_start.record()

    # ── Phase 1 (real graph) ─────────────────────────────────────────────────
    max_seq_len = max(len(item["input_ids"]) for item in batch_data)
    input_ids = torch.tensor(
        [item["input_ids"] + [0] * (max_seq_len - len(item["input_ids"]))
         for item in batch_data], dtype=torch.long, device=device,
    )
    set_gate_bias_for_layers(model, 0, gate_layer + 1, None)
    with torch.no_grad():
        hidden_states = inner.embed_tokens(input_ids)
        dtype = hidden_states.dtype
        hs_device = hidden_states.device
        position_ids = torch.arange(max_seq_len, device=hs_device).unsqueeze(0)
        if batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1)
        pe = inner.rotary_emb(hidden_states, position_ids)
        cache = DynamicCache(config=model.config)
        cache_pos = torch.arange(max_seq_len, device=hs_device)

    _gate_layer_input = [None]  # input to the gate layer (FastKV-rule q source)
    # a_j phase boundary for the 'mlp' scoring branch: Phase 1 ends at the gate
    # layer's attention residual; the layer's MLP runs on kept tokens at Phase-2
    # entry (gate_layer_mlp below). See split_gate_layer_mlp.
    _split = split_gate_layer_mlp(gate)

    def _run_layer(i):
        if i == gate_layer:
            _gate_layer_input[0] = hidden_states  # pre-layer hidden state (FastKV q source)
            if _split:
                return attn_only_forward(inner.layers[i], hidden_states,
                                         position_ids, cache, cache_pos, pe)
        return inner.layers[i](
            hidden_states, attention_mask=None, position_ids=position_ids,
            past_key_values=cache, use_cache=True, cache_position=cache_pos,
            position_embeddings=pe,
        )

    if not frozen_backbone:
        with torch.enable_grad():
            for i in range(gate_layer + 1):
                hidden_states = _run_layer(i)
    elif lora_start_layer >= 0:
        with torch.no_grad():
            for i in range(lora_start_layer):
                hidden_states = _run_layer(i)
        with torch.enable_grad():
            for i in range(lora_start_layer, gate_layer + 1):
                hidden_states = _run_layer(i)
    else:
        with torch.no_grad():
            for i in range(gate_layer + 1):
                hidden_states = _run_layer(i)
    hs_device = hidden_states.device
    if _prof:
        _e_p1fwd.record()

    # ── Boundary detach: gate + all Phase-2 read hs_d, not hidden_states ──────
    # With the a_j split active, `hidden_states` here IS a_j — the boundary moved
    # one sublayer earlier, everything below is unchanged.
    hs_d = hidden_states.detach().requires_grad_(True)

    # SCORING representation: the branch reads hs_d (= detached a_j) directly, so
    # its scoring gradient lands in hs_d.grad and reaches Phase 1 through the SAME
    # single reconnection as the LM gradient (`hidden_states.backward(hs_d.grad)`
    # below). Without a branch it falls back to hs_d (legacy tied gate).
    gate_hs_d = gate_branch_hs(gate, model, gate_layer, hs_d)

    # ── Gate forward (once) from the detached boundary ───────────────────────
    gate_param = next(gate.parameters(), None)
    b_per_sample: List[torch.Tensor] = []
    article_meta: List[dict] = []
    for b_idx, item in enumerate(batch_data):
        art_start = item["article_start"]; art_end = item["article_end"]
        n_art = art_end - art_start
        gdevice = gate_param.device if gate_param is not None else hs_d.device
        gdtype = gate_param.dtype if gate_param is not None else hs_d.dtype
        h_article = gate_hs_d[b_idx, art_start:art_end, :]
        b_per_sample.append(gate(h_article.to(device=gdevice, dtype=gdtype)))
        article_meta.append({"article_start": art_start, "article_end": art_end,
                             "n_art": n_art, "seq_len": len(item["input_ids"])})

    # ── Phase-1 KV snapshot (detached; Phase 2 never backprops through it) ────
    phase1_kv_snapshot: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
    for layer_idx in range(gate_layer + 1):
        layer = cache.layers[layer_idx]
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            phase1_kv_snapshot.append(None)
        else:
            phase1_kv_snapshot.append((layer.keys.detach(), layer.values.detach()))
    if _prof:
        _e_gate.record()

    n_valid_total = batch_size * M
    inv = 1.0 / (n_valid_total * grad_accum)
    batch_loss_val = torch.zeros((), device=device)
    n_valid = 0
    batch_loss_letter = torch.zeros((), device=device)
    batch_loss_eos = torch.zeros((), device=device)

    _per_k_loss_sum = []      # per-K summed CE (phase-2 train loss per length target)
    _per_k_n = []             # per-K valid-sample count
    def _prep(K):
        all_keep_indices = []
        max_n_kept = 0
        for b_idx, item in enumerate(batch_data):
            meta = article_meta[b_idx]; b = b_per_sample[b_idx]
            K_clamped = max(1, min(K, meta["n_art"]))
            b_agg_grad = gate.aggregate(b)
            with torch.no_grad():
                topk_local = _select_indices(b_agg_grad.detach(), K_clamped, gate)
            _accumulate_gate_sparsity(gate, b_agg_grad, int(topk_local.shape[0]))
            seq_len = meta["seq_len"]
            keep_indices = _keep_indices(topk_local, meta["article_start"], meta["article_end"], seq_len, hs_device)
            all_keep_indices.append((b_idx, keep_indices, topk_local,
                                     meta["article_start"], meta["article_end"], b,
                                     int(topk_local.shape[0])))
            max_n_kept = max(max_n_kept, keep_indices.shape[0])

        _b_sample = all_keep_indices[0][5]
        per_head = _b_sample.dim() >= 2  # 2=(n_art,H) per-head; 3=(n_art,L',H) per_layer_gate
        n_gate_heads = _b_sample.shape[-1] if per_head else 1

        hs_pruned = torch.zeros(batch_size, max_n_kept, hidden_states.shape[2],
                                dtype=hidden_states.dtype, device=hs_device)
        kept_positions = torch.zeros(batch_size, max_n_kept, dtype=torch.long, device=hs_device)
        # trailing shape from the gate output: () scalar, (H,) per-head, (L',H) per_layer_gate
        _tail = tuple(_b_sample.shape[1:])
        gate_bias_batched = torch.zeros((batch_size, max_n_kept, *_tail), dtype=dtype, device=hs_device)
        for b_idx, keep_indices, topk_local, art_start, art_end, b_scores, _nk in all_keep_indices:
            n_kept = keep_indices.shape[0]
            seq_len = len(batch_data[b_idx]["input_ids"])
            hs_pruned[b_idx, :n_kept, :] = hs_d[b_idx, keep_indices.to(hs_device), :]
            kept_positions[b_idx, :n_kept] = keep_indices.to(hs_device)
            b_pruned = build_gate_bias_vectorized(
                b_scores, topk_local, art_start, art_end,
                keep_indices.to(b_scores.device), seq_len, dtype,
            )
            if getattr(gate, "zero_gate_bias", False):
                b_pruned = torch.zeros_like(b_pruned)
            gate_bias_batched[b_idx, :n_kept] = b_pruned.to(hs_device)
            if n_kept < max_n_kept:
                gate_bias_batched[b_idx, n_kept:] = -1e9

        return all_keep_indices, max_n_kept, hs_pruned, kept_positions, gate_bias_batched, per_head

    # ── Packed Phase 2 (PACK_PHASE2=1): the targets of one sample concatenated into one
    # sequence and run as ONE variable-length launch per layer, one backward. The short target
    # is launch-bound on its own; packed it rides the full target's launches. Token-local ops
    # see the concatenation, attention sees the target boundaries (fa2_bias_attention_packed),
    # so every loss and gradient is the sequential path's.
    if PACK_PHASE2 and batch_size == 1 and M > 1 and not phase2_grad_checkpoint:
        _preps = [_prep(K) for K in K_list]
        _lens = [p[1] for p in _preps]
        if sum(_lens) <= PACK_PHASE2_MAX_TOKENS:
            per_head = _preps[0][5]
            hs_pruned = torch.cat([p[2] for p in _preps], dim=1)
            kept_positions = torch.cat([p[3] for p in _preps], dim=1)
            gate_bias_batched = torch.cat([p[4] for p in _preps], dim=1)
            if _split:
                hs_pruned = gate_layer_mlp(inner.layers[gate_layer], hs_pruned)
            _cu = torch.tensor([0] + [sum(_lens[:_i + 1]) for _i in range(M)], dtype=torch.int32, device=hs_device)
            set_packed_cu_for_layers(model, gate_layer + 1, len(inner.layers), (_cu, max(_lens), _lens))
            _apply_gate_bias(model, gate, gate_layer, len(inner.layers), gate_bias_batched, per_head,
                             ssmax_ratio=torch.cat([_ssmax_ratio(p[3]) for p in _preps], dim=1)
                             if getattr(gate, 'scaled_softmax_scale', False) else None)
            if gate.n_layers_bias == 0:
                _attn_scale = gate.attn_scale_eff() if hasattr(gate, 'attn_scale') else None
                set_attn_scale_for_layers(model, gate_layer + 1, len(inner.layers), _attn_scale)
            pe_pruned = inner.rotary_emb(hs_pruned, kept_positions)
            cache_pos_pruned = torch.arange(sum(_lens), device=hs_device)
            for i in range(gate_layer + 1, len(inner.layers)):
                hs_pruned = inner.layers[i](
                    hs_pruned, attention_mask=None, position_ids=kept_positions,
                    past_key_values=None, use_cache=False,
                    cache_position=cache_pos_pruned, position_embeddings=pe_pruned,
                )
            set_packed_cu_for_layers(model, gate_layer + 1, len(inner.layers), None)
            hs_pruned = inner.norm(hs_pruned)
            item = batch_data[0]
            ctx_len = item["ctx_len"]; n_ans = item["n_ans"]
            labels = torch.tensor(item["input_ids"][ctx_len:ctx_len + n_ans], dtype=torch.long, device=hs_device)
            loss_all = None
            _off = 0
            for p in _preps:
                _, keep_indices, _, art_start, art_end, _, K_clamped = p[0][0]
                ans_start_logit = art_start + K_clamped + (ctx_len - art_end) - 1
                if ans_start_logit + n_ans > keep_indices.shape[0]:
                    raise AssertionError("packed Phase 2: answer span falls outside the kept sequence")
                sample_logits = model.lm_head(hs_pruned[0, _off + ans_start_logit:_off + ans_start_logit + n_ans])
                ce = F.cross_entropy(sample_logits, labels)
                loss_all = ce if loss_all is None else loss_all + ce
                batch_loss_val += ce.detach()
                if n_ans >= 2:
                    batch_loss_letter += F.cross_entropy(sample_logits[:-1], labels[:-1]).detach()
                    batch_loss_eos += F.cross_entropy(sample_logits[-1:], labels[-1:]).detach()
                n_valid += 1
                _per_k_loss_sum.append(ce.detach()); _per_k_n.append(1)
                _off += p[1]
            if _prof:
                _ekf = torch.cuda.Event(enable_timing=True); _ekf.record()
            (loss_all * inv).backward()
            del hs_pruned, loss_all
            if _prof:
                _ek = torch.cuda.Event(enable_timing=True); _ek.record()
                _e_k_fwd.extend([_ekf] + [_ek] * (M - 1)); _e_k.extend([_ek] * M); _p2_ctx.extend(int(x) for x in _lens)
            K_list_run = []
        else:
            K_list_run = K_list
    else:
        K_list_run = K_list
        _preps = None

    for ki, K in enumerate(K_list_run):
        all_keep_indices, max_n_kept, hs_pruned, kept_positions, gate_bias_batched, per_head = _preps[ki] if _preps is not None else _prep(K)
        if _split:
            # Phase-2 entry: the gate layer's own MLP on KEPT tokens only (padding
            # rows stay zero — gate_layer_mlp). Reads hs_pruned ← hs_d, so its
            # backward feeds hs_d.grad; a delta-adapted layer-L MLP trains here on
            # the forward path exactly as before the split — only the token set shrank.
            hs_pruned = gate_layer_mlp(inner.layers[gate_layer], hs_pruned)

        _apply_gate_bias(model, gate, gate_layer, len(inner.layers), gate_bias_batched, per_head,
                     ssmax_ratio=_ssmax_ratio(kept_positions) if getattr(gate, 'scaled_softmax_scale', False) else None)
        if gate.n_layers_bias == 0:
            _attn_scale = gate.attn_scale_eff() if hasattr(gate, 'attn_scale') else None
            set_attn_scale_for_layers(model, gate_layer + 1, len(inner.layers), _attn_scale)
        pe_pruned = inner.rotary_emb(hs_pruned, kept_positions.to(hs_pruned.device))
        cache_pos_pruned = torch.arange(max_n_kept, device=hs_device)
        if phase2_grad_checkpoint:
            # Per-layer activation checkpointing for the FULL-CONTEXT targets
            #: recompute on backward instead of storing ~35 layers
            # of kept-length activations. Equivalence rests on cache_K being
            # WRITE-ONLY here — layers > gate_layer start empty and nothing
            # reads the phase-1 snapshot (it lives at layers <= gate_layer) —
            # so the checkpointed call runs cache-free; no dropout in the
            # backbone, biases stay set on the modules through the per-K
            # backward, so recompute is deterministic and bias grads flow via
            # closure captures (use_reentrant=False).
            from torch.utils.checkpoint import checkpoint as _ckpt
            _pos_dev = kept_positions.to(hs_device)
            for i in range(gate_layer + 1, len(inner.layers)):
                def _layer_fwd(_h, _i=i):
                    return inner.layers[_i](
                        _h, attention_mask=None, position_ids=_pos_dev,
                        past_key_values=None, use_cache=False,
                        cache_position=cache_pos_pruned, position_embeddings=pe_pruned,
                    )
                hs_pruned = _ckpt(_layer_fwd, hs_pruned, use_reentrant=False)
        else:
            for i in range(gate_layer + 1, len(inner.layers)):
                hs_pruned = inner.layers[i](
                    hs_pruned, attention_mask=None, position_ids=kept_positions.to(hs_device),
                    past_key_values=None, use_cache=False,
                    cache_position=cache_pos_pruned, position_embeddings=pe_pruned,
                )
        hs_pruned = inner.norm(hs_pruned)

        # — loss mask + per-sample CE (matches the trainer's multi-K accumulation) —
        loss_K = None
        _nv_before = n_valid
        for b_idx, item in enumerate(batch_data):
            _, keep_indices, _, art_start, art_end, _, K_clamped = all_keep_indices[b_idx]
            ctx_len = item["ctx_len"]; n_ans = item["n_ans"]
            post_article_len = ctx_len - art_end
            pruned_ctx_len = art_start + K_clamped + post_article_len
            n_kept = keep_indices.shape[0]
            ans_start_logit = pruned_ctx_len - 1
            ans_end_logit = ans_start_logit + n_ans
            if ans_end_logit > n_kept:
                continue
            sample_logits = model.lm_head(hs_pruned[b_idx, ans_start_logit:ans_end_logit])
            answer_ids = item["input_ids"][ctx_len:ctx_len + n_ans]
            labels = torch.tensor(answer_ids, dtype=torch.long, device=sample_logits.device)
            if sample_logits.shape[0] != labels.shape[0]:
                continue
            ce = F.cross_entropy(sample_logits, labels)
            loss_K = ce if loss_K is None else loss_K + ce
            batch_loss_val += ce.detach()
            if sample_logits.shape[0] >= 2:
                batch_loss_letter += F.cross_entropy(sample_logits[:-1], labels[:-1]).detach()
                batch_loss_eos += F.cross_entropy(sample_logits[-1:], labels[-1:]).detach()
            n_valid += 1

        _per_k_loss_sum.append(loss_K.detach() if loss_K is not None else torch.zeros((), device=device))
        _per_k_n.append(n_valid - _nv_before)

        if _prof:
            # fwd boundary: Phase-2 layers + norm + lm_head + loss CE are done;
            # only the per-K .backward() remains → splits Phase-2 fwd vs bwd.
            _ekf = torch.cuda.Event(enable_timing=True)
            _ekf.record()
            _e_k_fwd.append(_ekf)
        if loss_K is not None:
            (loss_K * inv).backward(retain_graph=(ki < M - 1))
        del hs_pruned, loss_K
        if phase2_grad_checkpoint:
            # Release cached allocator segments between targets: with the
            # powers-of-two ladder + full-context sentinel, the sequential
            # per-target shapes otherwise accumulate reserved memory to their
            # UNION (measured 78.9GB vs 66GB single-shape peak on the 16k
            # band). Costs ~ms per target; only when checkpointing is on.
            torch.cuda.empty_cache()
        if _prof:
            _p2_ctx.append(int(max_n_kept))
            _ek = torch.cuda.Event(enable_timing=True)
            _ek.record()
            _e_k.append(_ek)

    global LAST_PER_K_TRAIN_LOSS
    LAST_PER_K_TRAIN_LOSS = {"K": [int(k) for k in K_list],
                             "loss_sum": torch.stack(_per_k_loss_sum).tolist(), "n": _per_k_n}

    if n_valid != n_valid_total:
        raise AssertionError(
            f"detach backward expected all {n_valid_total} (sample,K) pairs valid, "
            f"got {n_valid}; per-K scaling assumes n_valid == B*M."
        )

    # ── ONE Phase-1 backward fed the accumulated boundary gradient ───────────
    if _prof:
        _e_p1bwd_s.record()
    # requires_grad guard: when the forward path is FULLY FROZEN (no adapter on any
    # Phase-1 layer — e.g. ADAPTER_TYPE=lora + LORA_RANK=0), Phase 1 runs under no_grad,
    # so `hidden_states` has no grad_fn and there is nothing to backprop into: the gate
    # is trained purely via the Phase-2 bias and its own scoring branch. Without this:
    # "element 0 of tensors does not require grad and does not have a grad_fn".
    if hs_d.grad is not None and hidden_states.requires_grad:
        hidden_states.backward(hs_d.grad)
    if _prof:
        _e_p1bwd_e.record()
        # Stash raw events + scalars; the trainer (rank 0) reads elapsed_time
        # after a single cuda.synchronize() so non-master ranks pay nothing.
        global PHASE_TIMING
        PHASE_TIMING = {
            "events": {
                "start": _e_start, "p1fwd": _e_p1fwd, "gate": _e_gate,
                "k_fwd": _e_k_fwd, "k": _e_k,
                "p1bwd_s": _e_p1bwd_s, "p1bwd_e": _e_p1bwd_e,
            },
            "p1_ctx": int(max_seq_len),                        # full Phase-1 seq len
            "n_art": sum(m["n_art"] for m in article_meta) / len(article_meta),
            "K": [int(k) for k in K_list],                     # per-K requested K
            "p2_ctx": _p2_ctx,                                 # per-K pruned Phase-2 seq len
        }

    batch_loss_letter, batch_loss_eos = torch.stack([batch_loss_letter, batch_loss_eos]).tolist()
    return batch_loss_val, n_valid, batch_loss_letter, batch_loss_eos
