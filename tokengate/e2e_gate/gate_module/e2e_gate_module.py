"""E2E Gate module: the finalized learned token gate.

The gate scores every article token at the gate layer L and (a) selects the
top-K article tokens to keep and (b) biases Phase-2 attention toward the kept
tokens.  The architecture below is FINAL — the research-era variants (tied
linear/mlp readouts, tanh head, frozen qk-norm scale, softplus-SSS, straight-
through bias, per-(layer,head) readouts) were removed after the config freeze;
`load_gate` rejects checkpoints that used them.

The finalized computation
-------------------------
Scoring (the "a_j split", absorbed shared readout):
    Phase 1 ends at the gate layer's attention output boundary a_j.  A dedicated
    SCORING BRANCH — a trainable clone of the gate layer's MLP whose down_proj is
    composed with the readout row W_g into a single (1, intermediate) row —
    maps a_j to ONE raw score per token:
        score_j = (W_g W_down) · SwiGLU(LN2(a_j))
    (see gate_forward.build_gate_branch / gate_branch_hs).  There is no residual
    readout half and no separate `proj`: the branch IS the whole readout.

Bias construction (E2EGate.forward):
    The scalar score is broadcast to every Phase-2 (layer, head) slot, RMS-
    normalized over the SEQUENCE axis per (layer, head) (Qwen3RMSNorm mechanics,
    eps on the variance — see _RMS_EPS), and scaled by a learnable positive
    per-(layer, head) magnitude:
        b_j[l, h] = softplus(gate_scale[l, h]) * RMSNorm_seq(score)_j
    gate_scale stores the RAW beta; it is initialized to softplus^-1(s_l) where
    s_l = ||g_q ∘ g_k||_2 / sqrt(d) from the layer's pretrained qk-norm gains
    (set_qk_norm_scales), i.e. the bias starts at about one native logit-sigma.
    W_g owns the PATTERN, gate_scale owns the SCALE.

Attention temperature (scaled-softmax cooling):
    attn_scale stores a RAW sigma per (layer, head); the effective temperature is
    S_eff = softplus(sigma) (attn_scale_eff), init softplus^-1(1) so step 0 is
    exactly the base model.  The Phase-2 kernel applies
        s_eff(l, h, q) = r_q * S_eff[l, h],   r_q = log(m_q) / log(p_q)
    (pruning-conditional cooling: exactly S_eff at full context).

Selection:
    top-K over aggregate(b) — the first Phase-2 slot's score, mean over heads.
    The bias added to Phase-2 logits is the LIVE b_j (no straight-through:
    the forward bias steers attention among kept tokens, ste_factor == 0).

Per-head / per-layer shape
--------------------------
n_heads should equal the model's num_attention_heads and n_layers_bias the
number of FULL-attention layers above the gate layer (every layer above L for
Qwen3; the full-attn subset for hybrid Qwen3.5).  The shared readout emits one
score per token; gate_scale / attn_scale give each (layer, head) its own
magnitude and temperature.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Epsilon for the per-(layer,head) RMS normalization in normed_gate — Qwen3RMSNorm's
# exact construction: full backprop, eps added to the VARIANCE inside the sqrt,
# b * rsqrt(mean(b^2) + eps). Value matches Qwen3-8B's rms_norm_eps.
#
# The eps PLACEMENT is the stabilizer. rsqrt(rho^2 + 1e-6) <= 1e3, i.e. the standard
# form carries an implicit SMOOTH denominator floor of sqrt(eps) = 1e-3 — worst-case
# amplification gate_scale/sqrt(eps) = 100x at init values. Two earlier variants were
# both wrong in different ways (measured, 2026-08-15): (a) b/(rho + 1e-6) put the eps
# on rho itself, capping at 1e6 and amplifying ~1e5x near zero-init; (b) a .detach()
# on rho was added to fix the resulting fp32 overflow but could not (the singular
# prefactor survives the detach) and broke the backward's scale-invariance
# (|<dL/db, b>| ~ 3.3 vs ~1e-6). Trained rho per (layer,head), measured on the frozen
# and adapter checkpoints: [2.3e-2, 27.8] — far above sqrt(eps), so the eps is
# negligible after early training and the forward matches the historical form to
# <1e-3 relative, well below bf16 resolution.
_RMS_EPS = 1e-6


# The per-call W&B diagnostics in E2EGate.forward (_last_tanh_stats / _last_rms_stats)
# each pull three or four scalars to the host -- a device sync every prefill, in
# every inference caller.  Off by default; the trainer, their only reader, turns
# them on (set_gate_diag_stats(True)).
_GATE_DIAG_STATS = False


def set_gate_diag_stats(enabled: bool) -> None:
    global _GATE_DIAG_STATS
    _GATE_DIAG_STATS = bool(enabled)


class CastLinear(nn.Module):
    """Linear with float32 PARAMETER STORAGE whose matmul runs in the INPUT's dtype.

    Same split as DeltaLinear / LoRALinear (lora.py): a bf16 parameter rounds away
    every per-step update below its ULP -- at |w|~0.02 that is ~80% of Muon's
    elements at lr=1e-4 and ~97% at lr=1.4e-5 -- and a bf16 running mean (SWA) of
    such params decays as n0/n instead of averaging.  fp32 storage keeps both exact
    while the compute stays at backbone precision (bf16 activations in).
    State-dict keys (`weight` / `bias`) match nn.Linear, so checkpoints written with
    bf16 or fp32 nn.Linear weights load unchanged (upcast on copy_).  The bare
    constructor zero-initialises (no uninitialised memory); build from weights with
    `from_tensors`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features,
                                               dtype=torch.float32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32,
                                                 device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_tensors(cls, weight: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> "CastLinear":
        m = cls(weight.shape[1], weight.shape[0], bias is not None, device=weight.device)
        with torch.no_grad():
            m.weight.copy_(weight.to(torch.float32))
            if bias is not None:
                m.bias.copy_(bias.to(torch.float32))
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cast = self._frozen_cast
        if cast is not None and cast[0] is x.dtype:
            return F.linear(x, cast[1], cast[2])
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.weight.to(x.dtype), b)

    _frozen_cast = None

    def freeze_cast(self, dtype: torch.dtype) -> None:
        """Inference: keep the compute-dtype copy of the weights instead of
        re-casting the fp32 storage on every call (the same values -- one
        rounding of the same numbers -- without the per-call pass over the
        parameters).  Never call while training: the copy would go stale."""
        with torch.no_grad():
            self._frozen_cast = (dtype, self.weight.to(dtype),
                                 None if self.bias is None else self.bias.to(dtype))

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, storage=float32, compute=input dtype")


def freeze_cast_linears(module: nn.Module, dtype: torch.dtype) -> int:
    """freeze_cast every CastLinear under `module`; returns how many."""
    n = 0
    for m in module.modules():
        if isinstance(m, CastLinear):
            m.freeze_cast(dtype)
            n += 1
    return n


def to_cast_linears(module: nn.Module) -> nn.Module:
    """Replace every nn.Linear under `module` (recursively, in place) with a CastLinear
    holding the same weights in fp32."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, CastLinear.from_tensors(
                child.weight.detach(), None if child.bias is None else child.bias.detach()))
        else:
            to_cast_linears(child)
    return module


def assert_fp32_params(module: nn.Module, what: str) -> None:
    bad = [(n, str(p.dtype)) for n, p in module.named_parameters() if p.dtype != torch.float32]
    if bad:
        raise RuntimeError(f"{what}: expected float32 parameter storage, found "
                           f"{bad[:4]}{' ...' if len(bad) > 4 else ''}")


@torch.no_grad()
def swa_inplace_update(avg_params, cur_params, n: int) -> None:
    """avg <- avg + (cur - avg) / (n + 1): the running mean over n+1 snapshots, exact
    in fp32.  `avg_params` / `cur_params` are aligned (name, tensor) sequences; the
    average must be float32 (asserted).  The previous form, mul_(n).add_(cur).div_(n+1)
    in the parameter's own dtype, loses `cur` entirely in bf16 once n > ~300 and then
    multiplies by n/(n+1) every step, so the average decays as n0/n toward zero
    (tests/test_swa_precision.py).
    """
    avg_params = list(avg_params)
    cur_params = list(cur_params)
    if len(avg_params) != len(cur_params):
        raise RuntimeError(f"swa_inplace_update: {len(avg_params)} averaged tensors vs "
                           f"{len(cur_params)} live tensors")
    for (n_avg, p_avg), (n_cur, p_cur) in zip(avg_params, cur_params):
        if n_avg != n_cur:
            raise RuntimeError(f"swa_inplace_update: parameter mismatch {n_avg!r} vs {n_cur!r}")
        if p_avg.dtype != torch.float32:
            raise RuntimeError(f"swa_inplace_update: SWA buffer {n_avg!r} must be float32, "
                               f"got {p_avg.dtype}")
        if p_avg.shape != p_cur.shape:
            raise RuntimeError(f"swa_inplace_update: shape mismatch for {n_avg!r}: "
                               f"{tuple(p_avg.shape)} vs {tuple(p_cur.shape)}")
        p_avg.add_(p_cur.detach().to(device=p_avg.device, dtype=torch.float32) - p_avg,
                   alpha=1.0 / (n + 1))


class E2EGate(nn.Module):
    """Finalized gate: absorbed shared readout + seq-RMSNorm + softplus scales.

    Parameters
    ----------
    d_model : int
        Hidden size of the base model (the scoring branch's input width).
    n_heads : int
        Attention heads at the Phase-2 layers (bias is per-(layer, head)).
    n_layers_bias : int
        Number of Phase-2 (full-attention) layers that receive the bias.
        Must be > 0: gate_scale / attn_scale are (n_layers_bias, n_heads).
    selection_mode : str
        ``'topk'`` (fixed K) or ``'threshold'`` (keep sigmoid(b) > tau).

    The scoring branch is attached by the trainer / loader as
    ``self.gate_branch`` (it must be cloned from the base model's gate layer,
    which the constructor cannot see) — see gate_forward.build_gate_branch.
    """

    # ── Finalized architecture constants (formerly configurable) ────────────
    # Kept as attributes so downstream readers (gate_forward, trainer logging,
    # benches) keep working; they can no longer be changed.
    normed_gate = True            # seq-RMSNorm per (layer, head)
    softplus_gate_scale = True    # learnable positive magnitude softplus(beta)
    scaled_softmax_scale = True   # ssmax r-cooling at Phase 2
    use_attn_scale = True         # learnable temperature exists
    shared_readout = True         # one readout broadcast to every (layer, head)
    absorbed_readout = True       # readout folded into the branch's down_proj
    readout_residual = False      # no residual W_g·a_j half; branch is the readout
    gate_branch_mode = "mlp"      # scoring branch = clone of the gate layer's MLP
    per_layer_gate = True         # bias emitted as (n_art, n_layers_bias, n_heads)
    learnable_sss_scale = True    # temperature S_eff = softplus(raw sigma)
    head_agg = "avg"              # selection score = uniform mean over heads
    ste_factor = 0.0              # live bias (no straight-through); kept for
                                  # introspection by legacy tooling

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        n_layers_bias: int = 0,
        selection_mode: str = "topk",
        tau: float = 0.5,
    ):
        super().__init__()
        if selection_mode not in ("topk", "threshold"):
            raise ValueError(
                f"selection_mode must be 'topk' or 'threshold', got {selection_mode!r}"
            )
        if n_layers_bias <= 0:
            raise ValueError(
                "n_layers_bias must be > 0: the finalized gate scales its bias per "
                "Phase-2 (layer, head) (set it to the number of full-attention "
                "layers above the gate layer)")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers_bias = n_layers_bias
        # Adaptive-K selection mode.  "topk" is the lanes' configuration; in
        # "threshold" mode the gate keeps article tokens whose sigmoid(b) > tau,
        # producing a sample-dependent K.
        self.selection_mode = selection_mode
        self.tau = float(tau)

        # The scoring branch (attached post-construction by trainer / loader).
        self.gate_branch = None

        # ── Shared parameters ─────────────────────────────────────────────────
        if n_heads > 1:
            # Fixed uniform weights for mean aggregation: score_j = mean_h b_j^h.
            # Stored as a buffer (not a parameter) — no gradient path through top-K.
            self.register_buffer('agg_weights', torch.full((n_heads,), 1.0 / n_heads))

        # RAW beta Parameter; the effective magnitude is softplus(beta), applied
        # in forward.  Zero-init here fails loudly if set_qk_norm_scales is
        # forgotten (softplus(0)=0.693: wrong magnitude AND wrong per-layer
        # profile vs the intended s_l init).
        self.gate_scale = nn.Parameter(torch.zeros(n_layers_bias, n_heads))

        # Temperature parameter: RAW sigma; effective S = softplus(sigma)
        # (attn_scale_eff), init softplus^-1(1) => S_eff == 1 at step 0 (base
        # model exactly).  The Phase-2 kernel applies s_eff = r_q * S_eff.
        import math as _m
        self.attn_scale = nn.Parameter(torch.full(
            (n_layers_bias, n_heads), _m.log(_m.expm1(1.0))))

        # Qwen's native RMSNorm class for the b-normalization (single (1,)-gain).
        # Constructed with normalized_shape=1: its forward means over the LAST
        # dim, so the caller moves the sequence axis last; the (1,) weight
        # broadcasts.  The gain is FROZEN at 1: gate_scale is the single scale
        # on the bias — a learnable global gain would be multiplicatively
        # degenerate with it.  Old checkpoints may carry gain != 1; load_gate
        # restores it faithfully (eval fidelity) but it can no longer train.
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        self.b_norm = Qwen3RMSNorm(1, eps=_RMS_EPS)
        self.b_norm.weight.requires_grad_(False)

    def attn_scale_eff(self) -> torch.Tensor:
        """The temperature actually applied to the softmax input: softplus(raw),
        positive by construction, init exactly 1.  Differentiable."""
        return F.softplus(self.attn_scale)

    @torch.no_grad()
    def set_qk_norm_scales(self, model: torch.nn.Module, gate_layer: int) -> list:
        """Initialize gate_scale from the model's pretrained qk_norm gains.

        s_l = ||g_q ∘ g_k||_2 / sqrt(head_dim): under the isotropic
        model this is exactly the model-implied logit std, landing at 2.05-3.60
        across Qwen3-8B's layers — the same range a LEARNED gate_scale converged
        to on its own.  The raw beta is set to softplus^-1(s_l) so the effective
        magnitude starts at s_l (~one native logit-sigma).  Broadcast over heads
        (qk_norm gains are per-layer).  Returns the per-layer values for logging.
        """
        import math
        layers = model.model.layers
        # Phase-2 rows exist only for FULL-attention layers above the gate: on
        # hybrid Qwen3.5 the DeltaNet layers carry no bias row (and no
        # self_attn.q_norm to read); on homogeneous Qwen3 this is the full range.
        from tokengate.e2e_gate.attn.qwen3_flex_attention import (
            get_full_attn_layer_indices,
        )
        idxs = [i for i in get_full_attn_layer_indices(model) if i > gate_layer]
        if len(idxs) != self.n_layers_bias:
            raise ValueError(f"n_layers_bias={self.n_layers_bias} != phase-2 layer count {len(idxs)}")
        vals = []
        for row, li in enumerate(idxs):
            attn = layers[li].self_attn
            gq = attn.q_norm.weight.detach().float()
            gk = attn.k_norm.weight.detach().float()
            # ||g_q ∘ g_k||_2 / sqrt(d): the L2 norm of the ELEMENTWISE gain
            # product, scaled like the logits — positive by construction,
            # upweights peaky gain coordinates instead of averaging them away.
            s_l = float((gq * gk).norm() / math.sqrt(gq.numel()))
            self.gate_scale.data[row].fill_(math.log(math.expm1(s_l)))
            vals.append(s_l)
        return vals

    def _seq_rms_norm_qwen(self, b: torch.Tensor, seq_dim: int) -> torch.Tensor:
        """RMS-normalize over the sequence axis via Qwen's native RMSNorm class
        (which means over the LAST dim, so the sequence axis is moved there and
        back). Includes the frozen (1,)-gain."""
        return self.b_norm(b.movedim(seq_dim, -1)).movedim(-1, seq_dim)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize and scale the branch's raw scores into the Phase-2 bias.

        Parameters
        ----------
        scores : Tensor, shape (n_article, 1)
            Raw per-token scores from the absorbed scoring branch
            (gate_forward.gate_branch_hs output — already width 1).

        Returns
        -------
        bias : Tensor
            per_layer_gate : (n_article, n_layers_bias, n_heads)
            else           : (n_article, n_heads), or (n_article,) if n_heads == 1
        """
        raw = scores
        _gs = F.softplus(self.gate_scale)
        # (n_art, 1) -> (n_art, L', H): the SAME score at every (layer, head).
        # expand, not repeat: no copy, and the gradient sums back into the one
        # readout.  The RMS normalization below is over the sequence axis, so
        # it maps identical slices to identical slices — what differentiates
        # the layers and heads downstream is gate_scale, which stays (L', H).
        b = raw.view(*raw.shape[:-1], 1, 1).expand(
            *raw.shape[:-1], self.n_layers_bias, self.n_heads)
        # RMS-normalize per (layer, head) over the SEQUENCE axis (dim -3).
        # W_g owns the PATTERN, gate_scale owns the SCALE.
        _bn = self._seq_rms_norm_qwen(b, -3)
        if _GATE_DIAG_STATS:
          with torch.no_grad():
            # W&B train/rms_bj/*.  Shared readout: every (l, h)
            # slice is the same pattern, one slice suffices.
            _b0 = _bn[..., 0, 0] if _bn.dim() >= 3 else _bn
            self._last_rms_stats = {
                "mean": float(_b0.mean()),
                "var": float(_b0.var(unbiased=False)),
                "abs_mean": float(_b0.abs().mean()),
            }
        return _bn * _gs  # (n_art, L', H)

    def aggregate(self, b: torch.Tensor) -> torch.Tensor:
        """Reduce per-head scores to a scalar for top-K selection.

        The (seq_len, L', H) bias uses the FIRST Phase-2 layer's slot (mirrors
        the legacy path where gate_scale[0] fed the selection score), reduced
        by a uniform mean over heads.
        """
        if b.dim() == 3:
            b = b[:, 0, :]  # (seq_len, n_heads)
        if self.n_heads == 1:
            return b
        # uniform mean across heads
        return (b * self.agg_weights).sum(-1)  # (seq_len,)


def save_gate(gate: E2EGate, path: str, metadata: Dict[str, Any] = None) -> None:
    """Save gate checkpoint with metadata dict.  The finalized architecture
    constants are recorded explicitly so the checkpoint is self-describing and
    pre-freeze tooling can still identify it."""
    torch.save(
        {
            "state_dict": gate.state_dict(),
            "d_model": gate.d_model,
            "n_heads": gate.n_heads,
            "n_layers_bias": gate.n_layers_bias,
            "selection_mode": gate.selection_mode,
            "tau": gate.tau,
            # finalized constants (validated, not configurable, on load):
            "head_agg": "avg",
            "per_layer_gate": True,
            "learnable_sss_scale": True,
            "normed_gate": True,
            "use_attn_scale": True,
            "scaled_softmax_scale": True,
            "softplus_gate_scale": True,
            "shared_readout": True,
            "absorbed_readout": True,
            "readout_residual": False,
            "ste_factor": 0.0,
            "gate_branch_mode": "mlp",
            "metadata": metadata or {},
        },
        path,
    )


def load_gate(path: str) -> Tuple[E2EGate, Dict[str, Any]]:
    """Load gate from checkpoint. Returns (gate, metadata).

    Only the finalized architecture is supported (absorbed shared readout on an
    'mlp' scoring branch, seq-RMSNorm + softplus gate_scale, live bias).
    Checkpoints from the removed research variants (tied linear/mlp readouts,
    tanh head, frozen qk-norm scale, softplus-SSS, straight-through bias,
    gate_cond conditioning) are rejected loudly rather than silently mis-loaded;
    evaluate those from a pre-freeze commit.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    def _require(key, expected, default):
        got = obj.get(key, default)
        if got != expected:
            raise ValueError(
                f"Checkpoint {path} has {key}={got!r}; the finalized gate "
                f"requires {key}={expected!r} (research variants were removed — "
                f"evaluate this checkpoint from a pre-freeze commit).")

    _gate_cond = obj.get("gate_cond", "none")
    if _gate_cond not in ("none", None) or obj.get("use_boundary_cond", False):
        raise ValueError(
            f"Checkpoint {path} uses removed gate_cond={_gate_cond!r}; only "
            "gate_cond='none' gates are supported."
        )
    _require("n_input_layers", 1, 1)
    _require("score_source", "hidden", "hidden")
    _require("learned_agg", False, False)
    _require("tanh_gate", False, False)
    _require("qk_norm_scale", False, False)
    _require("softplus_sss", False, False)
    _require("sss_frozen_gate_scale", False, False)
    _require("ste_factor", 0.0, 0.0)
    _require("normed_gate", True, True)
    _require("use_attn_scale", True, True)
    _require("scaled_softmax_scale", True, True)
    _require("softplus_gate_scale", True, True)
    _require("shared_readout", True, True)
    _require("absorbed_readout", True, True)
    _require("readout_residual", False, False)
    _require("gate_branch_mode", "mlp", "mlp")
    _require("head_agg", "avg", "avg")
    _require("per_layer_gate", True, True)
    _require("learnable_sss_scale", True, True)
    # The post-top-K projection was removed long before the freeze; loading such
    # a checkpoint would evaluate a DIFFERENT model.
    if "post_proj.weight" in obj["state_dict"]:
        raise ValueError(f"checkpoint {path!r} carries post_proj.weight (removed "
                         "post-top-K projection) — evaluate it from a pre-freeze commit.")

    gate = E2EGate(
        d_model=obj["d_model"],
        n_heads=obj.get("n_heads", 1),
        n_layers_bias=obj.get("n_layers_bias", 0),
        selection_mode=obj.get("selection_mode", "topk"),
        tau=obj.get("tau", 0.5),
    )
    sd = dict(obj["state_dict"])  # copy so we don't mutate the loaded object
    # The scoring branch carries the trained weights.  E2EGate.__init__ cannot
    # build it (train-time build_gate_branch clones it from the model), so
    # reconstruct an equivalent Qwen3MLP from the saved shapes; dropping these
    # keys would silently evaluate an UNTRAINED branch.
    _br_keys = sorted(k for k in sd if k.startswith("gate_branch."))
    if not _br_keys:
        raise ValueError(
            f"checkpoint {path!r} carries no gate_branch.* weights — the "
            "finalized gate scores through the 'mlp' branch; legacy tied-readout "
            "checkpoints must be evaluated from a pre-freeze commit.")
    from types import SimpleNamespace
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    _inter, _dm = sd["gate_branch.gate_proj.weight"].shape
    gate.gate_branch = Qwen3MLP(
        SimpleNamespace(hidden_size=_dm, intermediate_size=_inter,
                        hidden_act="silu")
    )
    # The absorbed readout folded W_g into this branch's last layer, so down_proj
    # is (1, intermediate) rather than (hidden, intermediate) and a stock Qwen3MLP
    # is the wrong shape.  Rebuild that one layer from the saved weight.
    _dp = sd.get("gate_branch.down_proj.weight")
    if _dp is not None and _dp.shape[0] != _dm:
        gate.gate_branch.down_proj = nn.Linear(_dp.shape[1], _dp.shape[0], bias=False)
    # fp32 storage, compute in the input dtype — the same modules build_gate_branch
    # trains with, so bench / compressor numerics match training.  bf16 weights from
    # a pre-fp32-storage checkpoint upcast exactly on load.
    to_cast_linears(gate.gate_branch)
    _missing, _unexpected = gate.load_state_dict(sd, strict=False)
    if _missing or _unexpected:
        raise RuntimeError(
            f"load_gate({path!r}): state_dict mismatch after compat shims — "
            f"missing={_missing}, unexpected={_unexpected}")
    assert_fp32_params(gate.gate_branch, f"load_gate({path!r}) scoring branch")

    return gate, obj.get("metadata", {})
