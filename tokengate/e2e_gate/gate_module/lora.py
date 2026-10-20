"""Adapter utilities for E2E gate training: factored LoRA and additive delta.

Two adapter parameterizations over frozen nn.Linear targets:

* LoRALinear — trainable rank-r A and B factors:
      output = linear(x) + (x @ lora_A @ lora_B) * (alpha / rank)
  Kept as the r<d ablation path and the control arm for the delta A/B.

* DeltaLinear — direct additive full-matrix delta (Muon-native; the factored
  form breaks Muon's spectral-flat update property):
      output = F.linear(x, W + delta, bias + delta_bias)
  delta is zero-init (bit-identical to the base model at step 0).  delta_bias
  exists only when the wrapped linear has a bias (Qwen3 has none — the rule is
  model portability); being 1D it routes to aux-AdamW under Muon, per the
  2D-only routing in muon.py.

The collect/load helpers below are the shared adapter API — the "lora" in
their names (and in the lora_step{N}.pt checkpoint filenames) is legacy
naming that now covers both adapter types; gate_final.pt meta's
"adapter_type" disambiguates.

Workflow:
1. apply_lora_to_layers(...) / apply_delta_to_layers(...) — wrap target
   linears in-place.
2. collect_lora_params(model) — returns all adapter Parameters.
   collect_lora_named_params(model) — the same as (name, Parameter) pairs.
3. collect_lora_state_dict(model) — returns {name: tensor} with only adapter weights.
4. load_lora_state_dict(model, state_dict) — loads saved adapter weights.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Delta targets are paths relative to the decoder layer (attn + MLP).
_DELTA_DEFAULT_TARGETS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with trainable low-rank A and B matrices."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        linear.requires_grad_(False)

        in_features = linear.in_features
        out_features = linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        device = linear.weight.device
        # Always float32 regardless of model dtype: bfloat16 ULP (~0.003 at
        # typical lora_A magnitudes ≈0.35) is ~100× larger than a 1e-5 LR
        # update, so every optimizer step rounds to zero and parameters freeze.
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, device=device) / rank ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mixed-precision forward: keep lora_A / lora_B in fp32 storage (the
        # original-comment concern at __init__ above is about parameter UPDATES
        # with lr~1e-5 vs bf16 ULP, which is a *param-dtype* issue) but do the
        # matmul itself in the model's compute dtype (bf16).  This avoids fp32
        # over the full pre-prune Phase-1 sequence at layer 0 — at full rank
        # 4096 that path was ~2x train-step on its own.  Gradients still
        # accumulate back to the fp32 params via autograd's cast handling.
        A = self.lora_A.to(x.dtype)
        B = self.lora_B.to(x.dtype)
        lora_out = x @ A @ B
        return self.linear(x) + lora_out * self.scaling


class DeltaLinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable additive full-matrix delta.

    forward = F.linear(x, W + delta, bias + delta_bias) — one O(out×in) weight
    add per call (negligible vs the seq×d² matmul), no extra matmuls.  delta
    (and delta_bias) storage is float32 for the same reason as LoRALinear's
    factors: per-step Muon/AdamW updates (~0.2·lr ≈ 3e-6) are far below the
    bf16 ULP at typical delta magnitudes (~4e-5 at |d|≈0.01), so bf16 params
    would round every update to zero and stall.  Compute runs in the model
    dtype (bf16) via a per-forward cast.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        linear.requires_grad_(False)
        device = linear.weight.device
        self.delta = nn.Parameter(torch.zeros(
            linear.out_features, linear.in_features,
            dtype=torch.float32, device=device))
        if linear.bias is not None:
            # Additive bias delta; 1D → routes to aux-AdamW under Muon
            # (build_muon_param_groups sends only 2D params to Muon).
            self.delta_bias = nn.Parameter(torch.zeros(
                linear.out_features, dtype=torch.float32, device=device))
        else:
            self.register_parameter("delta_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight + self.delta.to(self.linear.weight.dtype)
        bias = self.linear.bias
        if self.delta_bias is not None:
            bias = bias + self.delta_bias.to(bias.dtype)
        return F.linear(x, weight, bias)


def _resolve_layer_target(layer: nn.Module, target: str):
    """Resolve a dotted target path relative to a decoder layer.

    Returns (parent_module, attr_name, module).  Fails loudly on missing
    attributes — a typo'd target must not silently train nothing.
    """
    parts = target.split(".")
    parent = layer
    for p in parts[:-1]:
        if not hasattr(parent, p):
            raise AttributeError(f"adapter target {target!r}: no submodule {p!r}")
        parent = getattr(parent, p)
    attr = parts[-1]
    if not hasattr(parent, attr):
        raise AttributeError(f"adapter target {target!r}: no attribute {attr!r}")
    return parent, attr, getattr(parent, attr)


def apply_delta_to_layers(
    model: nn.Module,
    layer_ids: List[int],
    targets: Optional[List[str]] = None,
) -> None:
    """Replace target linears (paths relative to the decoder layer, e.g.
    "self_attn.q_proj", "mlp.down_proj") with DeltaLinear in-place."""
    if targets is None:
        targets = _DELTA_DEFAULT_TARGETS
    for layer_id in layer_ids:
        layer = model.model.layers[layer_id]
        for target in targets:
            parent, attr, linear = _resolve_layer_target(layer, target)
            if not isinstance(linear, nn.Linear):
                raise TypeError(
                    f"adapter target {target!r} is {type(linear).__name__}, "
                    "expected nn.Linear")
            setattr(parent, attr, DeltaLinear(linear))


def apply_lora_to_layers(
    model: nn.Module,
    layer_ids: List[int],
    rank: int,
    alpha: float,
    targets: Optional[List[str]] = None,
) -> None:
    """Replace target linears in layer_ids with LoRALinear in-place.

    Targets may be bare attention-projection names ("q_proj", legacy form, resolved
    under self_attn) or dotted paths relative to the decoder layer ("mlp.gate_proj",
    same form as the delta targets), so a low-rank run can cover the SAME module set
    as the delta adapter. A missing target raises — the old code silently `continue`d,
    so a typo'd target trained nothing without any signal.
    """
    if targets is None:
        targets = _DEFAULT_TARGETS
    for layer_id in layer_ids:
        layer = model.model.layers[layer_id]
        for target in targets:
            dotted = target if "." in target else f"self_attn.{target}"
            parent, attr, linear = _resolve_layer_target(layer, dotted)
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"LoRA target {dotted!r} on layer {layer_id} is "
                                f"{type(linear).__name__}, not nn.Linear")
            setattr(parent, attr, LoRALinear(linear, rank, alpha))


def resolve_all_adapter_layers(model: nn.Module) -> List[int]:
    """Layer ids meant by ``--lora-layers all``: every decoder layer on
    homogeneous models, only the full-attention layers on hybrid models
    (Qwen3.5 DeltaNet layers have no ``self_attn.q_proj``/``v_proj`` to wrap,
    so including them would raise in ``apply_lora_to_layers``)."""
    from tokengate.e2e_gate.attn.qwen3_flex_attention import (
        get_full_attn_layer_indices,
    )
    return get_full_attn_layer_indices(model)


def merge_adapters_into_layers(model: nn.Module, layer_ids: List[int]) -> int:
    """Fold every LoRALinear / DeltaLinear on the given decoder layers into its
    base weight and put the plain nn.Linear back; returns how many were merged.

    Inference only: the adapter is a fixed additive term, so W' = W + scaling*(A B)^T
    (or W + delta), rounded once to the backbone dtype, replaces two extra
    matmuls, two casts and an add per call.  The rounding differs from the
    unmerged bf16 path at the ulp level, so layers whose numerics must stay
    bit-identical to training (the gate layer and below, which decide the
    selection) are left alone by the caller.
    """
    merged = 0
    for layer_id in layer_ids:
        layer = model.model.layers[layer_id]
        for name, mod in list(layer.named_modules()):
            if not isinstance(mod, (LoRALinear, DeltaLinear)):
                continue
            parent, attr, _ = _resolve_layer_target(layer, name)
            lin = mod.linear
            with torch.no_grad():
                w = lin.weight.float()
                if isinstance(mod, LoRALinear):
                    w = w + (mod.lora_A.float() @ mod.lora_B.float()).t() * mod.scaling
                else:
                    w = w + mod.delta.float()
                    if mod.delta_bias is not None:
                        lin.bias.copy_((lin.bias.float() + mod.delta_bias.float()).to(lin.bias.dtype))
                lin.weight.copy_(w.to(lin.weight.dtype))
            setattr(parent, attr, lin)
            merged += 1
    return merged


_ADAPTER_PARAM_MARKERS = ("lora_A", "lora_B", "delta", "delta_bias")


def collect_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Return all adapter trainable parameters (LoRALinear + DeltaLinear).

    Order is guaranteed by model.modules() order: for each LoRALinear, lora_A
    before lora_B; for each DeltaLinear, delta then (if present) delta_bias.
    """
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend([m.lora_A, m.lora_B])
        elif isinstance(m, DeltaLinear):
            params.append(m.delta)
            if m.delta_bias is not None:
                params.append(m.delta_bias)
    return params


def collect_lora_named_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """(name, Parameter) for every adapter (LoRA / delta) weight, in
    model.named_parameters() order — the live tensors, no copy.  Same names and
    order as collect_lora_state_dict, which is built on it."""
    return [(name, param) for name, param in model.named_parameters()
            if name.rsplit(".", 1)[-1] in _ADAPTER_PARAM_MARKERS]


def collect_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return state dict containing only adapter (LoRA / delta) weights (CPU copies)."""
    return {name: param.detach().cpu() for name, param in collect_lora_named_params(model)}


def find_adapter_path(gate_path: str) -> Optional[str]:
    """Adapter file paired with a gate checkpoint, by name, in the same directory:
    gate_step{N}.pt -> lora_step{N}.pt, gate_final.pt -> lora_final.pt,
    swa_gate_step{N}.pt -> swa_lora_step{N}.pt, swa_gate_final.pt -> swa_lora_final.pt.
    Same-snapshot pairing only: an SWA gate never falls back to the live adapter and a
    step gate never to lora_final.pt (the old auto-discovery did both, silently
    evaluating a gate with an adapter it was not averaged / saved with).  Returns None
    when the paired file is absent; the caller decides from the gate metadata
    (`adapter_type`) whether that is an error.
    """
    d = os.path.dirname(os.path.abspath(gate_path))
    base = os.path.basename(gate_path)
    if base.startswith("swa_gate_"):
        cand = "swa_lora_" + base[len("swa_gate_"):]
    elif base.startswith("gate_"):
        cand = "lora_" + base[len("gate_"):]
    else:
        raise ValueError(f"cannot infer the paired adapter file for gate checkpoint "
                         f"name {base!r} (expected gate_*.pt or swa_gate_*.pt)")
    p = os.path.join(d, cand)
    return p if os.path.exists(p) else None


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load adapter weights from state dict into model.

    Every adapter tensor in ``state_dict`` must land on a parameter — a saved
    delta checkpoint loaded into a lora-wrapped model (or vice versa) fails
    loudly instead of silently training from scratch.
    """
    adapter_keys = {k for k in state_dict
                    if k.rsplit(".", 1)[-1] in _ADAPTER_PARAM_MARKERS}
    # strict=False: `missing` (model params absent from the checkpoint — the
    # whole frozen backbone) is expected; adapter keys the MODEL doesn't have
    # come back in `unexpected`.
    _missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unloaded = adapter_keys & set(unexpected)
    if unloaded:
        raise RuntimeError(
            f"adapter checkpoint has {len(unloaded)} tensors with no matching "
            f"model parameter (adapter_type/targets mismatch?): "
            f"{sorted(unloaded)[:4]}...")
