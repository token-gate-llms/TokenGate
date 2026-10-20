"""Muon optimizer wiring for the e2e-gate trainer (PyTorch-native).

We use the standard library ``torch.optim.Muon`` (2.9+) for 2D weight matrices
(LoRA A/B, FT weights), paired with ``torch.optim.AdamW`` for non-matrix / scale
params, bundled behind one ``CombinedOptimizer`` so the trainer keeps a single
optimizer object (one scheduler / step / state_dict). ``torch.optim.Muon`` always
runs with ``adjust_lr_fn="match_rms_adamw"`` (Moonshot RMS-matching: scales each
orthogonalized update to AdamW-comparable per-element RMS ~0.2), so Muon groups
use the SAME per-group lr they would under AdamW — there is no separate Muon LR.

DDP note: the e2e-gate trainer manually all-reduces (AVG) grads across ranks BEFORE
optimizer.step(), so ``.grad`` is identical on every rank and a plain single-device
Muon produces a consistent update (no sharded/all-gather variant needed).

Routing: ``build_muon_param_groups`` sends a param to Muon iff it is a 2D weight
matrix AND not in ``adamw_param_ids``; everything else (1D params, and explicitly
excluded 2D tensors like the gate's linear proj / gate_scale grids) goes to AdamW.
"""
from __future__ import annotations

import torch
def build_muon_param_groups(opt_groups, adamw_param_ids=None):
    """Convert AdamW-style param groups into Muon groups.

    A param goes to Muon iff it is a 2D weight matrix (gate projection, LoRA A/B,
    FT layer weights) AND not in ``adamw_param_ids``. Everything else (1D params;
    and explicitly-excluded 2D tensors such as the per-(layer,head) gate_scale /
    attn_scale grids, which are learned SCALES, not linear-transform weights —
    Newton–Schulz orthogonalization is meaningless for them) goes to aux-AdamW.
    Mixed groups are split. Preserves per-group lr (drops foreach — Muon ignores
    it). Returns a new list of groups, each with an explicit 'use_muon' flag.
    """
    adamw_param_ids = adamw_param_ids or set()
    new_groups = []
    for g in opt_groups:
        params = g["params"]
        muon_p = [p for p in params if p.ndim == 2 and id(p) not in adamw_param_ids]
        adam_p = [p for p in params if not (p.ndim == 2 and id(p) not in adamw_param_ids)]
        base = {k: v for k, v in g.items() if k not in ("params", "foreach", "use_muon")}
        if muon_p:
            new_groups.append({**base, "params": muon_p, "use_muon": True})
        if adam_p:
            new_groups.append({**base, "params": adam_p, "use_muon": False})
    return new_groups


# ──────────────────────────────────────────────────────────────────────────────
# Preferred path: PyTorch-native torch.optim.Muon (2.9+) + aux torch.optim.AdamW.
# torch.optim.Muon is 2D-only; adjust_lr_fn="match_rms_adamw" (Moonshot RMS-matching)
# makes each group's lr behave like an AdamW lr, so both sub-optimizers share the
# per-group lrs from the trainer's AdamW-style groups. We pair Muon with a standard
# AdamW for the non-2D / scale params and bundle both behind one Optimizer-compatible
# object so the existing single-optimizer trainer (LR scheduler, step, state_dict)
# works unchanged.
# ──────────────────────────────────────────────────────────────────────────────
class CombinedOptimizer(torch.optim.Optimizer):
    """Thin wrapper presenting several sub-optimizers as one Optimizer.

    Subclasses Optimizer (so torch LR schedulers accept it via isinstance) but does
    NOT call super().__init__; param_groups is a live view over the sub-optimizers'
    groups, so a scheduler mutating group['lr'] propagates. step/zero_grad/state_dict
    fan out to every sub-optimizer.
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("CombinedOptimizer needs at least one sub-optimizer")
        self.defaults = self.optimizers[0].defaults
        self.state = {}

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    @param_groups.setter
    def param_groups(self, _value):
        # Groups live on the sub-optimizers; schedulers mutate group dicts in place.
        pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self.optimizers:
            o.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"combined": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state_dict):
        for o, s in zip(self.optimizers, state_dict["combined"]):
            o.load_state_dict(s)


def build_torch_muon_optimizer(opt_groups, adamw_param_ids=None, momentum=0.95,
                               weight_decay=0.0, adamw_betas=(0.9, 0.95),
                               adamw_eps=1e-8):
    """Build CombinedOptimizer(torch.optim.Muon[2D matrices] + torch.optim.AdamW[rest]).

    The 2D weight matrices (gate proj, LoRA A/B, FT weights) — minus any
    ``adamw_param_ids`` (e.g. the gate_scale/attn_scale grids) — go to torch's Muon
    with ``adjust_lr_fn="match_rms_adamw"``, which makes the Muon lr AdamW-comparable,
    so EVERY group (Muon and aux-AdamW) keeps the per-group lr set in ``opt_groups``.
    ``weight_decay`` applies to both sub-optimizers (decoupled, AdamW-style).
    Returns (optimizer, n_muon_params, n_aux_params).
    """
    from torch.optim import Muon as TorchMuon, AdamW as TorchAdamW
    groups = build_muon_param_groups(opt_groups, adamw_param_ids=adamw_param_ids)
    muon_groups, adamw_groups = [], []
    for g in groups:
        if "lr" not in g:
            raise ValueError("build_torch_muon_optimizer: every param group must "
                             "carry an explicit 'lr' (shared Muon/AdamW LRs)")
        base = {k: v for k, v in g.items() if k not in ("use_muon",)}
        (muon_groups if g["use_muon"] else adamw_groups).append(base)
    optimizers, n_muon, n_aux = [], 0, 0
    if muon_groups:
        optimizers.append(TorchMuon(muon_groups, lr=muon_groups[0]["lr"],
                                    momentum=momentum, weight_decay=weight_decay,
                                    adjust_lr_fn="match_rms_adamw"))
        n_muon = sum(p.numel() for grp in muon_groups for p in grp["params"])
    if adamw_groups:
        optimizers.append(TorchAdamW(adamw_groups, lr=adamw_groups[0]["lr"],
                                     betas=adamw_betas, eps=adamw_eps,
                                     weight_decay=weight_decay))
        n_aux = sum(p.numel() for grp in adamw_groups for p in grp["params"])
    return CombinedOptimizer(optimizers), n_muon, n_aux
