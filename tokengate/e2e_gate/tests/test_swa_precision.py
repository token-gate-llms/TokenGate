"""SWA running-mean precision + fp32-storage scoring branch (CPU, no model needed).

1. swa_inplace_update is an exact running mean in fp32 (constant trajectory stays put;
   random trajectory matches an fp64 reference), and refuses a bf16 average.
2. The old op p.mul_(n).add_(cur).div_(n+1) in bf16 decays the average as ~n0/n
   (documents the bug it replaced).
3. CastLinear: fp32 storage, matmul in the input dtype, sub-bf16-ULP updates survive.
4. load_gate round-trip of a gate_branch checkpoint: branch params come back fp32 and
   compute in the input dtype; a bf16-saved branch loads as exact fp32 upcasts.
5. build_gate_branch on a tiny layer: fp32 CastLinear params, bf16 compute equals the
   materialised fp32 weights cast to bf16.
6. The adapter-SWA flow (delta adapter): fp32 clone seed, in-place update from the live
   params, swap-in for eval, exact restore.

Run: python -m tokengate.e2e_gate.tests.test_swa_precision
"""
import copy
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokengate.e2e_gate.gate_module.e2e_gate_module import (
    E2EGate, CastLinear, assert_fp32_params, swa_inplace_update, save_gate, load_gate,
)


def _check(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    print(f"  ok: {msg}")


def test_running_mean_exact():
    torch.manual_seed(0)
    w = torch.randn(256, 512) * 0.02
    avg = [("w", w.clone())]
    for n in range(1, 4000):
        swa_inplace_update(avg, [("w", w)], n)
    _check(torch.equal(avg[0][1], w), "constant trajectory: fp32 running mean unchanged after 4000 updates")

    traj_avg = [("w", w.clone())]
    ref = w.double().clone()
    cur = w.clone()
    for n in range(1, 2000):
        cur = cur + torch.randn_like(cur) * 2e-5
        swa_inplace_update(traj_avg, [("w", cur.to(torch.bfloat16))], n)
        ref = ref + (cur.to(torch.bfloat16).double() - ref) / (n + 1)
    rel = ((traj_avg[0][1].double() - ref).norm() / (ref - w.double()).norm()).item()
    _check(rel < 1e-4, f"random trajectory: fp32 running mean matches fp64 reference (rel err {rel:.2e})")

    try:
        swa_inplace_update([("w", w.to(torch.bfloat16))], [("w", w)], 1)
        raise SystemExit("FAIL: bf16 SWA buffer was accepted")
    except RuntimeError:
        print("  ok: bf16 SWA buffer rejected")
    try:
        swa_inplace_update([("w", w.clone())], [("v", w)], 1)
        raise SystemExit("FAIL: misaligned parameter names accepted")
    except RuntimeError:
        print("  ok: misaligned parameter names rejected")


def test_old_bf16_op_decays():
    torch.manual_seed(0)
    w = (torch.randn(256, 512) * 0.02).to(torch.bfloat16)
    p = w.clone()
    for n in range(1, 4000):
        p.mul_(n).add_(w).div_(n + 1)
    ratio = (p.float().abs().mean() / w.float().abs().mean()).item()
    _check(ratio < 0.2, f"old bf16 op on a CONSTANT trajectory decays the average to {ratio:.3f} of |w| by n=4000")


def test_cast_linear():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    cl = CastLinear.from_tensors(lin.weight.detach(), lin.bias.detach())
    assert_fp32_params(cl, "CastLinear")
    x = torch.randn(5, 64).to(torch.bfloat16)
    ref = F.linear(x, lin.weight.detach().to(torch.bfloat16), lin.bias.detach().to(torch.bfloat16))
    out = cl(x)
    _check(out.dtype == torch.bfloat16 and torch.equal(out, ref), "CastLinear computes in the input dtype (bf16) from fp32 storage")
    x32 = torch.randn(5, 64)
    _check(torch.equal(cl(x32), F.linear(x32, lin.weight.detach(), lin.bias.detach())), "CastLinear fp32 input -> fp32 compute")
    before = cl.weight.detach().clone()
    with torch.no_grad():
        cl.weight.add_(torch.full_like(cl.weight, 2e-6))
    _check(not torch.equal(before, cl.weight) and torch.equal(cl.weight, before + 2e-6),
           "sub-bf16-ULP update (2e-6 at |w|~0.1) survives in fp32 storage")
    wb = torch.full((32, 64), 0.1).to(torch.bfloat16)
    _check(torch.equal(wb + 2e-6, wb), "(same update is a no-op on a bf16 param at |w|=0.1)")
    sd = cl.state_dict()
    _check(set(sd) == {"weight", "bias"}, "state_dict keys match nn.Linear")
    cl2 = CastLinear(64, 32, bias=True)
    cl2.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()})
    _check(cl2.weight.dtype == torch.float32 and torch.equal(cl2.weight, sd["weight"].to(torch.bfloat16).float()),
           "bf16 state loads into fp32 storage (exact upcast)")


def test_load_gate_roundtrip():
    from types import SimpleNamespace
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    from tokengate.e2e_gate.gate_module.e2e_gate_module import to_cast_linears
    torch.manual_seed(0)
    d, inter = 32, 48
    gate = E2EGate(d_model=d, n_heads=4, n_layers_bias=2)
    br = Qwen3MLP(SimpleNamespace(hidden_size=d, intermediate_size=inter, hidden_act="silu"))
    br.down_proj = nn.Linear(inter, 1, bias=False)
    with torch.no_grad():
        for p_ in br.parameters():
            p_.normal_(0, 0.02)
    gate.gate_branch = to_cast_linears(br)
    assert_fp32_params(gate, "gate before save")
    z = torch.randn(7, d).to(torch.bfloat16)
    ref_out = gate.gate_branch(z)
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "gate_final.pt")
        save_gate(gate, pth, metadata={"adapter_type": None})
        g2, meta = load_gate(pth)
        assert_fp32_params(g2, "gate after load")
        _check(g2.gate_branch is not None, "load_gate reconstructs the branch")
        _check(torch.equal(g2.gate_branch(z), ref_out), "reloaded branch reproduces bf16 outputs bit-exactly")
        _check(all(torch.equal(a, b) for a, b in zip(g2.state_dict().values(), gate.state_dict().values())), "state_dict round-trips exactly")
        # a pre-fix checkpoint (bf16 branch weights) loads upcast, fp32 storage
        obj = torch.load(pth, map_location="cpu", weights_only=False)
        obj["state_dict"] = {k: (v.to(torch.bfloat16) if k.startswith("gate_branch.") else v) for k, v in obj["state_dict"].items()}
        pth2 = os.path.join(td, "gate_step5.pt")
        torch.save(obj, pth2)
        g3, _ = load_gate(pth2)
        assert_fp32_params(g3, "gate loaded from bf16 branch checkpoint")
        _check(torch.equal(g3.gate_branch.gate_proj.weight, gate.gate_branch.gate_proj.weight.to(torch.bfloat16).float()),
               "bf16-saved branch weights load as exact fp32 upcasts")


def test_build_gate_branch_tiny_layer():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    from tokengate.e2e_gate.gate_module.gate_forward import build_gate_branch, gate_branch_hs
    from tokengate.e2e_gate.gate_module.lora import apply_delta_to_layers
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=64, intermediate_size=96, num_hidden_layers=1, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, vocab_size=128)
    layer = Qwen3DecoderLayer(cfg, 0).to(torch.bfloat16)

    class M:
        pass
    mdl = M(); mdl.model = M(); mdl.model.layers = [layer]; mdl.config = cfg
    apply_delta_to_layers(mdl, [0])
    with torch.no_grad():
        layer.mlp.up_proj.delta.normal_(0, 0.01)
    gate = E2EGate(d_model=64, n_heads=4, n_layers_bias=1)
    build_gate_branch(mdl, gate, 0)
    assert_fp32_params(gate, "gate + branch")
    _check(isinstance(gate.gate_branch.gate_proj, CastLinear), "build_gate_branch produces fp32 CastLinears")
    w_eff = layer.mlp.up_proj.linear.weight.float() + layer.mlp.up_proj.delta
    _check(torch.equal(gate.gate_branch.up_proj.weight, w_eff), "delta adapter folded into fp32 branch weights exactly")
    _check(gate.gate_branch.down_proj.weight.shape == (1, 96) and float(gate.gate_branch.down_proj.weight.abs().sum()) == 0.0,
           "absorbed readout row is (1, inter) zero-init")
    a = torch.randn(3, 9, 64).to(torch.bfloat16)
    out = gate_branch_hs(gate, mdl, 0, a)
    _check(out.shape == (3, 9, 1) and out.dtype == torch.bfloat16 and float(out.abs().sum()) == 0.0,
           "gate_branch_hs: bf16 score of width 1, zero at init")
    with torch.no_grad():
        gate.gate_branch.down_proj.weight.normal_(0, 0.1)
    z = layer.post_attention_layernorm(a).to(torch.bfloat16)
    h = F.silu(F.linear(z, gate.gate_branch.gate_proj.weight.to(torch.bfloat16))) * F.linear(z, gate.gate_branch.up_proj.weight.to(torch.bfloat16))
    ref = F.linear(h, gate.gate_branch.down_proj.weight.to(torch.bfloat16))
    _check(torch.equal(gate_branch_hs(gate, mdl, 0, a), ref), "branch forward == bf16 matmuls of the fp32 weights cast per forward")
    swa = copy.deepcopy(gate)
    assert_fp32_params(swa, "deepcopy for SWA")
    _check(torch.equal(gate_branch_hs(swa, mdl, 0, a), gate_branch_hs(gate, mdl, 0, a)), "deepcopy computes identically (bf16 in, bf16 out)")
    n = 1
    for _ in range(50):
        with torch.no_grad():
            gate.gate_branch.up_proj.weight.add_(torch.randn_like(gate.gate_branch.up_proj.weight) * 1e-5)
        swa_inplace_update(swa.named_parameters(), gate.named_parameters(), n); n += 1
    _check(not torch.equal(swa.gate_branch.up_proj.weight, gate.gate_branch.up_proj.weight)
           and (swa.gate_branch.up_proj.weight - w_eff).abs().max() < (gate.gate_branch.up_proj.weight - w_eff).abs().max(),
           "SWA copy averages the fp32 branch (moves less than the live weights)")


def test_adapter_swa_in_place():
    """The trainer's adapter-SWA flow on a delta-wrapped tiny layer: seed the average as an
    fp32 clone of the live params (same device, no aliasing), update in place from the live
    tensors, swap it into the model for eval and restore the live weights exactly."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    from tokengate.e2e_gate.gate_module.lora import (
        apply_delta_to_layers, collect_lora_named_params, collect_lora_state_dict,
        load_lora_state_dict,
    )
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=64, intermediate_size=96, num_hidden_layers=1, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, vocab_size=128)
    layer = Qwen3DecoderLayer(cfg, 0).to(torch.bfloat16)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module(); self.model.layers = nn.ModuleList([layer])
    model = M()
    apply_delta_to_layers(model, [0])
    named = collect_lora_named_params(model)
    sd = collect_lora_state_dict(model)
    _check([n for n, _ in named] == list(sd.keys()) and len(named) == 7, "collect_lora_named_params: same names/order as collect_lora_state_dict")
    _check(all(p.dtype == torch.float32 and p.dim() == 2 for _, p in named), "delta params are fp32 matrices")
    swa = {k: p.detach().to(torch.float32).clone() for k, p in named}
    _check(all(swa[k].data_ptr() != p.data_ptr() for k, p in named), "seed is a clone, not an alias")
    ref = {k: p.detach().double().clone() for k, p in named}
    n = 1
    for _ in range(20):
        with torch.no_grad():
            for _, p in named:
                p.add_(torch.randn_like(p) * 1e-3)
        swa_inplace_update(swa.items(), collect_lora_named_params(model), n)
        for k, p in named:
            ref[k] += (p.detach().double() - ref[k]) / (n + 1)
        n += 1
    _check(max(float((swa[k].double() - ref[k]).abs().max()) for k in swa) < 1e-6, "in-place adapter average matches fp64 reference")
    live = {k: v.detach().clone() for k, v in collect_lora_state_dict(model).items()}
    load_lora_state_dict(model, swa)
    _check(all(torch.equal(p.detach(), swa[k]) for k, p in named), "SWA adapter swapped into the model")
    load_lora_state_dict(model, live)
    _check(all(torch.equal(p.detach(), live[k]) for k, p in named), "live adapter restored exactly")


if __name__ == "__main__":
    for t in (test_running_mean_exact, test_old_bf16_op_decays, test_cast_linear,
              test_load_gate_roundtrip, test_build_gate_branch_tiny_layer, test_adapter_swa_in_place):
        print(f"[{t.__name__}]")
        t()
    print("PASS: test_swa_precision")
