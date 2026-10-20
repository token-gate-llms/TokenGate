"""Packed Phase 2 (gate_forward.PACK_PHASE2) against the sequential detach path.

One real training step (forward_backward_with_gate_multi_K_detach, the short target +
the full-context sentinel, fa2 kernels, LoRA r16 on q/v of every layer, MLP scoring
branch) run both ways from the same weights: the loss and every gradient (adapters,
gate, scoring branch) must agree to bf16 kernel noise.

Run (1 GPU):  python -m tokengate.e2e_gate.tests.test_packed_phase2_grad
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch._functorch.config as _ft_cfg  # noqa: E402
_ft_cfg.donated_buffer = False

from transformers import AutoModelForCausalLM  # noqa: E402

from tokengate.e2e_gate.gate_module import gate_forward  # noqa: E402
from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate  # noqa: E402
from tokengate.e2e_gate.gate_module.gate_forward import (  # noqa: E402
    build_gate_branch, forward_backward_with_gate_multi_K_detach,
)
from tokengate.e2e_gate.gate_module.lora import apply_lora_to_layers, resolve_all_adapter_layers  # noqa: E402
from tokengate.e2e_gate.attn.qwen3_flex_attention import (  # noqa: E402
    patch_model_with_flex_attention, set_attention_backends_for_layers,
)
from tokengate.e2e_gate.attn.flex_attention_utils import warmup_flex_attention  # noqa: E402

BASE = "Qwen/Qwen3-1.7B"
GATE_LAYER = 1
SEQ, ART_S, ART_E, N_ANS = 3000, 40, 2900, 24
K_SHORT = 400


def main():
    torch.manual_seed(0)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map=device)
    for p in model.parameters():
        p.requires_grad_(False)
    patch_model_with_flex_attention(model)
    warmup_flex_attention(torch.device(device))
    apply_lora_to_layers(model, resolve_all_adapter_layers(model), 16, 32.0, ["q_proj", "v_proj"])
    n_above = len(model.model.layers) - GATE_LAYER - 1
    gate = E2EGate(d_model=model.config.hidden_size, n_heads=model.config.num_attention_heads,
                   n_layers_bias=n_above).to(device)
    gate.set_qk_norm_scales(model, GATE_LAYER)
    build_gate_branch(model, gate, GATE_LAYER)
    with torch.no_grad():
        # non-trivial scores (the folded readout row is zero-init)
        gate.gate_branch.down_proj.weight.normal_(std=0.01)
        for n, p in model.named_parameters():
            if n.endswith("lora_B"):
                p.normal_(std=0.01)
    params = ([(n, p) for n, p in model.named_parameters() if p.requires_grad]
              + [(f"gate.{n}", p) for n, p in gate.named_parameters() if p.requires_grad])

    ids = torch.randint(100, 50000, (SEQ,)).tolist()
    batch = [{"input_ids": ids, "article_start": ART_S, "article_end": ART_E,
              "ctx_len": SEQ - N_ANS, "n_ans": N_ANS, "K": K_SHORT}]
    k_list = [K_SHORT, ART_E - ART_S]

    out = {}
    for packed in (False, True, "flex"):
        # "flex": the sequential path on the FlexAttention kernels -- the same math on different
        # kernels, i.e. the size of pure bf16 kernel noise in these gradients (the yardstick).
        gate_forward.PACK_PHASE2 = packed is True
        gate_forward.PACK_PHASE2_MAX_TOKENS = 1 << 20
        set_attention_backends_for_layers(model, "flex" if packed == "flex" else "fa2")
        for _, p in params:
            p.grad = None
        torch.manual_seed(1)
        loss, n_valid, *_ = forward_backward_with_gate_multi_K_detach(
            model, gate, batch, GATE_LAYER, K_list=k_list, prune_phase1_kv=True,
            lora_start_layer=0, grad_accum=1)
        assert n_valid == 2, n_valid
        out[packed] = (float(loss), {n: p.grad.detach().float().clone() for n, p in params if p.grad is not None})
        print(f"[packed={packed}] loss={float(loss):.6f} params with grad: {len(out[packed][1])}/{len(params)}")

    (l0, g0), (l1, g1) = out[False], out[True]
    assert g0.keys() == g1.keys(), "different parameters received gradients"
    den = sum(float(g0[n].pow(2).sum()) for n in g0) ** 0.5
    noise = sum(float((out["flex"][1][n] - g0[n]).pow(2).sum()) for n in g0) ** 0.5 / den
    print(f"yardstick: sequential flex vs sequential fa2, global grad rel diff {noise:.2e}, "
          f"loss {out['flex'][0]:.6f}")
    num = sum(float((g1[n] - g0[n]).pow(2).sum()) for n in g0) ** 0.5
    worst = max(g0, key=lambda n: float((g1[n] - g0[n]).norm() / (g0[n].norm() + 1e-30)))
    wrel = float((g1[worst] - g0[worst]).norm() / (g0[worst].norm() + 1e-30))
    print(f"loss rel diff {abs(l1 - l0) / abs(l0):.2e}; global grad rel diff {num / den:.2e}; "
          f"worst param {worst}: {wrel:.2e} (|g|={float(g0[worst].norm()):.2e})")
    # The per-target CE is a bf16 scalar (random tokens: ~15, spacing 0.0625), so the summed loss
    # moves in quanta of ~2e-3 relative; allow two. Gradients: within twice the flex-vs-fa2 yardstick.
    ok = abs(l1 - l0) / abs(l0) < 5e-3 and num / den < 2 * noise
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
