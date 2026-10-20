"""GPU smoke test for the a_j phase-boundary split, at real 8B scale.

Exercises exactly the configuration the next run uses — FlexAttention-patched
Qwen3-8B, delta adapter at layer 0 (unfrozen layer 0), MLP scoring branch, live gate
config (per_layer_gate, H=32, n_layers_bias=35) — through the
REAL training path (forward_backward_with_gate_multi_K_detach, M=2 K values):

  1. one full fused fwd+bwd step completes (the old severed-hook crash scenario);
  2. loss is finite;
  3. the layer-0 ATTENTION deltas get nonzero grad  — carries LM + the restored
     scoring gradient (severed pre-refactor);
  4. the layer-0 MLP deltas get nonzero grad       — now trained on kept tokens only;
  5. the branch (MLP_gate) params get nonzero grad;
  6. a second call with a FROZEN copy of the gate config still runs (frozen-path
     regression: reconnection skipped, no crash).

Run (1 GPU):  python -m tokengate.e2e_gate.tests.test_split_gpu_smoke
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch._functorch.config as _ft_cfg  # noqa: E402
_ft_cfg.donated_buffer = False   # required by the detach path (train_e2e_gate.py §1b)

from transformers import AutoModelForCausalLM  # noqa: E402

from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate  # noqa: E402
from tokengate.e2e_gate.gate_module.gate_forward import (  # noqa: E402
    build_gate_branch, forward_backward_with_gate_multi_K_detach, split_gate_layer_mlp,
)
from tokengate.e2e_gate.gate_module.lora import apply_delta_to_layers  # noqa: E402
from tokengate.e2e_gate.attn.qwen3_flex_attention import (  # noqa: E402
    patch_model_with_flex_attention,
)
from tokengate.e2e_gate.attn.flex_attention_utils import (  # noqa: E402
    warmup_flex_attention,
)

BASE = "Qwen/Qwen3-8B"
GATE_LAYER = 0
SEQ, ART_S, ART_E, N_ANS = 600, 50, 500, 20
K_LIST = [64, 128]


def _grad_report(params, label):
    nz = [(n, float(p.grad.abs().max())) for n, p in params
          if p.grad is not None and float(p.grad.abs().sum()) > 0]
    print(f"  {label}: {len(nz)}/{len(params)} params with nonzero grad"
          + (f"  (max|g| {max(g for _, g in nz):.2e})" if nz else ""))
    return len(nz)


def main():
    torch.manual_seed(0)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map=device)
    model.model.embed_tokens.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)
    patch_model_with_flex_attention(model)
    warmup_flex_attention(torch.device(device))

    # unfrozen layer 0 == full-rank delta adapter (q/k/v/o + MLP), as the run script does
    apply_delta_to_layers(model, [GATE_LAYER])
    deltas = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    attn_deltas = [(n, p) for n, p in deltas if "self_attn" in n]
    mlp_deltas = [(n, p) for n, p in deltas if "mlp" in n]
    print(f"[smoke] delta params: {sum(p.numel() for _, p in deltas)/1e6:.2f}M "
          f"({len(attn_deltas)} attn, {len(mlp_deltas)} mlp)")

    # live gate config
    gate = E2EGate(d_model=model.config.hidden_size, n_heads=32,
                   n_layers_bias=35).to(device)
    gate.set_qk_norm_scales(model, GATE_LAYER)
    build_gate_branch(model, gate, GATE_LAYER)
    assert split_gate_layer_mlp(gate), "split must be active"
    # The folded readout row is zero-init -> dL/d(branch)=0 at exactly step 0
    # (harness artifact). Perturb as a trained gate would be.
    with torch.no_grad():
        gate.gate_branch.down_proj.weight.normal_(std=0.01)

    ids = torch.randint(100, 50000, (SEQ,)).tolist()
    batch = [{"input_ids": ids, "article_start": ART_S, "article_end": ART_E,
              "ctx_len": SEQ - N_ANS, "n_ans": N_ANS, "K": K_LIST[0]}]

    # ── adapter (unfrozen layer 0) pass ──────────────────────────────────────
    loss, n_valid, *_ = forward_backward_with_gate_multi_K_detach(
        model, gate, batch, GATE_LAYER, K_list=K_LIST,
        prune_phase1_kv=True, lora_start_layer=GATE_LAYER, grad_accum=1)
    print(f"[smoke] adapter pass: loss={float(loss):.4f}  n_valid={n_valid} "
          f"(expect {len(batch) * len(K_LIST)})")
    ok = torch.isfinite(loss) and n_valid == len(batch) * len(K_LIST)
    n_attn = _grad_report(attn_deltas, "layer-0 ATTN deltas")
    n_mlp = _grad_report(mlp_deltas, "layer-0 MLP  deltas")
    branch = [(f"branch.{i}", p) for i, p in enumerate(gate.gate_branch.parameters())]
    n_br = _grad_report(branch, "branch (MLP_gate)")
    ok = ok and n_attn == len(attn_deltas) and n_mlp == len(mlp_deltas) and n_br == len(branch)

    # ── frozen-path regression: same step with grads cleared + adapter off ───
    for _, p in deltas:
        p.requires_grad_(False)
        p.grad = None
    loss_f, n_valid_f, *_ = forward_backward_with_gate_multi_K_detach(
        model, gate, batch, GATE_LAYER, K_list=K_LIST,
        prune_phase1_kv=True, lora_start_layer=-1, grad_accum=1)
    frozen_ok = torch.isfinite(loss_f) and n_valid_f == len(batch) * len(K_LIST)
    print(f"[smoke] frozen pass:  loss={float(loss_f):.4f}  ok={frozen_ok}")
    ok = ok and frozen_ok

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
