"""Regression test: multi-K backwards through the scoring branch must not crash
with a TRAINABLE layer 0, and the scoring gradient must reach layer-0 attention.

The scoring branch is an MLP clone (a_j phase-boundary split): Phase 1
ends at the attention residual a_j; the branch reads the DETACHED-then-RECONNECTED
boundary hs_d directly (no hooks), and the gate layer's MLP runs on Phase-2 tokens.
The historical double-backward crash ("Trying to backward through the graph a second
time", which killed all 8 ranks of the first qasper-gbmlpadapter run under the old
hook-capture design) is structurally impossible here — asserted by running the same
multi-K backward pattern the trainer uses.

Run: PYTHONPATH=<repo> python tokengate/e2e_gate/tests/test_gate_branch_graph.py
"""
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
from tokengate.e2e_gate.gate_module.gate_forward import (
    build_gate_branch, gate_branch_hs, attn_only_forward, gate_layer_mlp,
    split_gate_layer_mlp)

cfg=AutoConfig.from_pretrained("Qwen/Qwen3-8B"); cfg.num_hidden_layers=1
cfg.hidden_size=128; cfg.intermediate_size=256; cfg.num_attention_heads=8
cfg.num_key_value_heads=2; cfg.head_dim=16
torch.manual_seed(0)
layer=Qwen3DecoderLayer(cfg,0).to(torch.float32)
# TRAINABLE layer 0 -- what the adapter does and what broke the old hook design
for p in layer.parameters(): p.requires_grad_(True)
class M: pass
mdl=M(); mdl.model=M(); mdl.model.layers=[layer]; mdl.config=cfg
B,S,d=1,24,cfg.hidden_size
e=torch.randn(B,S,d); pos=torch.arange(S).unsqueeze(0)
th=getattr(cfg,"rope_theta",1e6)
inv=1.0/(th**(torch.arange(0,cfg.head_dim,2).float()/cfg.head_dim))
ang=pos.float().unsqueeze(-1)*inv; pe=(torch.cat([ang,ang],-1).cos(),torch.cat([ang,ang],-1).sin())

torch.manual_seed(0)
g=E2EGate(d_model=d,n_heads=8,n_layers_bias=1)
build_gate_branch(mdl,g,0)
# the folded readout row is zero-init (score==0 => zero grads everywhere);
# randomize it so the loss actually reaches the branch parameters
torch.nn.init.normal_(g.gate_branch.down_proj.weight,std=0.1)
assert split_gate_layer_mlp(g)
for p in layer.parameters():
    p.grad=None
hidden=attn_only_forward(layer,e,pos,None,None,pe)         # a_j (Phase-1 boundary)
hs_d=hidden.detach().requires_grad_(True)
gate_hs=gate_branch_hs(g,mdl,0,hs_d)                       # branch reads the boundary
h_fwd=gate_layer_mlp(layer,hs_d)                           # Phase-2 entry
# multi-K loop: several backwards touching the shared branch graph, as the trainer does
NK=3
for k in range(NK):
    loss=(gate_hs[:, k::NK, :].float()**2).mean()+(h_fwd[:, k::NK, :].float()**2).mean()
    loss.backward(retain_graph=(k<NK-1))
# the ONE Phase-1 reconnection
hidden.backward(hs_d.grad)
br=[p for p in g.gate_branch.parameters() if p.requires_grad]
brg=sum(1 for p in br if p.grad is not None and float(p.grad.abs().sum())>0)
l0g=sum(1 for p in layer.parameters() if p.grad is not None and float(p.grad.abs().sum())>0)
qg=layer.self_attn.q_proj.weight.grad
q_ok=qg is not None and float(qg.abs().sum())>0
print(f"mlp: phase1 backward OK | branch grads {brg}/{len(br)} | layer0 grads {l0g} | "
      f"q_proj grad nonzero {q_ok}")
assert brg==len(br) and l0g>0 and q_ok, "FAIL: missing gradients"
print("PASS: no double-backward; all gradient paths live")
