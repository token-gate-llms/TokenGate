import sys

import torch

from tokengate.e2e_gate.attn.flex_attention_utils import attention_causal, attention_gate_bias


def reference(Q, K, V, bias):
    B, H, S, D = Q.shape
    G = H // K.shape[1]
    Kf = K.repeat_interleave(G, dim=1)
    Vf = V.repeat_interleave(G, dim=1)
    s = Q @ Kf.transpose(-1, -2) / D ** 0.5
    if bias is not None:
        s = s + bias.t().view(1, H, 1, S)
    causal = torch.ones(S, S, dtype=torch.bool, device=Q.device).tril()
    s = s.masked_fill(~causal, float("-inf"))
    return torch.softmax(s, -1) @ Vf


def run(causal_backend, bias_backend, S, B=1, H=16, Hkv=8, D=128, n_pad=64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def mk(h):
        return (torch.randn(B, h, S, D, device="cuda", generator=g) * 0.5).to(torch.bfloat16).requires_grad_(True)

    Q, K, V = mk(H), mk(Hkv), mk(Hkv)
    bias = torch.randn(S, H, device="cuda", generator=g) * 2.0
    bias[-n_pad:] = -1e9
    bias = bias.requires_grad_(True)
    dO = torch.randn(B, H, S, D, device="cuda", generator=g).to(torch.bfloat16)

    o = attention_causal(Q, K, V, backend=causal_backend)
    o.backward(dO)
    got_c = [o.detach().float()] + [t.grad.float().clone() for t in (Q, K, V)]
    for t in (Q, K, V):
        t.grad = None
    o = attention_gate_bias(Q, K, V, bias, per_head=True, backend=bias_backend)
    o.backward(dO)
    got_b = [o.detach().float()] + [t.grad.float().clone() for t in (Q, K, V, bias)]

    Qr, Kr, Vr = [t.detach().float().requires_grad_(True) for t in (Q, K, V)]
    br = bias.detach().float().requires_grad_(True)
    o = reference(Qr, Kr, Vr, None)
    o.backward(dO.float())
    ref_c = [o.detach()] + [t.grad.clone() for t in (Qr, Kr, Vr)]
    for t in (Qr, Kr, Vr):
        t.grad = None
    o = reference(Qr, Kr, Vr, br)
    o.backward(dO.float())
    ref_b = [o.detach()] + [t.grad.clone() for t in (Qr, Kr, Vr, br)]
    return got_c, ref_c, got_b, ref_b


def relerr(x, r):
    return ((x - r).norm() / r.norm().clamp_min(1e-30)).item()


def main():
    ok = True
    names_c = ["out", "dQ", "dK", "dV"]
    names_b = ["out", "dQ", "dK", "dV", "dbias"]
    from kernels.fa2bias.keybias import available as fa2_available
    arms = [("flash", "aug-flash")] + ([("flash", "fa2-bias")] if fa2_available() else [])
    for S in (1024, 4096, 4093):
        fc, rc, fb, rb = run("flex", "flex", S)
        for causal_backend, bias_backend in arms:
            hc, _, hb, _ = run(causal_backend, bias_backend, S)
            print(f"\nS={S}  relative error against the fp32 reference (flex | {bias_backend})")
            for name, f, h, r in zip(names_c, fc, hc, rc):
                ef, eh = relerr(f, r), relerr(h, r)
                flag = "" if eh <= 1.5 * ef + 1e-3 else "  <-- FAIL"
                ok &= not flag
                print(f"  causal {name:6s} {ef:.2e} | {eh:.2e}{flag}")
            for name, f, h, r in zip(names_b, fb, hb, rb):
                ef, eh = relerr(f, r), relerr(h, r)
                limit = max(1.5 * ef + 1e-3, 4e-3 if (name == "dbias" and bias_backend == "aug-flash") else 0.0)
                flag = "" if eh <= limit else "  <-- FAIL"
                ok &= not flag
                print(f"  bias   {name:6s} {ef:.2e} | {eh:.2e}{flag}")
            print(f"  bias grad on padded (-1e9) slots: flex {fb[4][-64:].abs().max().item():.1e}, "
                  f"{bias_backend} {hb[4][-64:].abs().max().item():.1e}")
    print("\ndbias: flex accumulates the captured bias's gradient in fp32; the flash kernel returns dK' in bf16, "
          "so the bias gradient carries bf16 rounding (~2e-3 relative per element, unbiased) like every other "
          "activation gradient in bf16 training")
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
