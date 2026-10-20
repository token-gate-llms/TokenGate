import json
import os
import site

import torch

from kernels.fa2bias.site_path import ABI, BUILD_INFO, SITE
_ext = None


def available() -> bool:
    return os.path.isdir(SITE) and any(f.startswith("flash_attn_keybias_cuda") for f in os.listdir(SITE))


def ext():
    global _ext
    if _ext is None:
        if not available():
            raise RuntimeError(f"flash_attn_keybias_cuda is not built for this machine ({SITE}): "
                               f"build it with `python kernels/fa2bias/build.py`, or run on the "
                               f"no-build flex kernels (gate_attn='flex' at inference, "
                               f"attn_backend='flex' in training)")
        site.addsitedir(SITE)
        import flash_attn_keybias_cuda
        _ext = flash_attn_keybias_cuda
    return _ext


def require_varlen() -> None:
    """The variable-length entry points carry the bias only from ABI 2 on; an older build
    rejects it inside the kernel launch with an unhelpful message, so check the stamp."""
    ext()
    path = os.path.join(SITE, BUILD_INFO)
    abi = json.load(open(path))["abi"] if os.path.exists(path) else 1
    if abi < ABI:
        raise RuntimeError(f"flash_attn_keybias_cuda at {SITE} is ABI {abi}, packed Phase 2 needs {ABI}: "
                           f"rebuild with `python kernels/fa2bias/build.py`")


@torch.library.custom_op("recursive_kvs::fa2_keybias_fwd", mutates_args=())
def _fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias_raw: torch.Tensor,
         scale: float, causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _ = ext().fwd(q, k, v, None, bias_raw, 0.0, scale, causal, -1, -1, 0.0, False, None)
    return out, lse


@_fwd.register_fake
def _fwd_fake(q, k, v, bias_raw, scale, causal):
    return torch.empty_like(q), q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32)


@torch.library.custom_op("recursive_kvs::fa2_keybias_bwd", mutates_args=())
def _bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
         lse: torch.Tensor, bias_raw: torch.Tensor, scale: float,
         causal: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv, _, dbias = ext().bwd(dout, q, k, v, out, lse, None, None, None, bias_raw,
                                     0.0, scale, causal, -1, -1, 0.0, False, None, None)
    return dq, dk, dv, dbias


@_bwd.register_fake
def _bwd_fake(dout, q, k, v, out, lse, bias_raw, scale, causal):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(bias_raw)


def _setup_context(ctx, inputs, output):
    q, k, v, bias_raw, scale, causal = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, bias_raw)
    ctx.scale = scale
    ctx.causal = causal


def _backward(ctx, dout, dlse):
    q, k, v, out, lse, bias_raw = ctx.saved_tensors
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    dq, dk, dv, dbias = torch.ops.recursive_kvs.fa2_keybias_bwd(dout, q, k, v, out, lse, bias_raw, ctx.scale, ctx.causal)
    return dq, dk, dv, dbias, None, None


torch.library.register_autograd("recursive_kvs::fa2_keybias_fwd", _backward, setup_context=_setup_context)


def flash_attn_key_bias(q, k, v, key_bias, softmax_scale, causal=True):
    bias_raw = (key_bias.float() / softmax_scale).contiguous()
    return torch.ops.recursive_kvs.fa2_keybias_fwd(q, k, v, bias_raw, softmax_scale, causal)[0]


@torch.library.custom_op("recursive_kvs::fa2_keybias_varlen_fwd", mutates_args=())
def _vfwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias_raw: torch.Tensor, cu_seqlens: torch.Tensor,
          max_seqlen: int, scale: float, causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _ = ext().varlen_fwd(q, k, v, None, cu_seqlens, cu_seqlens, None, None, None, bias_raw,
                                      max_seqlen, max_seqlen, 0.0, scale, False, causal, -1, -1, 0.0, False, None)
    return out, lse


@_vfwd.register_fake
def _vfwd_fake(q, k, v, bias_raw, cu_seqlens, max_seqlen, scale, causal):
    return torch.empty_like(q), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)


@torch.library.custom_op("recursive_kvs::fa2_keybias_varlen_bwd", mutates_args=())
def _vbwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
          lse: torch.Tensor, bias_raw: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, scale: float,
          causal: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv, _, dbias = ext().varlen_bwd(dout, q, k, v, out, lse, None, None, None, cu_seqlens, cu_seqlens,
                                            bias_raw, max_seqlen, max_seqlen, 0.0, scale, False, causal,
                                            -1, -1, 0.0, False, None, None)
    return dq, dk, dv, dbias


@_vbwd.register_fake
def _vbwd_fake(dout, q, k, v, out, lse, bias_raw, cu_seqlens, max_seqlen, scale, causal):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(bias_raw)


def _vsetup_context(ctx, inputs, output):
    q, k, v, bias_raw, cu_seqlens, max_seqlen, scale, causal = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, bias_raw, cu_seqlens)
    ctx.max_seqlen = max_seqlen
    ctx.scale = scale
    ctx.causal = causal


def _vbackward(ctx, dout, dlse):
    q, k, v, out, lse, bias_raw, cu_seqlens = ctx.saved_tensors
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    dq, dk, dv, dbias = torch.ops.recursive_kvs.fa2_keybias_varlen_bwd(
        dout, q, k, v, out, lse, bias_raw, cu_seqlens, ctx.max_seqlen, ctx.scale, ctx.causal)
    return dq, dk, dv, dbias, None, None, None, None


torch.library.register_autograd("recursive_kvs::fa2_keybias_varlen_fwd", _vbackward, setup_context=_vsetup_context)


def flash_attn_key_bias_varlen(q, k, v, key_bias, cu_seqlens, max_seqlen, softmax_scale, causal=True):
    bias_raw = (key_bias.float() / softmax_scale).contiguous()
    return torch.ops.recursive_kvs.fa2_keybias_varlen_fwd(q, k, v, bias_raw, cu_seqlens, max_seqlen, softmax_scale, causal)[0]
