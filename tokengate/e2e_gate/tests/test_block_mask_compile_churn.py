"""Regression test: compiled create_block_mask illegal-access under length churn.

Root cause of the biodex mixed-K (K_LOW=10) cudaErrorIllegalAddress crash that
killed both the STE and no-STE runs around step ~120-190:

`get_causal_block_mask(..., use_bucketing=False)` (the Phase-2 gate-bias path)
builds a *distinct* BlockMask for every kept-length n_kept.  biodex's
log-uniform K in [10, 5000] produces hundreds of distinct short n_kept values,
so the LRU cache (max 128) churns and evicts.  With `_compile=True`, evicting a
compiled BlockMask while continuing to churn new compiled masks triggers an
illegal memory access in the inductor static_cuda_launcher (reproduces below
around iter ~179 — matching the training crash at step ~184).

The fix (flex_attention_utils.py): only use `_compile=True` above
`_COMPILE_BLOCK_MASK_MIN_LEN`.  Below it the eager path materializes a small
n²·8-byte grid (16384²·8 ≈ 2 GiB) and is the battle-tested code path that every
pre-mixed-K run used.  The compiled path is reserved for genuinely long
sequences that would OOM the eager grid.

Run:
    python -m tokengate.e2e_gate.tests.test_block_mask_compile_churn
"""

import random
import torch

from tokengate.e2e_gate.attn import flex_attention_utils as fu
from tokengate.e2e_gate.attn.flex_attention_utils import (
    flex_attention_gate_bias,
    _BLOCK_MASK_CACHE,
)


def _churn(n_iter: int, h: int = 32, d: int = 128, seed: int = 0) -> int:
    """Run n_iter flex_attention_gate_bias calls over varied short lengths.

    Returns the number of calls that completed before any crash.  Injects
    NaN/inf into the bias periodically to mimic a drifted gate (the bias is
    exactly kv_len long with use_bucketing=False, so this is in-range — it does
    NOT cause the crash; included only to prove the crash is the block mask,
    not the score_mod).
    """
    dev = "cuda"
    rng = random.Random(seed)
    lengths = [
        min(int(10 * (500 ** rng.random())) + rng.randint(20, 120), 6000)
        for _ in range(n_iter)
    ]
    ok = 0
    for i, s in enumerate(lengths):
        q = torch.randn(1, h, s, d, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, h, s, d, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, h, s, d, device=dev, dtype=torch.bfloat16)
        bias = torch.randn(s, device=dev, dtype=torch.bfloat16)
        if i % 7 == 0:
            bias[0] = float("nan")
        flex_attention_gate_bias(q, k, v, bias, per_head=False)
        torch.cuda.synchronize()
        ok += 1
    return ok


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA required for this regression test")
        return

    n_iter = 400
    print(f"[churn] threshold _COMPILE_BLOCK_MASK_MIN_LEN={fu._COMPILE_BLOCK_MASK_MIN_LEN}")

    # 1) Current (fixed) behavior: eager below threshold — must complete clean.
    _BLOCK_MASK_CACHE.clear()
    ok = _churn(n_iter)
    passed = ok == n_iter
    print(f"[fixed]  {ok}/{n_iter} calls completed  -> {'PASS' if passed else 'FAIL'}")

    # 2) Old behavior (_compile=True everywhere): documents the crash.  We do
    #    NOT fail the suite on this — it intentionally provokes a CUDA illegal
    #    access that taints the context, so it is run last and its outcome is
    #    only reported.  Skip by default unless DEMONSTRATE_OLD_CRASH=1, because
    #    a tainted context aborts the process.
    import os
    if os.environ.get("DEMONSTRATE_OLD_CRASH") == "1":
        _BLOCK_MASK_CACHE.clear()
        saved = fu._COMPILE_BLOCK_MASK_MIN_LEN
        fu._COMPILE_BLOCK_MASK_MIN_LEN = 0  # force _compile=True for all lengths
        try:
            ok_old = _churn(n_iter)
            print(f"[old]    {ok_old}/{n_iter} (did NOT reproduce — env-dependent)")
        except RuntimeError as e:
            print(f"[old]    CRASHED after some calls: {str(e)[:80]} (expected)")
        finally:
            fu._COMPILE_BLOCK_MASK_MIN_LEN = saved

    print("RESULT:", "PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
