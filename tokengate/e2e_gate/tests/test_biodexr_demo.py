"""End-to-end test of the released-checkpoint demo (examples/biodexr_demo.py).

Runs the demo's own main() on the committed BioDEX-Reactions val samples with
the Hub checkpoint (or TOKENGATE_DEMO_CKPT_DIR to skip the download) and
asserts that the pipeline produces non-empty predictions with a mean
exact-match set F1 above a conservative floor — the trained checkpoint scores
far above it, so a failure means the demo path (loading, boundary computation,
pruned decode, or scoring) broke, not that the model got unlucky.

Run (single GPU):

    python -m tokengate.e2e_gate.tests.test_biodexr_demo

Not a pytest test (repo convention): prints a pass/fail summary and exits
non-zero on failure.
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

F1_FLOOR = 0.25  # checkpoint scores well above this; floor catches wiring bugs


def main():
    sys.path.insert(0, os.path.join(_REPO_ROOT, "examples"))
    import biodexr_demo

    argv = ["test_biodexr_demo", "--n", "4"]
    _ckpt = os.environ.get("TOKENGATE_DEMO_CKPT_DIR")
    if _ckpt:
        argv += ["--checkpoint", _ckpt]
    _attn = os.environ.get("TOKENGATE_DEMO_GATE_ATTN")
    if _attn:
        argv += ["--gate-attn", _attn]
    sys.argv = argv

    mean_f1 = biodexr_demo.main()
    assert mean_f1 is not None and mean_f1 >= F1_FLOOR, (
        f"demo mean F1 {mean_f1} below floor {F1_FLOOR} — the demo pipeline "
        "(checkpoint load / state-span computation / pruned decode / scoring) "
        "is broken.")
    print(f"\n[test_biodexr_demo] PASS (mean F1 {mean_f1:.3f} >= {F1_FLOOR})")


if __name__ == "__main__":
    main()
