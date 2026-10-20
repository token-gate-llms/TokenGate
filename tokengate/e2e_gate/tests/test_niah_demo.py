"""End-to-end test of the NIAH released-checkpoint demo (examples/niah_demo.py).

Runs the demo's own main() on the committed 32k-token haystacks with the Hub
checkpoint (or TOKENGATE_DEMO_CKPT_DIR to skip the download) at K=2000 and
asserts needle-retrieval accuracy above a conservative floor — the trained
checkpoint retrieves essentially every needle at this budget, so a failure
means the demo path (loading, state-span computation, pruned decode, or
scoring) broke, not that the model got unlucky.

Run (single GPU):

    python -m tokengate.e2e_gate.tests.test_niah_demo

Not a pytest test (repo convention): prints a pass/fail summary and exits
non-zero on failure.
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

ACC_FLOOR = 0.6  # checkpoint scores ~1.0 at K=2000; floor catches wiring bugs


def main():
    sys.path.insert(0, os.path.join(_REPO_ROOT, "examples"))
    import niah_demo

    argv = ["test_niah_demo", "--k", "2000"]
    _ckpt = os.environ.get("TOKENGATE_DEMO_CKPT_DIR")
    if _ckpt:
        argv += ["--checkpoint", _ckpt]
    _attn = os.environ.get("TOKENGATE_DEMO_GATE_ATTN")
    if _attn:
        argv += ["--gate-attn", _attn]
    sys.argv = argv

    results = niah_demo.main()
    acc = results["2000"]
    assert results["all"] >= ACC_FLOOR, (
        f"no-pruning accuracy {results['all']} below floor — the base demo "
        "pipeline is broken before pruning even starts.")
    assert acc >= ACC_FLOOR, (
        f"K=2000 accuracy {acc} below floor {ACC_FLOOR} — the demo pipeline "
        "(checkpoint load / state-span computation / pruned decode / scoring) "
        "is broken.")
    print(f"\n[test_niah_demo] PASS (acc@K=2000 {acc:.2f} >= {ACC_FLOOR}, "
          f"acc@all {results['all']:.2f})")


if __name__ == "__main__":
    main()
