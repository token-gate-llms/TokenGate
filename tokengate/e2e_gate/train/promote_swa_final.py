"""Promote a training run's SWA average to canonical swa_gate_final.pt /
swa_lora_final.pt.

Why this exists (2026-09-14): SWA weights are maintained during training
(swa_freq/swa_start) and checkpointed *inside* train_state_step{N}.pt
(swa_gate_state / swa_adapter_state / swa_n) for resume — but a run killed
before completion writes no standalone swa_gate_final.pt.  When a benchmark pin
targets the SWA average (checkpoints.py SWA_FINAL), the paired
swa_gate_final.pt / swa_lora_final.pt must exist on disk; these files are
gitignored, so this tool makes them reproducible from the (tracked-by-sha)
run's train_state.

The gate file needs the run's architecture metadata, which the raw
swa_gate_state dict lacks — so we load a gate_step{N}.pt from the SAME run as a
template and replace only its `state_dict` (asserting the key sets match).  The
adapter file is the bare swa_adapter_state dict, matching the lora_step{N}.pt
format that find_adapter_path pairs by name (swa_gate_final <-> swa_lora_final).

Usage:
  python -m tokengate.e2e_gate.train.promote_swa_final \
      --run-dir <.../baseline/L2> [--train-state-step 18000] [--out-dir <dir>]

--train-state-step defaults to the latest train_state_step*.pt in the run dir.
"""
import argparse
import glob
import os
import re

import torch

from tokengate.e2e_gate.gate_module.e2e_gate_module import load_gate
from tokengate.e2e_gate.gate_module.lora import find_adapter_path

_TS_RE = re.compile(r"train_state_step(\d+)\.pt$")


def _latest_train_state_step(run_dir):
    steps = [int(_TS_RE.search(p).group(1))
             for p in glob.glob(os.path.join(run_dir, "train_state_step*.pt"))
             if _TS_RE.search(p)]
    if not steps:
        raise FileNotFoundError(f"no train_state_step*.pt under {run_dir}")
    return max(steps)


def promote(run_dir, step, out_dir):
    ts_path = os.path.join(run_dir, f"train_state_step{step}.pt")
    if not os.path.isfile(ts_path):
        raise FileNotFoundError(f"missing {ts_path}")
    ts = torch.load(ts_path, map_location="cpu", weights_only=False)
    for k in ("swa_gate_state", "swa_adapter_state", "swa_n"):
        if k not in ts:
            raise KeyError(f"{ts_path} has no {k!r} — was SWA enabled for this run? "
                           f"(keys: {sorted(ts)})")
    swa_gate, swa_lora, swa_n = (ts["swa_gate_state"], ts["swa_adapter_state"],
                                 ts["swa_n"])

    tmpl_path = os.path.join(run_dir, f"gate_step{step}.pt")
    if not os.path.isfile(tmpl_path):
        raise FileNotFoundError(
            f"missing arch template {tmpl_path} (need a gate_step*.pt from the "
            f"same run to carry the gate architecture metadata)")
    tmpl = torch.load(tmpl_path, map_location="cpu", weights_only=False)
    tk, sk = set(tmpl["state_dict"].keys()), set(swa_gate.keys())
    if tk != sk:
        raise ValueError(f"SWA gate keys != template {os.path.basename(tmpl_path)}: "
                         f"missing {tk - sk}, extra {sk - tk}")
    tmpl["state_dict"] = swa_gate

    gpath = os.path.join(out_dir, "swa_gate_final.pt")
    lpath = os.path.join(out_dir, "swa_lora_final.pt")
    torch.save(tmpl, gpath)
    torch.save(swa_lora, lpath)

    # verify the loader accepts them and pairs them
    paired = find_adapter_path(gpath)
    if os.path.abspath(paired) != os.path.abspath(lpath):
        raise RuntimeError(f"find_adapter_path({gpath}) -> {paired}, expected {lpath}")
    load_gate(gpath)  # raises if the injected state_dict is not loadable
    print(f"[promote_swa] SWA@step{step} (swa_n={swa_n}) -> {gpath}\n"
          f"[promote_swa]                              -> {lpath}\n"
          f"[promote_swa] verified: load_gate OK, adapter pairs by name", flush=True)
    return gpath, lpath


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help=".../baseline/L{n} training output dir")
    p.add_argument("--train-state-step", type=int, default=None,
                   help="which train_state_step{N}.pt to average from (default: latest)")
    p.add_argument("--out-dir", default=None, help="output dir (default: --run-dir)")
    a = p.parse_args()
    step = a.train_state_step if a.train_state_step is not None else \
        _latest_train_state_step(a.run_dir)
    out_dir = a.out_dir or a.run_dir
    os.makedirs(out_dir, exist_ok=True)
    promote(a.run_dir, step, out_dir)


if __name__ == "__main__":
    main()
