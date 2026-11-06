"""Checkpoint policy survives normal and averaged saves; legacy stays unknown.

Run with pytest alongside the training regression tests.
"""

import copy
import tempfile
from pathlib import Path

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from tokengate.e2e_gate.gate_module.e2e_gate_module import (
    E2EGate,
    load_gate,
    save_gate,
    swa_inplace_update,
    to_cast_linears,
)


def test_checkpoint_policy_round_trip():
    gate = E2EGate(d_model=16, n_heads=2, n_layers_bias=2)
    gate.gate_branch = Qwen3MLP(Qwen3Config(hidden_size=16, intermediate_size=32))
    gate.gate_branch.down_proj = torch.nn.Linear(32, 1, bias=False)
    to_cast_linears(gate.gate_branch)
    gate.training_provenance = {"run_id": "test-run", "params_sha256": "a" * 64}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for random_policy in (False, True):
            gate.random_topk = random_policy
            averaged = copy.deepcopy(gate)
            swa_inplace_update(
                list(averaged.named_parameters()), list(gate.named_parameters()), 1
            )
            for name, source in (
                ("gate_step7.pt", gate),
                ("gate_final.pt", gate),
                ("swa_gate_step7.pt", averaged),
                ("swa_gate_final.pt", averaged),
            ):
                path = root / name
                save_gate(source, str(path), metadata={"step": 7})
                restored, metadata = load_gate(str(path))
                assert restored.selection_mode == "topk"
                assert restored.random_topk is random_policy
                assert metadata == {**gate.training_provenance, "step": 7}
                assert restored.training_provenance == gate.training_provenance
        legacy = torch.load(root / "gate_final.pt", weights_only=True)
        del legacy["random_topk"]
        torch.save(legacy, root / "legacy.pt")
        restored, _ = load_gate(str(root / "legacy.pt"))
        assert restored.random_topk is None
        try:
            save_gate(restored, str(root / "uncertified.pt"))
        except ValueError:
            pass
        else:
            raise AssertionError("Legacy policy must not be certified as False")
