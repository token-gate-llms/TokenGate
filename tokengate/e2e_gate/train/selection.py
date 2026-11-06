"""Select and save validation checkpoints, including state for exact training resume."""

import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tokengate.e2e_gate.artifacts import (
    atomic_json,
    atomic_save,
    file_hash,
    verify_source,
)
from tokengate.e2e_gate.gate_module.e2e_gate_module import save_gate
from tokengate.e2e_gate.gate_module.lora import (
    collect_lora_named_params,
    collect_lora_state_dict,
)


@dataclass
class CheckpointSelection:
    """Rank by accuracy, answer loss, and earlier step; stop after warmup.

    A cycle can evaluate both raw and averaged weights. Patience is counted
    once per cycle, after all candidates have been considered.
    """

    patience: int = 3
    warmup_steps: int = 1000
    best: dict | None = None
    stale_cycles: int = 0
    last_cycle: int = -1
    improved: bool = False

    def consider(self, step: int, variant: str, accuracy: float, loss: float) -> bool:
        """Record a candidate and return whether it improves the best result."""
        if not math.isfinite(accuracy) or not math.isfinite(loss):
            raise ValueError("Validation produced nonfinite accuracy or answer loss")
        if not 0 <= accuracy <= 1 or loss < 0:
            raise ValueError("Validation metrics are outside their valid range")
        candidate = dict(step=step, variant=variant, accuracy=accuracy, loss=loss)

        def key(item):
            return (item["accuracy"], -item["loss"], -item["step"])

        if self.best is None or key(candidate) > key(self.best):
            self.best = candidate
            self.improved = True
            return True
        return False

    def finish_cycle(self, step: int) -> bool:
        """Count one completed validation cycle and return whether to stop."""
        if step <= self.last_cycle:
            raise ValueError("Validation cycles must have increasing optimizer steps")
        if step >= self.warmup_steps:
            self.stale_cycles = 0 if self.improved else self.stale_cycles + 1
        self.improved = False
        self.last_cycle = step
        return self.patience > 0 and self.stale_cycles >= self.patience

    def state_dict(self) -> dict:
        """Return JSON-serializable state for exact checkpoint resume."""
        return asdict(self)


def save_weights(
    model,
    gate,
    config: dict,
    *,
    step: int,
    variant: str,
    adapter_layers: list[int],
    destination: Path,
) -> None:
    """Save a matched pair; a baseline snapshot contains only its own adapter."""
    destination.mkdir(parents=True, exist_ok=True)
    prefix = "swa_" if variant == "swa" else ""
    atomic_save(
        collect_lora_state_dict(model), destination / f"{prefix}lora_step{step}.pt"
    )
    if gate is not None:
        metadata = dict(
            step=step,
            gate_layer=config["gate_layer"],
            base_model=config["base_model"],
            run_id=config["run_id"],
            adapter_type="lora",
            lora_layer_ids=adapter_layers,
            lora_rank=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_targets=config["lora_targets"],
            lora_start_layer=min(adapter_layers),
            swa=variant == "swa",
        )
        path = destination / f"{prefix}gate_step{step}.pt"
        temporary = path.with_suffix(".pt.partial")
        save_gate(gate, str(temporary), metadata=metadata)
        temporary.replace(path)


def selected_checkpoint(
    run: Path, *, diagnostic: bool = False, training_source: Path | None = None
) -> tuple[dict, dict]:
    """Verify training provenance before returning the selected checkpoint.

    Inference revisions may explicitly identify the original training snapshot.
    That snapshot must still match the recorded manifest exactly. Callers record
    their inference source separately; this option does not certify equivalence.
    """
    config = json.loads((run / "params.json").read_text())
    status = json.loads((run / "status.json").read_text())
    best = json.loads((run / "selection.json").read_text())["best"]
    if bool(config["smoke"]) != diagnostic:
        raise ValueError("Diagnostic evaluation is reserved for smoke checkpoints")
    if not diagnostic and not status["complete"]:
        raise ValueError("Measured evaluation requires completed training")
    if best is None or best != status["selection"]:
        raise ValueError("Training status and checkpoint selection disagree")
    verify_source(
        training_source or Path(__file__).resolve().parents[3],
        Path(config["code_manifest"]),
        config["code_manifest_sha256"],
    )
    if best["variant"] not in {"raw", "swa"}:
        raise ValueError("Unsupported selected checkpoint variant")
    prefix = "swa_" if best["variant"] == "swa" else ""
    adapter = run / "checkpoints" / f"{prefix}lora_step{best['step']}.pt"
    gate = run / "checkpoints" / f"{prefix}gate_step{best['step']}.pt"
    evidence = {
        "run_id": config["run_id"],
        "diagnostic": diagnostic,
        "selection": best,
        "params_path": str(run / "params.json"),
        "params_sha256": file_hash(run / "params.json"),
        "adapter_path": str(adapter),
        "adapter_sha256": file_hash(adapter),
        "gate_path": str(gate) if config["gate_layer"] is not None else None,
        "gate_sha256": file_hash(gate) if config["gate_layer"] is not None else None,
    }
    return config, evidence


def trainable_parameters(model, gate) -> dict:
    parameters = {"adapter." + name: p for name, p in collect_lora_named_params(model)}
    if gate is not None:
        parameters.update(
            {
                "gate." + name: p
                for name, p in gate.named_parameters()
                if p.requires_grad
            }
        )
    if any(p.dtype != torch.float32 for p in parameters.values()):
        raise ValueError("All trainable parameters must use FP32 storage")
    return parameters


@contextmanager
def averaged_parameters(parameters: dict, average: dict):
    """Temporarily install SWA weights and restore live weights even on failure."""
    live = {name: p.detach().clone() for name, p in parameters.items()}
    try:
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(average[name])
        yield
    finally:
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(live[name])


def rng_state() -> dict:
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(),
    )


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def assert_equal(actual, expected, path: str) -> None:
    """Require exact state equality, including optimizer and per-rank RNG state."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"Different state keys: {path}")
        for key in expected:
            assert_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(f"Different state length: {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_equal(left, right, f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"Different state: {path}: {actual!r} != {expected!r}")


def verify_resume(resumed: Path, control: Path) -> None:
    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    state = load(resumed / "train_state.pt")
    reference = load(control / "train_state.pt")
    for key in (
        "step",
        "parameters",
        "optimizer",
        "scheduler",
        "average",
        "average_count",
        "selection",
        "rng_by_rank",
    ):
        assert_equal(state[key], reference[key], key)
    initial = load(resumed / "initial_parameters.pt")
    assert_equal(initial, load(control / "initial_parameters.pt"), "initial")
    changed = []
    for name, value in state["parameters"].items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite trained parameter: {name}")
        if not torch.equal(value, initial[name]):
            changed.append(name)
    if not any(name.startswith("adapter.") for name in changed):
        raise AssertionError("No adapter parameter learned an update")
    if state["config"]["gate_layer"] is not None:
        if not any(name.startswith("gate.gate_branch.") for name in changed):
            raise AssertionError("The gate scoring branch did not learn an update")
    atomic_json(
        dict(exact_match=True, step=state["step"], changed_parameters=changed),
        resumed / "resume-verification.json",
    )


def load_recipe(path=None):
    """Read the single checked-in reproduction configuration (source checkout)."""
    import yaml

    recipe = Path(
        path
        or os.environ.get("TOKENGATE_RECIPE")
        or Path(__file__).resolve().parents[3] / "train-scripts/niah/reproduction.yaml"
    )
    with recipe.open() as stream:
        return yaml.safe_load(stream)
