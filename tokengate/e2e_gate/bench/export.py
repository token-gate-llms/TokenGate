"""Export a selected NIAH checkpoint and verify the weights used for engine comparisons.

The vLLM fork's exporter merges the trained LoRA adapter into the base model.
The resulting checkpoint either includes TokenGate for vLLM or leaves its gate
separate for the original implementation's merged-weight reference.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokengate.e2e_gate.artifacts import atomic_json, capture_source, file_hash
from tokengate.e2e_gate.gate_module.lora import resolve_all_adapter_layers
from tokengate.e2e_gate.train.selection import selected_checkpoint


def load_exporter(source: Path | None = None):
    """Load the fork's export tool without importing a generic ``tools`` package.

    An editable vLLM installation exposes its checkout through ``vllm.__file__``.
    A wheel installation needs an explicit checkout containing the export tool.
    """
    if source is None:
        import vllm

        source = Path(vllm.__file__).resolve().parents[1]
    path = source / "tools/export_token_gate.py"
    if not path.is_file():
        raise FileNotFoundError(
            "TokenGate export requires the vLLM fork's tools/export_token_gate.py; "
            "pass --vllm-source with the fork checkout"
        )
    spec = importlib.util.spec_from_file_location("tokengate_vllm_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_export(directory: Path, checkpoint: dict) -> tuple[Path, dict]:
    """Require the selected checkpoint and every saved output hash to match."""
    status = json.loads((directory / "status.json").read_text())
    evidence_path = directory / "evidence.json"
    if not status["complete"] or file_hash(evidence_path) != status["evidence_sha256"]:
        raise ValueError("Export is incomplete or its evidence changed")
    evidence = json.loads(evidence_path.read_text())
    if evidence["checkpoint"] != checkpoint:
        raise ValueError("Export belongs to a different selected checkpoint")
    model_path = directory / "model"
    files = {
        str(path.relative_to(model_path))
        for path in model_path.rglob("*")
        if path.is_file()
    }
    if files != set(status["files"]):
        raise ValueError("Export file inventory changed")
    for name, expected in status["files"].items():
        if file_hash(model_path / name) != expected:
            raise ValueError(f"Export file changed: {name}")
    return model_path, evidence


def export_sft(
    base: Path, adapter: Path, output: Path, config: dict, *, exporter
) -> None:
    """Merge only the separately trained SFT adapter into the original backbone."""
    model, loading = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading.get("mismatched_keys"):
        raise ValueError(f"Incomplete base language-model weights: {loading}")
    if model.config.tie_word_embeddings and not torch.equal(
        model.get_input_embeddings().weight, model.get_output_embeddings().weight
    ):
        raise ValueError("Base checkpoint contradicts tied-embedding configuration")
    if hasattr(model.config, "token_gate"):
        raise ValueError("The SFT backbone must not contain a TokenGate configuration")
    metadata = {
        "adapter_type": "lora",
        "lora_layer_ids": resolve_all_adapter_layers(model),
        "lora_rank": config["lora_rank"],
        "lora_alpha": config["lora_alpha"],
        "lora_targets": config["lora_targets"],
    }
    state = torch.load(adapter, map_location="cpu", weights_only=True)
    exporter.merge_fixed_adapter(model, state, metadata)
    model.config.architectures = [type(model).__name__]
    model.save_pretrained(output, safe_serialization=True, save_original_format=False)
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(output)


def export_run(args: argparse.Namespace) -> None:
    exporter = load_exporter(args.vllm_source)

    config, checkpoint = selected_checkpoint(
        args.run,
        diagnostic=args.diagnostic,
        training_source=getattr(args, "training_source", None),
    )
    destination = args.output.resolve()
    # Claim the entire attempt before any writes. A failure remains inspectable.
    destination.mkdir(parents=True, exist_ok=False)
    source = capture_source(Path(__file__).resolve().parents[3], destination)
    base = Path(config["base_model"])
    adapter = Path(checkpoint["adapter_path"])
    evidence = {
        "format_version": exporter.EXPORT_EVIDENCE_FORMAT_VERSION,
        "export_tool_sha256": file_hash(Path(exporter.__file__)),
        "base_model": str(base),
        "run_id": config["run_id"],
        "reference_revision": "source-manifest-sha256:"
        + config["code_manifest_sha256"],
        "adapter_step": checkpoint["selection"]["step"],
        "adapter_sha256": checkpoint["adapter_sha256"],
        "params_sha256": checkpoint["params_sha256"],
        "tokenizer_sha256": exporter.tokenizer_sha256(base),
        "base_config_sha256": exporter.artifact_sha256(base / "config.json"),
        "base_weight_sha256": {
            path.name: exporter.artifact_sha256(path)
            for path in sorted(base.glob("*.safetensors"))
        },
        "base_source": json.loads(Path(config["source_manifest"]).read_text())[
            "models"
        ][base.name],
        "checkpoint": checkpoint,
        "export_source_manifest_sha256": file_hash(source),
        "kind": args.kind,
    }
    output = destination / "model"
    evidence_path = destination / "evidence.json"
    if checkpoint["gate_path"]:
        evidence.update(
            gate_sha256=checkpoint["gate_sha256"],
            gate_step=checkpoint["selection"]["step"],
            selection_mode="topk",
            random_topk=False,
        )
        atomic_json(evidence, evidence_path)
        exporter.export_checkpoint(
            base,
            Path(checkpoint["gate_path"]),
            adapter,
            Path(checkpoint["params_path"]),
            evidence_path,
            output,
            include_gate=args.kind == "serving",
        )
    else:
        atomic_json(evidence, evidence_path)
        export_sft(base, adapter, output, config, exporter=exporter)
    hashes = {
        str(path.relative_to(output)): file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    atomic_json(
        {
            "complete": True,
            "evidence_sha256": file_hash(evidence_path),
            "files": hashes,
        },
        destination / "status.json",
    )
    print(f"Export complete: {output}", flush=True)
