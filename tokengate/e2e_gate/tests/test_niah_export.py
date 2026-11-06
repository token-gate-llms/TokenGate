"""Load the vLLM export tool explicitly and reject changed checkpoint artifacts."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokengate.e2e_gate.artifacts import atomic_json, file_hash
from tokengate.e2e_gate.bench.export import load_exporter, verify_export


@pytest.mark.parametrize("explicit", [True, False])
def test_export_tool_loads_from_checkout_without_tools_package(
    tmp_path, monkeypatch, explicit
):
    path = tmp_path / "tools/export_token_gate.py"
    path.parent.mkdir()
    path.write_text("EXPORT_EVIDENCE_FORMAT_VERSION = 1\n")
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(__file__=str(tmp_path / "vllm/__init__.py")),
    )
    # An unrelated installed tools package must not shadow the exporter.
    monkeypatch.setitem(sys.modules, "tools", SimpleNamespace())
    exporter = load_exporter(tmp_path if explicit else None)
    assert exporter.EXPORT_EVIDENCE_FORMAT_VERSION == 1
    assert Path(exporter.__file__) == path


def test_missing_export_tool_explains_required_checkout(tmp_path):
    with pytest.raises(FileNotFoundError, match="--vllm-source"):
        load_exporter(tmp_path)


@pytest.mark.parametrize("change", [None, "weights", "inventory", "checkpoint"])
def test_export_verification_binds_every_weight_to_selected_checkpoint(
    tmp_path, change
):
    checkpoint = {"run_id": "selected", "adapter_sha256": "adapter"}
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"exported weights")
    atomic_json(
        {"checkpoint": checkpoint, "kind": "serving"}, tmp_path / "evidence.json"
    )
    atomic_json(
        {
            "complete": True,
            "evidence_sha256": file_hash(tmp_path / "evidence.json"),
            "files": {"model.safetensors": file_hash(weights)},
        },
        tmp_path / "status.json",
    )
    if change == "weights":
        weights.write_bytes(b"changed")
    elif change == "inventory":
        (model / "extra.bin").write_bytes(b"extra")
    elif change == "checkpoint":
        checkpoint = {"run_id": "different"}
    if change is None:
        assert verify_export(tmp_path, checkpoint)[0] == model
    else:
        with pytest.raises(ValueError):
            verify_export(tmp_path, checkpoint)
