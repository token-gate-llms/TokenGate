"""Verify original training sources when inference instrumentation changes."""

import json
import shutil

import pytest
from tokengate.e2e_gate.artifacts import capture_source, file_hash

from tokengate.e2e_gate.train import selection as checkpoint


def test_inference_revision_still_requires_intact_training_source(
    tmp_path, monkeypatch
):
    original = tmp_path / "original"
    module = original / "tokengate/e2e_gate/decode/gate_decode.py"
    module.parent.mkdir(parents=True)
    module.write_text("# Original generation implementation.\n")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    manifest = capture_source(original, artifacts, training=True)
    revised = tmp_path / "inference"
    shutil.copytree(original, revised)
    (revised / module.relative_to(original)).write_text("# With token observation.\n")
    monkeypatch.setattr(
        checkpoint,
        "__file__",
        str(revised / "tokengate/e2e_gate/train/selection.py"),
    )
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    config = {
        "run_id": "completed",
        "smoke": False,
        "gate_layer": None,
        "code_manifest": str(manifest),
        "code_manifest_sha256": file_hash(manifest),
    }
    best = {"step": 600, "variant": "raw", "accuracy": 1.0, "loss": 0.01}
    (run / "params.json").write_text(json.dumps(config))
    (run / "status.json").write_text(json.dumps({"complete": True, "selection": best}))
    (run / "selection.json").write_text(json.dumps({"best": best}))
    (run / "checkpoints/lora_step600.pt").write_bytes(b"checkpoint-content")

    with pytest.raises(ValueError, match="Recorded source changed"):
        checkpoint.selected_checkpoint(run)
    actual, evidence = checkpoint.selected_checkpoint(run, training_source=original)
    assert actual == config
    assert evidence["params_sha256"] == file_hash(run / "params.json")
    assert evidence["selection"] == best
    assert evidence["adapter_sha256"] == file_hash(run / "checkpoints/lora_step600.pt")

    module.write_text("# Tampered training source.\n")
    with pytest.raises(ValueError, match="Recorded source changed"):
        checkpoint.selected_checkpoint(run, training_source=original)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="Training source manifest changed"):
        checkpoint.selected_checkpoint(run, training_source=original)


@pytest.mark.parametrize("gated", [False, True])
def test_selected_checkpoint_uses_shared_loader_without_relaxing_validation(
    tmp_path, monkeypatch, gated
):
    from types import SimpleNamespace

    import torch
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

    from tokengate import inference
    from tokengate.e2e_gate.bench import runtime
    from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
    from tokengate.e2e_gate.gate_module.gate_forward import build_gate_branch
    from tokengate.e2e_gate.gate_module.lora import (
        apply_lora_to_layers,
        collect_lora_state_dict,
    )
    from tokengate.e2e_gate.train.selection import save_weights

    def base_model(*args, **kwargs):
        torch.manual_seed(4)
        return Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
            )
        ).eval()

    config = dict(
        base_model="fixture",
        gate_layer=0 if gated else None,
        attn_backend="flex",
        run_id="fixture",
        lora_rank=4,
        lora_alpha=8,
        lora_targets=["q_proj", "v_proj"],
    )
    model = base_model()
    apply_lora_to_layers(model, [0, 1], 4, 8, config["lora_targets"])
    adapter = collect_lora_state_dict(model)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.1)
    gate = None
    if gated:
        gate = E2EGate(d_model=16, n_heads=2, n_layers_bias=1)
        build_gate_branch(model, gate, 0)
        gate.training_provenance = {"params_sha256": "fixture-params"}
    save_weights(
        model,
        gate,
        config,
        step=4,
        variant="swa",
        adapter_layers=[0, 1],
        destination=tmp_path,
    )
    adapter_path = tmp_path / "swa_lora_step4.pt"
    gate_path = tmp_path / "swa_gate_step4.pt"
    evidence = dict(
        adapter_path=str(adapter_path),
        adapter_sha256=file_hash(adapter_path),
        gate_path=str(gate_path) if gated else None,
        gate_sha256=file_hash(gate_path) if gated else None,
        selection=dict(step=4, variant="swa"),
        params_sha256="fixture-params",
    )
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", base_model)
    monkeypatch.setattr(
        runtime.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(eos_token="EOS"),
    )
    monkeypatch.setattr(
        inference, "patch_model_with_flex_attention", lambda model: model
    )
    monkeypatch.setattr(inference, "set_attention_backends_for_layers", lambda *a: None)
    loader = inference.load_gate_model
    calls = []

    def on_cpu(*args, **kwargs):
        calls.append(kwargs)
        return loader(*args, **kwargs, device="cpu")

    monkeypatch.setattr(runtime, "load_gate_model", on_cpu)
    loaded, loaded_gate, _ = runtime.load_checkpoint(config, evidence)
    assert len(calls) == 1 and calls[0]["warmup"] is False
    assert not any(p.requires_grad for p in loaded.parameters())
    for name, value in collect_lora_state_dict(loaded).items():
        torch.testing.assert_close(value, adapter[name], rtol=0, atol=0)
    if gated:
        assert not any(p.requires_grad for p in loaded_gate.parameters())
        for name, value in gate.state_dict().items():
            torch.testing.assert_close(
                loaded_gate.state_dict()[name], value, rtol=0, atol=0
            )
        with pytest.raises(ValueError, match="provenance differ: step"):
            runtime.load_checkpoint(
                config, {**evidence, "selection": dict(step=5, variant="swa")}
            )
    broken = torch.load(adapter_path, weights_only=True)
    broken.pop(next(iter(broken)))
    torch.save(broken, adapter_path)
    with pytest.raises(ValueError, match="Selected adapter changed"):
        runtime.load_checkpoint(config, evidence)
    evidence["adapter_sha256"] = file_hash(adapter_path)
    with pytest.raises(ValueError, match="every expected tensor"):
        runtime.load_checkpoint(config, evidence)
