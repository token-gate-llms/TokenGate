"""Check cache accounting and the link between quality and timing inputs."""

import json

import pytest
import torch
from tokengate.e2e_gate.artifacts import file_hash
from tokengate.e2e_gate.bench.runtime import research_cache, storage_bytes
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from tokengate.e2e_gate.bench.benchmark_gate_e2e import (
    peak_memory,
    read_quality,
    verify_completed_training,
)


def test_peak_memory_separates_preallocated_pool_from_request_allocations(monkeypatch):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 120)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    # A request can reuse the allocator's existing reservation entirely.
    assert peak_memory({"allocated_bytes": 100, "reserved_bytes": 200}) == {
        "before_request_allocated_bytes": 100,
        "before_request_reserved_bytes": 200,
        "peak_allocated_bytes": 120,
        "peak_reserved_bytes": 200,
        "peak_allocated_increase_bytes": 20,
        "peak_reserved_increase_bytes": 0,
    }


@pytest.mark.parametrize("allocated,reserved", [(99, 200), (120, 199)])
def test_peak_memory_rejects_invalid_counter_interval(monkeypatch, allocated, reserved):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: reserved)
    with pytest.raises(RuntimeError, match="Peak counters"):
        peak_memory({"allocated_bytes": 100, "reserved_bytes": 200})


def test_cache_accounting_distinguishes_content_from_shared_storage():
    backing = torch.zeros(2, 1, 2, 10, 4, dtype=torch.bfloat16)
    assert storage_bytes([backing, backing[0], backing[1, :, :, :3]]) == 320
    cache = DynamicCache()
    cache.update(backing[0], backing[1], 0)
    cache.update(backing[0], backing[1], 1)
    cache.layers[1].crop(-7)
    memory = research_cache(cache, prompt_length=10, retained_length=3, gate_layer=0)
    assert memory["attention_kv_content_bytes"] == 2 * 1 * 2 * 13 * 4 * 2
    # Cropping produces views: the unused tail still occupies backing storage.
    assert memory["cache_storage_bytes"] == 2 * backing.untyped_storage().nbytes()
    assert memory["recurrent_state_content_bytes"] == 0
    assert memory["attention_page_bytes"] is None
    with pytest.raises(RuntimeError, match="cache length at layer 1"):
        research_cache(cache, prompt_length=10, retained_length=4, gate_layer=0)


def test_benchmark_rejects_quality_from_different_inputs(tmp_path):
    expected = {
        "engine": "research",
        "checkpoint": {"params_sha256": "checkpoint-content-hash"},
        "execution": "eager",
        "ids": ["same-id-in-both-tiers"],
        "tier": "16k",
        "partition": "validation",
        "data": {"tokenized_sha256": "16k-input-hash"},
    }
    answer = {"id": expected["ids"][0], "prediction": "needle"}
    (tmp_path / "full").mkdir()
    answers = tmp_path / "full/answers.jsonl"
    answers.write_text(json.dumps(answer) + "\n")
    (tmp_path / "status.json").write_text(
        json.dumps(
            {
                "complete": True,
                "results": [{"budget": None, "answers_sha256": file_hash(answers)}],
            }
        )
    )
    manifest = tmp_path / "evaluation.json"
    manifest.write_text(json.dumps(expected))
    assert read_quality(tmp_path, expected, None) == {answer["id"]: answer}
    for key, wrong in (
        ("tier", "32k"),
        ("data", {"tokenized_sha256": "different-input-hash"}),
        ("engine", "vllm"),
        ("execution", "compiled"),
    ):
        manifest.write_text(json.dumps({**expected, key: wrong}))
        with pytest.raises(ValueError, match="matching completed quality"):
            read_quality(tmp_path, expected, None)
    manifest.write_text(json.dumps(expected))
    answers.write_text(json.dumps({**answer, "prediction": "changed"}) + "\n")
    with pytest.raises(ValueError, match="Saved answers changed"):
        read_quality(tmp_path, expected, None)


def test_hybrid_cache_counts_actual_conv_and_recurrent_tensors():
    config = Qwen3_5TextConfig(
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"]
    )
    cache = DynamicCache(config=config)
    cache.update_conv_state(
        torch.zeros(1, 8, 6, dtype=torch.bfloat16), 0, conv_kernel_size=4
    )
    cache.update_recurrent_state(torch.zeros(1, 2, 3, 5, dtype=torch.float32), 0)
    kv = torch.zeros(1, 1, 3, 4, dtype=torch.bfloat16)
    cache.update(kv, kv, 1)
    memory = research_cache(cache, prompt_length=10, retained_length=3, gate_layer=0)
    assert memory["recurrent_state_content_bytes"] == 8 * 4 * 2 + 2 * 3 * 5 * 4
    assert memory["attention_kv_content_bytes"] == 2 * 3 * 4 * 2
    assert memory["cache_storage_bytes"] == (
        memory["recurrent_state_content_bytes"] + memory["attention_kv_content_bytes"]
    )


@pytest.fixture
def recovered_training(tmp_path):
    """A completed recovery plan retains a baseline from a failed pipeline."""
    previous = tmp_path / "failed"
    root = tmp_path / "recovery"
    (root / "configs").mkdir(parents=True)
    configs, completed = [], []
    for run_id, run_root in (("baseline", previous), ("gate", root)):
        output = run_root / "training" / run_id
        output.mkdir(parents=True)
        config = {
            "run_id": run_id,
            "root": str(run_root),
            "output_dir": str(output),
            "seed": 0,
        }
        (output / "params.json").write_text(json.dumps(config))
        status = output / "status.json"
        status.write_text(json.dumps({"complete": True}))
        configs.append(config)
        completed.append({"run_id": run_id, "status_sha256": file_hash(status)})
    (previous / "measured-training.json").write_text(json.dumps({"state": "failed"}))
    plan = root / "configs/measured-plan.json"
    plan.write_text(json.dumps({"runs": configs}))
    (root / "measured-training.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "plan_sha256": file_hash(plan),
                "completed": completed,
            }
        )
    )
    return root, configs


def test_benchmark_accepts_completed_recovery_without_changing_failed_plan(
    recovered_training,
):
    root, configs = recovered_training
    for config in configs:
        evidence = verify_completed_training(root, config)
        assert evidence == {
            "root": str(root),
            "plan_sha256": file_hash(root / "configs/measured-plan.json"),
            "status_sha256": file_hash(root / "measured-training.json"),
        }
    previous = root.parent / "failed"
    assert json.loads((previous / "measured-training.json").read_text()) == {
        "state": "failed"
    }
    with pytest.raises(RuntimeError, match="pipeline completes"):
        verify_completed_training(previous, configs[0])


@pytest.mark.parametrize("state", ["running", "failed"])
def test_benchmark_rejects_unfinished_recovery(recovered_training, state):
    root, configs = recovered_training
    path = root / "measured-training.json"
    status = json.loads(path.read_text())
    status["state"] = state
    path.write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match="pipeline completes"):
        verify_completed_training(root, configs[0])


def test_benchmark_rejects_changed_plan(recovered_training):
    root, configs = recovered_training
    path = root / "configs/measured-plan.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="plan changed"):
        verify_completed_training(root, configs[0])


def test_benchmark_rejects_unrelated_completed_plan(recovered_training):
    root, configs = recovered_training
    with pytest.raises(ValueError, match="not in the completed training plan"):
        verify_completed_training(root, {**configs[0], "seed": 1})


def test_benchmark_requires_every_planned_run_to_complete(recovered_training):
    root, configs = recovered_training
    path = root / "measured-training.json"
    status = json.loads(path.read_text())
    status["completed"].pop()
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="every planned run"):
        verify_completed_training(root, configs[0])


@pytest.mark.parametrize("change", ["status_hash", "unfinished", "params"])
def test_benchmark_rejects_changed_run_evidence(recovered_training, change):
    root, configs = recovered_training
    # Verify the other run too, not just the checkpoint being benchmarked.
    output = root / "training/gate"
    path = output / "status.json"
    if change == "params":
        (output / "params.json").write_text(json.dumps({**configs[1], "seed": 1}))
    else:
        path.write_text(json.dumps({"complete": False}))
        if change == "unfinished":
            pipeline = root / "measured-training.json"
            status = json.loads(pipeline.read_text())
            status["completed"][1]["status_sha256"] = file_hash(path)
            pipeline.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="Training completion evidence changed: gate"):
        verify_completed_training(root, configs[0])
