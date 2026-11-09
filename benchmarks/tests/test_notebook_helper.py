"""Protect notebook replay, full-test coverage, and benchmark launch boundaries."""

import json
import subprocess
import sys
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest
import yaml

from benchmarks.run_vllm_benchmark import file_hash, read_samples


@pytest.fixture
def helpers():
    return run_path("examples/inference-vllm-helper.py")


@pytest.fixture
def launch(tmp_path):
    config = tmp_path / "qwen.yaml"
    config.write_text("name: Qwen\n")
    source = tmp_path / "vllm"
    (source / "tools").mkdir(parents=True)
    (source / "tools/export_token_gate.py").touch()
    return {
        "configs": [config],
        "output": tmp_path / "results",
        "saved_runs": [],
        "engines": ["original", "vllm_tiled"],
        "vllm_source": source,
        "gpu": "2",
        "warmups": 1,
        "repeats": 3,
    }


@pytest.mark.parametrize("split,n,profile", [("val", 1, True), ("test", None, False)])
def test_launch_keeps_full_test_unlimited(
    helpers, launch, monkeypatch, split, n, profile
):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    runs = helpers["run_benchmarks"](**launch, split=split, n=n, profile=profile)
    assert runs == [launch["output"] / "qwen"]
    command, options = calls.pop()
    assert not calls
    assert command[:3] == [sys.executable, "-m", "benchmarks.run_vllm_benchmark"]
    assert command[command.index("--split") + 1] == split
    assert command[command.index("--gpu") + 1] == "2"
    assert command[command.index("--warmups") + 1] == "1"
    assert command[command.index("--repeats") + 1] == "3"
    start = command.index("--engines") + 1
    assert command[start : start + 2] == launch["engines"]
    assert ("--profile" in command) == profile
    assert ("--n" in command) == (n is not None)
    if n is not None:
        assert command[command.index("--n") + 1] == str(n)
    assert options["check"] is True
    assert options["cwd"] == Path.cwd()


def test_existing_later_run_blocks_all_launches(helpers, launch, monkeypatch):
    later = launch["configs"][0].with_name("later.yaml")
    later.write_text("name: Later\n")
    launch["configs"].append(later)
    (launch["output"] / "later").mkdir(parents=True)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("Started a run")
    )
    with pytest.raises(FileExistsError):
        helpers["run_benchmarks"](**launch)
    assert not (launch["output"] / "qwen").exists()


def test_saved_replay_never_launches_or_requires_vllm(
    helpers, launch, tmp_path, monkeypatch
):
    run = tmp_path / "saved"
    run.mkdir()
    for name, content in {
        "manifest": {"engines": ["vllm_tiled"], "warmups": 1, "repeats": 1},
        "validation": {"status": "behavior_checked"},
        "inputs": {"config": {"budgets": [64]}, "samples": [{"id": "a"}]},
    }.items():
        (run / f"{name}.json").write_text(json.dumps(content))
    (run / "measurements.jsonl").write_text(
        json.dumps(
            {
                "model": "Qwen",
                "engine": "vllm_tiled",
                "budget": 64,
                "id": "a",
                "repeat": 0,
                "token_ids": [7, 2],
                "correct": True,
                "score": 1.0,
                "ttft_s": 0.1,
                "final_s": 0.2,
                "wall_s": 0.3,
            }
        )
        + "\n"
    )
    launch.update(saved_runs=[run], vllm_source=tmp_path / "unavailable")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("Started a run")
    )
    assert helpers["run_benchmarks"](**launch) == [run]
    assert not launch["output"].exists()


def test_failed_subprocess_preserves_log(helpers, tmp_path):
    log = tmp_path / "failed.log"
    with pytest.raises(RuntimeError, match="failed.log") as error:
        helpers["_run_logged"](
            [sys.executable, "-c", "print('failure details'); raise SystemExit(3)"],
            log,
            "Test failure",
        )
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)
    assert "failure details" in log.read_text()


def test_prepare_rejects_changed_pool_before_configs(helpers, launch, tmp_path):
    samples = tmp_path / "samples.jsonl"
    samples.write_text("changed data\n")
    with pytest.raises(ValueError, match="differs from the verified"):
        helpers["prepare_niah_configs"](launch["configs"], samples, launch["output"])
    assert not list(launch["output"].glob("*.yaml"))


def test_prepared_configs_select_only_reserved_ids(
    helpers, launch, tmp_path, monkeypatch
):
    split = helpers["NIAH_SPLIT"]
    partitions = json.loads(split.read_text())["niah_kamradt_32k"]
    samples = tmp_path / "samples.jsonl"
    examples = [
        {"id": key, "body": "text", "question": "question", "answer": "answer"}
        for key in partitions["test"] + partitions["val"]
    ]
    samples.write_text("".join(json.dumps(row) + "\n" for row in examples))
    published = tmp_path / "examples/data/niah_32k_samples.jsonl"
    published.parent.mkdir(parents=True)
    published.write_text(json.dumps(examples[0]) + "\n")
    state = helpers["prepare_niah_configs"].__globals__
    monkeypatch.setitem(state, "ROOT", tmp_path)
    monkeypatch.setitem(state, "NIAH_POOL_SHA256", file_hash(samples))
    launch["configs"][0].write_text("name: Qwen\ntask: niah_kamradt\n")

    configs = helpers["prepare_niah_configs"](
        launch["configs"], samples, launch["output"]
    )
    cfg = yaml.safe_load(configs[0].read_text())
    ids = [row["id"] for row in read_samples(cfg, "test", None)]
    assert ids == partitions["test"]
    assert len(ids) == 184 and not set(ids) & set(partitions["val"])
    with pytest.raises(FileExistsError):
        helpers["prepare_niah_configs"](launch["configs"], samples, launch["output"])
