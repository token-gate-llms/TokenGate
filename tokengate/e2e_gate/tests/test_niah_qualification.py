"""Measured timings must not outlive or exceed their numerical qualification."""

import argparse
import copy
import json
from pathlib import Path

import pytest
from tokengate.e2e_gate.artifacts import atomic_json, capture_source, file_hash
from tokengate.e2e_gate.bench.qualification import verify_qualification

from tokengate.e2e_gate.bench.benchmark_gate_e2e import run_serial_benchmark


def save(value, path):
    atomic_json(value, path)
    return file_hash(path)


@pytest.fixture(params=["research", "vllm"])
def qualification(tmp_path, request):
    source = tmp_path / "source"
    (source / "tokengate").mkdir(parents=True)
    (source / "tokengate/model.py").write_text("# frozen inference source\n")
    directory = tmp_path / "qualification"
    directory.mkdir()
    checkpoint = {
        "diagnostic": False,
        "gate_path": "gate.pt",
        "adapter_sha256": "adapter",
    }
    runtime = {
        "packages": {"torch": "test"},
        "python": "test",
        "cuda": "test",
        "device": "test",
        "capability": [8, 0],
        "vllm_files": {"runner.py": "hash"},
    }
    settings = dict(
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
        generation_config="vllm",
        max_num_seqs=1,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
        kv_cache_memory_bytes=4096,
        long_prefill_token_threshold=0,
        resolved_backends=[["TRITON_ATTN"]],
    )
    common = dict(
        checkpoint=checkpoint,
        tier="16k",
        data={"sha256": "inputs"},
        ids=["a", "b"],
        budgets=[64, 500, None],
    )
    kwargs = dict(
        engine=request.param,
        checkpoint=checkpoint,
        tier="16k",
        data=common["data"],
        budget=500,
        execution="eager",
        source=source,
        runtime=runtime,
        settings=settings,
        export_status_sha256="export",
        max_gen_tokens=32,
    )
    if request.param == "research":
        modes = {}
        for mode in ("original", "revised", "callback"):
            child = directory / mode
            child.mkdir()
            manifest = capture_source(source, child)
            (child / "records.pt").write_bytes(b"validated records")
            record = {
                **common,
                "complete": True,
                "max_gen_tokens": 32,
                "partition": "validation",
                "runtime": runtime,
                "source_manifest_sha256": file_hash(manifest),
                "records_sha256": file_hash(child / "records.pt"),
            }
            save(record, child / "status.json")
            modes[mode] = record
        save(
            dict(complete=True, exact_match=True, atol=0, rtol=0, modes=modes),
            directory / "status.json",
        )
    else:
        trace = tmp_path / "trace"
        trace.mkdir()
        manifest = capture_source(source, trace)
        scope = {
            **common,
            "code_manifest_sha256": file_hash(manifest),
            "export_status_sha256": "export",
            "runtime": runtime,
            "settings": settings,
            "execution": "eager",
        }
        save(scope, trace / "trace.json")
        (trace / "cases.pt").write_bytes(b"checked distributions")
        (trace / "scorer.pt").write_bytes(b"checked gate scores")
        save(
            dict(
                complete=True,
                trace_sha256=file_hash(trace / "trace.json"),
                cases_sha256=file_hash(trace / "cases.pt"),
                scorer_sha256=file_hash(trace / "scorer.pt"),
                resolved_backends=settings["resolved_backends"],
            ),
            trace / "status.json",
        )
        manifest = capture_source(source, directory)
        save(
            dict(
                checkpoint=checkpoint,
                trace=str(trace),
                trace_status_sha256=file_hash(trace / "status.json"),
                source_manifest_sha256=file_hash(manifest),
                atol=0.02,
                rtol=0.02,
                exact_selection_required=True,
                exact_shared_scorer_required=True,
            ),
            directory / "comparison.json",
        )
        cases = [
            dict(id=i, budget=k, qualified=True)
            for i in common["ids"]
            for k in common["budgets"]
        ]
        save(
            dict(
                qualified=True,
                tier="16k",
                merge=cases,
                serving=cases,
                shared_scorer={
                    mode: dict(scores_equal=True, selected_rows_equal=True)
                    for mode in ("original", "merged")
                },
            ),
            directory / "results.json",
        )
        save(
            dict(
                complete=True,
                qualified=True,
                files={p.name: file_hash(p) for p in directory.glob("*.json")},
            ),
            directory / "status.json",
        )
    return directory, kwargs


def test_matching_qualification_is_recorded(qualification):
    directory, kwargs = qualification
    evidence = verify_qualification(directory, **kwargs)
    assert evidence["status_sha256"] == file_hash(directory / "status.json")
    assert evidence["fixture_ids"] == ["a", "b"]
    assert evidence["budgets"] == [64, 500, None]


@pytest.mark.parametrize(
    "key,value",
    [
        ("checkpoint", {"diagnostic": False, "gate_path": "different.pt"}),
        ("tier", "32k"),
        ("data", {"sha256": "other"}),
        ("budget", 2048),
        ("execution", "optimized"),
    ],
)
def test_qualification_rejects_other_measurements(qualification, key, value):
    directory, kwargs = qualification
    with pytest.raises(ValueError):
        verify_qualification(directory, **{**kwargs, key: value})


def test_qualification_rejects_changed_inference_code(qualification):
    directory, kwargs = qualification
    (kwargs["source"] / "tokengate/model.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="Recorded source changed"):
        verify_qualification(directory, **kwargs)


def test_qualification_rejects_changed_runtime(qualification):
    directory, kwargs = qualification
    runtime = copy.deepcopy(kwargs["runtime"])
    runtime["packages"]["torch"] = "changed"
    with pytest.raises(ValueError, match="runtime differs"):
        verify_qualification(directory, **{**kwargs, "runtime": runtime})


def test_qualification_rejects_changed_raw_evidence(qualification):
    directory, kwargs = qualification
    path = (
        directory / "callback/records.pt"
        if kwargs["engine"] == "research"
        else directory.parent / "trace/cases.pt"
    )
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_qualification(directory, **kwargs)


def test_qualification_rejects_failed_status(qualification):
    directory, kwargs = qualification
    status = json.loads((directory / "status.json").read_text())
    status["exact_match" if kwargs["engine"] == "research" else "qualified"] = False
    save(status, directory / "status.json")
    with pytest.raises(ValueError):
        verify_qualification(directory, **kwargs)


def test_serving_qualification_checks_export_and_backend(qualification):
    directory, kwargs = qualification
    if kwargs["engine"] != "vllm":
        return
    with pytest.raises(ValueError, match="different serving export"):
        verify_qualification(directory, **{**kwargs, "export_status_sha256": "changed"})
    settings = {**kwargs["settings"], "resolved_backends": [["FLASH_ATTN"]]}
    with pytest.raises(ValueError, match="settings/backends differ"):
        verify_qualification(directory, **{**kwargs, "settings": settings})


def test_measured_benchmark_requires_qualification_before_loading_model():
    args = argparse.Namespace(
        budget="64",
        partition="test",
        tier="16k",
        diagnostic=False,
        decision=Path("decision.json"),
        quality=Path("quality"),
        qualification=None,
    )
    with pytest.raises(ValueError, match="numerical qualification"):
        run_serial_benchmark(args)
