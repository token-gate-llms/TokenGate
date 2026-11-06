"""Require matching numerical evidence before reporting measured benchmark results.

Qualification covers the recorded validation fixtures, budgets and execution
mode. It does not replace held-out quality evaluation. Changed checkpoints,
exports, source code, runtimes or attention backends require new evidence.
"""

import json
from pathlib import Path

from tokengate.e2e_gate.artifacts import file_hash, verify_source


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _source(directory: Path, source: Path, expected: str) -> None:
    matches = [p for p in directory.glob("source-*.json") if file_hash(p) == expected]
    if len(matches) != 1:
        raise ValueError("Qualification source manifest is missing or ambiguous")
    verify_source(source, matches[0], expected)


def _same_runtime(recorded: dict, current: dict, *, serving: bool) -> None:
    keys = ["packages", "python", "cuda", "device", "capability"]
    if serving:
        keys.append("vllm_files")
    for key in keys:
        # JSON serializes the CUDA capability tuple as a list.
        if json.dumps(recorded[key]) != json.dumps(current[key]):
            raise ValueError(f"Qualification runtime differs: {key}")


def verify_qualification(
    directory: Path,
    *,
    engine: str,
    checkpoint: dict,
    tier: str,
    data: dict,
    budget: int | None,
    execution: str,
    source: Path,
    runtime: dict,
    settings: dict,
    export_status_sha256: str | None = None,
    max_gen_tokens: int,
) -> dict:
    """Check numerical evidence against the actual benchmark runner and inputs.

    Args:
        directory: A completed ``qualify`` or ``compare`` output directory.
        engine: ``research`` for original TokenGate, or ``vllm``.
        checkpoint: Verified selected gate and adapter identity.
        tier: Context-length tier being measured.
        data: Dataset and split hashes returned by the shared data loader.
        budget: Retained token count; None means full retention.
        execution: Eager or optimized engine execution.
        source: Current TokenGate source checkout.
        runtime: Package, device and source identities of the active runner.
        settings: Resolved settings of the active runner.
        export_status_sha256: Matched vLLM export status hash, if applicable.
        max_gen_tokens: Generation cap used by the benchmark.

    Returns:
        Qualification identity and fixture scope to save with the benchmark.

    Raises:
        ValueError: Evidence failed, changed, or does not cover this benchmark.
    """
    status = _read(directory / "status.json")
    if not status["complete"] or checkpoint["diagnostic"]:
        raise ValueError("Measured benchmarks require completed qualification")
    expected = {"checkpoint": checkpoint, "tier": tier, "data": data}
    if engine == "research":
        if (
            execution != "eager"
            or not status["exact_match"]
            or status["atol"] != 0
            or status["rtol"] != 0
        ):
            raise ValueError("Original TokenGate requires exact decode qualification")
        if set(status["modes"]) != {"original", "revised", "callback"}:
            raise ValueError("Qualification requires all three decode modes")
        for mode, record in status["modes"].items():
            child = directory / mode
            if _read(child / "status.json") != record or not record["complete"]:
                raise ValueError("Decode qualification status changed")
            if file_hash(child / "records.pt") != record["records_sha256"]:
                raise ValueError("Decode qualification records changed")
            if (
                any(record[k] != v for k, v in expected.items())
                or budget not in record["budgets"]
            ):
                raise ValueError(
                    "Decode qualification does not cover these inputs/budget"
                )
            if (
                record["max_gen_tokens"] != max_gen_tokens
                or record["partition"] != "validation"
                or len(record["ids"]) != 2
            ):
                raise ValueError("Decode qualification fixture scope differs")
            _same_runtime(record["runtime"], runtime, serving=False)
            if mode != "original":
                _source(child, source, record["source_manifest_sha256"])
        scope = status["modes"]["callback"]
    elif engine == "vllm":
        if (
            status["qualified"] is not True
            or not {"comparison.json", "results.json"} <= status["files"].keys()
        ):
            raise ValueError("vLLM requires a passing strict numerical comparison")
        for name, digest in status["files"].items():
            if file_hash(directory / name) != digest:
                raise ValueError(f"Numerical comparison changed: {name}")
        comparison = _read(directory / "comparison.json")
        results = _read(directory / "results.json")
        if (
            comparison["checkpoint"] != checkpoint
            or results["tier"] != tier
            or results["qualified"] is not True
        ):
            raise ValueError("Numerical comparison belongs to different inputs")
        if (
            comparison["atol"] != 0.02
            or comparison["rtol"] != 0.02
            or comparison["exact_selection_required"] is not True
            or comparison["exact_shared_scorer_required"] is not True
        ):
            raise ValueError("Numerical comparison criteria differ")
        _source(directory, source, comparison["source_manifest_sha256"])
        trace_dir = Path(comparison["trace"])
        if file_hash(trace_dir / "status.json") != comparison["trace_status_sha256"]:
            raise ValueError("Serving trace status changed")
        trace_status = _read(trace_dir / "status.json")
        for name, key in [("trace.json", "trace_sha256"), ("cases.pt", "cases_sha256")]:
            if file_hash(trace_dir / name) != trace_status[key]:
                raise ValueError(f"Serving trace changed: {name}")
        scope = _read(trace_dir / "trace.json")
        if (
            not trace_status["complete"]
            or any(scope[k] != v for k, v in expected.items())
            or budget not in scope["budgets"]
        ):
            raise ValueError("Serving qualification does not cover these inputs/budget")
        if (
            export_status_sha256 is None
            or scope["export_status_sha256"] != export_status_sha256
        ):
            raise ValueError("Qualification uses a different serving export")
        _source(trace_dir, source, scope["code_manifest_sha256"])
        _same_runtime(scope["runtime"], runtime, serving=True)
        if scope["execution"] != execution:
            raise ValueError("Qualification execution mode differs")
        keys = (
            "dtype",
            "seed",
            "enforce_eager",
            "generation_config",
            "max_num_seqs",
            "enable_chunked_prefill",
            "enable_prefix_caching",
            "async_scheduling",
            "kv_cache_memory_bytes",
            "long_prefill_token_threshold",
        )
        if (
            any(scope["settings"][key] != settings[key] for key in keys)
            or trace_status["resolved_backends"] != settings["resolved_backends"]
        ):
            raise ValueError("Qualification engine settings/backends differ")
        cases = [(i, k) for i in scope["ids"] for k in scope["budgets"]]
        if len(scope["ids"]) != 2 or len(set(cases)) != len(cases):
            raise ValueError("Invalid numerical fixture coverage")
        for stage in ("merge", "serving"):
            if [(r["id"], r["budget"]) for r in results[stage]] != cases or not all(
                r["qualified"] is True for r in results[stage]
            ):
                raise ValueError("Numerical comparison has missing or failing cases")
        if checkpoint["gate_path"]:
            if file_hash(trace_dir / "scorer.pt") != trace_status["scorer_sha256"]:
                raise ValueError("Shared scorer evidence changed")
            scorers = results["shared_scorer"]
            if set(scorers) != {"original", "merged"} or not all(
                r["scores_equal"] is True and r["selected_rows_equal"] is True
                for r in scorers.values()
            ):
                raise ValueError("Shared scorer comparison did not pass")
    else:
        raise ValueError(f"Unknown benchmark engine: {engine}")
    return {
        "directory": str(directory.resolve()),
        "status_sha256": file_hash(directory / "status.json"),
        "fixture_ids": scope["ids"],
        "budgets": scope["budgets"],
        "tier": tier,
    }
