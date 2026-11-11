"""Export a published TokenGate checkpoint and measure vLLM token delivery.

The benchmark prepares prompts and verifies checkpoint identities. This adapter
only packages those artifacts with the vLLM CLI and drives its generation API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from benchmarks.batching import BatchingConfig, measure_async


def memory_snapshot(worker, reset_peak: bool = False) -> dict:
    """Read device allocation and reserved cache bytes outside the timed loop."""
    import torch

    if reset_peak:
        torch.cuda.reset_peak_memory_stats(worker.device)
    from vllm.model_executor.layers.attention import Attention

    runner = worker.model_runner
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in runner.kv_caches
    }
    return dict(
        allocated_bytes=torch.cuda.memory_allocated(worker.device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(worker.device),
        reserved_kv_bytes=sum(storages.values()),
        attention_implementations=sorted(
            {
                type(layer.impl).__name__
                for layer in runner.model.modules()
                if isinstance(layer, Attention)
            }
        ),
        token_gate_tiled_prefill=sorted(
            {
                layer.impl.use_tiled_prefill
                for layer in runner.model.modules()
                if isinstance(layer, Attention)
                and hasattr(layer.impl, "use_tiled_prefill")
            }
        ),
    )


def exported_model(
    inputs: dict, source: Path, cache: Path, *, include_gate: bool = True
) -> Path:
    """Reuse only a complete, unchanged export; invoke the existing CLI otherwise."""
    from benchmarks.run_vllm_benchmark import file_hash, write_json

    exporter = source / "tools/export_token_gate.py"
    identity = {
        "gate_sha256": inputs["gate_sha256"],
        "adapter_sha256": inputs["adapter_sha256"],
        "base_files": inputs["base_files"],
        "exporter_sha256": file_hash(exporter),
        "include_gate": include_gate,
    }
    for name in ("gate", "adapter"):
        if file_hash(Path(inputs[name])) != inputs[name + "_sha256"]:
            raise ValueError(f"Changed {name} since input preparation")
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = cache / "exports" / key
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        attempt = directory.with_name(key + ".attempt-" + uuid.uuid4().hex)
        attempt.mkdir()
        subprocess.run(
            [
                sys.executable,
                str(exporter),
                "--base",
                inputs["base"],
                "--gate",
                inputs["gate"],
                "--adapter",
                inputs["adapter"],
                "--output",
                str(attempt / "model"),
                *([] if include_gate else ["--without-token-gate"]),
            ],
            cwd=source,
            check=True,
        )
        write_json(attempt / "identity.json", identity)
        attempt.rename(directory)
    if json.loads((directory / "identity.json").read_text()) != identity:
        raise ValueError("Export identity changed")
    model = directory / "model"
    record = json.loads((model / "token_gate_export_manifest.json").read_text())
    manifest = record["files"]
    actual = {
        str(path.relative_to(model))
        for path in model.rglob("*")
        if path.is_file() and path.name != "token_gate_export_manifest.json"
    }
    if actual != set(manifest):
        raise ValueError("Export file inventory changed")
    for name, expected in manifest.items():
        path = (model / name).resolve()
        if not path.is_relative_to(model.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Invalid or changed export file: {name}")
    provenance = record["sources"]
    expected_sources = {
        "gate_sha256": inputs["gate_sha256"],
        "adapter_sha256": inputs["adapter_sha256"],
        "base_config_sha256": inputs["base_files"]["config.json"],
        "base_weight_sha256": {
            name: digest
            for name, digest in inputs["base_files"].items()
            if name.endswith(".safetensors")
        },
    }
    if any(provenance.get(key) != value for key, value in expected_sources.items()):
        raise ValueError("Export sources differ from the prepared inputs")
    return model


def run_vllm(
    inputs: dict,
    budgets: list,
    warmups: int,
    repeats: int,
    source: Path,
    cache: Path,
    *,
    variant: str = "vllm",
    profile: Path | None = None,
    batching: BatchingConfig = BatchingConfig(),
    runs: list[dict] | None = None,
    metadata: dict | None = None,
) -> list[dict]:
    """Compare gated, adapter-only, and unmodified Qwen with matched scheduling.

    Default-backend and forced-Triton controls separate backend choice from
    TokenGate's learned attention. Optional traces use an extra request after
    the measured loop, so profiler timings never enter benchmark aggregates.
    """
    import vllm
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.metrics.loggers import StatLoggerBase

    observations = []

    class BatchStats(StatLoggerBase):
        """Collect native scheduler counters without printing per-step logs."""

        def __init__(self, vllm_config, engine_index=0):
            pass

        def log_engine_initialized(self):
            pass

        def record(
            self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0
        ):
            if scheduler_stats is not None:
                details = scheduler_stats.iteration_details
                observations.append(
                    dict(
                        time_s=time.perf_counter(),
                        occupied_block_fraction=scheduler_stats.kv_cache_usage,
                        prefix_cache_hits=scheduler_stats.prefix_cache_stats.hits,
                        **(asdict(details) if details is not None else {}),
                    )
                )

    if not Path(vllm.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError(
            "Installed vLLM is not the --vllm-source checkout; install it editable"
        )
    gated = variant in ("vllm", "vllm_tiled")
    model = (
        Path(inputs["base"])
        if variant.startswith("vllm_base")
        else exported_model(inputs, source, cache, include_gate=gated)
    )
    length = inputs["max_model_len"]
    profiler = (
        {}
        if profile is None
        else {
            "profiler": "torch",
            "torch_profiler_dir": str(profile.resolve()),
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_stack": False,
        }
    )
    if batching.base_settings == "normal" and not variant.startswith("vllm_base"):
        raise ValueError("Normal vLLM settings are only a base-model control")
    scheduling = (
        {}
        if batching.base_settings == "normal"
        else dict(
            enforce_eager=True,
            async_scheduling=False,
            enable_chunked_prefill=False,
            long_prefill_token_threshold=0,
            mamba_cache_mode="none",
        )
    )
    token_budget = batching.token_budget or length * batching.size
    if batching.base_settings == "matched" and token_budget < length:
        raise ValueError("The scheduling token budget must fit a complete prompt")
    token_scheduling = (
        {}
        if batching.base_settings == "normal" and batching.token_budget is None
        else dict(
            max_num_batched_tokens=token_budget,
            max_num_scheduled_tokens=token_budget,
        )
    )
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=str(model),
            dtype="bfloat16",
            language_model_only=True,
            max_model_len=length,
            max_num_seqs=batching.size,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.8,
            attention_config={
                "backend": "TRITON_ATTN" if variant.endswith("_triton") else None,
                "token_gate_prefill_backend": (
                    "tiled" if variant == "vllm_tiled" else "triton"
                ),
            },
            profiler_config=profiler,
            disable_log_stats=batching.mode != "sequential",
            enable_logging_iteration_details=batching.mode != "sequential",
            **scheduling,
            **token_scheduling,
        ),
        stat_loggers=[BatchStats] if batching.mode != "sequential" else None,
    )
    if metadata is not None:
        config = engine.vllm_config
        metadata.update(
            max_num_seqs=config.scheduler_config.max_num_seqs,
            max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
            max_num_scheduled_tokens=config.scheduler_config.max_num_scheduled_tokens,
            chunked_prefill=config.scheduler_config.enable_chunked_prefill,
            async_scheduling=config.scheduler_config.async_scheduling,
            enforce_eager=config.model_config.enforce_eager,
            prefix_caching=config.cache_config.enable_prefix_caching,
            attention_backend=str(config.attention_config.backend),
            token_gate_prefill_backend=config.attention_config.token_gate_prefill_backend,
            compilation_mode=str(config.compilation_config.mode),
            static_admission=(
                "pause, enqueue the complete batch, resume; pause time included in throughput"
                if batching.mode == "static"
                else None
            ),
        )

    admission_barrier = None

    async def before_batch(size: int) -> None:
        nonlocal admission_barrier
        await engine.pause_generation(mode="keep", clear_cache=False)
        admission_barrier = asyncio.Barrier(size)

    async def stream(prompt, sampling, request_id):
        if admission_barrier is None:
            async for output in engine.generate(prompt, sampling, request_id):
                yield output
            return
        queue, finished = None, False
        try:
            queue = await engine.add_request(request_id, prompt, sampling)
            if await admission_barrier.wait() == 0:
                await engine.resume_generation()
            while not finished:
                output = await queue.get()
                finished = output.finished
                yield output
        finally:
            if not finished:
                await admission_barrier.abort()
                if queue is not None:
                    await engine.abort(queue.request_id, internal=True)

    async def request(sample: dict, budget, repeat: int, request_id: str) -> dict:
        start, end = sample["span_start"], sample["span_end"]
        extra = (
            {
                "token_gate_version": 1,
                "token_gate_span_start": start,
                "token_gate_span_end": end,
                "token_gate_keep_tokens": end - start if budget == "All" else budget,
            }
            if gated
            else None
        )
        sampling = SamplingParams(
            temperature=0,
            max_tokens=inputs["config"]["max_new_tokens"],
            ignore_eos=True,
            stop_token_ids=[inputs["eos_token_id"]],
            detokenize=False,
            output_kind=RequestOutputKind.DELTA,
            extra_args=extra,
        )
        tokens, arrivals = [], []
        submitted = time.perf_counter()
        async for output in stream(
            {"prompt_token_ids": sample["input_ids"]},
            sampling,
            request_id,
        ):
            if output.prompt_token_ids != sample["input_ids"]:
                raise ValueError("vLLM changed the prepared prompt")
            chunk = output.outputs[0].token_ids
            tokens.extend(chunk)
            arrivals.extend([time.perf_counter()] * len(chunk))
        completed = time.perf_counter()
        if not tokens or not output.finished:
            raise ValueError("vLLM did not return a completed token stream")
        return dict(
            id=sample["id"],
            budget=budget,
            repeat=repeat,
            token_ids=tokens,
            input_tokens=len(sample["input_ids"]),
            submitted_s=submitted,
            first_token_s=arrivals[0],
            final_token_s=arrivals[-1],
            returned_s=completed,
            token_arrivals_s=arrivals,
            ttft_s=arrivals[0] - submitted,
            final_s=arrivals[-1] - submitted,
            wall_s=completed - submitted,
            prefill_s=None,
            decode_s=None,
            kept_positions=None,
        )

    async def generate() -> list[dict]:
        nonlocal admission_barrier
        rows = []
        if batching.mode == "sequential":
            for budget in budgets:
                for sample in inputs["samples"]:
                    for repeat in range(-warmups, repeats):
                        result = await request(
                            sample, budget, repeat, f"{sample['id']}-{budget}-{repeat}"
                        )
                        if repeat >= 0:
                            rows.append(result)
        else:
            for budget in budgets:
                for repeat in range(-warmups, repeats):

                    async def execute(sample, request_id):
                        return await request(sample, budget, repeat, request_id)

                    config = batching
                    if repeat < 0 and batching.duration_s:
                        from dataclasses import replace

                        config = replace(batching, duration_s=0)
                    observations.clear()
                    await engine.collective_rpc(
                        memory_snapshot, kwargs={"reset_peak": True}
                    )
                    measured, record = await measure_async(
                        execute,
                        inputs["samples"],
                        config,
                        budget=budget,
                        repeat=repeat,
                        before_batch=before_batch
                        if batching.mode == "static"
                        else None,
                    )
                    admission_barrier = None
                    record["memory"] = await engine.collective_rpc(memory_snapshot)
                    record["scheduler_steps"] = [
                        row
                        for row in observations
                        if record["start_s"] <= row["time_s"] <= record["end_s"]
                    ]
                    if repeat >= 0:
                        rows.extend(measured)
                        if runs is not None:
                            runs.append(record)
                    if any(row.get("error") for row in measured):
                        if repeat < 0:
                            raise RuntimeError(f"Warmup failed: {measured}")
                        return rows
        if profile is not None:
            from dataclasses import replace

            await engine.start_profile(profile_prefix="full-retention")
            try:
                if batching.mode == "sequential":
                    await request(inputs["samples"][0], "All", 0, "profile-full")
                else:

                    async def profile_request(sample, request_id):
                        return await request(sample, "All", 0, "profile-" + request_id)

                    measured, _ = await measure_async(
                        profile_request,
                        inputs["samples"][: 2 * batching.size],
                        replace(batching, duration_s=0),
                        budget="All",
                        repeat=0,
                        before_batch=before_batch
                        if batching.mode == "static"
                        else None,
                    )
                    if any(row.get("error") for row in measured):
                        raise RuntimeError("Untimed batching profile failed")
            finally:
                admission_barrier = None
                await engine.stop_profile()
        return rows

    try:
        return asyncio.run(generate())
    finally:
        engine.shutdown()
