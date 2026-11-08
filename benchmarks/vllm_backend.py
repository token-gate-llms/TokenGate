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
from pathlib import Path


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
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=str(model),
            dtype="bfloat16",
            enforce_eager=True,
            language_model_only=True,
            max_model_len=length,
            max_num_batched_tokens=length,
            max_num_scheduled_tokens=length,
            max_num_seqs=1,
            async_scheduling=False,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            long_prefill_token_threshold=0,
            mamba_cache_mode="none",
            gpu_memory_utilization=0.8,
            attention_config={
                "backend": "TRITON_ATTN" if variant.endswith("_triton") else None,
                "token_gate_prefill_backend": (
                    "tiled" if variant == "vllm_tiled" else "triton"
                ),
            },
            profiler_config=profiler,
        )
    )

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
        async for output in engine.generate(
            {"prompt_token_ids": sample["input_ids"]},
            sampling,
            request_id=request_id,
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
            ttft_s=arrivals[0] - submitted,
            final_s=arrivals[-1] - submitted,
            wall_s=completed - submitted,
            prefill_s=None,
            decode_s=None,
            kept_positions=None,
        )

    async def generate() -> list[dict]:
        rows = []
        for budget in budgets:
            for sample in inputs["samples"]:
                for repeat in range(-warmups, repeats):
                    result = await request(
                        sample, budget, repeat, f"{sample['id']}-{budget}-{repeat}"
                    )
                    if repeat >= 0:
                        rows.append(result)
        if profile is not None:
            await engine.start_profile(profile_prefix="full-retention")
            try:
                await request(inputs["samples"][0], "All", 0, "profile-full")
            finally:
                await engine.stop_profile()
        return rows

    try:
        return asyncio.run(generate())
    finally:
        engine.shutdown()
