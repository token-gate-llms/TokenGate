"""Compare original TokenGate and vLLM on the same checkpoint and tokenized inputs.

Run with ``python -m benchmarks.run_vllm_benchmark --config ... --output ...``.
Each engine executes in a fresh process. Inputs, source hashes, generated tokens,
and individual timings are saved before any aggregate or figure is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "original": "Original TokenGate",
    "vllm": "TokenGate in vLLM",
    "base": "Qwen in Transformers",
    "vllm_base": "Base Qwen in vLLM (auto)",
    "vllm_base_triton": "Base Qwen in vLLM (Triton)",
    "vllm_adapter": "Qwen + LoRA in vLLM (auto)",
    "vllm_adapter_triton": "Qwen + LoRA in vLLM (Triton)",
}


def file_hash(path: Path) -> str:
    """Hash a file without loading model weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Publish a JSON document atomically in its existing parent directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def hardware_info() -> dict:
    """Read CPU, memory, GPU, and driver details from the executing host."""
    info = {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    }
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                info["ram_bytes"] = int(line.split()[1]) * 1024
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        info["gpus"] = [
            dict(
                zip(
                    ("index", "uuid", "name", "vram_mib", "driver"),
                    (value.strip() for value in line.split(",")),
                    strict=True,
                )
            )
            for line in query.stdout.splitlines()
        ]
    except (FileNotFoundError, subprocess.CalledProcessError):
        info["gpus"] = []
    info["packages"] = {}
    for name in (
        "torch",
        "transformers",
        "vllm",
        "flash-attn",
        "flash-linear-attention",
    ):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if "torch" in info["packages"]:
        import torch

        info["pytorch_cuda_version"] = torch.version.cuda
    return info


def read_samples(cfg: dict, split: str, limit: int | None) -> list[dict]:
    """Normalize committed JSONL examples or the existing prepared eval store."""
    if "samples" in cfg:
        path = ROOT / cfg["samples"]
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        import pickle

        path = (
            ROOT
            / cfg.get("data_root", "datasets")
            / cfg["dataset"]
            / cfg["task"]
            / "tokenized"
            / Path(cfg["base_model"]).name
            / "val.pkl"
        )
        with path.open("rb") as stream:
            rows = pickle.load(stream)
    if "eval_split" in cfg:
        manifest = json.loads((ROOT / cfg["eval_split"]).read_text())
        ids = manifest[cfg.get("split_task", cfg["task"])][split]
        indexed = {row["id"]: row for row in rows}
        if len(indexed) != len(rows) or not set(ids) <= indexed.keys():
            raise ValueError("Evaluation samples have duplicate or missing IDs")
        rows = [indexed[sample_id] for sample_id in ids]
    if limit is not None:
        if limit < 1:
            raise ValueError("--n must be positive")
        rows = rows[:limit]
    normalized = []
    for row in rows:
        if "body" in row:
            article = row["body"] + "\n\nQuestion: " + row["question"]
        else:
            article = row["article"]
        target = row.get("target", row.get("answer"))
        if not isinstance(target, str) or not target:
            raise ValueError(f"Missing answer for {row['id']}")
        normalized.append(dict(id=row["id"], article=article, target=target))
    if not normalized or len({r["id"] for r in normalized}) != len(normalized):
        raise ValueError("Samples must be nonempty and have unique IDs")
    return normalized


def prepare(cfg: dict, split: str, limit: int | None) -> dict:
    """Resolve immutable sources and tokenize each sample once for both engines."""
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    from tokengate.datasets.tasks import get_task
    from tokengate.e2e_gate.bench.benchmark_gate_e2e import build_metas

    resolved = dict(cfg)
    for key, revision_key in (
        ("checkpoint", "checkpoint_revision"),
        ("base_model", "base_revision"),
    ):
        if Path(resolved[key]).exists():
            resolved[key] = str(Path(resolved[key]).resolve())
            resolved[revision_key] = None
            continue
        revision = resolved.get(revision_key)
        if revision is None:
            revision = HfApi().model_info(resolved[key]).sha
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{revision_key} must be a full commit hash")
        resolved[revision_key] = revision
    checkpoint = (
        Path(resolved["checkpoint"])
        if resolved["checkpoint_revision"] is None
        else Path(
            snapshot_download(
                resolved["checkpoint"], revision=resolved["checkpoint_revision"]
            )
        )
    )
    gate_name = resolved.get("gate_file")
    if gate_name is None:
        from tokengate.inference import _resolve_gate_path

        gate_name = Path(_resolve_gate_path(str(checkpoint))).name
    from tokengate.e2e_gate.gate_module.lora import find_adapter_path

    gate = checkpoint / gate_name
    paired = find_adapter_path(str(gate))
    if not paired or not gate.is_file():
        raise ValueError("The checkpoint must contain its paired gate and adapter")
    adapter = Path(paired)
    if cfg.get("adapter_file", adapter.name) != adapter.name:
        raise ValueError("Configured adapter is not paired with the gate")
    base = (
        Path(resolved["base_model"])
        if resolved["base_revision"] is None
        else Path(
            snapshot_download(
                resolved["base_model"],
                revision=resolved["base_revision"],
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.jinja",
                    "*.txt",
                    "*.model",
                ],
            )
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    task = get_task(cfg["task"])
    samples = read_samples(cfg, split, limit)
    for sample in samples:
        sample["input"] = task.format_prompt(sample["article"])
    metas = build_metas(tokenizer, task, samples)
    if len(metas) != len(samples):
        raise ValueError("Prompt preparation filtered requested samples")
    rows = []
    for meta in metas:
        ids, start, end = meta["prompt_ids"], meta["article_start"], meta["article_end"]
        if not 0 <= start < end < len(ids):
            raise ValueError("Prunable span must protect the final prompt token")
        rows.append(
            dict(
                id=meta["sample"]["id"],
                target=meta["sample"]["target"],
                input_ids=ids,
                span_start=start,
                span_end=end,
            )
        )
    import torch

    gate_data = torch.load(gate, map_location="cpu", weights_only=True)
    metadata = gate_data["metadata"]
    if (
        resolved["base_revision"] is not None
        and metadata["base_model"] != cfg["base_model"]
    ) or metadata["gate_layer"] != cfg["gate_layer"]:
        raise ValueError(
            "Checkpoint base model or gate layer differs from configuration"
        )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source_hashes = {
        str(p.relative_to(ROOT)): file_hash(p)
        for folder in (ROOT / "tokengate", ROOT / "benchmarks")
        for p in sorted(folder.rglob("*.py"))
    }
    return dict(
        config=resolved,
        base=str(base),
        gate=str(gate),
        adapter=str(adapter),
        gate_sha256=file_hash(gate),
        adapter_sha256=file_hash(adapter),
        gate_metadata=metadata,
        gate_policy={
            k: gate_data[k] for k in ("selection_mode", "random_topk") if k in gate_data
        },
        base_files={
            p.name: file_hash(p) for p in sorted(base.iterdir()) if p.is_file()
        },
        reference_revision=revision,
        source_sha256=source_hashes,
        samples=rows,
        sample_sha256=hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        eos_token_id=tokenizer.eos_token_id,
        max_model_len=max(len(r["input_ids"]) for r in rows) + cfg["max_new_tokens"],
    )


@dataclass
class TokenCollector:
    """Collect generated CPU tokens and their arrival times without decoding text."""

    skip_prompt: bool = False
    tokens: list[int] = field(default_factory=list)
    arrival: list[float] = field(default_factory=list)
    ended: bool = False

    def put(self, value) -> None:
        if self.skip_prompt:
            self.skip_prompt = False
            return
        tokens = value.reshape(-1).tolist()
        self.tokens.extend(tokens)
        self.arrival.extend([time.perf_counter()] * len(tokens))

    def end(self) -> None:
        self.ended = True


def original_runs(
    inputs: dict,
    budgets: list,
    warmups: int,
    repeats: int,
    *,
    base_only: bool = False,
    profile: Path | None = None,
) -> list[dict]:
    """Measure the public TokenGate API (or the existing unmodified-base arm)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tokengate import STATE_END_ID, STATE_START_ID, load_model

    cfg = inputs["config"]
    tokenizer = AutoTokenizer.from_pretrained(inputs["base"], local_files_only=True)
    if base_only:
        model = (
            AutoModelForCausalLM.from_pretrained(
                inputs["base"],
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .cuda()
            .eval()
        )
        budgets = ["All"]
    else:
        if cfg["gate_attn"] == "fa2":
            from kernels.fa2bias.keybias import available

            if not available():
                raise RuntimeError(
                    "Build kernels/fa2bias before measuring the FA2 reference"
                )
        model = load_model(
            inputs["gate"], base_model=inputs["base"], gate_attn=cfg["gate_attn"]
        )
    rows = []
    for budget in budgets:
        for sample in inputs["samples"]:
            start, end = sample["span_start"], sample["span_end"]
            ids = sample["input_ids"]
            k = end - start if budget == "All" else budget
            marked = (
                ids[:start]
                + [STATE_START_ID]
                + ids[start:end]
                + [STATE_END_ID]
                + ids[end:]
            )
            if not base_only:
                plain = model.generate(
                    marked,
                    tokenizer,
                    max_kept_prefill_tokens=k,
                    max_new_tokens=cfg["max_new_tokens"],
                )
            for repeat in range(-warmups, repeats):
                collector = TokenCollector(skip_prompt=base_only)
                tensor = torch.tensor([ids], device="cuda") if base_only else None
                torch.cuda.synchronize()
                submitted = time.perf_counter()
                if base_only:
                    model.generate(
                        tensor,
                        do_sample=False,
                        max_new_tokens=cfg["max_new_tokens"],
                        eos_token_id=tokenizer.eos_token_id,
                        streamer=collector,
                    )
                    result = None
                else:
                    result = model.generate(
                        marked,
                        tokenizer,
                        max_kept_prefill_tokens=k,
                        max_new_tokens=cfg["max_new_tokens"],
                        streamer=collector,
                    )
                completed = time.perf_counter()
                if not collector.ended or not collector.tokens:
                    raise ValueError(
                        "Generation did not deliver a complete token stream"
                    )
                if result is not None:
                    from tokengate.generators import _strip_thinking

                    streamed_answer = _strip_thinking(
                        tokenizer.decode(collector.tokens, skip_special_tokens=True)
                    )
                    if (
                        collector.tokens != plain.token_ids[0]
                        or result.token_ids != plain.token_ids
                        or result.answers != plain.answers
                        or result.kept_pos != plain.kept_pos
                        or streamed_answer != plain.answers[0]
                    ):
                        raise ValueError(
                            "Streaming changed generation or retained positions"
                        )
                if repeat >= 0:
                    rows.append(
                        dict(
                            id=sample["id"],
                            budget=budget,
                            repeat=repeat,
                            token_ids=collector.tokens,
                            ttft_s=collector.arrival[0] - submitted,
                            final_s=collector.arrival[-1] - submitted,
                            wall_s=completed - submitted,
                            prefill_s=result.prefill_s if result else None,
                            decode_s=result.decode_s if result else None,
                            kept_positions=result.kept_pos[0]
                            if result
                            else list(range(len(ids))),
                        )
                    )
    if profile is not None:
        profile.mkdir(parents=True, exist_ok=True)
        sample = inputs["samples"][0]
        ids, start, end = sample["input_ids"], sample["span_start"], sample["span_end"]
        marked = (
            ids[:start] + [STATE_START_ID] + ids[start:end] + [STATE_END_ID] + ids[end:]
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as trace:
            if base_only:
                model.generate(
                    torch.tensor([ids], device="cuda"),
                    do_sample=False,
                    max_new_tokens=cfg["max_new_tokens"],
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                model.generate(
                    marked,
                    tokenizer,
                    max_kept_prefill_tokens=end - start,
                    max_new_tokens=cfg["max_new_tokens"],
                )
        trace.export_chrome_trace(str(profile / "full-retention.json"))
    return rows


def validate_rows(rows: list[dict], inputs: dict, budgets: list, repeats: int) -> None:
    """Reject incomplete runs, duplicate measurements, and invalid timing records."""
    expected = {
        (s["id"], str(k), r)
        for s in inputs["samples"]
        for k in budgets
        for r in range(repeats)
    }
    keys = [(r["id"], str(r["budget"]), r["repeat"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Incomplete or duplicated measurement coverage")
    import math

    for row in rows:
        if not row["token_ids"] or any(type(t) is not int for t in row["token_ids"]):
            raise ValueError("Invalid generated token stream")
        if not all(math.isfinite(row[k]) for k in ("ttft_s", "final_s", "wall_s")):
            raise ValueError("Nonfinite latency")
        if not 0 < row["ttft_s"] <= row["final_s"] <= row["wall_s"]:
            raise ValueError("Inconsistent latency boundaries")


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate accuracy over examples and sequential throughput over request walls."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["engine"], str(row["budget"]))].append(row)
    result = []
    for (model, engine, budget), measurements in grouped.items():
        examples = defaultdict(list)
        for row in measurements:
            examples[row["id"]].append(row)
        consistent = all(
            len({tuple(r["token_ids"]) for r in rs}) == 1 for rs in examples.values()
        )
        # A nondeterministic example is not counted as correct by choosing its best repeat.
        correct = sum(all(r["correct"] for r in rs) for rs in examples.values())
        score = statistics.mean(min(r["score"] for r in rs) for rs in examples.values())
        wall = sum(r["wall_s"] for r in measurements)
        result.append(
            dict(
                model=model,
                engine=engine,
                budget=budget,
                examples=len(examples),
                correct=correct,
                accuracy=correct / len(examples),
                score=score,
                repeats_consistent=consistent,
                ttft_ms=1000 * statistics.median(r["ttft_s"] for r in measurements),
                final_ms=1000 * statistics.median(r["final_s"] for r in measurements),
                output_tokens=statistics.median(
                    len(r["token_ids"]) for r in measurements
                ),
                requests_per_s=len(measurements) / wall,
                output_tokens_per_s=sum(len(r["token_ids"]) for r in measurements)
                / wall,
                numerical_qualification="not_run",
            )
        )
    return result


def worker(args) -> None:
    """Load one engine, save raw rows, then score outside the timed region."""
    from transformers import AutoTokenizer

    from tokengate.datasets.tasks import get_task
    from tokengate.generators import _strip_thinking

    inputs = json.loads(args.worker.read_text())
    engine = args.engines[0]
    budgets = inputs["config"]["budgets"] if engine in ("original", "vllm") else ["All"]
    profile = args.output.parent / "profiles" / engine if args.profile else None
    if engine.startswith("vllm"):
        from benchmarks.vllm_backend import run_vllm

        rows = run_vllm(
            inputs,
            budgets,
            args.warmups,
            args.repeats,
            args.vllm_source,
            args.cache,
            variant=engine,
            profile=profile,
        )
    else:
        rows = original_runs(
            inputs,
            budgets,
            args.warmups,
            args.repeats,
            base_only=engine == "base",
            profile=profile,
        )
    validate_rows(rows, inputs, budgets, args.repeats)
    tokenizer = AutoTokenizer.from_pretrained(inputs["base"], local_files_only=True)
    task = get_task(inputs["config"]["task"])
    targets = {s["id"]: s["target"] for s in inputs["samples"]}
    for row in rows:
        answer = _strip_thinking(
            tokenizer.decode(row["token_ids"], skip_special_tokens=True)
        )
        score = float(task.evaluate(answer, targets[row["id"]]))
        row.update(
            engine=engine,
            model=inputs["config"].get("name", inputs["config"]["base_model"]),
            answer=answer,
            score=score,
            correct=score == 1.0,
            hit_output_cap=(
                len(row["token_ids"]) == inputs["config"]["max_new_tokens"]
                and row["token_ids"][-1] != inputs["eos_token_id"]
            ),
        )
    write_json(args.output, {"rows": rows, "hardware": hardware_info()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engines", nargs="+", choices=list(LABELS))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int, nargs="+")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture an untimed full-retention trace after measurements",
    )
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument(
        "--cache", type=Path, default=Path.home() / ".cache/tokengate-benchmark"
    )
    parser.add_argument("--original-python", default=sys.executable)
    parser.add_argument("--vllm-python", default=sys.executable)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats positive")
    if args.worker:
        if args.output is None:
            parser.error("worker requires --output")
        worker(args)
        return
    if args.config is None:
        parser.error("--config is required")
    cfg = yaml.safe_load(args.config.read_text())
    engines = (
        args.engines
        or cfg.get("engines")
        or (
            ["original", "vllm"]
            if cfg["task"] == "niah_kamradt"
            else ["base", "original"]
        )
    )
    if len(set(engines)) != len(engines):
        parser.error("duplicate engines")
    if not engines or set(engines) - LABELS.keys():
        parser.error("configuration contains an unknown implementation")
    if any(e.startswith("vllm") for e in engines) and args.vllm_source is None:
        parser.error("--vllm-source is required for the vLLM exporter")
    values = args.k or cfg["k_values"]
    if any(type(k) is not int or k < 1 for k in values) or len(set(values)) != len(
        values
    ):
        parser.error("budgets must be distinct positive integers")
    cfg["budgets"] = [*values, "All"]
    if type(cfg["max_new_tokens"]) is not int or cfg["max_new_tokens"] < 1:
        parser.error("max_new_tokens must be positive")
    args.output = (
        args.output
        or ROOT
        / "results/vllm-benchmark"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    ).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    from rich.console import Console

    console = Console()
    with console.status("Prepare published checkpoint and shared prompts"):
        inputs = prepare(cfg, args.split, args.n)
    write_json(args.output / "inputs.json", inputs)
    vllm_source = None
    if args.vllm_source:
        source = args.vllm_source.resolve()
        vllm_source = {
            "path": str(source),
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip(),
            "source_sha256": {
                str(path.relative_to(source)): file_hash(path)
                for folder in (source / "vllm", source / "tools")
                for path in sorted(folder.rglob("*.py"))
            },
        }
    write_json(
        args.output / "manifest.json",
        {
            "config": inputs["config"],
            "sample_sha256": inputs["sample_sha256"],
            "reference_revision": inputs["reference_revision"],
            "source_sha256": inputs["source_sha256"],
            "vllm_source": vllm_source,
            "hardware": hardware_info(),
            "engines": engines,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "token_counting": "generated tokens including EOS",
            "accuracy_unit": "distinct examples",
            "numerical_qualification": "not_run",
            "memory": {
                "peak_allocation": None,
                "reserved_capacity": None,
                "occupied_kv": None,
            },
            "throughput_denominator": "sum of measured request wall times; excludes warmup and scoring",
            "selected_gpu": args.gpu,
            "profiles": "untimed full-retention requests after measurement"
            if args.profile
            else None,
            "baseline_settings": "BF16, eager, single request, no chunking/prefix cache; backend auto or forced Triton",
        },
    )
    all_rows = []
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpu,
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        TOKENIZERS_PARALLELISM="false",
    )
    # Resolve worker imports from this checkout.
    env.pop("PYTHONPATH", None)
    for engine in engines:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                args.gpu,
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if gpu:
            raise RuntimeError(
                f"GPU {args.gpu} is occupied (PIDs: {gpu}); no worker started"
            )
        python = args.vllm_python if engine.startswith("vllm") else args.original_python
        command = [
            str(python),
            "-m",
            "benchmarks.run_vllm_benchmark",
            "--worker",
            str(args.output / "inputs.json"),
            "--output",
            str(args.output / f"{engine}.json"),
            "--engines",
            engine,
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--cache",
            str(args.cache.resolve()),
        ]
        if args.profile:
            command.append("--profile")
        if args.vllm_source:
            command += ["--vllm-source", str(args.vllm_source.resolve())]
        log_path = args.output / "logs" / f"{engine}.log"
        with console.status(f"Run {LABELS[engine]}"), log_path.open("w") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            write_json(
                args.output / "validation.json",
                {"status": "failed", "engine": engine, "log": str(log_path)},
            )
            raise RuntimeError(f"{LABELS[engine]} failed; see {log_path}")
        all_rows.extend(
            json.loads((args.output / f"{engine}.json").read_text())["rows"]
        )
        console.print(f"[green]Finished[/green] {LABELS[engine]}")
    by_case = defaultdict(dict)
    for row in all_rows:
        by_case[(row["id"], str(row["budget"]), row["repeat"])][row["engine"]] = row[
            "token_ids"
        ]
    agreement = all(
        pair["original"] == pair["vllm"]
        for pair in by_case.values()
        if "original" in pair and "vllm" in pair
    )
    write_json(
        args.output / "validation.json",
        {
            "status": "behavior_checked",
            "cross_engine_token_agreement": agreement
            if {"original", "vllm"} <= set(engines)
            else None,
            "numerical_qualification": "not_run",
            "qualified_speedups": False,
        },
    )
    (args.output / "measurements.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in all_rows)
    )
    summary = summarize(all_rows)
    write_json(args.output / "summary.json", summary)
    from rich.table import Table

    table = Table(title="Published TokenGate checkpoints")
    for col in (
        "Implementation",
        "K",
        "Correct",
        "TTFT (ms)",
        "Final token (ms)",
        "Output tokens",
        "Tokens/s",
    ):
        table.add_column(col)
    for row in summary:
        table.add_row(
            LABELS[row["engine"]],
            row["budget"],
            f"{row['correct']}/{row['examples']}",
            f"{row['ttft_ms']:.1f}",
            f"{row['final_ms']:.1f}",
            f"{row['output_tokens']:g}",
            f"{row['output_tokens_per_s']:.1f}",
        )
    console.print(table)
    console.print("Timing diagnostics; numerical qualification is recorded separately.")


if __name__ == "__main__":
    main()
