#!/usr/bin/env python3
"""Prepare pinned NIAH inputs and launch the fixed SFT/TokenGate training matrix.

Evaluation, export, numerical checks, and timing use benchmarks/run_benchmark.py.
This driver contains orchestration only; model and metric logic live in TokenGate.
"""

import argparse
import fcntl
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))

from tokengate.e2e_gate.artifacts import (
    atomic_json,
    capture_source,
    file_hash,
    verify_source,
)
from tokengate.e2e_gate.resources import available_gpus

from tokengate.datasets import pack
from tokengate.e2e_gate.train.selection import load_recipe, verify_resume
from tokengate.trainer import (
    TokenGateTrainer,
)


def prepare(root, recipe):
    """Reuse the checked-in generator and tokenizers with immutable Hub revisions."""
    from huggingface_hub import snapshot_download

    manifest = recipe["sources"]
    if set(manifest["models"]) != {run["model"] for run in recipe["runs"]}:
        raise ValueError("Model sources do not cover the training matrix")
    for entry in [*manifest["models"].values(), manifest["corpus"]]:
        if not re.fullmatch(r"[0-9a-f]{40}", entry["revision"]):
            raise ValueError("Every source must use an immutable revision")
    manifest_path = root / "configs/sources.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Pinned sources changed; choose a new output root")
    atomic_json(manifest, manifest_path)
    for name, model in manifest["models"].items():
        snapshot_download(
            model["repo"],
            revision=model["revision"],
            local_dir=root / "models" / name,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
            max_workers=4,
        )
    hashes_path = root / "configs/data-hashes.json"
    if hashes_path.exists():
        for name, digest in json.loads(hashes_path.read_text()).items():
            if file_hash(root / name) != digest:
                raise ValueError(f"Prepared input changed: {name}")
        return
    env = dict(
        os.environ,
        NIAH_DATA_DIR=str(root / "data"),
        NIAH_TOKENIZER=str(root / "models/Qwen3-1.7B"),
        NIAH_CORPUS_REVISION=manifest["corpus"]["revision"],
        PRETRAIN_DATA_ROOT=str(root / "data/prepared"),
        TOKENIZERS_PARALLELISM="false",
    )
    for filename, count, target, jitter in (
        ("niah_kamradt_all.jsonl", 12000, 16000, 150),
        ("niah_kamradt_32k.jsonl", 0, 32000, 300),
    ):
        destination = root / "data" / filename
        if not destination.exists():
            temporary = destination.with_suffix(".partial")
            if temporary.exists():
                raise ValueError(
                    f"Inspect the incomplete generation first: {temporary}"
                )
            subprocess.run(
                [
                    sys.executable,
                    str(SOURCE / "datasets/niah_kamradt/generate_niah_kamradt.py"),
                    "--output-file",
                    str(temporary),
                    "--n-train",
                    str(count),
                    "--seed",
                    "0",
                    "--test-target",
                    str(target),
                    "--test-jit",
                    str(jitter),
                ],
                env=env,
                check=True,
            )
            temporary.replace(destination)
    for model in manifest["models"]:
        env["PREPARE_MODEL"] = str(root / "models" / model)
        for tier in ("niah_kamradt", "niah_kamradt_32k"):
            subprocess.run(
                [sys.executable, str(SOURCE / "datasets" / tier / tier / "prepare.py")],
                env=env,
                check=True,
            )
        directory = root / "data/prepared/niah_kamradt/niah_kamradt/tokenized" / model
        pack.pack_one(directory / "train.pkl", directory, model=model)
    atomic_json(
        {
            str(p.relative_to(root)): file_hash(p)
            for p in sorted((root / "data").rglob("*"))
            if p.is_file()
        },
        hashes_path,
    )


def configurations(root, recipe, *, smoke):
    """Resolve the six configurations for one smoke seed or all measured seeds."""
    for seed in recipe["seeds"][:1] if smoke else recipe["seeds"]:
        for run in recipe["runs"]:
            model, layer = run["model"], run["gate_layer"]
            name = f"{model}-{'sft' if layer is None else f'gate{layer}'}-seed{seed}"
            if smoke:
                name = "smoke-" + name
            settings = {
                **recipe["training"],
                **recipe["kernels"][run["kernels"]],
                **(recipe["smoke"] if smoke else {}),
            }
            yield dict(
                **settings,
                run_id=name,
                root=str(root),
                base_model=str(root / "models" / model),
                output_dir=str(root / "training" / name),
                seed=seed,
                gate_layer=layer,
                eval_split_manifest=str(
                    SOURCE
                    / "tokengate/datasets/eval_splits/longbench_eval_split_niah_iso_v1.json"
                ),
                source_manifest=str(root / "configs/sources.json"),
                smoke=smoke,
            )


def invoke(config, root, env, *, resume=False, stop_after=None):
    path = root / "configs" / f"{config['run_id']}.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f"Recorded run configuration changed: {path}")
    atomic_json(config, path)
    available_gpus(env["CUDA_VISIBLE_DEVICES"])
    command = TokenGateTrainer.reproduction_command(
        path, resume=resume, stop_after=stop_after
    )
    with (root / "logs" / f"{config['run_id']}.log").open("a") as stream:
        stream.write(json.dumps({"command": command}) + "\n")
        stream.flush()
        subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
        )


def smoke_control(config, root, env):
    """Compare resumed and uninterrupted training, including per-rank RNG state."""
    name = "control-" + config["run_id"]
    control = {**config, "run_id": name, "output_dir": str(root / "training" / name)}
    pinned = []
    for rank in range(config["world_size"]):
        cache = root / "cache" / config["run_id"] / f"rank{rank}/triton"
        if any(cache.rglob("*.autotune.json")):
            snapshot = root / "kernel-cache" / config["run_id"] / f"rank{rank}"
            target = root / "cache" / name / f"rank{rank}/triton"
            kernel_cache_freeze(cache, snapshot)
            kernel_cache_seed(snapshot, target)
            pinned.append((snapshot, target))
    invoke(control, root, env, stop_after=4)
    for snapshot, target in pinned:
        kernel_cache_verify(snapshot, target)
    verify_resume(Path(config["output_dir"]), Path(control["output_dir"]))


def train(root, recipe, args):
    smoke = args.stage == "smoke"
    if json.loads((root / "configs/sources.json").read_text()) != recipe["sources"]:
        raise ValueError("Prepared sources differ from the recipe")
    manifest = capture_source(SOURCE, root / "configs", training=True)
    data_manifest = root / "configs/data-hashes.json"
    for name, digest in json.loads(data_manifest.read_text()).items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Prepared data changed: {name}")
    runs = [
        dict(
            c,
            code_manifest=str(manifest),
            code_manifest_sha256=file_hash(manifest),
            data_hash_manifest_sha256=file_hash(data_manifest),
        )
        for c in configurations(root, recipe, smoke=smoke)
    ]
    if args.only and not any(
        c["run_id"].removeprefix("smoke-") == args.only for c in runs
    ):
        raise ValueError(f"Unknown run: {args.only}")
    if args.resume and (smoke or not args.only):
        raise ValueError("Explicit resume requires --stage train --only RUN_ID")
    plan_path = root / "configs/measured-plan.json"
    if not smoke:
        plan = {"runs": runs}
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Measured plan changed; use a new output root")
        atomic_json(plan, plan_path)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpus,
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="3600",
    )
    for config in runs:
        if args.only and config["run_id"].removeprefix("smoke-") != args.only:
            continue
        output = Path(config["output_dir"])
        status = output / "status.json"
        if (output / "params.json").exists():
            if json.loads((output / "params.json").read_text()) != config:
                raise ValueError(f"Recorded parameters changed: {output}")
            saved = json.loads(status.read_text()) if status.exists() else {}
            if saved.get("complete") or (
                smoke
                and saved.get("step") == 4
                and (output / "resume-verification.json").exists()
            ):
                continue
            if not args.resume:
                raise ValueError(
                    f"Inspect the incomplete run before explicit resume: {output}"
                )
        if not smoke:
            smoke_id = (
                "smoke-"
                + config["run_id"].rsplit("seed", 1)[0]
                + f"seed{recipe['seeds'][0]}"
            )
            previous = root / "training" / smoke_id
            if not (previous / "resume-verification.json").exists():
                raise ValueError(f"Missing exact-resume smoke: {previous}")
            smoke_config = json.loads((previous / "params.json").read_text())
            verify_source(
                SOURCE,
                Path(smoke_config["code_manifest"]),
                smoke_config["code_manifest_sha256"],
            )
        invoke(config, root, env, resume=args.resume, stop_after=2 if smoke else None)
        if smoke:
            invoke(config, root, env, resume=True, stop_after=4)
            smoke_control(config, root, env)
    if not smoke:
        completed = []
        for config in runs:
            output = Path(config["output_dir"])
            status = output / "status.json"
            if not status.exists() or not json.loads(status.read_text())["complete"]:
                break
            if json.loads((output / "params.json").read_text()) != config:
                raise ValueError(f"Completed run differs from plan: {output}")
            completed.append(
                {"run_id": config["run_id"], "status_sha256": file_hash(status)}
            )
        atomic_json(
            {
                "state": "complete" if len(completed) == len(runs) else "incomplete",
                "plan_sha256": file_hash(plan_path),
                "completed": completed,
            },
            root / "measured-training.json",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "smoke", "train"), required=True)
    parser.add_argument("--gpus", help="Physical GPU indices, e.g. 0,1,2,3")
    parser.add_argument(
        "--only", help="Run one configuration, e.g. Qwen3-1.7B-sft-seed0"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without downloads, writes, or GPU work",
    )
    args = parser.parse_args()
    root, recipe = args.root.resolve(), load_recipe()
    if recipe["training"]["batch_size"] != 1 or recipe["training"]["grad_accum"] != 1:
        parser.error(
            "The fixed protocol uses one example per rank without accumulation"
        )
    if args.dry_run:
        print(
            json.dumps(
                recipe["sources"]
                if args.stage == "prepare"
                else [
                    c
                    for c in configurations(root, recipe, smoke=args.stage == "smoke")
                    if not args.only or c["run_id"].removeprefix("smoke-") == args.only
                ],
                indent=2,
            )
        )
        return
    if args.stage != "prepare":
        ids = args.gpus.split(",") if args.gpus else []
        if (
            len(ids) != recipe["training"]["world_size"]
            or len(set(ids)) != len(ids)
            or not all(i.isdecimal() for i in ids)
        ):
            parser.error(
                f"Select {recipe['training']['world_size']} distinct physical GPUs"
            )
    for name in ("configs", "data", "models", "logs", "training", "results"):
        (root / name).mkdir(parents=True, exist_ok=True)
    with (root / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "prepare":
            prepare(root, recipe)
        else:
            train(root, recipe, args)


def kernel_cache_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ("torch", "triton")}


def kernel_cache_freeze(source: Path, destination: Path) -> None:
    """Preserve the exact autotuning records produced by a completed probe."""
    paths = sorted(source.rglob("*.autotune.json"))
    if not paths:
        raise ValueError("No completed Triton autotuning records found")
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in paths:
        record = json.loads(path.read_text())
        if not record.get("configs_timings") or "key" not in record:
            raise ValueError(f"Invalid autotuning record: {path}")
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = file_hash(target)
    atomic_json(
        {
            "source": str(source.resolve()),
            "versions": kernel_cache_versions(),
            "files": hashes,
        },
        destination / "manifest.json",
    )


def kernel_cache_verify(
    snapshot: Path, destination: Path, *, allow_extra: bool = False
) -> dict:
    """Verify both the frozen snapshot and all seeded records in a runtime cache."""
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if kernel_cache_versions() != manifest["versions"]:
        raise ValueError("Triton/PyTorch versions differ from the frozen kernel cache")
    for name, digest in manifest["files"].items():
        if (
            file_hash(snapshot / name) != digest
            or file_hash(destination / name) != digest
        ):
            raise ValueError(f"Pinned autotuning record changed: {name}")
    extra = sorted(
        str(path.relative_to(destination))
        for path in destination.rglob("*.autotune.json")
        if str(path.relative_to(destination)) not in manifest["files"]
    )
    if extra and not allow_extra:
        raise ValueError(f"Unpinned autotuning records appeared: {extra}")
    return {"verified_records": len(manifest["files"]), "extra_records": extra}


def kernel_cache_seed(snapshot: Path, destination: Path) -> None:
    """Populate a new cache or verify an identically seeded one; never overwrite."""
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if kernel_cache_versions() != manifest["versions"]:
        raise ValueError("Triton/PyTorch versions differ from the frozen kernel cache")
    for name, digest in manifest["files"].items():
        source, target = snapshot / name, destination / name
        if file_hash(source) != digest:
            raise ValueError(f"Frozen autotuning record changed: {name}")
        if target.exists() and file_hash(target) != digest:
            raise ValueError(
                f"Refusing to replace an existing autotuning choice: {target}"
            )
    for name in manifest["files"]:
        target = destination / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot / name, target)
    kernel_cache_verify(snapshot, destination)
    atomic_json(
        {
            "snapshot": str(snapshot.resolve()),
            "manifest_sha256": file_hash(snapshot / "manifest.json"),
        },
        destination / "kernel-cache-seed.json",
    )


if __name__ == "__main__":
    main()
