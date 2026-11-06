"""Save experiment artifacts and verify their source, file, and runtime identities."""

import hashlib
import importlib.metadata
import json
import os
import platform
import tarfile
from pathlib import Path

import torch

_TRAINING_FILES = {
    "__init__.py",
    "recipe.json",
    "launch.py",
    "train.py",
    "runtime.py",
    "checkpoint.py",
    "data.py",
    "evaluate.py",
    "artifacts.py",
}


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(".pt.partial")
    torch.save(value, temporary)
    temporary.replace(path)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _source_files(source: Path, *, training: bool, version: int = 3) -> dict[str, str]:
    """Hash training dependencies, or all experiment sources for other stages."""
    files = {}
    directories = (
        ("tokengate", "kernels")
        if training
        else ("tokengate", "kernels", "datasets", "train-scripts")
    )
    if version == 3:
        directories = (
            "tokengate",
            "kernels",
            "datasets",
            "train-scripts",
            "benchmarks",
        )
    for directory in directories:
        for path in sorted((source / directory).rglob("*")):
            if not path.is_file() or path.suffix not in {
                ".py",
                ".sh",
                ".patch",
                ".json",
                *({".yaml", ".toml"} if version == 3 else set()),
            }:
                continue
            relative = path.relative_to(source)
            if version == 3 and relative.parts[:2] == ("tokengate", "experiments"):
                continue
            if training:
                if "tests" in relative.parts:
                    continue
                if relative.parts[:2] == ("tokengate", "experiments"):
                    runner_file = (
                        relative.parent == Path("tokengate/experiments/niah")
                        and relative.name in _TRAINING_FILES
                    )
                    package_init = str(relative) == "tokengate/experiments/__init__.py"
                    if not runner_file and not package_init:
                        continue
            files[str(relative)] = file_hash(path)
    if version == 3 and (source / "pyproject.toml").is_file():
        files["pyproject.toml"] = file_hash(source / "pyproject.toml")
    return files


def capture_source(source: Path, destination: Path, *, training: bool = False) -> Path:
    """Save a content-addressed manifest and a reconstructable source archive."""
    files = _source_files(source, training=training)
    if not files:
        raise ValueError("No experiment source files found")
    record = {"version": 3, "scope": "training" if training else "all", "files": files}
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    manifest = destination / f"source-{digest}.json"
    archive = destination / f"source-{digest}.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".gz.partial")
        with tarfile.open(temporary, "w:gz") as stream:
            for name in files:
                stream.add(source / name, arcname=name)
        temporary.replace(archive)
    atomic_json(record, manifest)
    return manifest


def verify_source(source: Path, manifest: Path, expected_hash: str) -> None:
    """Reject changed sources; retain support for earlier flat hash manifests."""
    if file_hash(manifest) != expected_hash:
        raise ValueError("Training source manifest changed")
    record = json.loads(manifest.read_text())
    if "version" in record:
        if record["version"] not in {2, 3} or record["scope"] not in {
            "training",
            "all",
        }:
            raise ValueError("Unsupported source manifest")
        files = record["files"]
        current = _source_files(
            source, training=record["scope"] == "training", version=record["version"]
        )
        if files.keys() != current.keys():
            raise ValueError("Recorded source file inventory changed")
    else:
        files = record
    for name, digest in files.items():
        if file_hash(source / name) != digest:
            raise ValueError(f"Recorded source changed: {name}")


def runtime_evidence(*, serving: bool) -> dict:
    packages = {}
    for name in ("torch", "transformers", "triton", "flash-linear-attention", "vllm"):
        packages[name] = importlib.metadata.version(name)
    evidence = {
        "packages": packages,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "CUBLAS_WORKSPACE_CONFIG",
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
            )
        },
    }
    if serving:
        import vllm

        source = Path(vllm.__file__).resolve().parent
        evidence["vllm_source"] = str(source)
        evidence["vllm_files"] = {
            str(path.relative_to(source)): file_hash(path)
            for path in sorted(source.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".so"}
        }
    return evidence
