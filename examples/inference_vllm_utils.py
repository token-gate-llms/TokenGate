"""Download verified NIAH artifacts and isolate the notebook's GPU runs.

The inference calls remain in inference-vllm.ipynb. This module handles file
verification and fresh Python processes so each engine releases its GPU memory
before the next comparison. It never embeds or saves an authentication token.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_checkpoint(repo_id, revision):
    """Download a pinned checkpoint package and verify its declared artifacts."""
    from huggingface_hub import snapshot_download

    package = Path(snapshot_download(repo_id, revision=revision))
    manifest = json.loads((package / "checkpoint.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(package / name) != expected:
            raise ValueError(f"Changed checkpoint artifact: {name}")
    base = snapshot_download(
        manifest["base_model"]["repo"],
        revision=manifest["base_model"]["revision"],
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
    )
    sample = json.loads((package / "sample.json").read_text())
    if sample["partition"] != "test":
        raise ValueError("The demonstration requires a held-out sample")
    start, end = sample["span_start"], sample["span_end"]
    if not 0 <= start < end < len(sample["input_ids"]):
        raise ValueError("Invalid prunable span")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "package": str(package),
        "base": base,
        "manifest": manifest,
        "sample": sample,
    }


def score_run(answer, token_ids, sample, ttft_s, total_s):
    """Apply the experiment's retrieval rule after the timing boundaries."""
    from tokengate.datasets.tasks import get_task
    from tokengate.generators import _strip_thinking

    answer = _strip_thinking(answer)
    return {
        "answer": answer.strip(),
        "token_ids": list(token_ids),
        "correct": bool(get_task("niah_kamradt").evaluate(answer, sample["answer"])),
        "ttft_ms": 1000 * ttft_s,
        "total_ms": 1000 * total_s,
        "generated_tokens": len(token_ids),
    }


def run_isolated(
    function,
    assets,
    *,
    keep_tokens,
    gpu,
    repeats,
    output_dir,
    python=None,
    max_new_tokens=1024,
):
    """Execute a notebook-defined inference function in a fresh GPU process.

    Only functions defined in this notebook are serialized. Logs and measured
    JSON stay in output_dir; a failed run raises with the log location.
    """
    import cloudpickle

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if type(keep_tokens) is not int or keep_tokens <= 0:
        raise ValueError("keep_tokens must be a positive integer")
    name = f"{assets['manifest']['run_id']}-k{keep_tokens}-{function.__name__}"
    log_path, result_path = output_dir / f"{name}.log", output_dir / f"{name}.json"
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if processes:
        raise RuntimeError(
            f"GPU {gpu} is occupied by processes {processes}; choose an idle GPU"
        )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        TOKENIZERS_PARALLELISM="false",
    )
    # Make this companion module importable when the notebook starts in examples/.
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).resolve().parent),
            str(Path(__file__).resolve().parents[1]),
        ]
    )
    worker = """
import json, sys, cloudpickle
from pathlib import Path
with open(sys.argv[1], 'rb') as stream:
    function, assets, budget, repeats, cap = cloudpickle.load(stream)
result = function(assets, budget, repeats, cap)
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\\n')
"""
    with tempfile.TemporaryDirectory(prefix="tokengate-notebook-") as scratch:
        payload = Path(scratch) / "function.pkl"
        payload.write_bytes(
            cloudpickle.dumps((function, assets, keep_tokens, repeats, max_new_tokens))
        )
        with log_path.open("w") as log:
            process = subprocess.run(
                [
                    str(python or sys.executable),
                    "-c",
                    worker,
                    str(payload),
                    str(result_path),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,
                check=False,
            )
    if process.returncode:
        raise RuntimeError(f"{name} failed; inspect {log_path}")
    result = json.loads(result_path.read_text())
    if result["K"] != keep_tokens:
        raise ValueError("Result does not match the requested retained-token budget")
    if len(result["runs"]) != repeats:
        raise ValueError("Incomplete timing repetitions")
    return result
