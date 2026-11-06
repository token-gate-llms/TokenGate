"""Install the notebook's pinned inference code in a separate Python environment.

The notebook downloads TokenGate before importing this module. This setup then
fetches the vLLM fork, installs its matching compiled wheel, and builds TokenGate's
FA2 key-bias kernel. Logs and dependencies stay in the chosen cache directory.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

VLLM_REVISION = "a6165a7a9ec2012d1017019e9a370468852a69b2"
VLLM_WHEEL_REVISION = "25fc60723cf6c92c041b0c665450023b96544e9a"


def prepare_runtime(root, cache, gpu, console):
    """Return an isolated worker Python with the required code and CUDA kernels.

    Args:
        root: TokenGate checkout downloaded by the notebook.
        cache: Writable directory for the environment, vLLM source, and build logs.
        gpu: Idle physical GPU index used to check CUDA and select the kernel target.
        console: Rich console for concise setup progress.
    """
    if platform.system() != "Linux" or not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError(
            "Use Linux with a Python 3.11, 3.12, or 3.13 notebook kernel"
        )
    root, cache = Path(root).resolve(), Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    runtime = cache / "runtime"
    python = runtime / "bin/python"
    vllm = cache / "vllm"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        VLLM_USE_PRECOMPILED="1",
        VLLM_PRECOMPILED_WHEEL_COMMIT=VLLM_WHEEL_REVISION,
        VLLM_PRECOMPILED_WHEEL_VARIANT="cu130",
        FA2BIAS_SITE=str(cache / "fa2bias/site"),
        FA2BIAS_BUILD=str(cache / "fa2bias/build"),
        FA2_SDIST=str(cache / "flash_attn-2.8.3.post1.tar.gz"),
        MAX_JOBS="4",
    )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    identity = {
        "tokengate": revision,
        "source": str(root),
        "vllm": VLLM_REVISION,
        "wheel": VLLM_WHEEL_REVISION,
        "python": list(sys.version_info[:2]),
        "setup_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    marker = cache / "installed.json"

    def run(title, command):
        log = cache / (title.lower().replace(" ", "-") + ".log")
        with console.status(title), log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"{title} failed; see {log}")
        console.print(f"[green]Ready[/green] {title}")

    if not marker.exists() or json.loads(marker.read_text()) != identity:
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(runtime)
        if not (vllm / ".git").exists():
            run(
                "Fetch vLLM",
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    "--filter=blob:none",
                    "https://github.com/luis-gasparschroeder/vllm.git",
                    str(vllm),
                ],
            )
        run("Pin vLLM", ["git", "-C", str(vllm), "checkout", "--detach", VLLM_REVISION])
        pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
        run(
            "Install CUDA PyTorch",
            pip
            + [
                "torch==2.14.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu130",
            ],
        )
        run(
            "Install build dependencies",
            pip + ["-r", str(vllm / "requirements/build/cuda.txt")],
        )
        run("Install vLLM", pip + ["--no-build-isolation", "-e", str(vllm)])
        run(
            "Install TokenGate",
            pip
            + [
                "-e",
                f"{root}[data]",
                "transformers==5.22.1",
                "flash-linear-attention==0.6.0",
                "cloudpickle==3.1.2",
            ],
        )
        architecture = subprocess.check_output(
            [
                str(python),
                "-c",
                "import torch; print('.'.join(map(str, torch.cuda.get_device_capability())))",
            ],
            env=env,
            text=True,
        ).strip()
        env["FA2BIAS_ARCHS"] = architecture + "+PTX"
        run(
            "Build TokenGate attention",
            [str(python), str(root / "kernels/fa2bias/build.py")],
        )
    # Import in a fresh worker, including the generated FlashMLA Python wrapper.
    run(
        "Check inference imports",
        [
            str(python),
            "-c",
            (
                "import vllm, tokengate, torch; "
                "from vllm.config.token_gate import TokenGateConfig; "
                "from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache; "
                "from tokengate.inference import load_gate_model; "
                "from tokengate.e2e_gate.decode.gate_decode import TokenDelivery; "
                "assert torch.cuda.is_available(); print(vllm.__file__); print(tokengate.__file__)"
            ),
        ],
    )
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    # These paths must also reach the isolated inference workers.
    os.environ["FA2BIAS_SITE"] = env["FA2BIAS_SITE"]
    return python
