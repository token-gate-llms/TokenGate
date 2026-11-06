"""Check GPU availability before starting an experiment without preempting other jobs."""

import subprocess


def available_gpus(gpus: str) -> None:
    """Refuse to overlap another workload; do not kill or preempt it."""
    result = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpus}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if result:
        raise RuntimeError(f"Selected GPUs have active compute processes: {result}")
