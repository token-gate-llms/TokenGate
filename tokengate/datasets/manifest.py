"""Pretraining-mixture manifest loader.

A "manifest" is a YAML file under
`tokengate/datasets/manifests/<version>.yaml` that
pins:
  * version: unique tag for the mix
  * datasets: ordered list of {dataset, task, prompt_version, ...} entries

The runner script (train-scripts/run_e2e_gate_pretrain.sh) calls `print_dataset_task_csvs`
to convert a manifest into the comma-separated `--dataset` and `--task`
strings expected by train_e2e_gate.py.  The trainer logs the manifest sha
+ version to W&B run config so a run can be reproduced exactly.

Why this exists: previously the (dataset, task) list lived only in the
shell script.  Two runs claiming the same MULTITASK_NAME but with edited
DATASET/TASK lists would conflate their W&B histories.  Manifests are
committed to git → reproducibility.
"""
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


MANIFESTS_DIR = Path(__file__).parent / "manifests"


def list_manifests() -> List[Path]:
    """Return paths of all *.yaml manifests in version-name order."""
    return sorted(MANIFESTS_DIR.glob("*.yaml"))


def resolve_manifest_path(version_or_path: str) -> Path:
    """Accept either a version name (e.g. `v2_accepted7_with_hybridqa`) or
    an absolute/relative path to a manifest file.  Returns the resolved Path
    or raises FileNotFoundError."""
    p = Path(version_or_path)
    if p.is_file():
        return p
    p2 = MANIFESTS_DIR / f"{version_or_path}.yaml"
    if p2.is_file():
        return p2
    raise FileNotFoundError(
        f"No manifest found for {version_or_path!r}.  Tried {p} and {p2}.  "
        f"Available manifests: {[p.stem for p in list_manifests()]}"
    )


def load_manifest(version_or_path: str) -> Dict[str, Any]:
    """Load + minimally validate a manifest.  Returns the parsed dict."""
    path = resolve_manifest_path(version_or_path)
    with path.open() as f:
        manifest = yaml.safe_load(f)
    # Required top-level keys.
    for key in ("version", "datasets"):
        if key not in manifest:
            raise ValueError(f"manifest {path} missing required key {key!r}")
    if not isinstance(manifest["datasets"], list) or not manifest["datasets"]:
        raise ValueError(f"manifest {path}: `datasets` must be a non-empty list")
    # Required per-row keys.
    for i, row in enumerate(manifest["datasets"]):
        for key in ("dataset", "task", "prompt_version"):
            if key not in row:
                raise ValueError(
                    f"manifest {path} datasets[{i}] missing required key {key!r}"
                )
    # Optional `eval_only_datasets`: subsets evaluated but never trained on.
    # Same required per-row keys as `datasets`; not added to --dataset/--task.
    _eo = manifest.get("eval_only_datasets")
    if _eo is not None:
        if not isinstance(_eo, list):
            raise ValueError(f"manifest {path}: `eval_only_datasets` must be a list")
        for i, row in enumerate(_eo):
            for key in ("dataset", "task", "prompt_version"):
                if key not in row:
                    raise ValueError(
                        f"manifest {path} eval_only_datasets[{i}] missing required "
                        f"key {key!r}"
                    )
    return manifest


def manifest_sha256(version_or_path: str) -> str:
    """SHA-256 of the manifest file (first 16 hex chars).  Used in W&B run
    config to identify which manifest a run came from."""
    path = resolve_manifest_path(version_or_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def dataset_task_csvs(version_or_path: str) -> Dict[str, str]:
    """Return {'dataset': 'd1,d2,...', 'task': 't1,t2,...'} from a manifest."""
    manifest = load_manifest(version_or_path)
    ds_list = [r["dataset"] for r in manifest["datasets"]]
    tk_list = [r["task"] for r in manifest["datasets"]]
    return {"dataset": ",".join(ds_list), "task": ",".join(tk_list)}


def eval_only_pairs_csv(version_or_path: str) -> str:
    """Return 'dataset:task,dataset:task,...' for the manifest's eval-only
    (held-out) subsets, or '' if the manifest has none.  This is the exact
    string the runner forwards to train_e2e_gate.py's --pretrain-eval-only-pairs.
    """
    manifest = load_manifest(version_or_path)
    eo = manifest.get("eval_only_datasets") or []
    return ",".join(f"{r['dataset']}:{r['task']}" for r in eo)


def expected_prompt_versions(version_or_path: str) -> Dict[str, str]:
    """Return {task: prompt_version} from a manifest, for trainer-side
    validation (cached tokens were built against the same prompt_version)."""
    manifest = load_manifest(version_or_path)
    return {r["task"]: r["prompt_version"] for r in manifest["datasets"]}


# CLI for shell-script use: `python -m tokengate.datasets.manifest <version> <key>`
#   keys: dataset (CSV), task (CSV), sha, version, json (full manifest)
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: python -m tokengate.datasets.manifest "
            "<version-or-path> <dataset|task|eval_only_pairs|sha|version|json>",
            file=sys.stderr,
        )
        print(f"available manifests: {[p.stem for p in list_manifests()]}", file=sys.stderr)
        sys.exit(1)
    version, field = sys.argv[1], sys.argv[2]
    if field == "dataset":
        print(dataset_task_csvs(version)["dataset"])
    elif field == "task":
        print(dataset_task_csvs(version)["task"])
    elif field == "eval_only_pairs":
        print(eval_only_pairs_csv(version))
    elif field == "sha":
        print(manifest_sha256(version))
    elif field == "version":
        print(load_manifest(version)["version"])
    elif field == "json":
        print(json.dumps(load_manifest(version), default=str, indent=2))
    else:
        print(f"unknown field {field!r}", file=sys.stderr)
        sys.exit(2)
