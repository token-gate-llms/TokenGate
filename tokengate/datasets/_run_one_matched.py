"""Run prepare_dataset + pack for ONE matched subset, to the canonical main-repo
pretraining_data root. Invoked once per subset (one GPU each) by the launcher.

Usage: python -m tokengate.datasets._run_one_matched <task>
where <task> is e.g. longbench_repobench_p.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from tokengate.datasets.prepare import prepare_dataset
from tokengate.datasets.pack import pack_one

MODEL = os.environ.get("PREPARE_MODEL", "Qwen/Qwen3-8B")


def _load_prepare_module(repo_root: Path, task: str):
    """The dataset folder (datasets/longbench-datasets) is data, not a package
    (and its name is not an importable identifier) — load prepare.py by path."""
    path = repo_root / "datasets" / "longbench-datasets" / task / "prepare.py"
    if not path.exists():
        raise FileNotFoundError(f"no matched prepare script at {path}")
    spec = importlib.util.spec_from_file_location(f"_matched_prepare_{task}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    task = sys.argv[1]

    main_repo = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], text=True).strip()
    main_repo = str(Path(main_repo).resolve().parent)
    out_root = Path(main_repo) / "datasets"

    mod = _load_prepare_module(Path(main_repo), task)

    print(f"[run] === START {task} === output_root={out_root}", flush=True)
    res = prepare_dataset(
        dataset_name=mod.DATASET,
        task_name=mod.TASK,
        load_train_fn=mod.load_train,
        load_val_fn=mod.load_val,
        model=MODEL,
        output_root=out_root,
        run_sanity=True,
        sanity_n=20,
        sanity_device="cuda:0",
    )
    print(f"[run] {task}: prepare done n_after_filter={res.get('n_after_filter')} "
          f"ctx_mean={res['token_lengths']['filtered']['mean']:.0f} "
          f"ans_mean={res['answer_lengths']['mean']:.1f}", flush=True)

    model_dir = out_root / mod.DATASET / mod.TASK / "tokenized" / os.path.basename(MODEL)
    pinfo = pack_one(model_dir / "train.pkl", model_dir, model=MODEL, force=True)
    print(f"[run] {task}: packed {pinfo['n_rows']} rows, {pinfo['n_tokens']} tokens", flush=True)
    print(f"[run] === DONE {task} ===", flush=True)


if __name__ == "__main__":
    main()
