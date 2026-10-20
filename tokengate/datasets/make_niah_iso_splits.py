"""Deterministic val/test iso splits for the generated NIAH + BioDEX datasets.

Same recipe as the other *_iso_v1 manifests (multinews/repobench): sort the
eval pool's row ids, first VAL_N become the in-training val partition, the rest
are the reserved test partition (scored via --eval-test-full-tasks as
<group>_heldout).  niah_kamradt test rows were seeded-shuffled before id
assignment, so the sorted order is a balanced city/depth mix.

The committed outputs live in tokengate/datasets/eval_splits/; rerun this only
to rebuild them from a freshly prepared tokenized store.

Usage:
    python -m tokengate.datasets.make_niah_iso_splits
"""
import json
import os
import pickle

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("PRETRAIN_DATA_ROOT", os.path.join(_REPO_ROOT, "datasets"))
MODEL = "Qwen3-1.7B"
VAL_N = 50

SPECS = [
    # (dataset_dir, task, group, out_json[, val_n — default VAL_N])
    ("niah_kamradt", "niah_kamradt", "niah", "longbench_eval_split_niah_iso_v1.json"),
    # official BioDEX-Reactions: 500-row draw from the official validation
    # pmids; val 50 rather than the default 16/rest
    ("biodex", "biodex_reactions", "biodexr", "longbench_eval_split_biodexr_iso_v1.json", 50),
]


def main():
    here = os.path.join(os.path.dirname(__file__), "eval_splits")
    for spec in SPECS:
        ds, task, group, out = spec[:4]
        val_n = spec[4] if len(spec) > 4 else VAL_N
        p = os.path.join(DATA_ROOT, ds, task, "tokenized", MODEL, "val.pkl")
        rows = pickle.load(open(p, "rb"))
        ids = [r["id"] for r in rows]
        assert all(ids) and len(set(ids)) == len(ids), f"{task}: ids missing/non-unique"
        ids = sorted(ids)
        split = {task: {"group": group, "val": ids[:val_n], "test": ids[val_n:]}}
        op = os.path.join(here, out)
        with open(op, "w") as f:
            json.dump(split, f, indent=0)
        print(f"{task}: total={len(ids)} val={val_n} test={len(ids)-val_n} -> {op}")


if __name__ == "__main__":
    main()
