# Training run scripts

Per-dataset bash configs for the released TokenGate runs: one lane per
dataset, with the gate at **layer 1** of the model, all sharing the same
finalized recipe.  The `coadaptN` token in filenames and run names is the
internal run-series tag of each recipe iteration — it is kept in the names
so output dirs and W&B histories stay addressable, but the lanes differ
only in dataset and model.

| dir | model | what |
|---|---|---|
| `2wiki/` | Qwen3-1.7B | 2WikiMQA (coadapt9), gate layer 1 |
| `biodexr/` | Qwen3-1.7B | BioDEX-Reactions (coadapt16), gate layer 1 |
| `multinews/` | Qwen3-1.7B | MultiNews (coadapt12), gate layer 1 |
| `niah/` | Qwen3-1.7B | NIAH (coadapt13), gate layer 1 |
| `niah-qwen3p5/` | Qwen3.5-2B hybrid | NIAH (coadapt17), gate layer 1 (a DeltaNet layer) |
| `repobench/` | Qwen3-1.7B | RepoBench-P (coadapt11), gate layer 1 |
| `example/` | Qwen3-1.7B | self-contained BioDEX starter config + multi-node SLURM launcher |

## Running one

Launch from anywhere — each script cds to the repo root itself:

```bash
bash train-scripts/biodexr/run_e2e_mlpgate_pretrain_biodexr_iso_coadapt16_l1_1p7b.sh
```

Every config sets its lane variables, cds to the repo root with

```bash
cd "$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
```

(which resolves correctly at any nesting depth, and to the worktree root
when run from a worktree), then calls the shared driver
`bash train-scripts/run_e2e_gate_pretrain.sh`, which sources
`train-scripts/e2e_gate_lib.sh`.  To change behavior for one lane, edit
its config script — not the driver or the lib.

A Python alternative to these configs is the `TokenGateConfig` /
`TokenGateTrainer` interface (`tokengate/trainer.py`), which builds the
same trainer CLI; see the root README's "Training interface" section.
