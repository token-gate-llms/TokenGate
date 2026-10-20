#!/bin/bash
#SBATCH --job-name=tokengate
#SBATCH --nodes=2                  # NNODES
#SBATCH --ntasks-per-node=1        # one lane-script invocation per node
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
# Standard SLURM multi-node launch: one task per node runs the SAME lane
# config; torchrun's static rendezvous (NNODES/NODE_RANK/MASTER_ADDR, wired by
# e2e_gate_lib.sh) joins them into one global DDP process group.
#
#   sbatch train-scripts/example/multinode.sbatch
#
# Edit RUN_SCRIPT (and the usual env overrides, e.g. WANDB_PROJECT) below.
# Requires the repo checkout, the python env (PYTHON) and the tokenized data
# at the same paths on every node — a shared filesystem does this for free.
set -euo pipefail

RUN_SCRIPT=${RUN_SCRIPT:-train-scripts/example/run_e2e_gate_biodexr_example.sh}

export NNODES="${SLURM_NNODES}"
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
export TORCHRUN_MASTER_PORT="${TORCHRUN_MASTER_PORT:-29500}"
export NUM_GPUS="${SLURM_GPUS_PER_NODE:-8}"

# One task per node; each learns its NODE_RANK from SLURM and runs the lane.
srun --ntasks-per-node=1 bash -c '
    export NODE_RANK="${SLURM_NODEID}"
    bash "'"${RUN_SCRIPT}"'"
'
