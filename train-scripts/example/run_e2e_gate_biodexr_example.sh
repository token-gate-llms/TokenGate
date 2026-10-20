#!/bin/bash
# End-to-end example: train TokenGate on BioDEX-Reactions (adverse-drug-reaction
# extraction from biomedical articles) — gate at layer 1, Qwen3-1.7B, the
# converged iso recipe.  The released demo checkpoint
# (hf.co/liana313/tokengate-biodexr-qwen3-1.7b) came from this exact config.
#
# This file IS the run config: every hyperparameter lives here, then the shared
# driver (train-scripts/run_e2e_gate_pretrain.sh → e2e_gate_lib.sh) launches
# `torchrun -m tokengate.e2e_gate.train.train_e2e_gate`.  Core training config has
# no hidden defaults — the driver fails loudly if a value below is deleted.
#
# One-time data prep (see README "End-to-end training"):
#   export PRETRAIN_DATA_ROOT=datasets
#   python -c "import datasets; datasets.load_dataset('BioDEX/BioDEX-Reactions')"
#   python datasets/biodex/biodex_reactions/prepare.py
#
# Launch:        WANDB_PROJECT=<your-project> bash train-scripts/example/run_e2e_gate_biodexr_example.sh
# Single GPU:    NUM_GPUS=1 bash train-scripts/example/run_e2e_gate_biodexr_example.sh
# Smoke run:     MAX_STEPS=50 EVAL_FREQ=25 bash train-scripts/example/run_e2e_gate_biodexr_example.sh
# No logging:    WANDB_MODE=disabled bash train-scripts/example/run_e2e_gate_biodexr_example.sh
set -euo pipefail

# ── W&B logging ───────────────────────────────────────────────────────────────
# Runs log to <WANDB_ENTITY>/<WANDB_PROJECT> (entity = your default unless set).
# Metrics: train loss, eval accuracy per K (val + held-out test groups),
# no-compression / random-selection baselines, gate diagnostics.
: "${WANDB_PROJECT:=token-gate}"      # set to your project (WANDB_MODE=disabled turns logging off)
export WANDB_PROJECT
export WANDB_TAGS="${WANDB_TAGS:-example,biodexr,1p7b}"

# ── Model / launch ────────────────────────────────────────────────────────────
MODEL=Qwen/Qwen3-1.7B
: "${TORCHRUN_MASTER_PORT:=29800}"    # parallel runs on one host need distinct ports
export MODEL TORCHRUN_MASTER_PORT
cd "$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"  # repo root

# ── Data ──────────────────────────────────────────────────────────────────────
PRETRAIN_MANIFEST=iso_biodexr         # pins the (dataset, task) list + data SHA in the run name
: "${PRETRAIN_DATA_ROOT:=datasets}"   # where prepare.py wrote tokenized/{train,val}.pkl
TARGET_BALANCED_MAP="tokengate/datasets/maps/eval_target_train_map_v20_biodexr_only.yaml"
: "${MIN_ARTICLE_TOKENS:=2048}"       # train-article (state) length window: rows outside
: "${MAX_ARTICLE_TOKENS:=16384}"      #   [min,max] are DROPPED, never truncated
EVAL_MIN_ARTICLE_TOKENS=0             # eval window is decoupled (frozen across runs)
EVAL_MAX_ARTICLE_TOKENS=39000         #   and capped to fit the 40,960 context
SHUFFLE_EPOCHS=1                      # epoch = one shuffled pass over the pool (no replacement)

# Fail with instructions if the tokenized data isn't there yet.
_TRAIN_PKL="${PRETRAIN_DATA_ROOT}/biodex/biodex_reactions/tokenized/$(basename "${MODEL}")/train.pkl"
if [ ! -f "${_TRAIN_PKL}" ]; then
    echo "[example] ERROR: no tokenized BioDEX-Reactions data at ${_TRAIN_PKL}" >&2
    echo "[example] Run the one-time data prep first (see the header of this script" >&2
    echo "[example] or README.md → 'Prepare the sample dataset')." >&2
    exit 1
fi

# ── K source (what the gate trains against) ───────────────────────────────────
PHASE2_LOGUNIFORM=1                   # per step: one target length L ~ LogUniform(128, seq/2)
                                      #   + a full-context sentinel (M=2 targets/step)
K_EVAL_VALUES="1000"                  # in-training eval K (median article ~5.6k tokens → keep ~18%)

# ── Schedule ──────────────────────────────────────────────────────────────────
LR_VALUES=1e-4                        # constant LR after warmup (driver: LR_SCHEDULE=constant,
EPOCHS=20                             #   WARMUP_STEPS=1000, OPTIMIZER=muon, WEIGHT_DECAY=0.1)
: "${EVAL_FREQ:=600}"                 # eval + checkpoint every N steps
SWA_FREQ=1                            # stochastic weight averaging ON (driver default is off)
SWA_START=0
SWA_EVAL_FREQ=1
GATE_FULL_EVAL_FREQ=5                 # gate@full-context diagnostic every Nth eval cycle
: "${GRAD_CHECKPOINT:=0}"             # OFF at 1.7B (faster, no OOM); set 1 for large models/K
export GRAD_CHECKPOINT

# ── Adapter (backbone co-adaptation; backbone itself stays frozen) ────────────
LORA_RANK=16                          # factored LoRA on q/v of all layers
LORA_ALPHA=32                         #   (ADAPTER_TYPE=lora, LORA_LAYERS=all are the defaults)
LORA_TARGETS="q_proj v_proj"
LORA_LR=1e-4

# ── Optimizer ─────────────────────────────────────────────────────────────────
MUON_GATE=1                           # scoring-branch matrices on Muon; scalars/readout on aux AdamW
NO_DECAY_SCALARS=1                    # weight decay on the two branch matrices only

# ── Batch / distribution ──────────────────────────────────────────────────────
: "${TRAIN_BATCH_SIZE:=1}"            # per-GPU micro-batch
: "${NUM_GPUS:=8}"                    # DDP world size on this node (NUM_GPUS=1 works)

# ── Eval split / naming ───────────────────────────────────────────────────────
EVAL_SPLIT_MANIFEST="tokengate/datasets/eval_splits/longbench_eval_split_biodexr_iso_v1.json"
                                      # REQUIRED: pins the val ids (scored as cat_biodexr) and the
                                      #   reserved test ids (scored as biodexr_heldout) explicitly
EVAL_GROUP_OVERRIDE="biodexr:biodex_reactions"
EVAL_TEST_FULL_TASKS="biodex_reactions:biodexr_heldout"
MULTITASK_SUFFIX="-example"           # distinct run name + output dir for this config
GATE_LAYERS="1"                       # the layer whose hidden states the gate scores
export GATE_LAYERS

export PRETRAIN_MANIFEST PRETRAIN_DATA_ROOT LORA_RANK
export LORA_ALPHA LORA_TARGETS LORA_LR
export GATE_FULL_EVAL_FREQ TRAIN_BATCH_SIZE
export MUON_GATE NO_DECAY_SCALARS PHASE2_LOGUNIFORM
export EVAL_SPLIT_MANIFEST EVAL_GROUP_OVERRIDE EVAL_TEST_FULL_TASKS
export MIN_ARTICLE_TOKENS MAX_ARTICLE_TOKENS EVAL_MIN_ARTICLE_TOKENS EVAL_MAX_ARTICLE_TOKENS
export SHUFFLE_EPOCHS
export LR_VALUES EPOCHS EVAL_FREQ SWA_FREQ SWA_START TARGET_BALANCED_MAP
export K_EVAL_VALUES SWA_EVAL_FREQ NUM_GPUS MULTITASK_SUFFIX

echo "[example] data : biodex/biodex_reactions via ${PRETRAIN_DATA_ROOT} (manifest ${PRETRAIN_MANIFEST})"
echo "[example] train: gate layer ${GATE_LAYERS}, ${NUM_GPUS} GPU(s), epochs=${EPOCHS}, eval every ${EVAL_FREQ} steps at K=${K_EVAL_VALUES}"
echo "[example] wandb: project=${WANDB_PROJECT} (WANDB_MODE=${WANDB_MODE:-online})"

bash train-scripts/run_e2e_gate_pretrain.sh
