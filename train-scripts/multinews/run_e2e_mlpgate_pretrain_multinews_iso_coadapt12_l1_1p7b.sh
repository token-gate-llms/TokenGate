#!/bin/bash
# MultiNews iso coadapt12 l1: token-gate training on the MultiNews pool (longbench_multi_news) — gate at layer 1, Qwen3-1.7B.
# Isolated single-dataset protocol: the manifest holds only the target pool; eval is the
# frozen iso split (val scored as cat_summarization, reserved test rows as summarization_heldout).
#
# Launch:  bash train-scripts/multinews/run_e2e_mlpgate_pretrain_multinews_iso_coadapt12_l1_1p7b.sh
set -euo pipefail

MODEL=Qwen/Qwen3-1.7B
: "${TORCHRUN_MASTER_PORT:=29711}"   # env-overridable (parallel runs need distinct ports)
export MODEL TORCHRUN_MASTER_PORT
cd "$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"  # repo root: script lives in a subdir

# ── Data ──────────────────────────────────────────────────────────────────────
PRETRAIN_MANIFEST=iso_multinews
TARGET_BALANCED_MAP="tokengate/datasets/maps/eval_target_train_map_v16_multinews_only.yaml"   # single-target sampling map (weight 1.0 on the pool)
: "${MIN_ARTICLE_TOKENS:=2048}"       # train-article window
: "${MAX_ARTICLE_TOKENS:=16384}"
EVAL_MIN_ARTICLE_TOKENS=0
EVAL_MAX_ARTICLE_TOKENS=39000         # eval protocol cap (fits the 40,960 context)
SHUFFLE_EPOCHS=1                      # map-consuming shuffled-epoch sampler
PHASE2_LOGUNIFORM=1                   # per-step phase-2 target: LogUniform(128, seq/2) + full-context sentinel

# ── Schedule ──────────────────────────────────────────────────────────────────
LR_VALUES=1e-4
EPOCHS=20
: "${EVAL_FREQ:=600}"                 # env-overridable (checkpoint freq follows)
SWA_FREQ=1                            # SWA on (driver default is off)
SWA_START=0
SWA_EVAL_FREQ=1
GATE_FULL_EVAL_FREQ=5                 # gate@full-context diagnostic every Nth eval cycle
: "${GRAD_CHECKPOINT:=0}"             # OFF at 1.7B: ~1.37x faster and no OOM; numerically exact either way
export GRAD_CHECKPOINT

# ── Adapter (backbone co-adaptation) ─────────────────────────────────────────
LORA_RANK=16
LORA_ALPHA=32
LORA_TARGETS="q_proj v_proj"
LORA_LR=1e-4

# ── Optimizer ─────────────────────────────────────────────────────────────────
MUON_GATE=1                           # scoring-branch matrices on Muon; scalars + readout on aux AdamW
NO_DECAY_SCALARS=1                    # weight decay on the two branch matrices only

# ── Batch / distribution ──────────────────────────────────────────────────────
: "${TRAIN_BATCH_SIZE:=1}"            # per-GPU micro-batch (GRAD_ACCUM is the effective-batch lever)
: "${NUM_GPUS:=8}"                    # per node; env-overridable

# ── Eval split / naming ───────────────────────────────────────────────────────
K_EVAL_VALUES="1000"
EVAL_SPLIT_MANIFEST="tokengate/datasets/eval_splits/longbench_eval_split_multinews_iso_v1.json"
EVAL_GROUP_OVERRIDE="summarization:longbench_multi_news"
EVAL_TEST_FULL_TASKS="longbench_multi_news:summarization_heldout"
MULTITASK_SUFFIX="-mlpgate-multinews-iso-coadapt12l1-rqslu"
GATE_LAYERS="1"
export GATE_LAYERS
export WANDB_TAGS="coadapt,coadapt12l1,multinews,layersweep,rmsnorm,learnS,lusamp,1p7b"
CHECKPOINT_KEEP_ALL_AFTER_WARMUP=1    # keep every gate/adapter ckpt past warmup (train_state stays keep-last-3)

export PRETRAIN_MANIFEST LORA_RANK
export LORA_ALPHA LORA_TARGETS LORA_LR
export GATE_FULL_EVAL_FREQ TRAIN_BATCH_SIZE
export MUON_GATE NO_DECAY_SCALARS PHASE2_LOGUNIFORM CHECKPOINT_KEEP_ALL_AFTER_WARMUP
export EVAL_SPLIT_MANIFEST EVAL_GROUP_OVERRIDE EVAL_TEST_FULL_TASKS
export MIN_ARTICLE_TOKENS MAX_ARTICLE_TOKENS EVAL_MIN_ARTICLE_TOKENS EVAL_MAX_ARTICLE_TOKENS
export SHUFFLE_EPOCHS
export LR_VALUES EPOCHS EVAL_FREQ SWA_FREQ SWA_START TARGET_BALANCED_MAP
export K_EVAL_VALUES SWA_EVAL_FREQ NUM_GPUS MULTITASK_SUFFIX

echo "[multinews-iso] data: longbench-datasets/longbench_multi_news via ${TARGET_BALANCED_MAP##*/}"
echo "[multinews-iso] eval: val 16 as cat_summarization + test 184 as summarization_heldout, every ${EVAL_FREQ} steps, K=${K_EVAL_VALUES}, gate layer ${GATE_LAYERS}"

bash train-scripts/run_e2e_gate_pretrain.sh
