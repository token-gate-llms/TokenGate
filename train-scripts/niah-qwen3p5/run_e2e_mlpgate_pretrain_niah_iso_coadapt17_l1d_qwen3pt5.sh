#!/bin/bash
# NIAH iso coadapt17 l1d (DeltaNet gate layer), Qwen3.5-2B hybrid: token-gate training on the Needle-in-a-Haystack (Kamradt) pool — gate at layer 1, Qwen3.5-2B (hybrid).
# Isolated single-dataset protocol: the manifest holds only the target pool; eval is the
# frozen iso split (val scored as cat_niah, reserved test rows as niah_heldout).
# Qwen3.5-2B is a 3:1 DeltaNet/full-attention hybrid, so gate-layer indices are
# constrained to its layer map: L3 is the earliest full-attention layer
# ({3,7,11,15,19,23}); the l1d/l2d lanes gate at DeltaNet layers — the scoring
# branch reads that layer's hidden state, the bias lands on the full-attention
# layers above it.
#
# Launch:  bash train-scripts/niah-qwen3p5/run_e2e_mlpgate_pretrain_niah_iso_coadapt17_l1d_qwen3pt5.sh
set -euo pipefail

MODEL=Qwen/Qwen3.5-2B
: "${TORCHRUN_MASTER_PORT:=29911}"   # env-overridable (parallel runs need distinct ports)
export MODEL TORCHRUN_MASTER_PORT
cd "$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"  # repo root: script lives in a subdir

# ── Data ──────────────────────────────────────────────────────────────────────
PRETRAIN_MANIFEST=iso_niah
TARGET_BALANCED_MAP="tokengate/datasets/maps/eval_target_train_map_v17_niah_only.yaml"   # single-target sampling map (weight 1.0 on the pool)
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
: "${EVAL_SAMPLES_PER_GROUP:=100}"
SWA_FREQ=1                            # SWA on (driver default is off)
SWA_START=0
SWA_EVAL_FREQ=1
: "${GATE_FULL_EVAL_FREQ:=5}"       # gate@full-context diagnostic every Nth eval cycle
: "${GRAD_CHECKPOINT:=0}"             # OFF: faster and no OOM on the hybrid (retime probe); numerically exact either way
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
K_EVAL_VALUES="2000"   # 16k tier: K2000 keeps ~12.5% of article tokens
EVAL_SPLIT_MANIFEST="tokengate/datasets/eval_splits/longbench_eval_split_niah_iso_v1.json"
EVAL_GROUP_OVERRIDE="niah:niah_kamradt"
EVAL_TEST_FULL_TASKS="niah_kamradt:niah_heldout"
MULTITASK_SUFFIX="-mlpgate-niah-iso-coadapt17l1d-rqslu"
GATE_LAYERS="1"             # DeltaNet layer; bias lands on the 6 full-attn layers above
export GATE_LAYERS
export WANDB_TAGS="coadapt,coadapt17l1d,niah,layersweep,rmsnorm,learnS,lusamp,2b,qwen3p5,hybrid"
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

echo "[niah-iso] data: niah_kamradt/niah_kamradt via ${TARGET_BALANCED_MAP##*/}"
echo "[niah-iso] eval: val 16 as cat_niah + test 184 as niah_heldout, every ${EVAL_FREQ} steps, K=${K_EVAL_VALUES}, gate layer ${GATE_LAYERS}"

bash train-scripts/run_e2e_gate_pretrain.sh
