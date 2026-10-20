#!/bin/bash
# e2e_gate_lib.sh — shared functions and main driver for e2e gate experiments.
#
# Source this file from a dataset config script after setting all config vars,
# then call run_e2e_gate_main.
#
# Required vars (no defaults): MODEL, DATASET, TASK,
#   EPOCHS, MAX_GEN_TOKENS, TRAIN_BATCH_SIZE,
#   CHECKPOINT_FREQ, EVAL_FREQ, GATE_LAYERS, WARMUP_STEPS, SEED_VALUES,
#   SWA_FREQ, SWA_START, LR_VALUES.

# ── Shared defaults (override in dataset script before sourcing) ───────────────
: "${PYTHON:=python3}"   # point at your env (conda/venv) for real runs
: "${NUM_GPUS:=8}"
: "${WANDB_PROJECT:=fast-context}"
: "${EVAL_GEN_BATCH_SIZE:=1}"
: "${SCALAR_LR_MULT:=1}"
: "${LOG_EXAMPLES:=20}"
: "${GRAD_ACCUM:=1}"
: "${LR_SCHEDULE:=constant}"
: "${PHASE2_LEN_LIST:=}"
: "${PHASE2_BUCKET_LISTS:=}"
: "${PHASE2_LOGUNIFORM:=}"
: "${K_EVAL_VALUES:=}"
: "${LORA_LR:=}"
: "${MAX_ARTICLE_TOKENS:=}"
: "${MAX_ANSWER_TOKENS:=}"
: "${TARGET_BALANCED_MAP:=}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
# Uncomment to get synchronous CUDA error tracebacks (slows training ~10-20%):
# export CUDA_LAUNCH_BLOCKING=1


# ── Canonical output roots (shared across git worktrees) ──────────────────────
# Per-run output trees (gate_step*.pt, ft_before_gate_step*.pt, lora_step*.pt
# + final/SWA gates + wandb) are large (10–100 GB per pretrain run).  Writing
# them under each worktree's e2e_gates/ duplicates the tree
# every time a new worktree is created.  Centralize on the main repo by
# default; override with E2E_GATE_OUTPUT_ROOT to direct a run elsewhere.
_e2e_gate_main_repo_dir() {
    local _gcd
    _gcd=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || {
        echo "[e2e_gate_lib] ERROR: not inside a git repo, cannot derive output root" >&2
        exit 1
    }
    dirname "$_gcd"
}
: "${E2E_GATE_OUTPUT_ROOT:=$(_e2e_gate_main_repo_dir)/e2e_gates}"


# ── Helper: train one gate variant ────────────────────────────────────────────

train_gate() {
    local VARIANT="$1"   # "baseline" or "curriculum"
    local L="$2"
    local LR="$3"        # learning rate for this run
    local DIR_NAME="${VARIANT}"
    local GATE_OUT="${GATE_ROOT}/${DIR_NAME}/L${L}"
    mkdir -p "${GATE_OUT}"

    echo ""
    echo "  --- Training [${DIR_NAME}] at layer L=${L} lr=${LR} ---"
    echo "  Output: ${GATE_OUT}/"

    local EXTRA_ARGS=""

    # "0" means OFF for the :+-forwarded boolean flags: a plain ${VAR:+--flag}
    # treats "0" as set-and-nonempty and passes the flag anyway. Normalize.
    local _bv
    for _bv in GRAD_CHECKPOINT SCALAR_LR_INCLUDE_GATE_SCALE \
               CHECKPOINT_KEEP_ALL_AFTER_WARMUP PHASE2_LOGUNIFORM; do
        if [ "$(eval echo "\${${_bv}:-}")" = "0" ]; then eval "${_bv}="; fi
    done

    if true; then
        if [ -n "${MAX_ARTICLE_TOKENS}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --max-article-tokens ${MAX_ARTICLE_TOKENS}"
        fi
        if [ -n "${MIN_ARTICLE_TOKENS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --min-article-tokens ${MIN_ARTICLE_TOKENS}"
        fi
        if [ -n "${EVAL_MAX_ARTICLE_TOKENS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --eval-max-article-tokens ${EVAL_MAX_ARTICLE_TOKENS}"
        fi
        if [ -n "${EVAL_MIN_ARTICLE_TOKENS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --eval-min-article-tokens ${EVAL_MIN_ARTICLE_TOKENS}"
        fi
        if [ -n "${MAX_ANSWER_TOKENS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --max-answer-tokens ${MAX_ANSWER_TOKENS}"
        fi
        if [ -n "${TARGET_BALANCED_MAP:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --target-balanced-map ${TARGET_BALANCED_MAP}"
        fi
        # Length-bucketed sampling: when set, every step picks a single length
        # bucket and all ranks sample within it.  Eliminates the cross-rank
        # seq-length variance that strands GPUs at the all-reduce barrier in
        # multitask / pretrain runs.
        if [ -n "${LENGTH_BUCKETS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --length-buckets ${LENGTH_BUCKETS}"
        fi
        # Per-length-bucket PHASE_2_LEN_LISTs (requires LENGTH_BUCKETS; one list
        # per bucket, ';'-separated).  Length-bucketed Phase-2 homogenization.
        if [ -n "${PHASE2_LOGUNIFORM:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --phase2-loguniform"
        fi
        if [ -n "${PHASE2_BUCKET_LISTS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --phase2-bucket-lists ${PHASE2_BUCKET_LISTS}"
        fi
        # Per-epoch shuffled iteration (without replacement) instead of the
        # default i.i.d. with-replacement draw.  Recommended for pretrain-mix.
        if [ "${SHUFFLE_EPOCHS:-0}" = "1" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --shuffle-epochs"
        fi
        # Profiling controls (throughput benchmarking).
        # MAX_STEPS caps the train loop for a cheap timing run; PROFILE prints a
        # per-step [profile] timing line and skips eval/checkpoint/SWA.
        if [ -n "${MAX_STEPS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --max-steps ${MAX_STEPS}"
        fi
        if [ "${PROFILE:-0}" = "1" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --profile"
        fi
        # Eval-side cost controls.
        if [ "${EVAL_ALTERNATE_DATASETS:-0}" = "1" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --eval-alternate-datasets"
        fi
        if [ -n "${EVAL_K_STRIDE:-}" ] && [ "${EVAL_K_STRIDE}" -gt 1 ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --eval-k-stride ${EVAL_K_STRIDE}"
        fi
        # torch.compile (in place) of the Phase-2 transformer layers.
        if [ "${COMPILE_PHASE2:-0}" = "1" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --compile-phase2"
        fi
        if [ -n "${TRAIN_ATTN_BACKENDS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --attn-backends ${TRAIN_ATTN_BACKENDS}"
        fi
        if [ "${PACK_PHASE2:-0}" = "1" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --pack-phase2-max-tokens ${PACK_PHASE2_MAX_TOKENS:?PACK_PHASE2=1 needs PACK_PHASE2_MAX_TOKENS}"
        fi
        # Experimental: packed-sequence training (segment-aware FlexAttention).
        if [ "${USE_PACKED_SEQUENCES:-0}" = "1" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --use-packed-sequences"
            if [ -n "${PACKED_MAX_TOTAL_LEN:-}" ]; then
                EXTRA_ARGS="${EXTRA_ARGS} --packed-max-total-len ${PACKED_MAX_TOTAL_LEN}"
            fi
        fi
        # Pre-prepped tokenized data (the only data path): the trainer loads
        # datasets/<ds>/<task>/tokenized/<model>/{train,val}.pkl.
        # Packed, memory-mapped token store (see datasets._lib.pack):
        # ms-scale load, ~10x less RAM, shared across ranks.  Requires the
        # packed/ dirs to exist (build with `python -m
        # tokengate.datasets.pack <manifest>`).
        if [ "${PRETRAIN_PACKED:-0}" = "1" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --pretrain-packed"
        fi
        if [ -n "${PRETRAIN_DATA_ROOT:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --pretrain-data-root ${PRETRAIN_DATA_ROOT}"
        fi
        # Held-out (eval-only) subsets: evaluated but never trained on, loaded
        # into the eval pool only and reported in separate "_heldout" groups.
        if [ -n "${EVAL_ONLY_PAIRS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --pretrain-eval-only-pairs ${EVAL_ONLY_PAIRS}"
        fi
        # Deterministic val/test split: REQUIRED by the trainer whenever eval
        # rows are loaded (the manifest pins each subset's val/test ids — the
        # explicit per-dataset eval/val size mechanism).
        if [ -n "${EVAL_SPLIT_MANIFEST:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --eval-split-manifest ${EVAL_SPLIT_MANIFEST}"
        fi
        # ALSO score the full reserved test partition of these subsets as extra
        # `<task>_test` eval groups (diagnostic runs only).
        if [ -n "${EVAL_TEST_FULL_TASKS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --eval-test-full-tasks ${EVAL_TEST_FULL_TASKS}"
        fi
        # In-distribution eval group(s) sampled from TRAINING rows
        # ("group:N:ds/task,ds/task"); rows remain in training.
        if [ -n "${EVAL_IND_TRAIN_GROUPS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --eval-ind-train-groups ${EVAL_IND_TRAIN_GROUPS}"
        fi
        # Keep trained subsets absent from the split manifest (e.g. the few-shot
        # packs) in the eval pool → in-distribution eval alongside the LongBench split.
        if [ "${EVAL_SPLIT_KEEP_UNTRACKED:-0}" = "1" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --eval-split-keep-untracked"
        fi
        # Pin eval subsets to v10 metric groups regardless of what is trained.
        if [ -n "${EVAL_GROUP_OVERRIDE:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --eval-group-override ${EVAL_GROUP_OVERRIDE}"
        fi
        if [ "${ATTN_SCALE:-1}" = "0" ]; then
            echo "ERROR: ATTN_SCALE=0 is no longer a configuration (the learnable temperature is part of the finalized gate)" >&2
            exit 1
        fi
        EXTRA_ARGS="${EXTRA_ARGS} --scalar-lr-mult ${SCALAR_LR_MULT}"
        # PHASE2_LEN_LIST = fixed list of M TARGET Phase-2 sequence lengths
        # (length-homogenized): per sample K_i = clamp(L_i - prompt_len, 1, n_art),
        # so every sample's Phase-2 ctx = L_i regardless of article length →
        # equal per-rank Phase-2 compute (straggler fix).  Mutually exclusive
        # with PHASE2_LOGUNIFORM / PHASE2_BUCKET_LISTS.
        if [ -n "${PHASE2_LEN_LIST:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --phase2-len-list ${PHASE2_LEN_LIST}"
        fi
        # OPTIMIZER = adamw (default) | muon (Muon for 2D matrices + aux-AdamW;
        # muon.py).  Muon always runs adjust_lr_fn=match_rms_adamw and shares the
        # per-group AdamW LRs (LR_VALUES / LORA_LR) — no separate Muon LR.
        if [ -n "${OPTIMIZER:-}" ] && [ "${OPTIMIZER}" != "adamw" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --optimizer ${OPTIMIZER}"
        fi
        # MUON_GATE=1: the gate's 2D linear maps go to Muon too (requires
        # OPTIMIZER=muon).  NO_DECAY_SCALARS=1: no weight decay on gate_scale /
        # attn_scale / vectors.  Both name-tagged below.
        if [ "${MUON_GATE:-0}" = "1" ]; then
            if [ "${OPTIMIZER:-adamw}" != "muon" ]; then
                echo "MUON_GATE=1 requires OPTIMIZER=muon (got '${OPTIMIZER:-adamw}')" >&2
                exit 1
            fi
            EXTRA_ARGS="${EXTRA_ARGS} --muon-gate"
        fi
        if [ "${NO_DECAY_SCALARS:-0}" = "1" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --no-decay-scalars"
        fi
        # WEIGHT_DECAY = decoupled weight decay for EVERY optimizer param group
        # (AdamW, and Muon + aux-AdamW under OPTIMIZER=muon).  Unset → trainer
        # legacy defaults (AdamW 1e-4, Muon path 0.0).  v11+ pretrain sets 0.1.
        if [ -n "${WEIGHT_DECAY:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --weight-decay ${WEIGHT_DECAY}"
        fi
        # LONG_CTX_FACTOR = YaRN RoPE scaling factor.  1.0 (default) = off.
        # Set to 4.0 for 128k ctx.
        if [ -n "${LONG_CTX_FACTOR:-}" ] && [ "${LONG_CTX_FACTOR}" != "1.0" ] && [ "${LONG_CTX_FACTOR}" != "1" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --long-ctx-factor ${LONG_CTX_FACTOR}"
        fi
        # Adapter at the gate layer: factored LoRA (LORA_RANK > 0) or additive
        # delta (ADAPTER_TYPE=delta; rank/alpha-free ΔW on attn q/k/v/o + MLP
        # gate/up/down; Muon-native).  Mutually exclusive.
        if [ -n "${ADAPTER_TYPE:-}" ] && [ "${ADAPTER_TYPE}" != "lora" ]; then
            if [ "${LORA_RANK:-0}" -gt 0 ]; then
                echo "[e2e_gate_lib] ERROR: ADAPTER_TYPE=${ADAPTER_TYPE} is mutually exclusive with LORA_RANK>0." >&2
                exit 1
            fi
            EXTRA_ARGS="${EXTRA_ARGS} --adapter-type ${ADAPTER_TYPE}"
            if [ -n "${LORA_LR:-}" ]; then
                EXTRA_ARGS="${EXTRA_ARGS} --lora-lr ${LORA_LR}"
            fi
        elif [ "${LORA_RANK:-0}" -gt 0 ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --lora-rank ${LORA_RANK} --lora-alpha ${LORA_ALPHA}"
            if [ -n "${LORA_LR:-}" ]; then
                EXTRA_ARGS="${EXTRA_ARGS} --lora-lr ${LORA_LR}"
            fi
            # LORA_TARGETS: space-separated module list (bare "q_proj" legacy form or
            # dotted "mlp.gate_proj", resolved per apply_lora_to_layers). Only
            # forwarded when set; the trainer's default is q/k/v/o. Runs that set it
            # get a _lt= suffix in RUN_NAME/GATE_ROOT (see the naming-collision pitfall).
            if [ -n "${LORA_TARGETS:-}" ]; then
                EXTRA_ARGS="${EXTRA_ARGS} --lora-targets ${LORA_TARGETS}"
            fi
            if [ -n "${LORA_LAYERS:-}" ]; then
                EXTRA_ARGS="${EXTRA_ARGS} --lora-layers ${LORA_LAYERS}"
            fi
            if [ "${LORA_MUON:-0}" = "1" ]; then
                EXTRA_ARGS="${EXTRA_ARGS} --lora-muon"
            fi
        fi
        # Random-K ablation: freeze gate, replace top-K with uniform random
        # K-subset per sample/step.  Lets a downstream LoRA / FT-below adapt to
        # a random subset, isolating gate selection from layer-0 adaptation.
        if [ "${RANDOM_TOPK:-0}" = "1" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --random-topk"
        fi
        # Adaptive-K (threshold + sparsity penalty).
        # GATE_SELECTION defaults to topk (legacy behaviour preserved).  In
        # threshold mode TAU controls the inference cutoff and SPARSITY_LAMBDA
        # the training penalty pulling sigmoid(b) down.
        if [ -n "${GATE_SELECTION:-}" ] && [ "${GATE_SELECTION}" != "topk" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --gate-selection ${GATE_SELECTION}"
        fi
        if [ -n "${TAU:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --tau ${TAU}"
        fi
        if [ -n "${SPARSITY_LAMBDA:-}" ] && [ "${SPARSITY_LAMBDA}" != "0" ] && [ "${SPARSITY_LAMBDA}" != "0.0" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --sparsity-lambda ${SPARSITY_LAMBDA}"
        fi
        # --sparsity-mode + --sparsity-sharpness only matter when the penalty is
        # active; pass them through whenever set so per-dataset configs can pick
        # soft_count without editing the lib.
        if [ -n "${SPARSITY_MODE:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --sparsity-mode ${SPARSITY_MODE}"
        fi
        if [ -n "${SPARSITY_SHARPNESS:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --sparsity-sharpness ${SPARSITY_SHARPNESS}"
        fi
        if [ -n "${TAU_EVAL_VALUES:-}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --tau-eval ${TAU_EVAL_VALUES}"
        fi
    fi
    if [ -n "${WANDB_PROJECT}" ]; then
        local MODEL_NAME RUN_NAME
        MODEL_NAME="$(basename "${MODEL}")"
        if true; then
            # RUN_PREFIX (default empty) tags a run's W&B name — e.g.
            # "retime-train-" for a throwaway sec/step timing run; pair with an
            # E2E_GATE_OUTPUT_ROOT override so its checkpoints land off the real tree.
            RUN_NAME="${RUN_PREFIX:-}${MODEL_NAME}_${MULTITASK_NAME:-${DATASET}_${TASK}}_${DIR_NAME}_L${L}_lr=${LR}_randseed=${SEED}_num-epochs=${EPOCHS}"
            if [ -n "${TRAIN_ATTN_BACKENDS:-}" ] && [ "${TRAIN_ATTN_BACKENDS}" != "flex" ]; then
                RUN_NAME="${RUN_NAME}_attn=${TRAIN_ATTN_BACKENDS}"
            fi
            if [ "${MUON_GATE:-0}" = "1" ]; then
                RUN_NAME="${RUN_NAME}_mg"
            fi
            if [ "${NO_DECAY_SCALARS:-0}" = "1" ]; then
                RUN_NAME="${RUN_NAME}_nds"
            fi
            if [ -n "${PHASE2_LOGUNIFORM:-}" ]; then
                RUN_NAME="${RUN_NAME}_p2lu"
            fi
            if [ -n "${SCALAR_LR_INCLUDE_GATE_SCALE:-}" ]; then
                RUN_NAME="${RUN_NAME}_gsl"
            fi
            if [ -n "${LORA_LAYERS:-}" ]; then
                RUN_NAME="${RUN_NAME}_ll=$(echo "${LORA_LAYERS}" | tr ',' '-')"
            fi
            if [ -n "${LORA_TARGETS:-}" ]; then
                RUN_NAME="${RUN_NAME}_lt=$(echo "${LORA_TARGETS}" | sed 's/_proj//g' | tr -d ' ')"
            fi
            if [ "${LORA_MUON:-0}" = "1" ]; then
                RUN_NAME="${RUN_NAME}_lmuon"
            fi
            if [ -n "${PHASE2_LEN_LIST:-}" ]; then
                RUN_NAME="${RUN_NAME}_p2len$(echo ${PHASE2_LEN_LIST} | wc -w | tr -d ' ')-$(printf '%s' "${PHASE2_LEN_LIST}" | md5sum | cut -c1-6)"
            fi
            if [ -n "${OPTIMIZER:-}" ] && [ "${OPTIMIZER}" != "adamw" ]; then
                RUN_NAME="${RUN_NAME}_opt=${OPTIMIZER}"
            fi
            if [ -n "${WEIGHT_DECAY:-}" ]; then
                RUN_NAME="${RUN_NAME}_wd=${WEIGHT_DECAY}"
            fi
            if [ -n "${ADAPTER_TYPE:-}" ] && [ "${ADAPTER_TYPE}" != "lora" ]; then
                RUN_NAME="${RUN_NAME}_ad=${ADAPTER_TYPE}"
            fi
            if [ "${LORA_RANK:-0}" -gt 0 ]; then
                RUN_NAME="${RUN_NAME}_lora_lin_r${LORA_RANK}_a${LORA_ALPHA}"
                if [ -n "${LORA_TARGETS:-}" ]; then
                    RUN_NAME="${RUN_NAME}_lt=$(echo "${LORA_TARGETS}" | wc -w | tr -d ' ')mod"
                fi
            fi
            if [ -n "${LENGTH_BUCKETS:-}" ]; then
                _LB_TAG="lb$(echo "${LENGTH_BUCKETS}" | tr ',' ' ' | wc -w | tr -d ' ')"  # compact: bucket count (255 NAME_MAX)
                RUN_NAME="${RUN_NAME}_${_LB_TAG}"
            fi
            if [ "${SHUFFLE_EPOCHS:-0}" = "1" ]; then
                RUN_NAME="${RUN_NAME}_shuf"
            fi
            if [ "${RANDOM_TOPK:-0}" = "1" ]; then
                RUN_NAME="${RUN_NAME}_randk"
            fi
            # Threshold mode: distinguish runs by selection scheme + tau + lambda.
            # Short-tag the sparsity mode to stay under the 255-char NFS leaf cap
            # for the GATE_ROOT (same tag is used in the matching dir name below).
            # NFS leaf cap is 255 chars; pretrain prefix is already ~190 so
            # any tag added here must be compact.  scf = soft_count_frac, sc =
            # soft_count, sm = soft_mean.  Old runs created with the longer
            # tags ("scfrac", "soft_mean") keep their existing dir names.
            case "${SPARSITY_MODE:-soft_mean}" in
                soft_count_frac_l2) _SMODE_TAG=scfl2 ;;
                soft_count_frac)    _SMODE_TAG=scf ;;
                soft_count)         _SMODE_TAG=sc ;;
                soft_mean)          _SMODE_TAG=sm ;;
                *)                  _SMODE_TAG="${SPARSITY_MODE}" ;;
            esac
            # Sharpness only appears when explicitly set AND non-default (1).
            if [ -n "${SPARSITY_SHARPNESS:-}" ] && [ "${SPARSITY_SHARPNESS}" != "1" ]; then
                _SHARP_TAG="_sh=${SPARSITY_SHARPNESS}"
            else
                _SHARP_TAG=""
            fi
            if [ -n "${GATE_SELECTION:-}" ] && [ "${GATE_SELECTION}" != "topk" ]; then
                RUN_NAME="${RUN_NAME}_sel=${GATE_SELECTION}_tau=${TAU:-0.5}_lam=${SPARSITY_LAMBDA:-0}_sm=${_SMODE_TAG}${_SHARP_TAG}"
            fi
        fi
        EXTRA_ARGS="${EXTRA_ARGS} --wandb-project ${WANDB_PROJECT} --wandb-run-name ${RUN_NAME}"
    fi
    # Inject specific adapter weights (gate proj + delta/LoRA ΔW) WITHOUT restoring
    # optimizer/step state — for eval-only checkpoint diagnostics (pairs with MAX_STEPS).
    if [ -n "${RESUME_GATE:-}" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --resume-gate ${RESUME_GATE}"
    fi
    if [ -n "${RESUME_LORA:-}" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --resume-lora ${RESUME_LORA}"
    fi

    local _TRAIN_MODULE="tokengate.e2e_gate.train.train_e2e_gate"

    local _n_gpus="${_TRAIN_GPUS:-${NUM_GPUS}}"
    # Resolve torchrun from the same env as ${PYTHON} so we don't pick up a
    # torchrun from a different conda env via PATH.
    local _TORCHRUN
    _TORCHRUN="$(dirname "${PYTHON}")/torchrun"
    if [ ! -x "${_TORCHRUN}" ]; then
        echo "[e2e_gate_lib] ERROR: torchrun not found at ${_TORCHRUN}.  " \
             "Install torchrun in the same env as ${PYTHON}." >&2
        exit 1
    fi
    # Torchrun rendezvous port.  Default 29500 collides when two runs share a
    # host; per-dataset configs can override TORCHRUN_MASTER_PORT to launch
    # parallel runs (e.g. two 4-GPU ablations on the same 8-GPU node).
    local _MASTER_PORT="${TORCHRUN_MASTER_PORT:-29500}"
    # Distributed launch args.  NNODES>1 (exported by launch_multinode.sh, a
    # SLURM task, or by hand on each node) switches torchrun to a static
    # multi-node rendezvous: every node passes the SAME --nnodes/--master_addr/
    # --master_port and its own --node_rank, and the one global process group
    # spans all nodes (the trainer already shards data and all-reduces grads by
    # global rank).  NNODES=1 is the plain single-node invocation.
    local _NNODES="${NNODES:-1}"
    local -a _DIST_ARGS
    if [ "${_NNODES}" -gt 1 ]; then
        _DIST_ARGS=(
            --nnodes="${_NNODES}"
            --node_rank="${NODE_RANK:?NNODES>1 requires NODE_RANK (0..NNODES-1, per node)}"
            --nproc_per_node="${_n_gpus}"
            --master_addr="${MASTER_ADDR:?NNODES>1 requires MASTER_ADDR (rank-0 node)}"
            --master_port="${_MASTER_PORT}"
        )
        echo "[e2e_gate_lib] multinode DP (static rendezvous): NNODES=${_NNODES}" \
             "NODE_RANK=${NODE_RANK} master=${MASTER_ADDR}:${_MASTER_PORT}" >&2
    else
        _DIST_ARGS=(
            --nproc_per_node="${_n_gpus}"
            --master_port="${_MASTER_PORT}"
        )
    fi
    CUBLAS_WORKSPACE_CONFIG=:4096:8 "${_TORCHRUN}" \
        "${_DIST_ARGS[@]}" \
        -m "${_TRAIN_MODULE}" \
        --base-model "${MODEL}" \
        --dataset "${DATASET}" \
        --task "${TASK}" \
        --gate-layer "${L}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --seed "${SEED}" \
        --batch-size "${TRAIN_BATCH_SIZE}" \
        --grad-accum "${GRAD_ACCUM}" \
        --checkpoint-freq "${CHECKPOINT_FREQ}" \
        ${CHECKPOINT_KEEP_LAST:+--checkpoint-keep-last ${CHECKPOINT_KEEP_LAST}} \
        ${CHECKPOINT_KEEP_BEST:+--checkpoint-keep-best ${CHECKPOINT_KEEP_BEST}} \
        ${CHECKPOINT_BEST_GROUP:+--checkpoint-best-group ${CHECKPOINT_BEST_GROUP}} \
        ${SAVE_SWA_STEP_CHECKPOINTS:+--save-swa-step-checkpoints} \
        ${GRAD_CHECKPOINT:+--grad-checkpoint} \
        --eval-freq "${EVAL_FREQ}" \
        --log-examples "${LOG_EXAMPLES}" \
        --max-gen-tokens "${MAX_GEN_TOKENS}" \
        --eval-gen-batch-size "${EVAL_GEN_BATCH_SIZE:-1}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --lr-schedule "${LR_SCHEDULE}" \
        --swa-freq "${SWA_FREQ}" \
        --swa-start "${SWA_START}" \
        ${SWA_EVAL_FREQ:+--swa-eval-freq ${SWA_EVAL_FREQ}} \
        ${GATE_FULL_EVAL_FREQ:+--gate-full-eval-freq ${GATE_FULL_EVAL_FREQ}} \
        ${CHECKPOINT_KEEP_ALL_AFTER_WARMUP:+--checkpoint-keep-all-after-warmup} \
        ${SCALAR_LR_INCLUDE_GATE_SCALE:+--scalar-lr-include-gate-scale} \
        $([ "${SWA_EVAL_DDP:-1}" = "0" ] && echo "--no-swa-eval-ddp") \
        --output-dir "${GATE_OUT}" \
        ${K_EVAL_VALUES:+--k-eval ${K_EVAL_VALUES}} \
        ${RESUME_FROM:+--resume-from "${RESUME_FROM}"} \
        ${EVAL_ON_RESUME:+--eval-on-resume} \
        ${EXTRA_ARGS}

    echo "  Gate saved to ${GATE_OUT}/gate_final.pt"
}



# ── Main driver ───────────────────────────────────────────────────────────────

run_e2e_gate_main() {
    # MUON_LR / MUON_MATCH_RMS_ADAMW were removed: torch.optim.Muon always runs
    # with adjust_lr_fn=match_rms_adamw, so Muon groups share the per-group AdamW
    # LRs (LR_VALUES / LORA_LR) — there is no separate Muon LR.  Fail loudly on
    # stale configs rather than silently ignoring them.
    if [ -n "${MUON_LR:-}" ] || [ -n "${MUON_MATCH_RMS_ADAMW:-}" ]; then
        echo "[e2e_gate_lib] ERROR: MUON_LR / MUON_MATCH_RMS_ADAMW were removed." >&2
        echo "  Muon now always uses adjust_lr_fn=match_rms_adamw and shares the" >&2
        echo "  per-group LRs (LR_VALUES / LORA_LR).  Unset these variables." >&2
        exit 1
    fi
    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  E2E Gate Experiment"
    echo "  Model        : ${MODEL}"
    echo "  Dataset      : ${DATASET}   Task: ${TASK}"
    echo "  Epochs       : ${EPOCHS}   Seeds: ${SEED_VALUES}"
    echo "  GPUs         : ${NUM_GPUS}  (torchrun DDP, eff. batch = $((TRAIN_BATCH_SIZE * NUM_GPUS)))"
    echo "  Train BS     : ${TRAIN_BATCH_SIZE} per GPU (batched with FlexAttention)"
    echo "  K eval       : ${K_EVAL_VALUES:?K_EVAL_VALUES is required}"
    if [ -n "${GATE_SELECTION:-}" ] && [ "${GATE_SELECTION}" != "topk" ]; then
        echo "  Selection    : ${GATE_SELECTION}   tau=${TAU:-0.5}   sparsity_lambda=${SPARSITY_LAMBDA:-0}   sparsity_mode=${SPARSITY_MODE:-soft_mean}${SPARSITY_SHARPNESS:+   sparsity_sharpness=${SPARSITY_SHARPNESS}}"
        echo "  Tau eval     : ${TAU_EVAL_VALUES:-${TAU:-0.5}}"
    fi
    echo "  Article cap  : min=${MIN_ARTICLE_TOKENS:-none}  max=${MAX_ARTICLE_TOKENS:-none} tokens"
    echo "  Answer cap   : max=${MAX_ANSWER_TOKENS:-none} tokens (drop rows with longer targets)"
    echo "  LR sweep     : ${LR_VALUES}"
    echo "  Layers       : ${GATE_LAYERS}"
    echo "  LR warmup    : ${WARMUP_STEPS} steps  schedule=${LR_SCHEDULE}  swa_freq=${SWA_FREQ}  swa_start=${SWA_START}"
    echo "  Train mode   : sft"
    echo "══════════════════════════════════════════════════════════════════════"

    local _TRAIN_GPUS="${NUM_GPUS}"

    # ── Step 1: Train ─────────────────────────────────────────────────────────

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step 1: train_e2e_gate"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local GATE_ROOT SEED LR _KSET_TAG
    for LR in ${LR_VALUES}; do
        for SEED in ${SEED_VALUES}; do
            # Dedupe redundant DATASET==TASK in the filesystem dir tag so the
            # leaf dir stays under the 255-char limit for long configs (e.g.
            # mkm=16 + _opt=muon_mlr suffixes overflow otherwise). Affects the
            # GATE_ROOT path only; the W&B RUN_NAME is left unchanged.
            local _DS_TAG="${MULTITASK_NAME:-${DATASET}_${TASK}}"
            if [ -z "${MULTITASK_NAME:-}" ] && [ "${DATASET}" = "${TASK}" ]; then
                _DS_TAG="${DATASET}"
            fi
            GATE_ROOT="${E2E_GATE_OUTPUT_ROOT}/${MODEL}/${_DS_TAG}_lr=${LR}_randseed=${SEED}_num-epochs=${EPOCHS}"
            # Keep GATE_ROOT and RUN_NAME suffixes in sync or runs silently share
            # an output tree.
            if [ -n "${TRAIN_ATTN_BACKENDS:-}" ] && [ "${TRAIN_ATTN_BACKENDS}" != "flex" ]; then
                GATE_ROOT="${GATE_ROOT}_attn=${TRAIN_ATTN_BACKENDS}"
            fi
            if [ -n "${PHASE2_LOGUNIFORM:-}" ]; then
                GATE_ROOT="${GATE_ROOT}_p2lu"
            fi
            if [ -n "${SCALAR_LR_INCLUDE_GATE_SCALE:-}" ]; then
                GATE_ROOT="${GATE_ROOT}_gsl"
            fi
            if [ -n "${LORA_LAYERS:-}" ]; then
                GATE_ROOT="${GATE_ROOT}_ll=$(echo "${LORA_LAYERS}" | tr ',' '-')"
            fi
            if [ -n "${LORA_TARGETS:-}" ]; then
                GATE_ROOT="${GATE_ROOT}_lt=$(echo "${LORA_TARGETS}" | sed 's/_proj//g' | tr -d ' ')"
            fi
            if [ "${LORA_MUON:-0}" = "1" ]; then
                GATE_ROOT="${GATE_ROOT}_lmuon"
            fi
            if [ -n "${PHASE2_LEN_LIST:-}" ]; then
                GATE_ROOT="${GATE_ROOT}_p2len$(echo ${PHASE2_LEN_LIST} | wc -w | tr -d ' ')-$(printf '%s' "${PHASE2_LEN_LIST}" | md5sum | cut -c1-6)"
            fi
            if [ -n "${OPTIMIZER:-}" ] && [ "${OPTIMIZER}" != "adamw" ]; then
                GATE_ROOT="${GATE_ROOT}_opt=${OPTIMIZER}"
            fi
            if [ -n "${WEIGHT_DECAY:-}" ]; then
                GATE_ROOT="${GATE_ROOT}_wd=${WEIGHT_DECAY}"
            fi
            # Adapter tag: _ad=delta for the additive adapter, or the LoRA
            # rank/alpha suffix when LORA_RANK > 0 (mutually exclusive).
            if [ -n "${ADAPTER_TYPE:-}" ] && [ "${ADAPTER_TYPE}" != "lora" ]; then
                GATE_ROOT="${GATE_ROOT}_ad=${ADAPTER_TYPE}"
            fi
            if [ "${LORA_RANK:-0}" -gt 0 ]; then
                GATE_ROOT="${GATE_ROOT}_lora_lin_r${LORA_RANK}_a${LORA_ALPHA}"
                if [ -n "${LORA_TARGETS:-}" ]; then
                    GATE_ROOT="${GATE_ROOT}_lt=$(echo "${LORA_TARGETS}" | wc -w | tr -d ' ')mod"
                fi
            fi
            if [ "${MUON_GATE:-0}" = "1" ]; then
                GATE_ROOT="${GATE_ROOT}_mg"
            fi
            if [ "${NO_DECAY_SCALARS:-0}" = "1" ]; then
                GATE_ROOT="${GATE_ROOT}_nds"
            fi
            if [ -n "${LENGTH_BUCKETS:-}" ]; then
                _LB_TAG="lb$(echo "${LENGTH_BUCKETS}" | tr ',' ' ' | wc -w | tr -d ' ')"  # compact: bucket count (255 NAME_MAX)
                GATE_ROOT="${GATE_ROOT}_${_LB_TAG}"
            fi
            if [ "${SHUFFLE_EPOCHS:-0}" = "1" ]; then
                GATE_ROOT="${GATE_ROOT}_shuf"
            fi
            if [ "${RANDOM_TOPK:-0}" = "1" ]; then
                GATE_ROOT="${GATE_ROOT}_randk"
            fi
            # Adaptive-K threshold mode.  Always append when active so
            # threshold runs never collide with legacy top-K runs.
            case "${SPARSITY_MODE:-soft_mean}" in
                soft_count_frac_l2) _SMODE_TAG=scfl2 ;;
                soft_count_frac)    _SMODE_TAG=scf ;;
                soft_count)         _SMODE_TAG=sc ;;
                soft_mean)          _SMODE_TAG=sm ;;
                *)                  _SMODE_TAG="${SPARSITY_MODE}" ;;
            esac
            if [ -n "${SPARSITY_SHARPNESS:-}" ] && [ "${SPARSITY_SHARPNESS}" != "1" ]; then
                _SHARP_TAG="_sh=${SPARSITY_SHARPNESS}"
            else
                _SHARP_TAG=""
            fi
            if [ -n "${GATE_SELECTION:-}" ] && [ "${GATE_SELECTION}" != "topk" ]; then
                GATE_ROOT="${GATE_ROOT}_thr_tau=${TAU:-0.5}_lam=${SPARSITY_LAMBDA:-0}_sm=${_SMODE_TAG}${_SHARP_TAG}"
            fi
            # NAME_MAX (255) guard: long pretrain configs (long manifest tag +
            # mkm + opt + lora suffixes) can overflow the 255-char filesystem
            # limit on the GATE_ROOT leaf.  Truncate the leaf and append a
            # deterministic 8-char hash of the full leaf so distinct configs
            # never collide AND resume reconstructs the same path.  Only the
            # on-disk dir is shortened; the W&B RUN_NAME keeps the full name.
            local _gr_leaf _gr_parent _gr_hash
            _gr_leaf="$(basename "${GATE_ROOT}")"
            if [ "${#_gr_leaf}" -gt 255 ]; then
                _gr_parent="$(dirname "${GATE_ROOT}")"
                _gr_hash="$(printf '%s' "${_gr_leaf}" | md5sum | cut -c1-8)"
                GATE_ROOT="${_gr_parent}/${_gr_leaf:0:240}_${_gr_hash}"
                echo "  [name] GATE_ROOT leaf >255 chars; truncated + hashed → $(basename "${GATE_ROOT}")"
            fi
            mkdir -p "${GATE_ROOT}"
            echo ""
            echo "  [ LR=${LR}  seed=${SEED}  warmup=${WARMUP_STEPS}  lr_schedule=${LR_SCHEDULE} ]"
            for L in ${GATE_LAYERS}; do
                train_gate baseline "${L}" "${LR}"
            done
        done
    done

    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  E2E gate runs complete."
    echo "  Gates : ${E2E_GATE_OUTPUT_ROOT}/${MODEL}/"
    echo "══════════════════════════════════════════════════════════════════════"
}
