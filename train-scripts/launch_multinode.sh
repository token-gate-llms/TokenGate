#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_multinode.sh — orchestrate a multinode (data-parallel) e2e-gate run.
#
# Fans out run_e2e_gate_pretrain.sh across nodes over ssh with a STATIC torchrun
# rendezvous (one global process group; global RANK/WORLD_SIZE span all nodes).
# The trainer already shards data + all-reduces grads by global rank, so this is
# pure launch plumbing — no Python change.
#
# Usage:
#   NODES="node-a node-b" RUN_SCRIPT=train-scripts/<target>/<lane>.sh \
#       [config env...] bash train-scripts/launch_multinode.sh start
#   NODES="node-a node-b" bash train-scripts/launch_multinode.sh stop
#   bash train-scripts/launch_multinode.sh start    # NODES defaults to this host (single node)
#
#   - The FIRST node in NODES is rank0 / the rendezvous master.
#   - All config env set in the calling shell (EVAL_FREQ, MAX_STEPS, PROFILE,
#     ...) is forwarded to EVERY node.  Extend the forwarded set with
#     MN_FORWARD="VAR1 VAR2 ...".
#   - On a SLURM cluster, prefer the sbatch route instead:
#     train-scripts/example/multinode.sbatch (one task per node; this script is
#     for clusters where you own the nodes directly).
#
# Requires: passwordless ssh between nodes (BatchMode); the repo checkout, the
# python env (PYTHON) and the tokenized data at the SAME paths on every node
# (a shared filesystem does this for free).  If NCCL's interface autodetection
# fails on your fabric, export NCCL_SOCKET_IFNAME=<iface[,iface]> — it is
# forwarded to every node.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

WORKTREE="$(cd "$(dirname "$0")/.." && pwd)"   # repo root (script lives in train-scripts/)
# Overridable so a multinode run can use a PER-DATASET config script (which sets its
# own ~40 knobs) instead of the bare driver + the FORWARD_VARS allowlist. Forwarding
# that many vars by hand is how a knob silently goes missing; running the same config
# script on every node cannot drop one.
RUN_SCRIPT="${RUN_SCRIPT:-train-scripts/run_e2e_gate_pretrain.sh}"
LOGDIR="${WORKTREE}/.opt_logs"
mkdir -p "${LOGDIR}"

ACTION="${1:-start}"
NODES="${NODES:-$(hostname -f)}"
read -r -a _NODE_ARR <<< "${NODES}"
NNODES="${#_NODE_ARR[@]}"

# Worker argv signature for targeted teardown.  Killing the torchrun launcher
# alone leaves the python workers holding VRAM (see memory: parallel-ddp-launch
# gotchas) — so we pkill the worker module path on every node.
WORKER_SIG="tokengate.e2e_gate.train.train_e2e_gate"

# Config vars forwarded to every node (only those currently set are sent).
DEFAULT_FORWARD="PYTHON PRETRAIN_DATA_ROOT OPTIMIZER WEIGHT_DECAY \
ADAPTER_TYPE LORA_RANK LORA_ALPHA LORA_LR LORA_TARGETS LORA_LAYERS \
MUON_GATE NO_DECAY_SCALARS PHASE2_LOGUNIFORM PHASE2_LEN_LIST \
EVAL_FREQ K_EVAL_VALUES MAX_STEPS PROFILE NUM_GPUS SEED \
EPOCHS LR_VALUES SHUFFLE_EPOCHS LENGTH_BUCKETS GRAD_CHECKPOINT \
GRAD_CHECKPOINT_MIN_LEN WANDB_PROJECT MAX_ARTICLE_TOKENS PYTORCH_CUDA_ALLOC_CONF \
NCCL_SOCKET_IFNAME RESUME_FROM EVAL_SPLIT_MANIFEST MIN_ARTICLE_TOKENS EVAL_MIN_ARTICLE_TOKENS EVAL_MAX_ARTICLE_TOKENS \
TRAIN_LOG_FREQ PER_SAMPLE_STATS_FREQ PROFILE_PHASES PHASE2_BUCKET_LISTS PRETRAIN_MANIFEST \
FASTKV_BASELINE FASTKV_TSP_LAYER SWA_START SWA_FREQ SWA_EVAL_FREQ SWA_EVAL_DDP \
TARGET_BALANCED_MAP EVAL_FASTKV_L0_BASELINE \
EVAL_GROUP_OVERRIDE EVAL_TEST_FULL_TASKS MULTITASK_SUFFIX \
WANDB_RUN_ID WANDB_RESUME GATE_FULL_EVAL_FREQ"
FORWARD_VARS="${MN_FORWARD:-${DEFAULT_FORWARD}}"

# Run a command on a node — locally if it is this host, else over ssh.
run_on_node() {  # $1=node  $2=command
    local node="$1" cmd="$2"
    local self_f self_s; self_f="$(hostname -f)"; self_s="$(hostname -s)"
    if [ "${node}" = "${self_f}" ] || [ "${node}" = "${self_s}" ] \
       || [ "${node}" = "${self_f%%.*}" ]; then
        bash -lc "${cmd}"
    else
        ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${node}" "${cmd}"
    fi
}

# ── Teardown ─────────────────────────────────────────────────────────────────
# PID-based (via `ps -C <python names>`), NOT `pkill -f`: a pkill -f pattern matches
# the argv of the very shell running it (the pattern string is on its command
# line), so `pkill -f train_e2e_gate` self-kills the teardown shell before it
# finishes.  `ps -C python3.10` lists only python procs (never this bash), so
# grepping its output and killing the PIDs is safe.
if [ "${ACTION}" = "stop" ]; then
    KILL="ps -C python,python3,python3.10,python3.11,python3.12,python3.13 -o pid=,args= 2>/dev/null | grep -e '${WORKER_SIG}' -e torchrun | awk '{print \$1}' | xargs -r kill -9; echo stopped"
    for node in "${_NODE_ARR[@]}"; do
        echo "[launch_multinode] stopping workers on ${node}"
        run_on_node "${node}" "${KILL}"
    done
    echo "[launch_multinode] stop issued — verify VRAM freed with: nvidia-smi"
    exit 0
fi

if [ "${ACTION}" != "start" ]; then
    echo "usage: bash train-scripts/launch_multinode.sh {start|stop}" >&2
    exit 2
fi

# ── Launch ───────────────────────────────────────────────────────────────────
MASTER_ADDR="${MASTER_ADDR:-${_NODE_ARR[0]}}"
MASTER_PORT="${TORCHRUN_MASTER_PORT:-29500}"

# Build an export prefix from the forward allowlist (set vars only, %q-quoted so
# space-separated lists like FRAC_LIST survive one shell-parse on the far side).
ENV_PREFIX=""
for v in ${FORWARD_VARS}; do
    if [ -n "${!v:-}" ]; then
        ENV_PREFIX+="export ${v}=$(printf '%q' "${!v}"); "
    fi
done

echo "[launch_multinode] NNODES=${NNODES} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[launch_multinode] nodes : ${NODES}"
echo "[launch_multinode] fwd   : $(for v in ${FORWARD_VARS}; do [ -n "${!v:-}" ] && printf '%s=%s ' "$v" "${!v}"; done)"

i=0
for node in "${_NODE_ARR[@]}"; do
    _short="$(echo "${node}" | cut -d. -f1)"
    LOG="${LOGDIR}/mn_${_short}_rank${i}.log"
    REMOTE_CMD="cd '${WORKTREE}' && ${ENV_PREFIX}export NNODES=${NNODES} \
NODE_RANK=${i} MASTER_ADDR='${MASTER_ADDR}' TORCHRUN_MASTER_PORT=${MASTER_PORT}; \
nohup bash ${RUN_SCRIPT} > '${LOG}' 2>&1 < /dev/null & \
echo \"[node ${node}] launched pid \$! → ${LOG}\""
    echo "[launch_multinode] node_rank=${i} → ${node}  (log: ${LOG})"
    run_on_node "${node}" "${REMOTE_CMD}"
    i=$((i + 1))
done

echo "[launch_multinode] all ${NNODES} node(s) launched."
echo "[launch_multinode] follow rank0: tail -f ${LOGDIR}/mn_$(echo "${_NODE_ARR[0]}" | cut -d. -f1)_rank0.log"
echo "[launch_multinode] teardown:    NODES=\"${NODES}\" bash train-scripts/launch_multinode.sh stop"
