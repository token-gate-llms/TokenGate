#!/bin/bash
# run_e2e_gate_pretrain.sh — pretraining-mixture training over accepted datasets.
#
# Loads pre-prepped tokenized .pkl files from
#   datasets/<dataset>/<task>/tokenized/Qwen3-8B/{train,val}.pkl
# (written by per-dataset prepare.py scripts) — the trainer's only data path.
#
# Sampling: per-epoch shuffled permutation over the pool (SHUFFLE_EPOCHS).
# Eval: per-dataset val.pkl, pinned to each subset's EVAL_SPLIT_MANIFEST
# `val` ids (the explicit per-dataset eval/val size mechanism).
# Held-out OOD test (post-training): the LongBench OOD subsets (not in the
# trained pairs) — evaluated via bench_grouped_eval / benchmark_gate_e2e.
#
# Usage:
#   bash run_e2e_gate_pretrain.sh
#   bash run_e2e_gate_pretrain.sh 2>&1 | tee run_e2e_gate_pretrain.log

set -euo pipefail

# ── Infrastructure ────────────────────────────────────────────────────────────
: "${PYTHON:=python3}"   # point at your env (conda/venv) for real runs
: "${NUM_GPUS:=8}"

# ── Multi-node (optional) ─────────────────────────────────────────────────────
# Single-node by default.  For NNODES>1, every node runs this same chain with
# NODE_RANK 0..N-1 and the same MASTER_ADDR (see launch_multinode.sh or the
# SLURM example train-scripts/example/multinode.sbatch).
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=}"
export NNODES NODE_RANK MASTER_ADDR

# ── Model ─────────────────────────────────────────────────────────────────────
: "${MODEL:=Qwen/Qwen3-8B}"   # env-overridable (was a hard assignment)

# ── Train-step implementation (kernels, compile, packing, diagnostics cadence) ──
# ONE place for every config that runs through this driver, and the configuration the
# paper's training-time table is timed with (kernels/README.md, "Train-step cost"): the
# timing harness runs these defaults and passes no switches of its own. Override per run
# from the environment only to study the switch itself.
#   TRAIN_ATTN_BACKENDS  fa2 = FlashAttention-2 + the key-bias fork (kernels/fa2bias);
#                        flex = FlexAttention (1.85-2.4x slower per step).
#   COMPILE_PHASE2       torch.compile (in place) of the layers above the gate layer.
#   PACK_PHASE2          the Phase-2 targets of a sample as one variable-length launch, up to
#                        PACK_PHASE2_MAX_TOKENS packed tokens (both targets' activations are
#                        alive at once; above it the targets run one after the other).
#   TRAIN_LOG_FREQ       rank-0 diagnostics + full W&B log every N steps (~860 host syncs per
#                        step at 1); loss / K / pbar still every step.
# The fork's backward is built for head_dim 128 (Qwen3 dense); Qwen3.5 (head_dim 256) trains
# on flex, unpacked.
case "${MODEL}" in
    Qwen/Qwen3.5-*) _FA2_TRAIN=0 ;;
    Qwen/Qwen3-*)   _FA2_TRAIN=1 ;;
    *) echo "[pretrain] no train-step kernel default for MODEL=${MODEL}: set TRAIN_ATTN_BACKENDS / PACK_PHASE2 / COMPILE_PHASE2 and add the model here" >&2; exit 1 ;;
esac
if [ "${_FA2_TRAIN}" = "1" ]; then
    TRAIN_ATTN_BACKENDS="${TRAIN_ATTN_BACKENDS:-fa2}"
    PACK_PHASE2="${PACK_PHASE2:-1}"
    COMPILE_PHASE2="${COMPILE_PHASE2:-1}"
else
    TRAIN_ATTN_BACKENDS="${TRAIN_ATTN_BACKENDS:-flex}"
    PACK_PHASE2="${PACK_PHASE2:-0}"
    COMPILE_PHASE2="${COMPILE_PHASE2:-0}"
fi
PACK_PHASE2_MAX_TOKENS="${PACK_PHASE2_MAX_TOKENS:-12288}"
TRAIN_LOG_FREQ="${TRAIN_LOG_FREQ:-50}"
export TRAIN_ATTN_BACKENDS PACK_PHASE2 PACK_PHASE2_MAX_TOKENS COMPILE_PHASE2 TRAIN_LOG_FREQ
echo "[pretrain] train step: attn=${TRAIN_ATTN_BACKENDS} compile_phase2=${COMPILE_PHASE2} pack_phase2=${PACK_PHASE2} (<=${PACK_PHASE2_MAX_TOKENS} tokens) log_freq=${TRAIN_LOG_FREQ}"

# ── Pretraining mode flags (consumed by e2e_gate_lib.sh) ──────────────────────
# Load training tokens from the packed, memory-mapped store (built by
# `python -m tokengate.datasets.pack <manifest>`, e.g. iso_repobench)
# instead of unpickling train.pkl.  ms-scale load, ~10x less host RAM, one
# physical copy shared across all 8 ranks via the OS page cache.  Eval still
# reads val.pkl.  Set to 0 to fall back to the legacy pickle loader.
: "${PRETRAIN_PACKED:=1}"   # env-overridable; set 0 to use the legacy per-dataset
                            # train.pkl loader (needed for subset runs whose datasets
                            # aren't in the packed store, e.g. the trec validation run).
: "${PRETRAIN_DATA_ROOT:=datasets}"   # repo-local tokenized stores (prepare.py output)
# Eval is reported as one pooled val set per metric group (qa / code_completion
# / summarization / ...), evaluated once with per-sample task resolution → one
# micro-averaged number per group (+ a free per-dataset breakdown).  Each
# group's SIZE is pinned explicitly by EVAL_SPLIT_MANIFEST's per-subset `val`
# id lists (the implicit per-group sampling budget was removed).  The one-time
# step-0 no-compression + oracle baselines run over these SAME per-group pools
# so their W&B reference lines are directly comparable to the gated
# eval/cat_<group> curves.

# ── Accepted datasets via manifest (reproducibility) ─────────────────────────
# The (dataset, task) list lives in a version-pinned YAML manifest under
# tokengate/datasets/manifests/<version>.yaml so the
# exact mix is reproducible from the script + git history.  Manifest also
# records per-row prompt_version + sanity numbers so a run can be re-derived
# without re-reading DATASETS.md.
#
# To launch a different mix: bump PRETRAIN_MANIFEST to a different version
# name (or write a new YAML file under _lib/manifests/).  Each unique
# manifest produces a distinct MULTITASK_NAME → distinct output dir + W&B
# run history.
: "${PRETRAIN_MANIFEST:?set PRETRAIN_MANIFEST (every train-scripts config pins one; see tokengate/datasets/manifests/)}"

_MANIFEST_CLI="${PYTHON} -m tokengate.datasets.manifest ${PRETRAIN_MANIFEST}"
# DATASET/TASK/EVAL_ONLY_PAIRS default to the manifest's full lists, but can be
# OVERRIDDEN by env (comma-sep DATASET/TASK; "ds:tk,..." EVAL_ONLY_PAIRS) to train on
# a SUBSET of the mix without a new manifest — e.g. the trec validation run trains on
# just qt_squad_fewshot and evals held-out on longbench_trec.  The manifest is still
# read for sha/version (→ run name); set MULTITASK_SUFFIX to keep subset runs distinct.
DATASET="${DATASET:-$(${_MANIFEST_CLI} dataset)}"
TASK="${TASK:-$(${_MANIFEST_CLI} task)}"
# Held-out (eval-only) subsets: full LongBench eval coverage.  These are evaluated
# but NEVER trained on, forwarded to the trainer via --pretrain-eval-only-pairs
# (NOT --dataset/--task), so train + pack are identical to the v5 trained mix.
# Empty for manifests with no eval_only_datasets section.
EVAL_ONLY_PAIRS="${EVAL_ONLY_PAIRS:-$(${_MANIFEST_CLI} eval_only_pairs)}"
_MANIFEST_SHA="$(${_MANIFEST_CLI} sha)"

# In-training eval split: every lane pins its val/test partition explicitly via
# a tokengate/datasets/eval_splits/*.json manifest — the single source of
# eval-set sizes (no silent default).
: "${EVAL_SPLIT_MANIFEST:?set EVAL_SPLIT_MANIFEST in the lane config (tokengate/datasets/eval_splits/<target>_iso_v1.json)}"
_MANIFEST_VERSION="$(${_MANIFEST_CLI} version)"
# Extract the major version (e.g., "v2") from the manifest's version field
# (e.g., "v2_accepted7_with_hybridqa") for a clean W&B-sortable prefix.
# Run name leads with `pretrain-vN-` so all pretrain runs cluster in W&B
# filtering / sorting regardless of MULTITASK_NAME content drift.
_MAJOR_VERSION="$(echo "${_MANIFEST_VERSION}" | grep -oE '^v[0-9]+' || echo 'v0')"
_VERSION_SUFFIX="${_MANIFEST_VERSION#${_MAJOR_VERSION}_}"
# Compact the suffix: the GATE_ROOT leaf dir encodes EVERY hyperparameter, and the
# mixed-K + full-rank-LoRA suffixes (_mkm=8_mkbw=detach_lora1_r4096_a8192)
# already push it to the 255-char filesystem NAME_MAX limit.  Abbreviate the long
# manifest tokens so the leaf stays under 255 (longbench_full_eval -> lbfull).
_VERSION_SUFFIX="${_VERSION_SUFFIX//longbench/lb}"
_VERSION_SUFFIX="${_VERSION_SUFFIX//_full_eval/full}"
# Append first 8 chars of manifest SHA so editing the manifest (even with
# the same `version:` string) yields a distinct run dir + W&B history
# rather than silently overwriting prior history.
: "${MULTITASK_NAME:=pretrain-${_MAJOR_VERSION}-${_VERSION_SUFFIX}-sha${_MANIFEST_SHA:0:8}}"   # env-overridable (subset runs set a clean leading name)
# Optional suffix for one-off verification / ablation runs that need a clean
# unique output dir (e.g. side-by-side bs=1 vs bs=4 init-eval comparison).
# Leave unset for normal training runs.
MULTITASK_NAME="${MULTITASK_NAME}${MULTITASK_SUFFIX:-}"
echo "[pretrain] manifest=${PRETRAIN_MANIFEST}  sha=${_MANIFEST_SHA}  → ${MULTITASK_NAME}"


# ── Required training config (FAIL LOUDLY when unset) ────────────────────────
# These are real training hyperparameters every lane must state explicitly; a
# silent default here would hide the value a run actually trained with.
: "${EPOCHS:?set EPOCHS in the lane config (passes over the training set)}"
: "${LR_VALUES:?set LR_VALUES in the lane config}"
: "${MUON_GATE:?set MUON_GATE (0/1) in the lane config}"
: "${NO_DECAY_SCALARS:?set NO_DECAY_SCALARS (0/1) in the lane config}"
: "${SHUFFLE_EPOCHS:?set SHUFFLE_EPOCHS (0/1) in the lane config}"
: "${EVAL_MIN_ARTICLE_TOKENS:?set EVAL_MIN_ARTICLE_TOKENS in the lane config}"
: "${EVAL_MAX_ARTICLE_TOKENS:?set EVAL_MAX_ARTICLE_TOKENS in the lane config}"
: "${LORA_RANK:?set LORA_RANK in the lane config (0 disables the factored adapter)}"
if [ "${ADAPTER_TYPE:-lora}" = "lora" ] && [ "${LORA_RANK}" -gt 0 ]; then
    : "${LORA_ALPHA:?set LORA_ALPHA in the lane config}"
    : "${LORA_LR:?set LORA_LR in the lane config}"
    : "${LORA_TARGETS:?set LORA_TARGETS in the lane config}"
fi
# v3 is summarization-heavy (answer p95 151–2228 tok: eur_lex 2228, klexikon 1574,
# summscreen 860, arxiv 648, …).  Eval decode EOS-breaks per group, so the short-
# answer groups (qa/code/classification, p95<=56) still finish early regardless of
# this cap; 1024 covers summarization p95 for all but the two legal outliers
# (klexikon, eur_lex), which are still covered near their means.
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-1024}"   # env-overridable (smoke runs shrink it)
: "${MAX_ARTICLE_TOKENS:=40960}"           # env-overridable. Qwen3-8B max context (max_position_embeddings); was
                                           # 30000, which dropped the 217 packed rows in (30k,45k]
                                           # — the longest articles.  Articles >40960 (a handful)
                                           # are still dropped (exceed model context); the longest
                                           # in-range ones may transient-OOM at M=8/r4096 → OOM-skip.
: "${MIN_ARTICLE_TOKENS:=0}"           # env-overridable (v10: length window floor)
# Drop (never truncate) training/eval rows whose TARGET answer exceeds this many
# tokens.  Distinct from MAX_GEN_TOKENS (which only bounds eval-time generation):
# this caps the actual training answer length.  The compiled Phase-2 sequence is
# ~K_kept + n_ans, so summarization's long, highly-variable answers (arxiv n_ans
# up to ~40k, thousands of distinct lengths) flood torch.compile/FlexAttention
# with distinct shapes and stall the first-step compile on multi-GPU.  Capping
# n_ans collapses that spread.  1024 keeps the bulk of every dataset (summ p95
# 179–971 for arxiv/booksum/mslr; only the legal-summary tails exceed it).
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-1024}"

# Mix all data and treat it as a single dataset, but iterate a per-epoch
# shuffled permutation (without replacement) instead of i.i.d. with-replacement
# sampling: the ranks consume disjoint shards whose union covers the pool
# exactly once per epoch, reshuffled each epoch.

# ── Gate architecture (same as quality/multitask) ─────────────────────────────
GATE_LAYERS="${GATE_LAYERS:-0}"

# ── K sampling ────────────────────────────────────────────────────────────────
# Per-step K targets come from PHASE2_BUCKET_LISTS (per length-bucket phase-2
# target lists; see --phase2-bucket-lists in train_e2e_gate.py).
# K_EVAL_VALUES = the in-training eval sweep (first value = the deployment /
# headline K; lanes set it).  Post-training benchmarking sweeps its own K list
# via benchmarks/run_benchmark.py configs.
K_EVAL_VALUES="${K_EVAL_VALUES:-2000}"
: "${TRAIN_BATCH_SIZE:=1}"                 # 8 GPUs × bs=1 = eff. batch 8 (env-overridable)

# How many eval-generation samples to attempt per (rank, K, group) at once.
# Default in e2e_gate_lib.sh is 1 (single-sample decode).  Workers had ~25-40 GB
# headroom on H100s; batching 4 cuts decode wall ~3-4x.  Env-overridable so
# verification runs can compare bs=1 vs bs=4 head-to-head.  Safe to keep at 4
# now that the batched-decode EOS-truncation bug (PR #63) is fixed.
EVAL_GEN_BATCH_SIZE=${EVAL_GEN_BATCH_SIZE:-4}

# ── Length bucketing (reduce cross-rank stragglers) ────────────────────────────
# All ranks draw from the SAME length bucket each step (seeded by step, so they
# agree), then sample within it.  With GATE_LAYERS=0 phase-1 runs ONE layer over
# the FULL article incl. quadratic attention, so a 19k-token narrativeqa sample
# costs far more phase-1 than an 850-token 2wikimqa one; uniform mixing strands
# ranks at the all-reduce barrier.  Bucketing equalises per-step article length
# across ranks and reduces BlockMask-cache thrash.  LINEAR 4k increments (not
# geometric/doubling): caps the within-bucket length spread at 4k EVERYWHERE, so
# the worst-case cross-rank phase-1 difference is a bounded ~4k tokens regardless
# of absolute length (geometric leaves the high buckets 8-16k wide -> 2x spread on
# exactly the expensive long samples).
# ⚠️ MUTUALLY EXCLUSIVE with SHUFFLE_EPOCHS (below) — the trainer errors if both
# are set.  Default OFF (shuffle-epochs is the v6 sampler); to use bucketing
# instead, run with SHUFFLE_EPOCHS=0 LENGTH_BUCKETS="4000,8000,...,32000".
# Pending the bucketing-vs-shuffle A/B on the v6 mix.
: "${LENGTH_BUCKETS=}"

# ── Adapter ───────────────────────────────────────────────────────────────────
# Default: factored LoRA on all layers (the converged co-adapt recipe); the
# rank/alpha/lr/targets values are REQUIRED (see the fail-loudly block above).
# ADAPTER_TYPE=delta selects the additive full-matrix ΔW adapter instead
# (rank/alpha-free, Muon-native); the two are mutually exclusive.
: "${ADAPTER_TYPE:=lora}"
: "${LORA_LAYERS:=all}"

# ── Optimizer ──────────────────────────────────────────────────────────────────
# Muon (torch.optim.Muon, adjust_lr_fn=match_rms_adamw — always) on the full-rank
# LoRA 2D matrices for faster convergence; the gate's own linear proj + scalars
# stay on AdamW (Muon lagged AdamW on the gate — project_muon_lags_adamw_gate).
# match_rms_adamw makes the Muon lr AdamW-comparable, so every group shares the
# per-group LRs (LR_VALUES / LORA_LR) — there is NO separate Muon LR anymore.
# Set OPTIMIZER=adamw to revert.
# NOTE: full-rank LoRA overfits/diverges on tiny SFT sets (verified on quality,
# ~3k samples — both AdamW and Muon); it is used here because the pretraining
# mix is large enough to support 109M adapter params.  Watch the early SWA loss
# curve and drop LR_VALUES (e.g. 3e-6) if it climbs.
: "${OPTIMIZER:=muon}"
# Decoupled weight decay on EVERY optimizer group (Muon + aux-AdamW).  v11+
# runs at 0.1 (Muon-standard); v10 and earlier ran the Muon path with wd=0.
# The _wd= tag in RUN_NAME/GATE_ROOT keeps output dirs distinct.
: "${WEIGHT_DECAY:=0.1}"

# ── Training hyperparameters ──────────────────────────────────────────────────
SEED_VALUES="${SEED_VALUES:-0}"
SCALAR_LR_MULT="${SCALAR_LR_MULT:-1}"
: "${WARMUP_STEPS:=1000}"   # env-overridable (lower for smoke tests)
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
# SWA is OFF by default (SWA_FREQ=-1); lanes that average weights set
# SWA_FREQ=1 (+ SWA_START / SWA_EVAL_FREQ) explicitly.
: "${SWA_FREQ:=-1}"
: "${SWA_START:=0}"
: "${SWA_EVAL_FREQ:=1}"
# EVAL_FREQ=1800 halves the number of in-training evals.  Mixed-K eval now sweeps
# 3 K values (K_EVAL_VALUES="500 1000 2000") instead of 1, so each eval cycle is
# ~3x the single-K cost; EVAL_GEN_BATCH_SIZE=4 keeps decode wall bounded.  Widen
# K_EVAL_VALUES or raise EVAL_FREQ if the eval share of wall time grows too large.
: "${EVAL_FREQ:=1800}"   # env-overridable (lower for more frequent eval signal on test runs)
# Checkpoint save cadence ALIGNED to the eval cadence by default (one save per eval
# point).  This is the disk lever when keeping all checkpoints: a periodic save is
# ~2.3GB (raw gate + lora + train_state) or ~3.1GB with SAVE_SWA_STEP_CHECKPOINTS=1
# (adds swa_gate + swa_lora).  To KEEP ALL checkpoints for a checkpoint-vs-accuracy
# sweep set CHECKPOINT_KEEP_LAST=1000000 (+ SAVE_SWA_STEP_CHECKPOINTS=1); with the
# eval-aligned cadence disk stays bounded (df -h the output partition before launch).
: "${CHECKPOINT_FREQ:=${EVAL_FREQ}}"   # env-overridable; default = EVAL_FREQ

# ── Ensure the packed, memory-mapped token store exists ───────────────────────
# Idempotent: skips datasets whose packed/ store is already fresh (matches the
# current train.pkl size+mtime), so this is a no-op on every run after the first.
# The first run (or after a train.pkl regen) pays a one-time ~15 min repack.
# Only NODE_RANK 0 builds (multi-node runs share the store over the common
# filesystem, so non-master nodes must not race to repack it).  Idempotent.
if [ "${PRETRAIN_PACKED:-0}" = "1" ] && [ "${NODE_RANK:-0}" = "0" ]; then
    echo "[pretrain] ensuring packed token stores exist (idempotent)…"
    "${PYTHON}" -m tokengate.datasets.pack \
        "${PRETRAIN_MANIFEST}" \
        --root "${PRETRAIN_DATA_ROOT}" \
        --model "$(basename "${MODEL}")"
fi

# ── Load shared library and run ───────────────────────────────────────────────
# shellcheck source=e2e_gate_lib.sh
source "$(dirname "$0")/e2e_gate_lib.sh"
run_e2e_gate_main
