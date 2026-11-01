#!/usr/bin/env bash
# NIAH recipe checks for both public demo models on four otherwise idle A100s.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${TOKENGATE_A100_RUN_ROOT:?set an output directory}"
DATA_ROOT="${TOKENGATE_NIAH_DATA_ROOT:?set the prepared 12k-row data root}"
QWEN3_PYTHON="${TOKENGATE_QWEN3_PYTHON:?set the fast-kernel environment Python}"
QWEN35_PYTHON="${TOKENGATE_QWEN35_PYTHON:?set the hybrid environment Python}"
HF_HOME="${HF_HOME:?set a writable Hugging Face cache}"
FA2BIAS_SITE="${FA2BIAS_SITE:?set the built key-bias kernel directory}"
mkdir -p "$RUN_ROOT"
export HF_HOME FA2BIAS_SITE PYTHONPATH="$ROOT" WANDB_PROJECT="" WANDB_MODE=disabled
export PRETRAIN_DATA_ROOT="$DATA_ROOT" EVAL_TEST_FULL_TASKS="" CHECKPOINT_BEST_GROUP=niah
export NUM_GPUS=4

run_recipe() {
    local label="$1" python="$2" lane="$3" steps="$4"
    local out="$RUN_ROOT/$label"
    mkdir -p "$out"
    echo "[$(date -u +%FT%TZ)] waiting for four idle A100s for $label" | tee -a "$RUN_ROOT/status.log"
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$("$QWEN35_PYTHON" "$ROOT/notebooks/a100_wait_idle.py" --count 4 \
        2>> "$RUN_ROOT/status.log")"
    echo "[$(date -u +%FT%TZ)] $label on GPUs $CUDA_VISIBLE_DEVICES" | tee -a "$RUN_ROOT/status.log"
    nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader > "$out/gpu_start.csv"
    export PYTHON="$python" E2E_GATE_OUTPUT_ROOT="$out/checkpoints" MAX_STEPS="$steps"
    export TORCHINDUCTOR_CACHE_DIR="$out/inductor"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
    (cd "$ROOT" && bash "$lane") > "$out/train.log" 2>&1
    echo "[$(date -u +%FT%TZ)] $label finished" | tee -a "$RUN_ROOT/status.log"
}

for model in Qwen3-1.7B Qwen3.5-2B; do
    test -s "$DATA_ROOT/niah_kamradt/niah_kamradt/tokenized/$model/packed/info.json"
    test -s "$DATA_ROOT/niah_kamradt/niah_kamradt/tokenized/$model/val.pkl"
done

# Short launch checks catch environment and kernel failures before long runs.
run_recipe qwen3_smoke "$QWEN3_PYTHON" \
    train-scripts/niah/run_e2e_mlpgate_pretrain_niah_iso_coadapt13_l1_1p7b.sh 2
run_recipe qwen35_smoke "$QWEN35_PYTHON" \
    train-scripts/niah-qwen3p5/run_e2e_mlpgate_pretrain_niah_iso_coadapt17_l1d_qwen3pt5.sh 2
run_recipe qwen3_step3000 "$QWEN3_PYTHON" \
    train-scripts/niah/run_e2e_mlpgate_pretrain_niah_iso_coadapt13_l1_1p7b.sh 3000
run_recipe qwen35_step18000 "$QWEN35_PYTHON" \
    train-scripts/niah-qwen3p5/run_e2e_mlpgate_pretrain_niah_iso_coadapt17_l1d_qwen3pt5.sh 18000
