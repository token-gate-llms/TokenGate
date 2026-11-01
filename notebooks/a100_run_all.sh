#!/usr/bin/env bash
# Persistent cluster driver. Each stage records a separate result and continues.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TOKENGATE_A100_RUN_ROOT="${TOKENGATE_A100_RUN_ROOT:-/lfs/guestrin-hgx-1/0/aryankatiyar/tokengate-demo/a100_paper_validation_20261101}"
export TOKENGATE_NIAH_DATA_ROOT="${TOKENGATE_NIAH_DATA_ROOT:-/lfs/guestrin-hgx-1/0/aryankatiyar/tokengate-reproduction/niah_release/data}"
export TOKENGATE_QWEN3_PYTHON="${TOKENGATE_QWEN3_PYTHON:-/lfs/guestrin-hgx-1/0/aryankatiyar/tokengate-reproduction/venv/bin/python}"
export TOKENGATE_QWEN35_PYTHON="${TOKENGATE_QWEN35_PYTHON:-/lfs/guestrin-hgx-1/0/aryankatiyar/tokengate-demo/venv/bin/python}"
export HF_HOME="${HF_HOME:-/lfs/guestrin-hgx-1/0/aryankatiyar/tokengate-demo/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/lfs/guestrin-hgx-1/0/aryankatiyar/tokengate-demo/cache}"
export FA2BIAS_SITE="${FA2BIAS_SITE:-/lfs/guestrin-hgx-1/0/aryankatiyar/tokengate-reproduction/niah_release/fa2_site}"
mkdir -p "$TOKENGATE_A100_RUN_ROOT" "$XDG_CACHE_HOME"
status="$TOKENGATE_A100_RUN_ROOT/status.log"
echo "[$(date -u +%FT%TZ)] validation driver started at $(git -C "$ROOT" rev-parse --short HEAD)" >> "$status"

run_stage() {
    local label="$1" script="$2" rc
    echo "[$(date -u +%FT%TZ)] $label started" | tee -a "$status"
    bash "$ROOT/notebooks/$script" >> "$TOKENGATE_A100_RUN_ROOT/${label}_driver.log" 2>&1
    rc=$?
    echo "[$(date -u +%FT%TZ)] $label exit=$rc" | tee -a "$status"
    return 0
}

run_stage released_inference a100_inference_validation.sh
while [[ ! -s "$TOKENGATE_NIAH_DATA_ROOT/niah_kamradt/niah_kamradt/tokenized/Qwen3.5-2B/packed/info.json" ]]; do
    echo "[$(date -u +%FT%TZ)] waiting for Qwen3.5 packed training data" >> "$status"
    sleep 300
done
run_stage training a100_train_validation.sh
run_stage trained_inference a100_trained_inference.sh
echo "[$(date -u +%FT%TZ)] validation driver finished" | tee -a "$status"
