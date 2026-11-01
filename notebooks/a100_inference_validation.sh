#!/usr/bin/env bash
# Released-checkpoint NIAH inference on the frozen 150-row test partition.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${TOKENGATE_A100_RUN_ROOT:?set an output directory}"
QWEN3_PYTHON="${TOKENGATE_QWEN3_PYTHON:?set the fast-kernel environment Python}"
QWEN35_PYTHON="${TOKENGATE_QWEN35_PYTHON:?set the hybrid environment Python}"
HF_HOME="${HF_HOME:?set a writable Hugging Face cache}"
FA2BIAS_SITE="${FA2BIAS_SITE:?set the built key-bias kernel directory}"
mkdir -p "$RUN_ROOT"
export HF_HOME FA2BIAS_SITE PYTHONPATH="$ROOT"
SPLIT="$ROOT/tokengate/datasets/eval_splits/longbench_eval_split_niah_iso_v1.json"

make_samples() {
    local length="$1" raw="$RUN_ROOT/niah_${1}_raw.jsonl" out="$RUN_ROOT/niah_${1}_test150.jsonl"
    if [[ ! -s "$out" ]]; then
        if [[ "$length" == 16384 ]]; then
            "$QWEN35_PYTHON" "$ROOT/datasets/niah_kamradt/generate_niah_kamradt.py" \
                --output-file "$raw" --n-train 0 --n-test-cities 20 --seed 0
        else
            "$QWEN35_PYTHON" "$ROOT/datasets/niah_kamradt/generate_niah_kamradt.py" \
                --output-file "$raw" --n-train 0 --n-test-cities 20 --seed 0 \
                --test-target "$length" --test-jit 0
        fi
        "$QWEN35_PYTHON" "$ROOT/notebooks/a100_prepare_heldout.py" "$raw" "$SPLIT" "$out"
    fi
}

run_model() {
    local label="$1" length="$2" python="$3" base="$4" checkpoint="$5" attn="$6"
    local out="$RUN_ROOT/$label" config="$RUN_ROOT/$label.yaml"
    mkdir -p "$out"
    cat > "$config" <<YAML
base_model: $base
task: niah_kamradt
gate_layer: 1
checkpoint: $checkpoint
samples: $RUN_ROOT/niah_${length}_test150.jsonl
k_values: [100, 500, 2000]
max_new_tokens: 32
gate_attn: $attn
YAML
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$("$QWEN35_PYTHON" "$ROOT/notebooks/a100_wait_idle.py" --count 1 \
        2>> "$RUN_ROOT/status.log")"
    echo "[$(date -u +%FT%TZ)] $label smoke on A100 $CUDA_VISIBLE_DEVICES" | tee -a "$RUN_ROOT/status.log"
    export TORCHINDUCTOR_CACHE_DIR="$out/inductor"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
    (cd "$ROOT" && "$python" -u benchmarks/run_benchmark.py --config "$config" --n 3) \
        > "$out/smoke3.log" 2>&1
    CUDA_VISIBLE_DEVICES="$("$QWEN35_PYTHON" "$ROOT/notebooks/a100_wait_idle.py" --count 1 \
        2>> "$RUN_ROOT/status.log")"
    echo "[$(date -u +%FT%TZ)] $label full test150 on A100 $CUDA_VISIBLE_DEVICES" | tee -a "$RUN_ROOT/status.log"
    (cd "$ROOT" && "$python" -u benchmarks/run_benchmark.py --config "$config") \
        > "$out/test150.log" 2>&1
    echo "[$(date -u +%FT%TZ)] $label finished" | tee -a "$RUN_ROOT/status.log"
}

make_samples 16384
make_samples 32768
run_model qwen3_release16k 16384 "$QWEN3_PYTHON" \
    Qwen/Qwen3-1.7B liana313/tokengate-niah-qwen3-1.7b fa2
run_model qwen35_release32k 32768 "$QWEN35_PYTHON" \
    Qwen/Qwen3.5-2B liana313/tokengate-niah-qwen3.5-2b flex
