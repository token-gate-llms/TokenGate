#!/usr/bin/env bash
# Benchmark the completed reproduction checkpoints on the same held-out rows.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${TOKENGATE_A100_RUN_ROOT:?set an output directory}"
QWEN3_PYTHON="${TOKENGATE_QWEN3_PYTHON:?set the fast-kernel environment Python}"
QWEN35_PYTHON="${TOKENGATE_QWEN35_PYTHON:?set the hybrid environment Python}"
HF_HOME="${HF_HOME:?set a writable Hugging Face cache}"
FA2BIAS_SITE="${FA2BIAS_SITE:?set the built key-bias kernel directory}"
export HF_HOME FA2BIAS_SITE PYTHONPATH="$ROOT"

run_model() {
    local label="$1" length="$2" python="$3" base="$4" attn="$5" parent="$6" layer="${7:-1}"
    local out="$RUN_ROOT/$label" config="$RUN_ROOT/$label.yaml" checkpoint
    checkpoint="$(find "$RUN_ROOT/$parent/checkpoints" -type f -name gate_final.pt -printf '%h\n' | head -n 1)"
    if [[ -z "$checkpoint" ]]; then
        echo "No completed checkpoint found for $parent" >&2
        return 1
    fi
    mkdir -p "$out"
    cat > "$config" <<YAML
base_model: $base
task: niah_kamradt
gate_layer: $layer
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

run_model qwen3_trained16k 16384 "$QWEN3_PYTHON" \
    Qwen/Qwen3-1.7B fa2 qwen3_step3000
run_model qwen35_trained32k 32768 "$QWEN35_PYTHON" \
    Qwen/Qwen3.5-2B flex qwen35_step18000
run_model qwen35_layer3_trained32k 32768 "$QWEN35_PYTHON" \
    Qwen/Qwen3.5-2B flex qwen35_layer3_step18000 3
