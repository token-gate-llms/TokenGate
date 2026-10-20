#!/usr/bin/env bash
# Prepare the LongBench-matched subsets (datasets/longbench-datasets/) in
# parallel (one GPU each), writing tokenized data + sanity to the canonical
# main-repo pretraining_data root.
#
# Prereq: build the LongBench eval signature index once:
#   python -m tokengate.datasets.longbench_index build
#
# Then launch training with the matching iso manifest (iso_2wiki /
# iso_multinews / iso_repobench) via its lane script in train-scripts/.
set -u

PYTHON=${PYTHON:-python}
export PYTORCH_ALLOC_CONF=expandable_segments:True

MAIN=$(dirname "$(dirname "$(git rev-parse --git-common-dir)")")
[ "$MAIN" = "." ] && MAIN=$(git rev-parse --show-toplevel)
LOGDIR="$MAIN/datasets/longbench-datasets/_logs"
mkdir -p "$LOGDIR"

# task:gpu — one GPU per subset.
JOBS=(
  "longbench_2wikimqa:0" "longbench_multi_news:1" "longbench_repobench_p:2"
)

for j in "${JOBS[@]}"; do
  task="${j%:*}"; gpu="${j#*:}"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" \
    -m tokengate.datasets._run_one_matched "$task" \
    > "$LOGDIR/$task.log" 2>&1 &
  echo "launched $task on GPU$gpu (pid $!)"
done
wait
echo "[run_matched_prepare] all subsets finished — see $LOGDIR"
