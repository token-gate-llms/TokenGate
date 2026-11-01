# A100 NIAH validation

The cluster driver is `notebooks/a100_run_all.sh`. Its output root is
`/lfs/guestrin-hgx-1/0/aryankatiyar/tokengate-demo/a100_paper_validation_20261101`;
`status.log` records each stage. The driver waits for otherwise idle A100s
before inference (one GPU) and training (four GPUs), and runs:

1. Released Qwen3-1.7B gate on 16k NIAH and released Qwen3.5-2B layer-1 gate
   on 32k NIAH, each with a 3-row smoke test and the frozen 150-row test set.
2. Two-step launch checks, then 3,000-step Qwen3-1.7B and 18,000-step
   Qwen3.5-2B layer-1 training on 12,000 synthetic NIAH rows. The 50-row
   validation partition is used for selection; the 150 test IDs are excluded.
3. An additional Qwen3.5-2B layer-3 run at 18,000 steps. The
   [paper's](https://arxiv.org/html/2610.26754v1) matched-accuracy 3.8×
   result uses layer 3, while the public checkpoint is
   at layer 1.
4. Inference of each newly trained checkpoint on the same 150 held-out rows.

Each benchmark measures single-pass, batch-one prefill to the first token for
the unmodified base model, gate with all tokens, and K=100/500/2000. The
paper's speedup denominator is a separately LoRA-tuned baseline and uses its
own optimized kernel stack. Therefore this is an independent consistency
check, not an exact paper reproduction. The scripts keep the raw logs and
checkpoint paths for inspection.

An earlier A100 fast-kernel run of the released Qwen3-1.7B checkpoint on 150
held-out 16k rows reached 150/150 accuracy at K=100 and 10.82× TTFT speedup
against the unmodified base model. The paper reports 11.3× TTFT speedup at
matched accuracy for this setting. The new queued run will measure both
models and their newly trained gates with one documented split and protocol.
