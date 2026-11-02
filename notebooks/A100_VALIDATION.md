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

## Completed released-checkpoint inference

The persistent driver finished its two released-checkpoint benchmarks on the
frozen 150-row NIAH test partition. Each speedup uses the **unmodified** base
model on the same A100 as its denominator, so it is a consistency check rather
than the paper's separately LoRA-tuned baseline comparison.

| Model / document length | Plain accuracy / TTFT | Gate K=100 | Gate K=500 | Gate K=2000 |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B / 16k | 149/150, 950.7 ms | 150/150, 82.1 ms (11.58×) | 150/150, 70.6 ms (13.46×) | 150/150, 92.9 ms (10.23×) |
| Qwen3.5-2B / 32k | 150/150, 5314.9 ms | 61/150, 917.1 ms (5.80×) | 128/150, 958.1 ms (5.55×) | 136/150, 1169.8 ms (4.54×) |

These numbers come from `qwen3_release16k/test150.log` and
`qwen35_release32k/test150.log` under the output root above. The Qwen3.5
released checkpoint gates at layer 1. The paper's 3.8× matched-accuracy result
uses a separately trained **layer-3** gate, so the row above does not test that
claim. Its layer-3 training and inference remain queued with the other training
stages. As of 2026-11-01 20:51 Pacific, the driver was waiting for four idle
A100s to begin the short Qwen3 launch check; all eight GPUs were occupied.

The optional LoRA-only baseline helper was checked against the prepared
cluster stores without a GPU: both models exposed 12,000 packed training rows
and exactly the 50 frozen validation rows, and their input/label padding
collated successfully. This verifies the data path, not the full training run.
