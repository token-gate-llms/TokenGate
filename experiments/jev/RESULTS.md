# JEV integration: exploratory results

This is an integration study, not a reproduction of the TokenGate paper. We used
`autotrust/JEV-9B` with its released decision adapter on an A100 80 GB. Its
Qwen3.5 backbone has fast decode; this experiment asks whether a TokenGate
prefill can also reduce decision latency without losing accuracy.

## Setup

The gate was trained for 300 steps at layer 3 and a 512-token budget on the
public JEV distillation corpus's training split. Each training state had one
marked target case inserted at a random position among 12 unrelated cases.
The decision question was repeated before the state so the causal gate could
see it. Only binary `noul` rows with a target-probability margin of at least
0.2 were used. We evaluated on a shuffled slice of the corpus's separate
`test_set_30k` split. The same augmented prompt was given to every method.
For latency, the environment had Flash Linear Attention and causal-conv1d.
Times are median end-to-end times for one decision, including prompt
preparation, and are not paper-style kernel measurements.

## Held-out comparison

| State length and method | Median time | Correct top choice |
| --- | ---: | ---: |
| 1k tokens, full prefill | 368 ms | 18/20 |
| 1k tokens, middle 512 | 331 ms | 19/20 |
| 1k tokens, random 512 | 329 ms | 18/20 |
| 1k tokens, TokenGate 512 | 339 ms | 17/20 |
| 1k tokens, TokenGate 512, mixed-length training | 359 ms | 17/20 |
| 6k tokens, full prefill | 821 ms | 28/30 |
| 6k tokens, middle 512 | 342 ms | 24/30 |
| 6k tokens, random 512 | 343 ms | 26/30 |
| 6k tokens, TokenGate 512 | 402 ms | 23/30 |
| 6k tokens, TokenGate 1024 | 427 ms | 23/30 |
| 6k tokens, TokenGate 512, mixed-length training | 432 ms | 24/30 |

The 6k-token gate is about 2.0× faster than full prefill, but this checkpoint
does not yet preserve accuracy as well as even random input selection. Larger
budgets did not fix the gap. The 6k test also extrapolates from training states
with 12 background cases to 72, so length generalization is a likely factor.
Disabling the gate's attention bias while keeping its selected tokens left
top-choice accuracy unchanged at 23/30 (the mean KL to full prefill improved
slightly from 0.274 to 0.271). This suggests the observed gap is mostly in
selection or training distribution, rather than the learned bias.
Training another 300-step gate with 12–72 background cases improved mean KL
to full prefill from 0.274 to 0.129 on the same 30 long examples, but only
raised top-choice accuracy from 23/30 to 24/30. This supports a length
generalization issue without closing the accuracy gap. On the original 1k
held-out prompts, the mixed-length gate stayed at 17/20.
These augmented cases are synthetic long contexts; they do not establish
performance on an unmodified JEV workload. The machine-readable results are
in [`results/heldout_noul_20_1k.json`](results/heldout_noul_20_1k.json) and
[`results/heldout_noul_30_6k.json`](results/heldout_noul_30_6k.json); the
ablation is in [`results/heldout_noul_30_6k_zero_bias.json`](results/heldout_noul_30_6k_zero_bias.json),
and the second checkpoint in [`results/heldout_noul_30_6k_mixed_train.json`](results/heldout_noul_30_6k_mixed_train.json).
Its 1k check is in [`results/heldout_noul_20_1k_mixed_train.json`](results/heldout_noul_20_1k_mixed_train.json).

## What to improve next

The gate's top-K decision is discrete. Decision-head distillation can adjust
the attention bias on kept tokens, but it gives limited direct feedback about
relevant tokens that were dropped. The next useful experiment is target-span
supervision or another selection loss, plus training across varying state
lengths for longer than 300 steps and held-out evaluation on natural JEV tasks. That work is needed
before claiming an accuracy-preserving speedup.
