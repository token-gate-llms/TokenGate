# TokenGate on JEV: initial integration

See [exploratory results](RESULTS.md) for the measured speed and accuracy tradeoff.

This experiment uses [JEV-9B](https://huggingface.co/autotrust/JEV-9B), whose Qwen3.5 backbone scores a decision through a separate 24-slot head. TokenGate can prune the text state after its early layers, then send the final prompt hidden state through that same decision head. The JEV LoRA adapter must be loaded first. The [Transformers JEV API](https://huggingface.co/docs/transformers/main/model_doc/jev) is included in the stable 5.23 release.

`tokengate.experimental.jev` provides:

- `standard_decide`: JEV's ordinary full prefill.
- `input_pruned_decide`: prefix, middle, suffix, and random token selection before JEV's ordinary forward. These are simple comparison methods.
- `gated_decide`: TokenGate's learned selection and attention bias, followed by JEV's decision head.
- `create_jev_gate` and `train_jev_gate_step`: initialize a gate and distill one full-prefill decision into it. A useful checkpoint needs many training examples and held-out validation.

The prototype handles one text question with its decision readout at the end of the prompt. It does not yet handle image assets, multiple questions in one forward, or arbitrary decision-model architectures. A gate checkpoint must be trained for the same JEV backbone and adapter; existing TokenGate NIAH gates are not compatible.

To compare full prefill with simple input pruning:

```bash
python -m pip install -e .
python -m pip install -r experiments/jev/requirements.txt
python experiments/jev/compare.py --budgets 256 512 1024
```

For latency measurements, install `flash-linear-attention` and `causal-conv1d` in the same environment. Without them, Transformers uses slower PyTorch fallbacks for JEV's DeltaNet layers.

The comparison script reports one-request end-to-end time, state tokens retained, and JEV's refund probability for a synthetic long state. It accepts `--state-file` for another text. `--gate-checkpoint` is available after training a matching gate with metadata `decision_model: autotrust/JEV-9B`. Its numbers are exploratory and should not be compared to the TokenGate paper benchmarks.
Use `--needle-position start|middle|end` and `--filler-repetitions N` to check different state lengths and fact positions.
JEV's default template places the state before the question, so early causal state representations cannot see that question. `--prefix-question` repeats the question at the start of the state; use it consistently for training and evaluation. The full-prefill reference receives the same modified prompt.
`--random-target-position` plants a clearly marked target case among unrelated excerpts at varying depths. The full model, gate, and fixed-position baselines then receive the same prompt.
`--kind noul --min-target-margin 0.2` focuses training on binary decisions with a clear corpus label; evaluate on held-out rows with the same filter and also inspect broader rows separately.
When a gate is supplied, it also reports `full-flex`: the full decision after TokenGate patches the attention layers. Compare its probability with `full` before interpreting gated results.

An initial training run can use `python experiments/jev/train_gate.py --steps 100 --budget 128 --output jev_gate.pt`, then `python experiments/jev/compare.py --gate-checkpoint jev_gate.pt --budgets 128 256 512`. The script distills full-prefill decisions from the public [JEV training corpus](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) into the gate and adds unrelated background excerpts so there is a meaningful state span to prune. Use this for integration testing; held-out accuracy and long-context latency must be measured before treating it as a useful trained gate.

For a length curriculum, set `--min-background-excerpts 12 --background-excerpts 72` to sample a different number of unrelated cases on each training step.

`python experiments/jev/evaluate.py --gate-checkpoint jev_gate.pt --rows 20` measures agreement and KL against full JEV prefill and the corpus targets on the corpus's held-out `test_set_30k` split. Prefix, middle, suffix, and random pruning run at the same budgets. All methods use the same patched backbone and augmented states.

For a less ambiguous top-choice check, add `--kind noul --min-target-margin 0.2`; KL remains useful even when the corpus's two highest probabilities are nearly tied.

Add `--zero-gate-bias` to keep the gate's selected positions while removing its attention bias. This ablation helps separate token selection errors from bias effects.
