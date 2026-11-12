# TokenGate on JEV: initial integration

This experiment uses [JEV-9B](https://huggingface.co/autotrust/AutoTrust-Decider-Mini-9B), whose Qwen3.5 backbone scores a decision through a separate 24-slot head. TokenGate can prune the text state after its early layers, then send the final prompt hidden state through that same decision head. The JEV LoRA adapter must be loaded first. The [Transformers JEV API](https://huggingface.co/docs/transformers/main/model_doc/jev) currently lives on the source branch; `requirements.txt` pins a revision that contains it.

`tokengate.experimental.jev` provides:

- `standard_decide`: JEV's ordinary full prefill.
- `input_pruned_decide`: prefix and random token selection before JEV's ordinary forward. These are simple comparison methods.
- `gated_decide`: TokenGate's learned selection and attention bias, followed by JEV's decision head.
- `create_jev_gate` and `train_jev_gate_step`: initialize a gate and distill one full-prefill decision into it. A useful checkpoint needs many training examples and held-out validation.

The prototype handles one text question with its decision readout at the end of the prompt. It does not yet handle image assets, multiple questions in one forward, or arbitrary decision-model architectures. A gate checkpoint must be trained for the same JEV backbone and adapter; existing TokenGate NIAH gates are not compatible.

To compare full prefill with prefix and random pruning after the ongoing two-GPU training run releases resources:

```bash
python -m pip install -e .
python -m pip install -r experiments/jev/requirements.txt
python experiments/jev/compare.py --budgets 256 512 1024
```

The comparison script reports one-request end-to-end time, state tokens retained, and JEV's refund probability for a synthetic long state. It accepts `--state-file` for another text. `--gate-checkpoint` is available after training a matching gate with metadata `decision_model: autotrust/JEV-9B`. Its numbers are exploratory and should not be compared to the TokenGate paper benchmarks.
When a gate is supplied, it also reports `full-flex`: the full decision after TokenGate patches the attention layers. Compare its probability with `full` before interpreting gated results.

An initial training run can use `python experiments/jev/train_gate.py --steps 100 --budget 128 --output jev_gate.pt`, then `python experiments/jev/compare.py --gate-checkpoint jev_gate.pt --budgets 128 256 512`. The script distills full-prefill decisions from the public [JEV training corpus](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) into the gate and adds unrelated background excerpts so there is a meaningful state span to prune. Use this for integration testing; held-out accuracy and long-context latency must be measured before treating it as a useful trained gate.

`python experiments/jev/evaluate.py --gate-checkpoint jev_gate.pt --rows 20` measures top-choice agreement and KL against full JEV prefill on held-out validation rows, alongside prefix and random pruning at the same budgets. All methods use the same patched backbone and the same augmented states.
