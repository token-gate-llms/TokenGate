# TokenGate — notes for coding agents

## Setup and tests

- Install: `uv venv && uv pip install -e ".[train]"` (Python >= 3.10, CUDA PyTorch >= 2.9).
- Optional fast kernel: `python kernels/fa2bias/build.py` (needs nvcc). Without it,
  inference falls back to FlexAttention with a warning; training requires an explicit
  `attn_backend="flex"`.
- Tests are **script-style `main()` modules, not pytest**:
  `python -m tokengate.e2e_gate.tests.test_trainer_api` (CPU-safe);
  `test_niah_demo` / `test_biodexr_demo` need one GPU each.
- Benchmarks: `python benchmarks/run_benchmark.py --config benchmarks/configs/<ds>.yaml
  --split val|test`.

## Layout

- `tokengate/` — the installable package (gate module, patched attention, trainer,
  inference API). Tasks, manifests, eval splits, and sampling maps are package data
  under `tokengate/datasets/`.
- `datasets/` — the data root: per-dataset `prepare.py` entry scripts (run by path)
  and gitignored tokenized stores. **Never add an `__init__.py` here** — a top-level
  `datasets` package shadows the HuggingFace `datasets` library.
- `train-scripts/` — one lane script per dataset over a shared driver.
- `kernels/fa2bias/` — the FlashAttention-2 per-key-bias fork (build.py installs the
  compiled .so to `~/.cache/tokengate/fa2bias/site`; `FA2BIAS_SITE` overrides).

## Conventions

- **Fail loudly**: no silent defaults; invalid configs raise with the fix named.
- Eval splits are frozen artifacts (`tokengate/datasets/eval_splits/*.json`): their
  ids pin the released benchmark rows — never regenerate them casually, and never pick
  release checkpoints from in-training eval curves.
- Rows are never truncated; length windows drop whole rows and live in the train
  config, not dataset loaders.
- Never commit tokens/credentials, tokenized stores, or run outputs.
