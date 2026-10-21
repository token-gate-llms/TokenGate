# Contributing to TokenGate

Thanks for your interest! Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/tokengate-models/TokenGate && cd TokenGate
uv venv && uv pip install -e ".[train]"      # Python >= 3.10, CUDA PyTorch >= 2.9
python kernels/fa2bias/build.py              # optional: the fast key-bias kernel (needs nvcc)
```

Without the kernel build, inference falls back to FlexAttention automatically and
training takes an explicit `attn_backend="flex"`.

## Running tests

Tests are script-style (`python -m <module>`), not pytest. CPU-safe:

```bash
python -m tokengate.e2e_gate.tests.test_trainer_api
```

GPU tests (one GPU each) exercise the released checkpoints end to end:

```bash
python -m tokengate.e2e_gate.tests.test_niah_demo
python -m tokengate.e2e_gate.tests.test_biodexr_demo
```

## Conventions

- **Fail loudly.** No silent defaults or fallbacks that change results — a config
  error should stop the run with a message naming the fix. The one exception is
  inference kernel fallback, which warns.
- **Never add an `__init__.py` under top-level `datasets/`.** It is a data directory,
  not a package — a top-level `datasets` package would shadow the HuggingFace
  `datasets` library for any process started from the repo root.
- **Code lives in `tokengate/`, data in `datasets/`.** Per-dataset `prepare.py`
  scripts are run by path; generated stores stay gitignored.
- Keep PRs focused, and include the test output for anything touching training or
  benchmark code paths.
