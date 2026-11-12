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

All tests live in the root `tests/` directory. They are script-style
(`python -m tests.<module>`), not pytest; run them from the repository root.
The same five CPU suites run in GitHub Actions on Python 3.10 and 3.12:

```bash
python -m tests.test_trainer_api
python -m tests.test_swa_precision
python -m tests.test_split_gate_layer
python -m tests.test_gate_branch_graph
python -m tests.test_inference_api
```

These tests use small, randomly initialized layers. The split/gradient tests
download a public Qwen config, and the inference API test downloads a public
tokenizer from Hugging Face; no model weights or credentials are required.

For a CPU-only development environment matching CI's installation:

```bash
uv venv --python 3.12
uv pip install --torch-backend=cpu -e .
```

CI resolves the dependency ranges in `pyproject.toml` with CPU PyTorch wheels.
It does not install the CUDA dependencies in the training `uv.lock`.

GPU tests (one GPU each) exercise the released checkpoints end to end:

```bash
python -m tests.test_niah_demo
python -m tests.test_biodexr_demo
```

The remaining attention, integration, training, and checkpoint suites are run
manually on a suitable CUDA machine, with the model/checkpoint/data requirements
listed in each script. They are not part of the hosted CPU check. When adding a
CPU suite, add its module command to `.github/workflows/tests.yml` as well.

### Required PR check (repository administrator)

The `Tests` workflow runs on every pull request (including stacked PRs), pushes
to `main`, merge queues, and manual dispatch. Its final **Tests** job fails if
any CPU matrix job fails, is cancelled, or is skipped. There are no path filters
that could leave a required check pending.

Adding a workflow alone does not block merges. After the workflow has run on a
PR, a repository administrator must configure the `main` branch's ruleset or
branch protection in **Settings → Rules → Rulesets** or **Settings → Branches**:

1. Require a pull request before merging.
2. Enable **Require status checks to pass before merging**, select **Tests**
   from GitHub Actions, and require the branch to be up to date before merging
   (or use a merge queue).
3. Enforce the rule without bypasses if it should apply to administrators too.

Keep the **Tests** job name stable as the matrix grows. This setting is managed
in GitHub, outside the pull request; see GitHub's
[required status check documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging).

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
