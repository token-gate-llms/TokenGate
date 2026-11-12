# Contributing to TokenGate

## Setup

TokenGate requires Python 3.10 or newer and PyTorch 2.9 or newer. To get started:

```bash
git clone https://github.com/token-gate-llms/TokenGate && cd TokenGate
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[train]"
```

For CPU tests, use `uv pip install --torch-backend=cpu -e .` instead.
For CUDA training, you can build the fast attention kernel with
`python kernels/fa2bias/build.py` (requires `nvcc`). Without it, inference
falls back to FlexAttention; training needs `attn_backend="flex"`.

## Tests

Tests live under `tests/`, grouped by subsystem. Run each script as a module
from the repository root. These five suites run on CPU:

```bash
python -m tests.training.test_trainer_api
python -m tests.training.test_swa_precision
python -m tests.gate.test_split_gate_layer
python -m tests.gate.test_gate_branch_graph
python -m tests.inference.test_inference_api
```

Some tests download public Qwen configs or tokenizers from Hugging Face.
The remaining suites need CUDA, model weights, or prepared data; each script
describes its requirements. For example, these test the released demos:

```bash
python -m tests.examples.test_niah_demo
python -m tests.examples.test_biodexr_demo
```

Performance benchmarks live in `benchmarks/`.

## Contributions

- Keep each PR focused and include the relevant test results.
- Raise clear errors for invalid configuration. Inference's documented
  attention fallback should warn.
- Keep code in `tokengate/` and prepared data in `datasets/`. Run dataset
  preparation scripts by path; leave generated stores out of Git.
- Keep `datasets/` without an `__init__.py`: making it a Python package would
  shadow the Hugging Face library.
