# Roadmap

TODOs:
- [ ] **CI** — GitHub Actions on push/PR: ruff lint + format check, a CPU test job
  (`python -m tokengate.e2e_gate.tests.test_trainer_api`, config validation, import
  checks), and a README link checker. Badges (CI, arXiv, license, HF models) at the
  top of the README.
- [ ] **PyPI package** — `pip install tokengate` with a torch-light core for inference
  and a `[train]` extra. Prerequisite: the `kernels.fa2bias` import path resolves from
  the repo root today, so a non-editable wheel install breaks on it — move the kernel
  sources under the package (or make the import optional with the flex fallback) first.
- [ ] **Colab quickstart notebook** — the inference quickstart on a free-tier GPU; no
  custom kernel build needed since inference falls back to FlexAttention automatically.
- [ ] **Full open-source roadmap**
- [ ] **pixi environment**
