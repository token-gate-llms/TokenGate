# Roadmap

TODOs:
- [x] **CPU test CI** — GitHub Actions on push/PR runs the five lightweight CPU
  suites on Python 3.10 and 3.12.
- [ ] **Require passing CI** — a repository administrator must require the
  `Tests` check on `main` (see CONTRIBUTING.md).
- [ ] **Additional CI** — ruff lint + format check, a README link checker, GPU
  test runners, and badges (CI, arXiv, license, HF models) on the README.
- [ ] **PyPI package** — `pip install tokengate` with a torch-light core for inference
  and a `[train]` extra. Prerequisite: the `kernels.fa2bias` import path resolves from
  the repo root today, so a non-editable wheel install breaks on it — move the kernel
  sources under the package (or make the import optional with the flex fallback) first.
- [ ] **Colab quickstart notebook** — the inference quickstart on a free-tier GPU; no
  custom kernel build needed since inference falls back to FlexAttention automatically.
- [ ] **Full open-source roadmap**
- [ ] **pixi environment**
