# TokenGate open-source roadmap

TokenGate today supports Qwen3 (1.7B) and hybrid Qwen3.5 (2B) through 🤗
transformers, with two attention backends (the `fa2bias` CUDA kernel and a
FlexAttention fallback). The theme of this roadmap is **extensibility first**:
build the extension points once, so that each new model and each new serving
platform becomes a small, self-contained contribution instead of a core change.

Items are ordered into tiers; within a tier, order is roughly by
leverage-per-effort. Near-term tasks are tracked in [TODO.md](TODO.md).

## Tier 0 — Foundation

See [TODO.md](TODO.md): CI, a PyPI package, and a Colab quickstart. One
addition that everything below depends on:

- [ ] **Benchmark harness** — reproduce the paper's results table with one
  command against the released checkpoints. Every integration and new model
  below is judged by whether its numbers match this harness.

## Tier 1 — Extension points

The spine of the roadmap. No new user-facing capability, but it is the
difference between porting TokenGate to each model ourselves and a contributor
doing it in an afternoon.

- [ ] **Model adapter registry** — TokenGate asks a model only a handful of
  questions: where the attention layers live, which of them are full-attention
  (vs. sliding-window or linear in hybrids), the RMSNorm class and epsilon, the
  MLP shape for the gate branch, and the head/GQA layout. Move those answers
  behind a per-architecture adapter keyed on `config.model_type`, failing
  loudly for unregistered architectures. The current Qwen-specific code
  becomes the first adapter, and "support model X" becomes one small file.
- [ ] **Attention backend protocol** — formalize `resolve_backends` into a
  registry: a backend declares an availability check, a biased-attention
  implementation, and its capabilities (inference-only vs. trainable).
  `fa2-bias` and FlexAttention become the two built-ins; an external engine's
  kernel registers itself without touching core code.
- [ ] **Versioned gate artifact spec** — a stable checkpoint format on the
  Hugging Face Hub: gate weights plus the LoRA adapter (exported as standard
  PEFT so other stacks can load it), with metadata — base model id *and
  revision*, gate layer, tokenizer hash, trained budget range. This is what
  makes a trained gate portable across serving platforms, not just across
  versions of this repo.

## Tier 2 — Model coverage

Each entry rides on the adapter registry and ships three things: the adapter,
at least one trained gate on the Hub, and eval numbers in the README table.

- [ ] **Llama 3.1 / 3.2** — the most-requested base; plain GQA architecture,
  so also the easiest adapter.
- [ ] **Larger Qwen3 (4B / 8B)** — same adapter as today, mostly training
  compute.
- [ ] **Gemma 3** — its sliding-window/full-attention mix is already derisked
  by the hybrid Qwen3.5 support.
- [ ] **Mistral / Ministral** — after the above.
- [ ] **Bring-your-own-model guide** — a training walkthrough for an
  unsupported base model, built on the adapter registry. This is where
  community-trained gates come from.

## Tier 3 — Platform integrations

These ride on the backend protocol and the gate artifact spec.

- [ ] **Hugging Face kernels hub** — publish the `fa2bias` kernel through the
  [`kernels`](https://github.com/huggingface/kernels) library so `pip install`
  users get the fast path with no local build step. Also dissolves the PyPI
  prerequisite in TODO.md.
- [ ] **NVIDIA kvpress** — contribute a `TokenGatePress` to
  [kvpress](https://github.com/NVIDIA/kvpress), the de-facto zoo of KV-cache
  compression methods. Inference-only, so cheap — and it buys apples-to-apples
  comparisons against SnapKV, H2O, and friends for free.
- [ ] **vLLM** — the real serving prize: prefill pruning composes naturally
  with chunked prefill and paged KV, but the gate bias needs an
  attention-level hook. Staged as design doc → upstream RFC issue → plugin,
  once the gate artifact spec is frozen.
- [ ] **SGLang** — same template as the vLLM integration, after it.
- [ ] **RAG framework examples** — LangChain / LlamaIndex long-context RAG
  examples under `examples/`. Not engine work, just documentation of the
  headline use case; cheap and worth doing early.
- [ ] **llama.cpp / Ollama** *(help wanted)* — a GGUF gate export and a C++
  port of token-gated attention. Large audience, large effort.

## Tier 4 — Community scaffolding

- [ ] **Docs site** — move long-form docs (training guide, adapter-authoring
  guide, gate format spec) out of the README.
- [ ] **Good-first-issue seeding** — model adapters and RAG examples are ideal
  first contributions.
- [ ] **Versioning and compatibility policy** — what a gate checkpoint
  guarantees across package versions.
- [ ] **Integration wishlist issue** — a pinned issue where platform and model
  requests get collected and triaged.

## Contributing

If you want to pick something up, open an issue first so we can agree on the
shape — see [CONTRIBUTING.md](CONTRIBUTING.md). Requests for new models or
platforms are welcome as issues even if you can't implement them yourself.
