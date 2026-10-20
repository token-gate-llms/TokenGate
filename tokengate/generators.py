"""Generation backends – local HuggingFace model or OpenAI-compatible API.

Both backends expose the same interface so that compressors (truncation,
lm_search) can be written once and work with either a local model or a
remote API endpoint such as vLLM / TGI / OpenAI.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── attention implementation selection ────────────────────────────────────────

def _best_attn_implementation() -> Optional[str]:
    """Return the best attention implementation available at runtime."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        pass
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return None  # will fall through to eager


# ── abstract interface ────────────────────────────────────────────────────────

class Generator(ABC):
    """Unified generation interface."""

    @property
    @abstractmethod
    def tokenizer(self) -> AutoTokenizer:
        """A tokenizer – always available for token counting / truncation."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in *text* (no special tokens)."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, int]:
        """Generate a single response.

        Returns ``(response_text, prompt_token_count)``.
        """
        ...

    @abstractmethod
    def generate_batch(
        self, prompts: List[str], max_tokens: int = 256,
    ) -> List[Tuple[str, int]]:
        """Batched generation.

        Returns a list of ``(response_text, prompt_token_count)`` in the
        same order as *prompts*.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release GPU / network resources."""
        ...


# ── model-size estimation ─────────────────────────────────────────────────────

def _estimate_model_size_bytes(model_name: str) -> Optional[int]:
    """Estimate total model weight size (bytes) from HuggingFace Hub metadata.

    Returns ``None`` when the size cannot be determined.
    """
    try:
        from huggingface_hub import model_info
        info = model_info(model_name)
        # Prefer the safetensors total reported by the Hub
        if hasattr(info, "safetensors") and info.safetensors:
            total = getattr(info.safetensors, "total", None)
            if total:
                return int(total)
        # Fall back to summing individual weight files
        total = 0
        for sib in info.siblings or []:
            if sib.rfilename.endswith((".safetensors", ".bin")):
                if sib.size:
                    total += sib.size
        if total > 0:
            return total
    except Exception:
        pass
    return None


# ── local HuggingFace generator ──────────────────────────────────────────────

class LocalGenerator(Generator):
    """Load a HuggingFace causal-LM locally.

    Strategy selection (``device`` parameter):

    - ``"auto"`` – **automatically** choose:
        * If model fits on a single GPU **and** multiple GPUs are visible →
          **data parallelism** (one replica per GPU, batches split across
          them, near-linear throughput scaling).
        * If model does **not** fit on a single GPU → **model parallelism**
          (``device_map="auto"`` from HuggingFace accelerate, layers are
          distributed across GPUs).
        * Single GPU → plain single-device inference.
    - ``"dp"`` – force data parallelism (OOM if model doesn't fit on one GPU).
    - ``"mp"`` – force model parallelism (``device_map="auto"``).
    - ``"cuda"``, ``"cuda:0"``, ``"cpu"`` – explicit single device.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype=torch.bfloat16,
    ):
        # -- tokenizer (shared, CPU-only) --
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True,
        )
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._models: List[AutoModelForCausalLM] = []
        self._devices: List[str] = []
        self._model_parallel = False  # True when layers span multiple GPUs

        if device == "auto":
            self._auto_init(model_name, dtype)
        elif device == "dp":
            self._init_data_parallel(model_name, dtype)
        elif device == "mp":
            self._init_model_parallel(model_name, dtype)
        else:
            self._init_single(model_name, dtype, device)

    # ── init strategies ───────────────────────────────────────────────────────

    def _auto_init(self, model_name: str, dtype):
        """Pick the best strategy based on model size vs GPU memory."""
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            self._init_single(model_name, dtype, "cpu")
            return

        model_bytes = _estimate_model_size_bytes(model_name)
        gpu_bytes = torch.cuda.get_device_properties(0).total_memory

        if model_bytes is not None:
            fits_on_one = model_bytes < gpu_bytes * 0.80
            print(f"[LocalGenerator] Model size ~{model_bytes / 1e9:.1f} GB, "
                  f"GPU mem {gpu_bytes / 1e9:.1f} GB, "
                  f"fits_on_one={fits_on_one}")
        else:
            # Can't determine size → assume it fits to avoid unnecessarily
            # slow model-parallel.  Will OOM and fall back if wrong.
            fits_on_one = True
            print("[LocalGenerator] Could not estimate model size, "
                  "assuming it fits on one GPU")

        if fits_on_one and n_gpus > 1:
            try:
                self._init_data_parallel(model_name, dtype)
                return
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Guess was wrong – clean up and fall through to model parallel
                print("[LocalGenerator] OOM during data-parallel init, "
                      "falling back to model parallelism")
                for m in self._models:
                    del m
                self._models.clear()
                self._devices.clear()
                torch.cuda.empty_cache()
        elif fits_on_one:
            self._init_single(model_name, dtype, "cuda:0")
            return

        # Model doesn't fit on one GPU → model parallel
        self._init_model_parallel(model_name, dtype)

    def _init_single(self, model_name: str, dtype, device: str):
        attn_impl = _best_attn_implementation()
        extra = {"attn_implementation": attn_impl} if attn_impl else {}
        print(f"[LocalGenerator] Loading {model_name} on {device} "
              f"(attn={attn_impl or 'eager'}) ...")
        m = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, **extra,
        ).to(device).eval()
        self._models = [m]
        self._devices = [device]
        print(f"[LocalGenerator] Ready (single device: {device}).")

    def _init_data_parallel(self, model_name: str, dtype):
        """One full replica per GPU – best throughput for models that fit."""
        n = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(n)]
        attn_impl = _best_attn_implementation()
        extra = {"attn_implementation": attn_impl} if attn_impl else {}
        for i, dev in enumerate(devices):
            print(f"[LocalGenerator] Loading replica {i + 1}/{n} "
                  f"on {dev} (attn={attn_impl or 'eager'}) ...")
            m = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, **extra,
            ).to(dev).eval()
            self._models.append(m)
            self._devices.append(dev)
        print(f"[LocalGenerator] Ready (data-parallel, "
              f"{n} replicas: {devices}).")

    def _init_model_parallel(self, model_name: str, dtype):
        """Split layers across GPUs via accelerate ``device_map='auto'``."""
        n = torch.cuda.device_count()

        # Reserve ~15 % of each GPU for activations / KV-cache headroom
        max_memory = {}
        for i in range(n):
            total = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = int(total * 0.85)

        attn_impl = _best_attn_implementation()
        extra = {"attn_implementation": attn_impl} if attn_impl else {}
        print(f"[LocalGenerator] Loading {model_name} with model parallelism "
              f"across {n} GPU(s) (attn={attn_impl or 'eager'}) ...")
        m = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto",
            max_memory=max_memory, **extra,
        ).eval()
        self._models = [m]
        self._model_parallel = True
        # Input tensors go on the device of the first parameter
        input_dev = str(next(m.parameters()).device)
        self._devices = [input_dev]
        print(f"[LocalGenerator] Ready (model-parallel, "
              f"input device: {input_dev}).")

    # ── Generator interface ───────────────────────────────────────────────────

    @property
    def tokenizer(self):
        return self._tokenizer

    def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, int]:
        return self.generate_batch([prompt], max_tokens)[0]

    def generate_batch(
        self, prompts: List[str], max_tokens: int = 256,
    ) -> List[Tuple[str, int]]:
        n_replicas = len(self._models)

        # Single model (single GPU or model-parallel) → no threading
        if n_replicas == 1:
            return self._generate_on_device(
                prompts, self._models[0], self._devices[0], max_tokens,
            )

        # Data-parallel: distribute prompts round-robin across replicas
        gpu_prompts: List[List[str]] = [[] for _ in range(n_replicas)]
        gpu_indices: List[List[int]] = [[] for _ in range(n_replicas)]
        for i, p in enumerate(prompts):
            g = i % n_replicas
            gpu_prompts[g].append(p)
            gpu_indices[g].append(i)

        results: List[Optional[Tuple[str, int]]] = [None] * len(prompts)

        def _work(gpu_idx: int):
            if not gpu_prompts[gpu_idx]:
                return
            sub = self._generate_on_device(
                gpu_prompts[gpu_idx],
                self._models[gpu_idx],
                self._devices[gpu_idx],
                max_tokens,
            )
            for j, global_idx in enumerate(gpu_indices[gpu_idx]):
                results[global_idx] = sub[j]

        with ThreadPoolExecutor(max_workers=n_replicas) as pool:
            futs = [pool.submit(_work, g) for g in range(n_replicas)]
            for f in futs:
                f.result()

        return results  # type: ignore[return-value]

    # ── per-device generation ─────────────────────────────────────────────────

    def _generate_on_device(
        self,
        prompts: List[str],
        model: AutoModelForCausalLM,
        device: str,
        max_tokens: int,
    ) -> List[Tuple[str, int]]:
        tokenized = []
        for p in prompts:
            try:
                ids = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=False,
                )
            except TypeError:
                ids = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            # Some tokenizers return BatchEncoding; ensure plain list of ints
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                # input_ids may be nested [[...]] for single example
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
            elif hasattr(ids, "tolist"):
                ids = ids.tolist()
            elif not isinstance(ids, list):
                ids = list(ids)
            tokenized.append(ids)

        max_len = max(len(ids) for ids in tokenized)
        pad_id = self._tokenizer.pad_token_id

        padded, masks, lengths = [], [], []
        for ids in tokenized:
            pad_len = max_len - len(ids)
            padded.append([pad_id] * pad_len + ids)
            masks.append([0] * pad_len + [1] * len(ids))
            lengths.append(len(ids))

        input_ids = torch.LongTensor(padded).to(device)
        attention_mask = torch.LongTensor(masks).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )

        gen_start = input_ids.size(1)
        results: List[Tuple[str, int]] = []
        for i in range(len(prompts)):
            # .cpu() handles model-parallel where output may be on any device
            new_ids = output_ids[i, gen_start:].cpu()
            text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            text = _strip_thinking(text)
            results.append((text, lengths[i]))
        return results

    def unload(self):
        for m in self._models:
            del m
        self._models.clear()
        del self._tokenizer
        torch.cuda.empty_cache()


# ── thinking-tag stripping ─────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (including truncated ones) from model output."""
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)  # handle truncated think blocks
    return text.strip()


# ── API (OpenAI-compatible) generator ─────────────────────────────────────────

class APIGenerator(Generator):
    """Thin wrapper around an OpenAI-compatible endpoint (vLLM, TGI, etc.).

    Calls ``/chat/completions`` by default, or ``/completions`` when
    *use_completions=True*.  Passes ``max_tokens`` and ``temperature``
    directly — no internal capping or post-processing.
    """

    def __init__(
        self,
        model_name: str,
        api_base_url: str,
        api_key: str = "EMPTY",
        tokenizer_name: Optional[str] = None,
        max_workers: int = 16,
        reasoning_effort: Optional[str] = None,
        use_completions: bool = False,
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=api_base_url)
        self._model_name = model_name
        self._max_workers = max_workers
        self._reasoning_effort = reasoning_effort
        self._use_completions = use_completions

        # Always keep a local tokenizer for counting / truncation
        tok = tokenizer_name or model_name
        print(f"[APIGenerator] Loading tokenizer {tok} (for token counting)...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tok, use_fast=True)
        except Exception:
            fallback = "mistralai/Mistral-7B-Instruct-v0.2"
            print(f"[APIGenerator] Tokenizer {tok} not found, "
                  f"falling back to {fallback}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                fallback, use_fast=True,
            )

        mode = "completions" if use_completions else "chat"
        print(f"[APIGenerator] Using model={model_name} at {api_base_url} "
              f"(mode={mode})")

    @property
    def tokenizer(self):
        return self._tokenizer

    # ── single generation ─────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, int]:
        if self._use_completions:
            return self._generate_completions(prompt, max_tokens)
        return self._generate_chat(prompt, max_tokens)

    def _generate_chat(self, prompt: str, max_tokens: int) -> Tuple[str, int]:
        """Generate via /chat/completions."""
        prompt_tokens_est = self.count_tokens(prompt)

        for attempt in range(3):
            try:
                kwargs: dict = dict(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                if self._reasoning_effort is not None:
                    kwargs["extra_body"] = {
                        "reasoning_effort": self._reasoning_effort,
                    }

                resp = self._client.chat.completions.create(**kwargs)

                msg = resp.choices[0].message
                content = msg.content or ""

                # Strip <think>...</think> blocks (e.g. Qwen3 thinking mode)
                content = _strip_thinking(content)

                # Some reasoning models put the answer in reasoning_content
                if not content.strip():
                    reasoning = getattr(msg, "reasoning_content", None)
                    if reasoning:
                        content = _strip_thinking(reasoning)

                n_tok = (
                    resp.usage.prompt_tokens
                    if resp.usage and resp.usage.prompt_tokens
                    else prompt_tokens_est
                )
                return content, n_tok

            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return "", prompt_tokens_est

    def _generate_completions(
        self, prompt: str, max_tokens: int,
    ) -> Tuple[str, int]:
        """Generate via /completions (raw prompt, no chat template)."""
        prompt_tokens_est = self.count_tokens(prompt)

        for attempt in range(3):
            try:
                resp = self._client.completions.create(
                    model=self._model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0,
                )
                text = resp.choices[0].text or ""

                n_tok = (
                    resp.usage.prompt_tokens
                    if resp.usage and resp.usage.prompt_tokens
                    else prompt_tokens_est
                )
                return text.strip(), n_tok

            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return "", prompt_tokens_est

    # ── batch generation ──────────────────────────────────────────────────────

    def generate_batch(
        self, prompts: List[str], max_tokens: int = 256,
    ) -> List[Tuple[str, int]]:
        results: List[Optional[Tuple[str, int]]] = [None] * len(prompts)

        def _call(idx: int):
            return idx, self.generate(prompts[idx], max_tokens)

        workers = min(self._max_workers, len(prompts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_call, i) for i in range(len(prompts))]
            for f in as_completed(futs):
                i, result = f.result()
                results[i] = result

        return results  # type: ignore[return-value]

    def unload(self):
        pass
