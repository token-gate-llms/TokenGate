"""Run the original TokenGate implementation and vLLM on the same NIAH inputs.

Both runners expose serial prefill and token-delivery timing. Cache inspection
reports occupied content separately from backing storage, outside the timed
region. Checkpoint hashes and protected prompt tokens are checked before use.
"""

import functools
import os
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from tokengate.e2e_gate.artifacts import file_hash
from tokengate.e2e_gate.decode.gate_decode import TokenDelivery, generate_batch
from tokengate.e2e_gate.gate_module.gate_forward import prefill_batch_for_decode
from tokengate.inference import load_gate_model


def validate_retained_positions(
    positions: list[int], row: dict, budget: int | None
) -> None:
    """Require sorted, unique retained positions and every protected token."""
    start, end, length = row["article_start"], row["article_end"], row["ctx_len"]
    retained = end - start if budget is None else min(budget, end - start)
    if positions != sorted(set(positions)) or any(
        index < 0 or index >= length for index in positions
    ):
        raise ValueError("Retained positions must be unique ordered prompt indices")
    protected = set(range(start)) | set(range(end, length))
    if not protected.issubset(positions) or len(positions) != len(protected) + retained:
        raise ValueError("Retained positions violate the budget or protected span")


def serving_settings(
    model_path: Path,
    config: dict,
    rows: list[dict],
    *,
    gate: bool,
    execution: str,
    max_new_tokens: int | None = None,
) -> dict:
    """Share serial engine settings across evaluation, timing and numerical traces."""
    if execution not in {"eager", "optimized"}:
        raise ValueError(f"Unknown execution mode: {execution}")
    if gate and execution != "eager":
        raise ValueError("TokenGate currently requires eager execution")
    limit = max(row["ctx_len"] for row in rows) + (
        config["max_gen_tokens"] if max_new_tokens is None else max_new_tokens
    )
    return {
        "model": str(model_path),
        "dtype": "bfloat16",
        "generation_config": "vllm",
        "seed": config["seed"],
        "enforce_eager": execution == "eager",
        "max_model_len": limit,
        "max_num_batched_tokens": limit,
        "max_num_scheduled_tokens": limit,
        "long_prefill_token_threshold": 0,
        "max_num_seqs": 1,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "kv_cache_memory_bytes": 4 * 1024**3,
    }


def describe_backends(worker):
    return [
        [group.backend.get_name() for group in groups]
        for groups in worker.model_runner.attn_groups
    ]


def prefill_selection(worker):
    """Copy the real prefill layout before the next engine step replaces it."""
    batch = worker.model_runner.model_state.history.batch
    if batch is None or not batch.is_prefill:
        raise RuntimeError("Expected an active TokenGate prefill batch")
    return {
        "retained": batch.retained_rows.detach().cpu(),
        "selected_scores": batch.prompt_scores.detach().cpu(),
    }


def request_options(row: dict, budget: int | None) -> dict:
    return {
        "token_gate_version": 1,
        "token_gate_span_start": row["article_start"],
        "token_gate_span_end": row["article_end"],
        "token_gate_keep_tokens": budget or row["article_end"] - row["article_start"],
    }


def load_checkpoint(config: dict, evidence: dict):
    """Validate a selected run, then use the public inference loader's path."""
    for kind in ("adapter", "gate"):
        path = evidence[f"{kind}_path"]
        if path is not None and file_hash(Path(path)) != evidence[f"{kind}_sha256"]:
            raise ValueError(f"Selected {kind} changed")
    expected = {
        "step": evidence["selection"]["step"],
        "swa": evidence["selection"]["variant"] == "swa",
        "run_id": config["run_id"],
        "params_sha256": evidence["params_sha256"],
        "gate_layer": config["gate_layer"],
        "base_model": config["base_model"],
    }
    model, gate, _ = load_gate_model(
        config["base_model"],
        evidence["gate_path"],
        config["gate_layer"],
        gate_attn=config["attn_backend"],
        adapter_path=evidence["adapter_path"],
        adapter_config=config,
        expected_metadata=expected,
        warmup=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"], local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, gate, tokenizer


def storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    """Count backing allocations once, including unused portions of tensor views."""
    storages = {
        (tensor.device, tensor.untyped_storage().data_ptr()): tensor.untyped_storage()
        for tensor in tensors
    }
    return sum(storage.nbytes() for storage in storages.values())


def research_cache(cache, prompt_length: int, retained_length: int, gate_layer) -> dict:
    """Inspect real attention tensors and recurrent states after research prefill."""
    attention, recurrent = [], []
    for index, layer in enumerate(cache.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            states = [
                state
                for group in (layer.conv_states, layer.recurrent_states)
                for state in group.values()
            ]
            if not states or any(
                not isinstance(state, torch.Tensor) for state in states
            ):
                raise RuntimeError(f"Incomplete recurrent cache at layer {index}")
            recurrent.extend(states)
            continue
        keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
        if keys is not None:
            expected = (
                prompt_length
                if gate_layer is None or index <= gate_layer
                else retained_length
            )
            if keys.shape[2] != expected or values.shape[2] != expected:
                raise RuntimeError(f"Unexpected research cache length at layer {index}")
            attention.extend((keys, values))
        else:
            raise RuntimeError(f"Unrecognized research cache at layer {index}")
    return {
        "attention_kv_content_bytes": sum(
            t.numel() * t.element_size() for t in attention
        ),
        "recurrent_state_content_bytes": sum(
            t.numel() * t.element_size() for t in recurrent
        ),
        "attention_page_bytes": None,
        "recurrent_page_bytes": None,
        "cache_storage_bytes": storage_bytes(attention + recurrent),
    }


def serving_cache(worker, *, prompt_length: int, gate: bool) -> dict:
    """Read the actual worker cache groups and their single-request block counts."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        UniformTypeKVCacheSpecs,
    )

    runner = worker.model_runner
    if runner.max_num_reqs != 1:
        raise RuntimeError("Prefill measurements require a single request slot")
    positions = list(range(prompt_length))
    if gate:
        batch = runner.model_state.history.batch
        if batch is None or not batch.is_prefill:
            raise RuntimeError("Expected the just-completed TokenGate prefill")
        positions = batch.retained_rows.cpu().tolist()
    totals = {
        "attention_kv_content_bytes": 0,
        "recurrent_state_content_bytes": 0,
        "attention_page_bytes": 0,
        "recurrent_page_bytes": 0,
        "cache_storage_bytes": storage_bytes(runner.kv_caches),
    }
    groups = []
    for index, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        specs = (
            spec.kv_cache_specs
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else dict.fromkeys(group.layer_names, spec)
        )
        blocks = int(runner.block_tables.num_blocks.np[index, 0])
        length = len(positions) if group.uses_compact_sequence else prompt_length
        for layer_spec in specs.values():
            if isinstance(layer_spec, FullAttentionSpec):
                if layer_spec.tokens_per_state != 1:
                    raise ValueError("Benchmark expects uncompressed attention KV")
                expected = (length + layer_spec.block_size - 1) // layer_spec.block_size
                if blocks != expected:
                    raise RuntimeError(
                        "Allocated attention blocks differ from the prompt layout"
                    )
                totals["attention_kv_content_bytes"] += (
                    length * layer_spec.num_heads * layer_spec.state_content_size_bytes
                )
                totals["attention_page_bytes"] += blocks * layer_spec.page_size_bytes
            elif isinstance(layer_spec, MambaSpec):
                if blocks != 1 or layer_spec.mamba_cache_mode != "none":
                    raise ValueError(
                        "Benchmark expects one recurrent-state block per layer"
                    )
                totals["recurrent_state_content_bytes"] += (
                    layer_spec.state_content_size_bytes
                )
                totals["recurrent_page_bytes"] += blocks * layer_spec.page_size_bytes
            else:
                raise TypeError(
                    f"Unsupported benchmark cache type: {type(layer_spec).__name__}"
                )
        groups.append({"layers": group.layer_names, "blocks": blocks, "tokens": length})
    return {
        **totals,
        "groups": groups,
        "kept_positions": positions,
        "worker_pid": os.getpid(),
    }


class ResearchRunner:
    """Reuse the research prefill and generation paths for serial inference."""

    def __init__(self, config: dict, checkpoint: dict):
        self.model, self.gate, self.tokenizer = load_checkpoint(config, checkpoint)
        self.layer = config["gate_layer"]
        self.settings = {
            "attention_backend": config["attn_backend"],
            "adapter": "factored",
        }

    @torch.no_grad()
    def prefill(self, row: dict, budget: int | None) -> tuple[int, tuple]:
        prompt = row["input_ids"][: row["ctx_len"]]
        if self.gate is None:
            inputs = torch.tensor([prompt], device=self.model.device)
            output = self.model(
                inputs,
                attention_mask=torch.ones_like(inputs),
                use_cache=True,
                logits_to_keep=1,
            )
            token = int(output.logits[0, -1].argmax().cpu())
            return token, (output.past_key_values, None)
        tokens, cache, positions, counts, _ = prefill_batch_for_decode(
            self.model,
            self.gate,
            [prompt],
            [row["article_start"]],
            [row["article_end"]],
            self.layer,
            budget if budget is not None else row["article_end"] - row["article_start"],
        )
        return int(tokens[0, 0].cpu()), (cache, positions[0, : counts[0]])

    @torch.no_grad()
    def decode(self, row: dict, budget: int | None, cap: int) -> dict:
        delivery = TokenDelivery()
        prompt = row["input_ids"][: row["ctx_len"]]
        start = time.perf_counter()
        generated = generate_batch(
            self.model,
            self.gate,
            self.tokenizer,
            [prompt],
            [row["article_start"]],
            [row["article_end"]],
            self.layer,
            budget if budget is not None else row["article_end"] - row["article_start"],
            cap,
            token_callback=delivery.put,
        )
        tokens = generated.token_ids[0]
        if tokens != delivery.token_ids:
            raise RuntimeError("CPU delivery differs from generated tokens")
        return delivery.result(start, self.tokenizer.eos_token_id, cap)

    def inspect(self, context: tuple, row: dict) -> dict:
        cache, positions = context
        if positions is None:
            positions = list(range(row["ctx_len"]))
        elif isinstance(positions, torch.Tensor):
            positions = positions.cpu().tolist()
        return {
            **research_cache(cache, row["ctx_len"], len(positions), self.layer),
            "kept_positions": positions,
        }


class ServingRunner:
    """Issue serial requests through the renderer and public vLLM engine API."""

    def __init__(self, settings: dict, gate: bool):
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        from vllm import LLM

        self.gate = gate
        self.settings = dict(settings)
        self.llm = LLM(**self.settings)
        self.eos_token_id = self.llm.get_tokenizer().eos_token_id
        self.settings["resolved_backends"] = self.llm.collective_rpc(describe_backends)[
            0
        ]
        self.settings["resolved_compilation"] = str(
            self.llm.llm_engine.vllm_config.compilation_config
        )

    def submit(self, row: dict, budget: int | None, cap: int, **sampling) -> None:
        from vllm import SamplingParams

        engine = self.llm.llm_engine
        if engine.has_unfinished_requests():
            raise RuntimeError("Serial generation requires an idle engine")
        (prompt,) = self.llm.renderer.render_cmpl(
            [{"prompt_token_ids": row["input_ids"][: row["ctx_len"]]}]
        )
        engine.add_request(
            "serial",
            prompt,
            SamplingParams(
                temperature=0,
                max_tokens=cap,
                stop_token_ids=[]
                if sampling.get("ignore_eos")
                else [self.eos_token_id],
                detokenize=False,
                extra_args=request_options(row, budget) if self.gate else None,
                **sampling,
            ),
        )

    def generate(self, row: dict, budget: int | None, cap: int, **sampling):
        """Yield each CPU completion; inspect prefill selection after the first.

        Callers must exhaust the iterator before issuing another request. Timing
        callers consume it without performing cache inspection or scoring.
        """
        self.submit(row, budget, cap, **sampling)
        engine = self.llm.llm_engine
        first_token = None
        for count in range(1, cap + 1):
            results = engine.step()
            if len(results) != 1 or len(results[0].outputs) != 1:
                raise RuntimeError("Expected one serial completion")
            result = results[0]
            completion = result.outputs[0]
            if result.request_id != "serial" or len(completion.token_ids) != count:
                raise RuntimeError("Expected one token per step for the serial request")
            if count == 1:
                first_token = completion.token_ids[0]
            elif completion.token_ids[0] != first_token:
                raise RuntimeError("Serial decode changed the prefill token")
            if result.finished == engine.has_unfinished_requests():
                raise RuntimeError("Serial completion and engine state disagree")
            yield completion
            if result.finished:
                return
        raise RuntimeError("Serial generation exceeded its output cap")

    def prefill(self, row: dict, budget: int | None) -> tuple[int, None]:
        (completion,) = self.generate(row, budget, 1)
        (token,) = completion.token_ids
        return token, None

    def decode(self, row: dict, budget: int | None, cap: int) -> dict:
        """Time first-to-last CPU token delivery, including the terminal EOS.

        No cache inspection or quality scoring occurs between timing boundaries.
        A response ending at its first token has no decode interval.
        """
        start = time.perf_counter()
        responses = self.generate(row, budget, cap)
        completion = next(responses)
        torch.cuda.synchronize()
        first_token_time = time.perf_counter()
        steps = 0
        for completion in responses:
            steps += 1
        torch.cuda.synchronize()
        last_token_time = time.perf_counter() if steps else first_token_time
        elapsed = last_token_time - first_token_time
        tokens = list(completion.token_ids)
        return {
            "token_ids": tokens,
            "generated_tokens": len(tokens),
            "decode_tokens": steps,
            "generation_seconds": last_token_time - start,
            "decode_seconds": elapsed if steps else None,
            "decode_seconds_per_token": elapsed / steps if steps else None,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
        }

    def inspect(self, context: None, row: dict) -> dict:
        result = self.llm.collective_rpc(
            functools.partial(
                serving_cache, prompt_length=row["ctx_len"], gate=self.gate
            )
        )[0]
        if result.pop("worker_pid") != os.getpid():
            raise RuntimeError("Peak memory accounting requires an in-process worker")
        return result
