"""Experimental TokenGate bridge for text-only JEV decisions.

JEV reads a decision head at the final prompt token. TokenGate's prefill path
normally sends that hidden state to a language-model head for the first decoded
token; this bridge sends it to JEV's decision head instead. The caller must
provide a gate trained for the *same* JEV backbone and adapter. A gate trained
on a different Qwen size or task is not interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tokengate.e2e_gate.gate_module.gate_forward import (
    build_gate_branch,
    compute_prunable_span,
    forward_with_gate_batched,
    prefill_batch_for_decode,
)
from tokengate.e2e_gate.gate_module.e2e_gate_module import E2EGate
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    get_full_attn_layer_indices,
    patch_model_with_flex_attention,
    set_attn_scale_for_layers,
    set_gate_bias_for_layers,
    set_ssmax_ratio_for_layers,
)


@dataclass
class DecisionResult:
    answers: dict[str, dict[str, Any]]
    scores: dict[str, torch.Tensor]
    kept_state_tokens: int
    state_tokens: int
    kept_positions: list[int]


class _DecisionPrefillModel(nn.Module):
    """Expose JEV's backbone/head with the interface of TokenGate's prefill."""

    def __init__(self, jev_model: nn.Module):
        super().__init__()
        self.model = jev_model.model
        self.config = jev_model.config.text_config
        self.proj = jev_model.proj
        indices = getattr(jev_model, "_full_attn_layer_indices", None)
        if indices is None and hasattr(self.model, "layers"):
            indices = get_full_attn_layer_indices(jev_model)
        self._full_attn_layer_indices = indices

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.to(self.proj.weight.dtype))


def _prepare_one(model, tokenizer, request: dict[str, Any]):
    """Prepare a single text question and locate its prunable state span."""
    if request.get("assets"):
        raise ValueError("This prototype supports text-only JEV requests.")
    prompts = model.decision_config.prepare_inputs(request, tokenizer)
    if len(prompts) != 1 or len(prompts[0][1]) != 1:
        raise ValueError("This prototype expects exactly one decision question.")
    inputs, questions = prompts[0]
    if set(inputs) - {"input_ids", "attention_mask", "marks"}:
        raise ValueError("This prototype supports token IDs only; multimodal inputs are unsupported.")
    ids = inputs["input_ids"][0].tolist()
    marks = inputs["marks"].reshape(-1).tolist()
    if marks != [len(ids)]:
        raise ValueError("The decision readout must be at the end of the prompt.")

    empty = {**request, "state": ""}
    empty_prompts = model.decision_config.prepare_inputs(empty, tokenizer)
    empty_ids = empty_prompts[0][0]["input_ids"][0].tolist()
    start, end = compute_prunable_span(ids, empty_ids)
    if end <= start:
        raise ValueError("No prunable state tokens found in the decision prompt.")
    return inputs, questions[0], ids, start, end


def _clear_gate_attention(model) -> None:
    """Restore ordinary full prefill after a prior gated decision."""
    if not hasattr(model.model, "layers"):
        return
    end = len(model.model.layers)
    set_gate_bias_for_layers(model, 0, end, None)
    set_ssmax_ratio_for_layers(model, 0, end, None)
    set_attn_scale_for_layers(model, 0, end, None)


def _option_scores(model, inputs, question, raw_scores: torch.Tensor) -> torch.Tensor:
    offset = model.config.offsets[question.type]
    n = len(question.options)
    if offset + n > raw_scores.numel():
        raise ValueError(f"{n} {question.type} options exceed the decision head width.")
    scores = raw_scores[offset : offset + n].float()
    if model.config.softcap is not None:
        cap = model.config.softcap
        scores = cap * torch.tanh(scores / cap)
    marks = inputs["marks"].reshape(-1)
    tail = inputs["input_ids"][0, int(marks[-1]) :]
    return model.decision_config.logits_processors[question.type](tail, scores)


def _answer(model, inputs, question, raw_scores: torch.Tensor) -> tuple[dict, torch.Tensor]:
    scores = _option_scores(model, inputs, question, raw_scores)
    return question.answer(scores.softmax(-1).tolist()), scores


@torch.no_grad()
def standard_decide(model, tokenizer, request: dict[str, Any]) -> DecisionResult:
    """The JEV full-prefill reference, using its own public ``decide`` API."""
    inputs, question, ids, start, end = _prepare_one(model, tokenizer, request)
    _clear_gate_attention(model)
    scores = model.decide(**inputs, decision_questions=[question])[0]
    answer = question.answer(scores.softmax(-1).tolist())
    return DecisionResult(
        answers={question.name: answer}, scores={question.name: scores},
        kept_state_tokens=end - start, state_tokens=end - start,
        kept_positions=list(range(len(ids))),
    )


@torch.no_grad()
def input_pruned_decide(
    model,
    tokenizer,
    request: dict[str, Any],
    *,
    token_budget: int,
    method: str = "prefix",
    seed: int = 0,
) -> DecisionResult:
    """Cheap comparison: prune state IDs before the ordinary JEV forward.

    ``prefix``, ``middle``, or ``suffix`` keeps one fixed part of the state; ``random`` samples
    state tokens without replacement and restores their original order. None scores
    token importance or applies TokenGate's Phase-2 attention bias.
    """
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    inputs, question, ids, start, end = _prepare_one(model, tokenizer, request)
    state_len = end - start
    k = min(token_budget, state_len)
    if method == "prefix":
        local = list(range(k))
    elif method == "middle":
        start_at = (state_len - k) // 2
        local = list(range(start_at, start_at + k))
    elif method == "suffix":
        local = list(range(state_len - k, state_len))
    elif method == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        local = torch.randperm(state_len, generator=generator)[:k].sort().values.tolist()
    else:
        raise ValueError("method must be 'prefix', 'middle', 'suffix', or 'random'")
    positions = list(range(start)) + [start + i for i in local] + list(range(end, len(ids)))
    shortened = inputs["input_ids"][:, positions]
    new_inputs = {"input_ids": shortened, "marks": torch.tensor([len(positions)])}
    _clear_gate_attention(model)
    scores = model.decide(**new_inputs, decision_questions=[question])[0]
    return DecisionResult(
        answers={question.name: question.answer(scores.softmax(-1).tolist())},
        scores={question.name: scores}, kept_state_tokens=k,
        state_tokens=state_len, kept_positions=positions,
    )


@torch.no_grad()
def gated_decide(
    model,
    gate,
    tokenizer,
    request: dict[str, Any],
    *,
    gate_layer: int,
    token_budget: int,
) -> DecisionResult:
    """Run a trained TokenGate on the state and score JEV's decision head.

    ``model`` must have its Qwen3.5 full-attention layers patched with
    ``patch_model_with_flex_attention``. Keep the JEV LoRA adapter active.
    The gate must be trained against this same model and decision task.
    """
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    inputs, question, ids, start, end = _prepare_one(model, tokenizer, request)
    prefill_model = _DecisionPrefillModel(model)
    logits, _, positions, n_kept, k_used = prefill_batch_for_decode(
        prefill_model, gate, [ids], [start], [end], gate_layer, token_budget,
        return_logits=True,
    )
    answer, scores = _answer(model, inputs, question, logits[0])
    return DecisionResult(
        answers={question.name: answer}, scores={question.name: scores},
        kept_state_tokens=k_used[0], state_tokens=end - start,
        kept_positions=positions[0, : n_kept[0]].tolist(),
    )


def create_jev_gate(model, *, gate_layer: int = 1) -> E2EGate:
    """Freeze a loaded JEV model, patch attention, and initialize its gate.

    Load the JEV LoRA adapter *before* calling this function. This initializes
    a new, untrained gate; its decisions need not match full prefill yet.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("TokenGate's JEV training path requires CUDA.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    full_layers = get_full_attn_layer_indices(model)
    if gate_layer < 0 or not any(i > gate_layer for i in full_layers):
        raise ValueError("gate_layer must precede at least one full-attention layer")
    patch_model_with_flex_attention(model)
    config = model.config.text_config
    gate = E2EGate(
        d_model=config.hidden_size,
        n_heads=config.num_attention_heads,
        n_layers_bias=sum(i > gate_layer for i in full_layers),
    ).to(next(model.parameters()).device)
    gate.set_qk_norm_scales(model, gate_layer)
    build_gate_branch(model, gate, gate_layer)
    return gate


def train_jev_gate_step(
    model,
    gate: E2EGate,
    tokenizer,
    request: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    gate_layer: int,
    token_budget: int,
) -> float:
    """Distill one full-prefill JEV decision into the gate at a fixed budget.

    The backbone and JEV adapter stay frozen. Only gate parameters are
    optimized. The caller should train on many separate requests and validate
    on held-out decisions and lengths before drawing speed/accuracy conclusions.
    """
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    inputs, question, ids, start, end = _prepare_one(model, tokenizer, request)
    if token_budget >= end - start:
        raise ValueError("token_budget must prune some state tokens during gate training")
    model.eval()
    gate.train()
    optimizer.zero_grad(set_to_none=True)
    _clear_gate_attention(model)
    with torch.no_grad():
        teacher = model.decide(**inputs, decision_questions=[question])[0].float().detach()
    prefill_model = _DecisionPrefillModel(model)
    logits, loss_mask, _ = forward_with_gate_batched(
        prefill_model, gate,
        [{"input_ids": ids, "article_start": start, "article_end": end,
          "ctx_len": len(ids), "n_ans": 1, "K": token_budget}],
        gate_layer, frozen_backbone=True,
    )
    selected = logits[0, loss_mask[0]]
    if selected.shape[0] != 1:
        raise RuntimeError("Expected exactly one decision readout from the gated prefill")
    student = _option_scores(model, inputs, question, selected[0])
    loss = F.kl_div(F.log_softmax(student, dim=-1), teacher.softmax(-1), reduction="sum")
    loss.backward()
    optimizer.step()
    return float(loss.detach())
