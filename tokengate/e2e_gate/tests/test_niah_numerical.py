"""Guard the numerical threshold and teacher-forced cached history."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from tokengate.e2e_gate.bench.benchmark_sanity_check import (
    compare_case,
    compare_logprobs,
    compare_selection,
    forced_history,
    trace,
)


def test_one_bad_vocabulary_entry_fails_even_with_small_average_error():
    expected = torch.full((2, 1000), -5.0)
    actual = expected.clone()
    actual[1, 17] += 0.13  # Declared tolerance is 0.02 + 0.02 * 5 = 0.12.
    result = compare_logprobs(actual, expected)
    assert not result["close"] and result["violating_entries"] == 1
    actual[1, 17] = float("nan")
    result = compare_logprobs(actual, expected)
    assert not result["close"] and result["nonfinite_entries"] == 1


def test_retained_overlap_does_not_replace_exact_positions():
    result = compare_selection(torch.tensor([0, 2, 4]), torch.tensor([0, 3, 4]))
    assert not result["selected_rows_equal"]
    assert result["retained_overlap"] == 0.5


def test_equal_distributions_do_not_qualify_different_retained_positions():
    reference = {
        "id": "example",
        "budget": 2,
        "tokens": [4, 5],
        "logprobs": torch.full((2, 32), -4.0),
        "retained": torch.tensor([0, 2, 3, 5]),
    }
    actual = {**reference, "retained": torch.tensor([0, 1, 3, 5])}
    result = compare_case(actual, reference)
    assert result["close"] and not result["qualified"]
    with pytest.raises(ValueError, match="same prompt and decode history"):
        compare_case({**actual, "tokens": [4, 6]}, reference)


def tiny_model():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def test_trace_uses_supplied_history_even_when_it_contains_eos():
    model = tiny_model()
    case = dict(id="example", prompt=[4, 5, 6], tokens=[2, 7], span=(1, 2), budget=None)
    result = trace(model, None, SimpleNamespace(pad_token_id=0), {}, case)
    with torch.no_grad():
        expected = torch.stack(
            [
                model(torch.tensor([[4, 5, 6]])).logits[0, -1],
                model(torch.tensor([[4, 5, 6, 2]])).logits[0, -1],
            ]
        )
    torch.testing.assert_close(result["logits"], expected)
    assert not model.lm_head._forward_hooks


def test_trace_removes_its_hook_after_failure():
    model = tiny_model()
    with pytest.raises(ValueError, match="invalid forced token"):
        with forced_history(model, [100]):
            model(torch.tensor([[4, 5, 6]]))
    assert not model.lm_head._forward_hooks


def test_serving_request_uses_current_span_contract():
    from tokengate.e2e_gate.bench.runtime import request_options

    row = {"article_start": 3, "article_end": 11}
    assert request_options(row, 2) == {
        "token_gate_version": 1,
        "token_gate_span_start": 3,
        "token_gate_span_end": 11,
        "token_gate_keep_tokens": 2,
    }
    assert request_options(row, None)["token_gate_keep_tokens"] == 8


def test_decode_qualification_keeps_exact_logit_requirement():
    from tokengate.e2e_gate.bench.benchmark_sanity_check import compare_records

    record = {"id": "sample", "token_ids": [1], "logits": torch.ones(1, 4)}
    compare_records([record], [record])
    changed = {**record, "logits": record["logits"].clone()}
    changed["logits"][0, 0] += 1e-6
    with pytest.raises(AssertionError):
        compare_records([record], [changed])
    nonfinite = {**record, "logits": torch.full((1, 4), float("nan"))}
    with pytest.raises(AssertionError):
        compare_records([nonfinite], [nonfinite])
