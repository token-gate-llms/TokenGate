"""Check first-token delivery, EOS truncation, and timing boundaries."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokengate.e2e_gate.decode import gate_decode


@pytest.fixture
def gate_generation(monkeypatch):
    """Keep the shared decode loop real; replace model math with fixed tokens."""
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(gate_decode.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(gate_decode, "_sync", lambda: None)

    class Head(nn.Module):
        def __init__(self, batches):
            super().__init__()
            self.batches = iter(batches)

        def forward(self, hidden):
            clock.now += 2
            tokens = torch.tensor(next(self.batches))
            return torch.nn.functional.one_hot(tokens, 16).float().unsqueeze(1)

    class FormState:
        def __init__(self, *args):
            clock.now += 5  # Decode setup must be inside the delivery interval.

        def step(self, *args):
            pass

        def finish(self):
            clock.now += 11  # Cleanup must be outside the delivery interval.

    monkeypatch.setattr(gate_decode, "FormDecodeState", FormState)

    def make(first, batches):
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList()
        model.model.embed_tokens = nn.Embedding(16, 4)
        model.model.norm = nn.Identity()
        model.model.rotary_emb = lambda *args: None
        model.lm_head = Head(batches)
        model.device = torch.device("cpu")

        def prefill(*args, **kwargs):
            clock.now += 3
            batch = len(first)
            return (
                torch.tensor(first).unsqueeze(1),
                SimpleNamespace(layers=[]),
                torch.tensor([[0, 2]] * batch),
                [2] * batch,
                [1] * batch,
            )

        def decode(tokens, **kwargs):
            clock.now += 13  # Detokenization must not be included either.
            return " ".join(str(token) for token in tokens)

        monkeypatch.setattr(gate_decode, "prefill_batch_for_decode", prefill)
        tokenizer = SimpleNamespace(eos_token_id=9, pad_token_id=0, decode=decode)
        return model, tokenizer, clock

    return make


def test_shared_decode_preserves_answers_and_first_eos(gate_generation):
    def generate(callback=None):
        model, tokenizer, _ = gate_generation([3, 2], [[9, 4], [7, 9]])
        return gate_decode.generate_batch(
            model,
            object(),
            tokenizer,
            [[1, 2, 3]] * 2,
            [1, 1],
            [2, 2],
            0,
            1,
            6,
            token_callback=callback,
        )

    expected = generate()
    delivered = []
    result = generate(lambda tokens: delivered.append(tokens.cpu().tolist()))
    assert delivered == [[[3], [2]], [[9], [4]], [[7], [9]]]
    assert result.token_ids == expected.token_ids == [[3, 9], [2, 4, 9]]
    assert result.answers == expected.answers == ["3", "2 4"]
    assert result.n_gen == expected.n_gen == [1, 2]
    assert result.steps == expected.steps == 2
    torch.testing.assert_close(result.kept_pos, expected.kept_pos)


def test_delivery_rejects_batching_and_invalid_termination():
    delivery = gate_decode.TokenDelivery()
    with pytest.raises(ValueError, match="one token"):
        delivery.put(torch.tensor([1, 2]))
    with pytest.raises(RuntimeError, match="no tokens"):
        delivery.result(0, 9, 4)
    delivery.put(torch.tensor([1]))
    with pytest.raises(RuntimeError, match="before EOS"):
        delivery.result(0, 9, 4)
    delivery.put(torch.tensor([9]))
    delivery.put(torch.tensor([2]))
    with pytest.raises(RuntimeError, match="first EOS"):
        delivery.result(0, 9, 4)


@pytest.mark.parametrize(
    "first,batches,cap,tokens,elapsed,reason",
    [
        ([3], [[9]], 8, [3, 9], 7, "stop"),
        ([9], [], 8, [9], 0, "stop"),
        ([3], [], 1, [3], 0, "length"),
        ([3], [[4], [5]], 3, [3, 4, 5], 9, "length"),
    ],
)
def test_delivery_timestamps_exclude_cleanup(
    gate_generation, first, batches, cap, tokens, elapsed, reason
):
    model, tokenizer, clock = gate_generation(first, batches)
    delivery = gate_decode.TokenDelivery()
    result = gate_decode.generate_batch(
        model,
        object(),
        tokenizer,
        [[1, 2, 3]],
        [1],
        [2],
        0,
        1,
        cap,
        token_callback=delivery.put,
    )
    metrics = delivery.result(0, 9, cap)
    assert result.token_ids[0] == delivery.token_ids == tokens
    assert delivery.first_time == 3
    assert delivery.last_time == 3 + elapsed
    assert delivery.last_time < clock.now
    assert metrics["finish_reason"] == reason
