"""Check token delivery, EOS handling, and research benchmark boundaries."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from tokengate.e2e_gate.bench import runtime as benchmark
from tokengate.e2e_gate.decode import gate_decode


@pytest.fixture
def gate_generation(monkeypatch):
    """Keep the shared decode loop real; replace model math with fixed tokens."""
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock.now)
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


@pytest.mark.parametrize(
    "first,batches,cap,tokens,elapsed,reason",
    [
        ([3], [[9]], 8, [3, 9], 7, "stop"),
        ([9], [], 8, [9], None, "stop"),
        ([3], [], 1, [3], None, "length"),
        ([3], [[4], [5]], 3, [3, 4, 5], 9, "length"),
    ],
)
def test_research_timing_includes_setup_and_excludes_cleanup(
    gate_generation, first, batches, cap, tokens, elapsed, reason
):
    model, tokenizer, clock = gate_generation(first, batches)
    runner = benchmark.ResearchRunner.__new__(benchmark.ResearchRunner)
    runner.model, runner.tokenizer = model, tokenizer
    runner.gate, runner.layer = object(), 0
    row = {"input_ids": [1, 2, 3], "ctx_len": 3, "article_start": 1, "article_end": 2}
    result = runner.decode(row, 1, cap)
    assert result["token_ids"] == tokens
    assert result["decode_seconds"] == elapsed
    assert result["decode_tokens"] == len(tokens) - 1
    assert result["generation_seconds"] == 3 + (elapsed or 0)
    assert result["generation_seconds"] < clock.now
    assert result["finish_reason"] == reason
    assert result["stop_reason"] == (9 if reason == "stop" else None)
    assert result["decode_seconds_per_token"] == (
        elapsed / (len(tokens) - 1) if elapsed else None
    )


def test_sft_streamer_preserves_hugging_face_generation():
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            eos_token_id=None,
            pad_token_id=0,
        )
    ).eval()
    runner = benchmark.ResearchRunner.__new__(benchmark.ResearchRunner)
    runner.model, runner.gate, runner.layer = model, None, None
    runner.tokenizer = SimpleNamespace(
        eos_token_id=None, pad_token_id=0, decode=lambda ids, **kw: str(ids)
    )
    row = {"input_ids": [1, 2, 3], "ctx_len": 3, "article_start": 1, "article_end": 2}
    inputs = torch.tensor([[1, 2, 3]])
    expected = model.generate(
        inputs,
        attention_mask=torch.ones_like(inputs),
        do_sample=False,
        max_new_tokens=4,
        eos_token_id=None,
        pad_token_id=0,
    )[0, 3:].tolist()
    result = runner.decode(row, None, 4)
    assert result["token_ids"] == expected
    assert result["generated_tokens"] == 4
    assert result["decode_tokens"] == 3
    assert 0 < result["decode_seconds"] < result["generation_seconds"]
    assert result["finish_reason"] == "length"


@pytest.mark.parametrize("tokens", [[9], [3, 4, 9]])
def test_serving_timing_uses_the_same_delivery_boundaries(monkeypatch, tokens):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: None)

    class Engine:
        def __init__(self):
            self.count = 0

        def has_unfinished_requests(self):
            return self.count < len(tokens)

        def step(self):
            clock.now += 3 if self.count == 0 else 2
            self.count += 1
            completion = SimpleNamespace(
                token_ids=tokens[: self.count], finish_reason="stop", stop_reason=9
            )
            return [
                SimpleNamespace(
                    request_id="serial",
                    finished=self.count == len(tokens),
                    outputs=[completion],
                )
            ]

    runner = benchmark.ServingRunner.__new__(benchmark.ServingRunner)
    runner.submit = lambda *args: None
    runner.llm = SimpleNamespace(llm_engine=Engine())
    result = runner.decode({}, None, 8)
    steps = len(tokens) - 1
    assert result["token_ids"] == tokens
    assert result["generation_seconds"] == 3 + 2 * steps
    assert result["decode_seconds"] == (2 * steps if steps else None)


@pytest.mark.parametrize(
    "outputs,unfinished,message",
    [
        ([], False, "one serial completion"),
        (
            [
                SimpleNamespace(
                    request_id="other", outputs=[SimpleNamespace(token_ids=[3])]
                )
            ],
            False,
            "serial request",
        ),
        (
            [
                SimpleNamespace(
                    request_id="serial", outputs=[SimpleNamespace(token_ids=[3, 4])]
                )
            ],
            False,
            "one token per step",
        ),
        (
            [
                SimpleNamespace(
                    request_id="serial",
                    finished=True,
                    outputs=[SimpleNamespace(token_ids=[3])],
                )
            ],
            True,
            "engine state disagree",
        ),
        (
            [
                SimpleNamespace(
                    request_id="serial",
                    finished=False,
                    outputs=[SimpleNamespace(token_ids=[3])],
                )
            ],
            True,
            "output cap",
        ),
    ],
)
def test_serial_serving_rejects_incomplete_or_mixed_responses(
    outputs, unfinished, message
):
    runner = benchmark.ServingRunner.__new__(benchmark.ServingRunner)
    runner.submit = lambda *args: None
    runner.llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            step=lambda: outputs,
            has_unfinished_requests=lambda: unfinished,
        )
    )
    with pytest.raises(RuntimeError, match=message):
        list(runner.generate({}, None, 1))


@pytest.mark.parametrize("gated", [False, True])
def test_quality_reads_selection_before_decode_and_handles_first_token_eos(gated):
    from benchmarks.run_benchmark import evaluate_serving_budget

    events = []
    row = dict(
        id="example",
        target="needle",
        partition="validation",
        city="test",
        depth_requested=50,
        depth_actual=50,
        article_start=1,
        article_end=2,
        input_ids=[1, 2, 3],
        ctx_len=3,
    )

    def responses(*args):
        events.append("prefill")
        yield SimpleNamespace(token_ids=[9], finish_reason="stop", stop_reason=9)
        events.append("finished")

    def selection(*args):
        assert events == ["prefill"]
        return [{"retained": torch.arange(3)}]

    runner = SimpleNamespace(
        gate=gated,
        generate=responses,
        llm=SimpleNamespace(
            get_tokenizer=lambda: SimpleNamespace(decode=lambda *a, **k: "needle"),
            collective_rpc=selection,
        ),
    )
    (answer,) = evaluate_serving_budget(runner, [row], budget=None, cap=4)
    assert events == ["prefill", "finished"]
    assert answer["token_ids"] == [9] and answer["accuracy"] == 1
    assert answer["kept_positions"] == [0, 1, 2]


@pytest.mark.parametrize("gated", [False, True])
def test_research_quality_saves_raw_tokens_with_eos(monkeypatch, tmp_path, gated):
    import json

    from tokengate.e2e_gate.train import train_e2e_gate as evaluate

    tokens = [3, 9]
    model = SimpleNamespace(
        device="cpu",
        eval=lambda: None,
        generate=lambda *args, **kwargs: torch.tensor([[1, 2, 3, 3, 9]]),
    )
    gate = SimpleNamespace(eval=lambda: None) if gated else None
    monkeypatch.setattr(
        evaluate,
        "generate_batch",
        lambda *args: SimpleNamespace(
            answers=["needle"],
            token_ids=[tokens],
            kept_pos=torch.tensor([[0, 1, 2]]),
            n_kepts=[3],
        ),
    )
    monkeypatch.setattr(evaluate, "answer_loss", lambda *args: 0.0)
    row = {
        "id": "validation-0",
        "input_ids": [1, 2, 3],
        "ctx_len": 3,
        "article_start": 1,
        "article_end": 2,
        "target": "needle",
    }
    tokenizer = SimpleNamespace(
        eos_token_id=9, pad_token_id=0, decode=lambda *args, **kwargs: "needle"
    )
    evaluate.evaluate_reproduction(
        model,
        gate,
        tokenizer,
        [row],
        {"max_gen_tokens": 4, "gate_layer": 0},
        budget=None,
        step=1,
        variant="raw",
        rank=0,
        world_size=1,
        output=tmp_path,
    )
    record = json.loads((tmp_path / "step1-raw-Kfull-rank0.jsonl").read_text())
    assert record["token_ids"] == tokens
    assert record["prediction"] == "needle"
    assert record["accuracy"] == 1.0

    # The shared decoder records the full prompt for SFT as well as an
    # unpruned gate. Those records must pass the independent quality audit.
    from tokengate.e2e_gate.bench.benchmark_sanity_check import verify_answer

    row.update(partition="validation", city="test", depth_requested=50, depth_actual=50)
    record.update({key: value for key, value in row.items() if key != "input_ids"})
    record.update(prompt_tokens=row["ctx_len"], budget=None)
    assert verify_answer(record, row, None, tokenizer, gated)["retained_tokens"] == 3
    for invalid in ([0, 2], [0, 0, 2], [2, 1, 0]):
        with pytest.raises(ValueError):
            verify_answer(
                {**record, "kept_positions": invalid}, row, None, tokenizer, gated
            )
    if not gated:
        legacy = {**record, "kept_positions": None}
        assert (
            verify_answer(legacy, row, None, tokenizer, False)["retained_tokens"] == 3
        )
