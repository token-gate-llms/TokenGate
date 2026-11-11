"""Protect benchmark coverage, scoring units, and saved-result integrity."""

import json
import asyncio
import time
from pathlib import Path

import pytest

from benchmarks.run_vllm_benchmark import read_samples, summarize, validate_rows
from benchmarks.batching import (
    BatchTokenCollector,
    BatchingConfig,
    measure_async,
    run_record,
)


@pytest.mark.parametrize("mode", ["static", "continuous"])
def test_batch_submission_barrier_and_concurrency(mode):
    async def scenario():
        started = [asyncio.Event() for _ in range(3)]
        release = [asyncio.Event() for _ in range(3)]
        finished = set()
        active = 0
        peak = 0
        group_sizes = []

        async def before_batch(size):
            assert active == 0
            group_sizes.append(size)

        async def request(sample, request_id):
            nonlocal active, peak
            index = int(request_id.rsplit("-", 1)[1])
            if index == 2:
                assert (1 in finished) == (mode == "static")
            active += 1
            peak = max(peak, active)
            started[index].set()
            submitted = time.perf_counter()
            await release[index].wait()
            ended = time.perf_counter()
            active -= 1
            finished.add(index)
            return dict(
                id=sample["id"],
                returned_s=ended,
                token_ids=[7],
                token_arrivals_s=[ended],
                ttft_s=ended - submitted,
                final_s=ended - submitted,
            )

        task = asyncio.create_task(
            measure_async(
                request,
                [{"id": str(i)} for i in range(3)],
                BatchingConfig(mode, 2),
                budget=64,
                repeat=0,
                before_batch=before_batch,
            )
        )
        await asyncio.gather(started[0].wait(), started[1].wait())
        release[0].set()
        if mode == "static":
            release[1].set()
        await started[2].wait()
        release[1].set()
        release[2].set()
        rows, run = await task
        assert peak == 2
        assert len(rows) == run["completed"] == 3
        assert {r["id"] for r in rows} == {"0", "1", "2"}
        assert run["requests_per_s"] == pytest.approx(3 / run["elapsed_s"])
        assert group_sizes == ([2, 1] if mode == "static" else [])

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_window_counts_tokens_from_requests_that_finish_during_drain():
    rows = [
        dict(
            returned_s=2.5,
            token_ids=[7, 8],
            token_arrivals_s=[1.5, 2.4],
            ttft_s=0.5,
            final_s=1.4,
        ),
        dict(returned_s=4.0, token_ids=[4, 5, 6], token_arrivals_s=[0.8, 2.0, 3.5]),
    ]
    run = run_record(
        rows,
        1.0,
        3.0,
        budget=64,
        repeat=0,
        config=BatchingConfig("continuous", 2, duration_s=2),
        drained=4,
    )
    assert run["completed"] == 1
    assert run["output_tokens"] == 3
    assert run["requests_per_s"] == 0.5
    assert run["output_tokens_per_s"] == 1.5
    assert run["drain_end_s"] == 4


def test_batch_collector_ignores_each_rows_post_eos_tokens():
    import torch

    collector = BatchTokenCollector(2, 2)
    collector.put(torch.tensor([[2], [7]]))
    collector.put(torch.tensor([[9], [8]]))
    collector.put(torch.tensor([[9], [2]]))
    collector.end()
    assert collector.tokens == [[2], [7, 8, 2]]
    assert list(map(len, collector.arrivals)) == [1, 3]
    assert collector.ended


def test_batch_failure_is_preserved_and_stops_admission():
    async def fail(sample, request_id):
        raise RuntimeError("engine unavailable")

    rows, run = asyncio.run(
        measure_async(
            fail,
            [{"id": str(i)} for i in range(5)],
            BatchingConfig("continuous", 2),
            budget=64,
            repeat=0,
        )
    )
    assert len(rows) == 2
    assert all("engine unavailable" in r["error"] for r in rows)
    assert run["failed"] == 2
    assert not run["valid"]


def row(**changes):
    value = dict(
        id="a",
        model="Qwen3",
        engine="original",
        budget=64,
        repeat=0,
        token_ids=[7, 2],
        ttft_s=0.1,
        final_s=0.2,
        wall_s=0.3,
        correct=True,
        score=1.0,
    )
    return value | changes


@pytest.mark.parametrize("rows", [[], [row(), row()], [row(repeat=1)]])
def test_missing_or_duplicate_cases_fail(rows):
    with pytest.raises(ValueError, match="coverage"):
        validate_rows(rows, {"samples": [{"id": "a"}]}, [64], 1)


@pytest.mark.parametrize(
    "changes",
    [
        dict(ttft_s=float("nan")),
        dict(final_s=0.4),
        dict(token_ids=[]),
        dict(token_ids=[True]),
    ],
)
def test_invalid_measurements_fail(changes):
    with pytest.raises(ValueError):
        validate_rows([row(**changes)], {"samples": [{"id": "a"}]}, [64], 1)


def test_repeats_do_not_increase_accuracy_sample_size():
    result = summarize(
        [
            row(),
            row(repeat=1),
            row(id="b", correct=False, score=0.5),
            row(id="b", repeat=1, correct=False, score=0.5),
        ]
    )[0]
    assert result["examples"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5
    assert result["score"] == 0.75  # Preserve nonbinary task metrics.
    assert result["requests_per_s"] == pytest.approx(4 / 1.2)


def test_disagreeing_repeats_do_not_pick_best_answer():
    result = summarize(
        [row(), row(repeat=1, token_ids=[8, 2], correct=False, score=0)]
    )[0]
    assert not result["repeats_consistent"]
    assert result["accuracy"] == 0


def test_split_requires_every_requested_sample(tmp_path):
    samples, split = tmp_path / "samples.jsonl", tmp_path / "split.json"
    samples.write_text(json.dumps(dict(id="a", body="text", question="q", answer="a")))
    split.write_text(json.dumps({"niah_kamradt": {"test": ["a", "missing"]}}))
    with pytest.raises(ValueError, match="missing IDs"):
        read_samples(
            dict(samples=str(samples), eval_split=str(split), task="niah_kamradt"),
            "test",
            1,
        )


def test_committed_needle_examples_use_existing_task_prompt():
    from tokengate.datasets.tasks import get_task

    rows = read_samples({"samples": "examples/data/niah_32k_samples.jsonl"}, "test", 1)
    original = json.loads(
        Path("examples/data/niah_32k_samples.jsonl").read_text().splitlines()[0]
    )
    assert rows[0]["target"] == original["answer"]
    prompt = get_task("niah_kamradt").format_prompt(rows[0]["article"])
    assert prompt.index(original["question"]) < prompt.index(
        original["body"].lstrip("\n")
    )


def test_tiled_results_require_every_pruning_budget(tmp_path):
    from runpy import run_path

    load_results = run_path("examples/inference-vllm-helper.py")["load_results"]
    (tmp_path / "inputs.json").write_text(
        json.dumps({"config": {"budgets": [64, 512, "All"]}, "samples": [{"id": "a"}]})
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"engines": ["vllm_tiled"], "repeats": 1})
    )
    (tmp_path / "validation.json").write_text(
        json.dumps({"status": "behavior_checked"})
    )
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text(json.dumps(row(engine="vllm_tiled", budget="All")) + "\n")
    with pytest.raises(ValueError, match="coverage"):
        load_results([tmp_path])
    measurements.write_text(
        "".join(
            json.dumps(row(engine="vllm_tiled", budget=k)) + "\n"
            for k in [64, 512, "All"]
        )
    )
    table, qualified = load_results([tmp_path])
    assert len(table) == 3
    assert not qualified
