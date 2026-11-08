"""Protect benchmark coverage, scoring units, and saved-result integrity."""

import json
from pathlib import Path

import pytest

from benchmarks.run_vllm_benchmark import read_samples, summarize, validate_rows


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
