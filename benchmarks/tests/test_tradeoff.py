"""Check test-set coverage and example-level uncertainty in notebook figures."""

import json
from runpy import run_path

import pytest

HELPERS = run_path("examples/inference-vllm-helper.py")


def write_run(path, samples, repeats=3):
    path.mkdir()
    inputs = {
        "config": {"name": "Qwen", "base_model": "Qwen", "budgets": [64]},
        "samples": [{"id": sample} for sample in samples],
        "gate_sha256": "gate",
        "adapter_sha256": "adapter",
        "base_files": {"config.json": "base"},
    }
    manifest = {
        "engines": ["vllm_tiled"],
        "repeats": repeats,
        "warmups": 1,
        "source_sha256": {"runner": "code"},
        "vllm_source": {"source_sha256": {"kernel": "code"}},
    }
    for name, value in [
        ("inputs", inputs),
        ("manifest", manifest),
        ("validation", {"status": "behavior_checked"}),
    ]:
        (path / f"{name}.json").write_text(json.dumps(value))
    rows = []
    for sample, latencies in samples.items():
        for repeat, latency in enumerate(latencies):
            rows.append(
                {
                    "id": sample,
                    "model": "Qwen",
                    "engine": "vllm_tiled",
                    "budget": 64,
                    "repeat": repeat,
                    "token_ids": [7, 2],
                    "correct": True,
                    "score": 1.0,
                    "ttft_s": latency,
                    "final_s": latency + 1,
                    "wall_s": latency + 2,
                }
            )
    (path / "measurements.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return path


def test_tradeoff_aggregates_examples_before_timings(tmp_path):
    # Median of example medians is 50.5 s; pooling repetitions gives 2 s.
    first = write_run(tmp_path / "first", {"a": [1, 1, 100]})
    second = write_run(tmp_path / "second", {"b": [2, 100, 100]})
    result = HELPERS["load_tradeoff"]([first, second], ["a", "b"]).iloc[0]
    assert result.examples == result.correct == 2
    assert result.accuracy == 100
    assert result.accuracy_low < 100
    assert result.accuracy_high == pytest.approx(100)
    assert result.ttft_ms == 50500
    assert result.ttft_ms_low <= result.ttft_ms <= result.ttft_ms_high
    assert result.inconsistent_examples == 0


@pytest.mark.parametrize("ids", [["a", "b"], ["wrong"], ["a", "a"]])
def test_tradeoff_rejects_incomplete_or_wrong_partition(tmp_path, ids):
    run = write_run(tmp_path / "run", {"a": [1, 1, 1]})
    with pytest.raises(ValueError):
        HELPERS["load_tradeoff"]([run], ids)


def test_tradeoff_rejects_overlapping_shards(tmp_path):
    run = write_run(tmp_path / "run", {"a": [1, 1, 1]})
    with pytest.raises(ValueError, match="Overlapping"):
        HELPERS["load_tradeoff"]([run, run], ["a"])


def test_tradeoff_rejects_changed_weights(tmp_path):
    first = write_run(tmp_path / "first", {"a": [1, 1, 1]})
    second = write_run(tmp_path / "second", {"b": [1, 1, 1]})
    p = second / "inputs.json"
    inputs = json.loads(p.read_text())
    inputs["gate_sha256"] = "different"
    p.write_text(json.dumps(inputs))
    with pytest.raises(ValueError, match="Inconsistent"):
        HELPERS["load_tradeoff"]([first, second], ["a", "b"])


def test_tradeoff_does_not_choose_best_repeat(tmp_path):
    run = write_run(tmp_path / "run", {"a": [1, 1, 1]})
    p = run / "measurements.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    rows[1].update(correct=False, score=0.0, token_ids=[8, 2])
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = HELPERS["load_tradeoff"]([run], ["a"]).iloc[0]
    assert result.examples == 1
    assert result.correct == 0
    assert result.inconsistent_examples == 1
    assert result.accuracy_low == result.accuracy == 0
    assert result.accuracy_high > 0


def test_perfect_test_set_has_exact_upper_endpoint(tmp_path):
    samples = {str(index): [1, 1, 1] for index in range(184)}
    run = write_run(tmp_path / "run", samples)
    result = HELPERS["load_tradeoff"]([run], list(samples)).iloc[0]
    # A rounded value just below 100 would give Matplotlib a negative error length.
    assert result.accuracy_high == result.accuracy == 100
    assert result.accuracy_low < result.accuracy
