"""Prevent held-out leakage and optimistic budget selection in NIAH reports."""

import json
import pickle

import pytest
from tokengate.e2e_gate.artifacts import file_hash
from tokengate.e2e_gate.bench.budget_selection import (
    choose_budget,
    evaluation_budgets,
    read_validation,
)
from tokengate.e2e_gate.bench.budget_selection import freeze_budget as freeze

from tokengate.datasets.dataset import partition_rows, validation_rows

BUDGETS = evaluation_budgets()
from tokengate.e2e_gate.bench.runtime import validate_retained_positions


def evaluation_pool():
    rows = [
        {
            "id": str(index),
            "full_ids": [1, 2, 3, 4],
            "ctx_len": 3,
            "n_ans": 1,
            "article_start": 1,
            "article_end": 2,
            "prompt_version": "niah_v1",
        }
        for index in range(200)
    ]
    return rows, {
        "val": [str(index) for index in range(50)],
        "test": [str(index) for index in range(50, 200)],
    }


def test_partitions_preserve_examples_and_reject_leakage():
    rows, manifest = evaluation_pool()
    result = partition_rows(rows, manifest, "16k")
    assert sum(row["partition"] == "validation" for row in result) == 50
    assert sum(row["partition"] == "test" for row in result) == 150
    assert [row["id"] for row in result] == [row["id"] for row in rows]
    manifest["test"][0] = manifest["val"][0]
    with pytest.raises(ValueError, match="disjoint"):
        partition_rows(rows, manifest, "16k")


def test_partitions_reject_missing_or_duplicate_examples():
    rows, manifest = evaluation_pool()
    with pytest.raises(ValueError, match="exactly cover"):
        partition_rows(rows[:-1], manifest, "16k")
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="exactly cover"):
        partition_rows(rows, manifest, "16k")


def test_32k_reserved_validation_is_not_the_16k_partition():
    rows, manifest = evaluation_pool()
    with pytest.raises(ValueError, match="partition sizes"):
        partition_rows(rows, manifest, "32k")
    manifest = {
        "val": [str(i) for i in range(16)],
        "test": [str(i) for i in range(16, 200)],
    }
    assert (
        sum(row["partition"] == "test" for row in partition_rows(rows, manifest, "32k"))
        == 184
    )


def test_training_validation_preserves_order_and_checks_the_complete_pool(tmp_path):
    rows, manifest = evaluation_pool()
    rows.reverse()
    prepared = tmp_path / "val.pkl"
    prepared.write_bytes(pickle.dumps(rows))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"niah_kamradt": manifest}))
    actual = validation_rows(tmp_path, splits)
    assert [row["id"] for row in actual] == [str(i) for i in range(49, -1, -1)]
    assert all(row["input_ids"] == row["full_ids"] for row in actual)
    # A malformed held-out partition must also invalidate training validation.
    manifest["test"][-1] = "missing-id"
    splits.write_text(json.dumps({"niah_kamradt": manifest}))
    with pytest.raises(ValueError, match="exactly cover"):
        validation_rows(tmp_path, splits)


def test_budget_selection_keeps_2000_and_2048_distinct():
    results = [{"budget": budget, "accuracy": 0.9, "count": 50} for budget in BUDGETS]
    for result in results:
        if result["budget"] in (2000, 2048, None):
            result["accuracy"] = 1.0
    assert choose_budget(results, 1.0)["budget"] == 2000
    results[BUDGETS.index(2000)]["accuracy"] = 0.98
    assert choose_budget(results, 1.0)["budget"] == 2048


def test_budget_selection_does_not_invent_accuracy_preservation():
    results = [{"budget": budget, "accuracy": 0.9, "count": 50} for budget in BUDGETS]
    assert choose_budget(results, 1.0) == {
        "matches_sft_accuracy": False,
        "budget": None,
        "accuracy": None,
    }
    results[-1]["accuracy"] = 1.0
    assert choose_budget(results, 1.0) == {
        "matches_sft_accuracy": True,
        "budget": None,
        "accuracy": 1.0,
    }
    with pytest.raises(ValueError, match="complete declared sweep"):
        choose_budget(results[:-1], 1.0)


def test_serving_selection_preserves_protected_tokens_and_budget():
    row = {"article_start": 2, "article_end": 7, "ctx_len": 9}
    validate_retained_positions([0, 1, 3, 5, 7, 8], row, 2)
    validate_retained_positions(list(range(9)), row, None)
    validate_retained_positions(list(range(9)), row, 100)
    for invalid in ([0, 1, 3, 5, 6, 7], [0, 1, 3, 7, 8]):
        with pytest.raises(ValueError, match="budget or protected"):
            validate_retained_positions(invalid, row, 2)
    for invalid in ([0, 1, 3, 3, 7, 8], [1, 0, 3, 5, 7, 8], [0, 1, 3, 5, 7, 9]):
        with pytest.raises(ValueError, match="unique ordered prompt indices"):
            validate_retained_positions(invalid, row, 2)


def test_serving_scores_cannot_change_the_research_budget_decision(tmp_path):
    params = tmp_path / "params.json"
    params.write_text("{}")
    (tmp_path / "full").mkdir()
    answers = tmp_path / "full/answers.jsonl"
    answers.write_text(
        "".join(
            json.dumps({"id": str(i), "partition": "validation", "accuracy": 1.0})
            + "\n"
            for i in range(50)
        )
    )
    manifest = {
        "engine": "vllm",
        "checkpoint": {
            "diagnostic": False,
            "params_path": str(params),
            "params_sha256": file_hash(params),
            "gate_path": None,
        },
        "partition": "validation",
        "tier": "16k",
        "ids": [str(i) for i in range(50)],
    }
    (tmp_path / "evaluation.json").write_text(json.dumps(manifest))
    (tmp_path / "status.json").write_text(
        json.dumps(
            {
                "complete": True,
                "results": [
                    {
                        "budget": None,
                        "count": 50,
                        "accuracy": 1.0,
                        "answers_sha256": file_hash(answers),
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="research validation"):
        read_validation(tmp_path)
    manifest["engine"] = "research"
    (tmp_path / "evaluation.json").write_text(json.dumps(manifest))
    assert read_validation(tmp_path)[2][0]["accuracy"] == 1.0


@pytest.mark.parametrize("ids", [["a", "a"], ["b", "a"], ["a"], []])
def test_answer_evidence_rejects_duplicate_reordered_or_missing_ids(tmp_path, ids):
    from tokengate.e2e_gate.bench.budget_selection import read_answers

    path = tmp_path / "answers.jsonl"
    path.write_text("".join(json.dumps({"id": value}) + "\n" for value in ids))
    with pytest.raises(ValueError, match="Answer IDs"):
        read_answers(tmp_path, {"answers_sha256": file_hash(path)}, ["a", "b"])


@pytest.mark.parametrize(
    "different_model,different_seed", [(False, False), (True, False), (False, True)]
)
def test_budget_selection_matches_resolved_backbone_and_seed(
    tmp_path, monkeypatch, different_model, different_seed
):
    model = tmp_path / "original/Qwen3-1.7B"
    model.mkdir(parents=True)
    recovery_model = tmp_path / "recovery/Qwen3-1.7B"
    recovery_model.parent.mkdir()
    if different_model:
        recovery_model.mkdir()
    else:
        recovery_model.symlink_to(model, target_is_directory=True)
    validation, baseline = tmp_path / "gate", tmp_path / "baseline"
    for directory in (validation, baseline):
        directory.mkdir()
        for name in ("evaluation.json", "status.json"):
            (directory / name).write_text("{}")

    # Answer-level validation is tested separately. Exercise the matched-model
    # guard using the path aliases created by the recovery pipeline.
    def validated_results(directory):
        gated = directory == validation
        return (
            {"checkpoint": {"gate_path": "gate.pt" if gated else None}, "ids": ["0"]},
            {
                "base_model": str(recovery_model if gated else model),
                "seed": int(gated and different_seed),
            },
            [
                {"budget": budget, "accuracy": 1.0, "count": 50}
                for budget in (BUDGETS if gated else (None,))
            ],
        )

    monkeypatch.setattr(
        "tokengate.e2e_gate.bench.budget_selection.read_validation", validated_results
    )
    output = tmp_path / "decision.json"
    if different_model or different_seed:
        with pytest.raises(ValueError, match="same backbone and training seed"):
            freeze(validation, baseline, output)
        assert not output.exists()
    else:
        freeze(validation, baseline, output)
        assert json.loads(output.read_text())["budget"] == min(
            budget for budget in BUDGETS if budget is not None
        )


def test_research_cli_routes_to_shared_evaluation_and_writes_evidence(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import torch

    from benchmarks import run_benchmark as cli
    from tokengate.e2e_gate.train import train_e2e_gate as training

    config = {
        "gate_layer": None,
        "max_gen_tokens": 4,
        "attn_backend": "sdpa",
        "code_manifest_sha256": "training-fixture",
    }
    checkpoint = {"gate_path": None, "selection": {"step": 1, "variant": "raw"}}
    row = {
        "id": "val-0",
        "input_ids": [1, 2, 3],
        "ctx_len": 3,
        "article_start": 1,
        "article_end": 2,
        "target": "needle",
        "partition": "validation",
        "city": "fixture",
        "depth_requested": 0.5,
        "depth_actual": 0.5,
    }

    def generate(inputs, *, streamer, **kwargs):
        streamer.put(inputs)
        for token in (4, 9):
            streamer.put(torch.tensor([token]))
        return torch.tensor([[1, 2, 3, 4, 9]])

    model = SimpleNamespace(device="cpu", eval=lambda: None, generate=generate)
    tokenizer = SimpleNamespace(
        eos_token_id=9, pad_token_id=0, decode=lambda *a, **k: "needle"
    )
    monkeypatch.setattr(
        cli, "selected_checkpoint", lambda *a, **k: (config, checkpoint)
    )
    monkeypatch.setattr(cli, "load_rows", lambda *a: ([row], {}))
    monkeypatch.setattr(cli, "load_checkpoint", lambda *a: (model, None, tokenizer))
    monkeypatch.setattr(training, "answer_loss", lambda *a: 0.25)
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text("{}")
    monkeypatch.setattr(cli, "capture_source", lambda *a: source_manifest)
    output = tmp_path / "results"
    cli.reproduction_main(
        [
            "--stage",
            "quality",
            "--run",
            str(tmp_path),
            "--output",
            str(output),
            "--tier",
            "16k",
            "--partition",
            "validation",
        ]
    )
    status = json.loads((output / "status.json").read_text())
    assert status["complete"] and status["results"][0]["accuracy"] == 1.0
    answer = json.loads((output / "full/answers.jsonl").read_text())
    assert answer["token_ids"] == [4, 9]
    assert answer["partition"] == "validation"
    assert answer["budget"] is None
    assert status["results"][0]["answers_sha256"] == file_hash(
        output / "full/answers.jsonl"
    )
