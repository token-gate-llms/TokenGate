"""Freeze an inference token budget using validation scores from matched TokenGate and SFT runs."""

import json
from pathlib import Path

from tokengate.e2e_gate.artifacts import atomic_json, file_hash
from tokengate.e2e_gate.train.selection import load_recipe


def evaluation_budgets():
    return tuple(load_recipe()["budgets"])


def validate_decision(path: Path, checkpoint: dict) -> str:
    """Require a frozen decision for this checkpoint before scoring held-out data."""
    decision = json.loads(path.read_text())
    if decision["checkpoint"] != checkpoint:
        raise ValueError("Deployment decision belongs to another checkpoint")
    if decision["validation_count"] != 50 or decision["tier"] != "16k":
        raise ValueError("Deployment choice must use the fixed 16K validation split")
    for source, expected_hash in decision["inputs"].items():
        if file_hash(Path(source)) != expected_hash:
            raise ValueError(f"Deployment decision input changed: {source}")
    return file_hash(path)


def choose_budget(results: list[dict], baseline_accuracy: float) -> dict:
    """Choose the smallest K matching SFT accuracy; retain a no-match outcome."""
    if {item["budget"] for item in results} != set(evaluation_budgets()):
        raise ValueError("Budget selection requires the complete declared sweep")
    if len(results) != len(evaluation_budgets()):
        raise ValueError("Duplicate evaluation budget")
    for item in results:
        if item["count"] != 50 or not 0 <= item["accuracy"] <= 1:
            raise ValueError("Budget selection requires 50 validation scores")
    candidates = [item for item in results if item["accuracy"] >= baseline_accuracy]
    if not candidates:
        return {"matches_sft_accuracy": False, "budget": None, "accuracy": None}
    selected = min(
        candidates,
        key=lambda item: float("inf") if item["budget"] is None else item["budget"],
    )
    return {
        "matches_sft_accuracy": True,
        "budget": selected["budget"],
        "accuracy": selected["accuracy"],
    }


def read_answers(directory: Path, summary: dict, ids: list[str]) -> list[dict]:
    """Read immutable answer evidence in the manifest's exact example order."""
    path = directory / "answers.jsonl"
    if file_hash(path) != summary["answers_sha256"]:
        raise ValueError(f"Saved answers changed: {path}")
    answers = [json.loads(line) for line in path.read_text().splitlines()]
    if not ids or len(set(ids)) != len(ids) or [a["id"] for a in answers] != ids:
        raise ValueError(f"Answer IDs do not match the manifest: {path}")
    return answers


def read_validation(directory: Path) -> tuple[dict, dict, list[dict]]:
    """Check answer-level evidence before accepting any validation aggregate."""
    manifest = json.loads((directory / "evaluation.json").read_text())
    if manifest["engine"] != "research":
        raise ValueError("Deployment budgets must be selected from research validation")
    status = json.loads((directory / "status.json").read_text())
    checkpoint = manifest["checkpoint"]
    if (
        not status["complete"]
        or checkpoint["diagnostic"]
        or manifest["partition"] != "validation"
        or manifest["tier"] != "16k"
        or len(manifest["ids"]) != 50
        or len(set(manifest["ids"])) != 50
    ):
        raise ValueError("Budget selection requires complete, measured 16K validation")
    params = Path(checkpoint["params_path"])
    if file_hash(params) != checkpoint["params_sha256"]:
        raise ValueError("Training configuration changed")
    config = json.loads(params.read_text())
    results = status["results"]
    expected = evaluation_budgets() if checkpoint["gate_path"] else (None,)
    if len(results) != len(expected) or {item["budget"] for item in results} != set(
        expected
    ):
        raise ValueError("Incomplete or duplicate validation budgets")
    for item in results:
        name = "full" if item["budget"] is None else f"K{item['budget']}"
        answers = read_answers(directory / name, item, manifest["ids"])
        if any(answer["partition"] != "validation" for answer in answers):
            raise ValueError("Held-out answers cannot select a budget")
        if any(answer["accuracy"] not in (0.0, 1.0) for answer in answers):
            raise ValueError("Expected binary NIAH retrieval scores")
        accuracy = sum(answer["accuracy"] for answer in answers) / 50
        if item["count"] != 50 or item["accuracy"] != accuracy:
            raise ValueError("Validation aggregate differs from per-example scores")
    return manifest, config, results


def freeze_budget(validation: Path, baseline: Path, output: Path) -> None:
    manifest, config, results = read_validation(validation)
    reference, baseline_config, baseline_results = read_validation(baseline)
    if reference["checkpoint"]["gate_path"] is not None:
        raise ValueError("Reference must be a separately trained SFT baseline")
    # Recovery runs can reference the same pinned model through different aliases.
    model = Path(config["base_model"]).resolve(strict=True)
    baseline_model = Path(baseline_config["base_model"]).resolve(strict=True)
    if model != baseline_model or config["seed"] != baseline_config["seed"]:
        raise ValueError(
            "TokenGate and SFT must use the same backbone and training seed"
        )
    if manifest["ids"] != reference["ids"]:
        raise ValueError("TokenGate and SFT validation IDs differ")
    baseline_accuracy = baseline_results[0]["accuracy"]
    selected = (
        choose_budget(results, baseline_accuracy)
        if manifest["checkpoint"]["gate_path"]
        else {
            "matches_sft_accuracy": True,
            "budget": None,
            "accuracy": baseline_accuracy,
        }
    )
    decision = {
        "checkpoint": manifest["checkpoint"],
        "baseline_checkpoint": reference["checkpoint"],
        "tier": "16k",
        "validation_count": 50,
        "rule": "smallest K with validation accuracy at least the matched SFT baseline",
        "baseline_accuracy": baseline_accuracy,
        **selected,
        "inputs": {
            str(directory / name): file_hash(directory / name)
            for directory in (validation, baseline)
            for name in ("evaluation.json", "status.json")
        },
    }
    if output.exists():
        if json.loads(output.read_text()) != decision:
            raise ValueError(
                "Existing deployment decision differs; refusing to replace it"
            )
    else:
        atomic_json(decision, output)
