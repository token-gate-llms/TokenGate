"""Run in the pinned Torch/Triton environment; no GPU execution is required."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "niah_reproduce",
    Path(__file__).resolve().parents[3] / "train-scripts/niah/reproduce.py",
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
freeze, seed, verify = (
    driver.kernel_cache_freeze,
    driver.kernel_cache_seed,
    driver.kernel_cache_verify,
)


def test_seed_preserves_choices_and_refuses_to_replace_an_existing_one(tmp_path):
    source = tmp_path / "source/hash"
    source.mkdir(parents=True)
    record = {"key": [16, 128], "configs_timings": [[{"num_warps": 4}, [0.1]]]}
    path = source / "kernel.autotune.json"
    path.write_text(json.dumps(record))
    snapshot, cache = tmp_path / "snapshot", tmp_path / "cache"
    freeze(source.parent, snapshot)
    seed(snapshot, cache)
    assert (cache / "hash/kernel.autotune.json").read_bytes() == path.read_bytes()
    assert verify(snapshot, cache) == {"verified_records": 1, "extra_records": []}
    (cache / "hash/kernel.autotune.json").write_text("{}")
    with pytest.raises(ValueError, match="Refusing to replace"):
        seed(snapshot, cache)


def test_verify_rejects_new_unpinned_tuning_decisions(tmp_path):
    source = tmp_path / "source/hash"
    source.mkdir(parents=True)
    (source / "kernel.autotune.json").write_text(
        json.dumps(
            {
                "key": [16, 128],
                "configs_timings": [[{"num_warps": 4}, [0.1]]],
            }
        )
    )
    snapshot, cache = tmp_path / "snapshot", tmp_path / "cache"
    freeze(source.parent, snapshot)
    seed(snapshot, cache)
    (cache / "new.autotune.json").write_text("{}")
    with pytest.raises(ValueError, match="Unpinned"):
        verify(snapshot, cache)


@pytest.fixture(autouse=True)
def synthetic_versions(monkeypatch):
    # These tests verify the storage protocol without requiring CUDA/Triton.
    monkeypatch.setattr(
        driver,
        "kernel_cache_versions",
        lambda: {"torch": "fixture", "triton": "fixture"},
    )


def test_cache_rejects_a_different_runtime(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.autotune.json").write_text(
        json.dumps({"key": [16], "configs_timings": [[{}, [0.1]]]})
    )
    snapshot = tmp_path / "snapshot"
    freeze(source, snapshot)
    monkeypatch.setattr(driver, "kernel_cache_versions", lambda: {})
    with pytest.raises(ValueError, match="versions differ"):
        seed(snapshot, tmp_path / "cache")
