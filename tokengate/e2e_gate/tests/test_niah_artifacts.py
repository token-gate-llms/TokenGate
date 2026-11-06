"""Keep training reproducible while allowing independent diagnostic development."""

import json
import tarfile

import pytest
from tokengate.e2e_gate.artifacts import capture_source, file_hash, verify_source


@pytest.fixture
def source_tree(tmp_path):
    source = tmp_path / "source"
    for name in (
        "tokengate/__init__.py",
        "tokengate/e2e_gate/train/train_e2e_gate.py",
        "train-scripts/niah/reproduction.yaml",
        "tokengate/e2e_gate/bench/benchmark_sanity_check.py",
        "tokengate/e2e_gate/tests/test_example.py",
        "tokengate/e2e_gate/gate_module/gate_forward.py",
        "tokengate/datasets/eval_splits/niah.json",
        "kernels/fa2bias/keybias.patch",
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# original\n")
    return source


def test_training_snapshot_excludes_tests(source_tree, tmp_path):
    manifest = capture_source(source_tree, tmp_path, training=True)
    digest = file_hash(manifest)
    for name in ("tokengate/e2e_gate/tests/test_example.py",):
        (source_tree / name).write_text("# diagnostic improvement\n")
    verify_source(source_tree, manifest, digest)
    assert capture_source(source_tree, tmp_path, training=True) == manifest
    record = json.loads(manifest.read_text())
    with tarfile.open(manifest.with_suffix(".tar.gz")) as archive:
        assert set(archive.getnames()) == record["files"].keys()
        for name, expected in record["files"].items():
            assert file_hash(source_tree / name) == expected
            assert archive.extractfile(name).read() == (source_tree / name).read_bytes()


@pytest.mark.parametrize(
    "name",
    [
        "tokengate/e2e_gate/train/train_e2e_gate.py",
        "train-scripts/niah/reproduction.yaml",
        "tokengate/e2e_gate/gate_module/gate_forward.py",
        "tokengate/datasets/eval_splits/niah.json",
        "kernels/fa2bias/keybias.patch",
    ],
)
def test_training_snapshot_rejects_changed_dependencies(source_tree, tmp_path, name):
    manifest = capture_source(source_tree, tmp_path, training=True)
    digest = file_hash(manifest)
    (source_tree / name).write_text("changed\n")
    with pytest.raises(ValueError, match="Recorded source changed"):
        verify_source(source_tree, manifest, digest)


@pytest.mark.parametrize("change", ["add", "delete"])
def test_training_snapshot_rejects_inventory_changes(source_tree, tmp_path, change):
    manifest = capture_source(source_tree, tmp_path, training=True)
    digest = file_hash(manifest)
    if change == "add":
        (source_tree / "tokengate/new_dependency.py").write_text("# new\n")
    else:
        (source_tree / "tokengate/__init__.py").unlink()
    with pytest.raises(ValueError, match="inventory changed"):
        verify_source(source_tree, manifest, digest)


def test_full_snapshot_includes_diagnostics(source_tree, tmp_path):
    manifest = capture_source(source_tree, tmp_path)
    digest = file_hash(manifest)
    (source_tree / "tokengate/e2e_gate/bench/benchmark_sanity_check.py").write_text(
        "# v2\n"
    )
    with pytest.raises(ValueError, match="Recorded source changed"):
        verify_source(source_tree, manifest, digest)


def test_legacy_manifest_is_still_checked_strictly(source_tree, tmp_path):
    name = "tokengate/__init__.py"
    manifest = tmp_path / "legacy.json"
    manifest.write_text(json.dumps({name: file_hash(source_tree / name)}))
    digest = file_hash(manifest)
    verify_source(source_tree, manifest, digest)
    (source_tree / name).write_text("# changed\n")
    with pytest.raises(ValueError, match="Recorded source changed"):
        verify_source(source_tree, manifest, digest)


def test_changed_manifest_is_rejected(source_tree, tmp_path):
    manifest = capture_source(source_tree, tmp_path, training=True)
    digest = file_hash(manifest)
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest changed"):
        verify_source(source_tree, manifest, digest)


def test_version_two_snapshots_retain_their_original_inventory_rules(
    source_tree, tmp_path
):
    from tokengate.e2e_gate.artifacts import _source_files

    manifest = tmp_path / "v2.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "scope": "training",
                "files": _source_files(source_tree, training=True, version=2),
            }
        )
    )
    digest = file_hash(manifest)
    verify_source(source_tree, manifest, digest)
    (source_tree / "tokengate/e2e_gate/gate_module/gate_forward.py").write_text(
        "changed"
    )
    with pytest.raises(ValueError, match="Recorded source changed"):
        verify_source(source_tree, manifest, digest)
