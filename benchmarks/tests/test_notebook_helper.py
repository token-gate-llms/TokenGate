"""Protect notebook replay, full-test coverage, and benchmark launch boundaries."""

import json
import subprocess
import sys
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest
import yaml

from benchmarks.run_vllm_benchmark import file_hash, read_samples


@pytest.fixture
def helpers():
    return run_path("examples/inference-vllm-helper.py")


@pytest.fixture
def launch(tmp_path):
    config = tmp_path / "qwen.yaml"
    config.write_text("name: Qwen\n")
    source = tmp_path / "vllm"
    (source / "tools").mkdir(parents=True)
    (source / "tools/export_token_gate.py").touch()
    return {
        "configs": [config],
        "output": tmp_path / "results",
        "saved_runs": [],
        "engines": ["original", "vllm_tiled"],
        "vllm_source": source,
        "gpu": "2",
        "warmups": 1,
        "repeats": 3,
    }


@pytest.mark.parametrize("split,n,profile", [("val", 1, True), ("test", None, False)])
def test_launch_keeps_full_test_unlimited(
    helpers, launch, monkeypatch, split, n, profile
):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    runs = helpers["run_benchmarks"](**launch, split=split, n=n, profile=profile)
    assert runs == [launch["output"] / "qwen"]
    command, options = calls.pop()
    assert not calls
    assert command[:3] == [sys.executable, "-m", "benchmarks.run_vllm_benchmark"]
    assert command[command.index("--split") + 1] == split
    assert command[command.index("--gpu") + 1] == "2"
    assert command[command.index("--warmups") + 1] == "1"
    assert command[command.index("--repeats") + 1] == "3"
    start = command.index("--engines") + 1
    assert command[start : start + 2] == launch["engines"]
    assert ("--profile" in command) == profile
    assert ("--n" in command) == (n is not None)
    if n is not None:
        assert command[command.index("--n") + 1] == str(n)
    assert options["check"] is True
    assert options["cwd"] == Path.cwd()


def test_batch_plot_keeps_accuracy_separate_from_repeated_load(helpers, tmp_path):
    import matplotlib.pyplot as plt
    import pandas as pd

    def record(mode, duration, throughput, accuracy):
        return dict(
            model="Qwen",
            engine="vllm_tiled",
            budget="64",
            mode=mode,
            size=2,
            duration_s=duration,
            base_settings="matched",
            accuracy=accuracy,
            accuracy_low=40 if accuracy is not None else None,
            accuracy_high=60 if accuracy is not None else None,
            run_statistics={
                "requests_per_s": dict(mean=throughput, low=None, high=None),
                "ttft_p95_ms": dict(mean=100, low=None, high=None),
            },
        )

    table = pd.DataFrame(
        [
            record("static", 0, 5, 50),
            record("continuous", 0, 6, 50),
            record("continuous", 60, 20, None),
        ]
    )
    unpruned = table.copy()
    unpruned["budget"] = "All"
    unpruned["run_statistics"] = [
        {
            "requests_per_s": dict(mean=value, low=None, high=None),
            "ttft_p95_ms": dict(mean=200, low=None, high=None),
        }
        for value in (3, 4, 8)
    ]
    table = pd.concat([table, unpruned], ignore_index=True)
    other = table.copy()
    other["model"] = "Qwen3.5"
    table = pd.concat([table, other], ignore_index=True)
    figure = helpers["plot_batching"](table, "64", tmp_path / "batching")
    assert len(figure.axes) == 8
    assert figure.axes[0].lines[0].get_ydata().tolist() == [5]
    assert figure.axes[4].lines[0].get_ydata().tolist() == [20]
    assert figure.axes[4].lines[1].get_ydata().tolist() == [8]
    assert [line.get_linestyle() for line in figure.axes[4].lines] == ["-", ":"]
    assert figure.axes[4].lines[0].get_color() == figure.axes[4].lines[1].get_color()
    assert figure.axes[0].get_ylim()[1] > 20
    assert "Qwen3.5" in figure.axes[2].get_ylabel()
    assert all("accuracy" not in ax.get_ylabel().lower() for ax in figure.axes)
    assert table.loc[table.duration_s == 0, "accuracy"].eq(50).all()
    assert table.loc[table.duration_s > 0, "accuracy"].isna().all()
    assert all(not ax.get_title() for ax in figure.axes)
    assert all(
        (tmp_path / f"batching.{ext}").is_file() for ext in ("png", "pdf", "svg")
    )
    plt.close(figure)
    with pytest.raises(ValueError, match="paired finite"):
        helpers["plot_batching"](
            table[table.duration_s > 0], "64", tmp_path / "invalid"
        )


@pytest.mark.parametrize(
    "changed", [None, "base_files", "sample_sha256", "source_sha256"]
)
def test_batch_reader_pairs_original_and_vllm_evidence(helpers, tmp_path, changed):
    """A separate original run must match the vLLM arm's inputs, weights and code."""
    from dataclasses import asdict
    from benchmarks.batching import BatchingConfig, run_record

    config = BatchingConfig(mode="static", size=2)
    directories = []
    for engine in ("original", "vllm_tiled"):
        directory = tmp_path / engine
        directory.mkdir()
        directories.append(directory)
        inputs = dict(
            config=dict(name="Qwen", budgets=[64]),
            samples=[dict(id="a")],
            sample_sha256="same-inputs",
            gate_sha256="gate",
            adapter_sha256="adapter",
            base_files={"weights": "base"},
            source_sha256={"inference.py": "code"},
        )
        if engine == "vllm_tiled" and changed:
            inputs[changed] = "different"
        row = dict(
            model="Qwen",
            engine=engine,
            id="a",
            budget=64,
            repeat=0,
            request_id="64-0-0",
            token_ids=[7, 2],
            correct=True,
            score=1.0,
            submitted_s=1.0,
            returned_s=1.3,
            token_arrivals_s=[1.1, 1.2],
            ttft_s=0.1,
            final_s=0.2,
            wall_s=0.3,
        )
        run = dict(
            run_record([row], 1.0, 1.4, budget=64, repeat=0, config=config),
            model="Qwen",
            engine=engine,
        )
        manifest = dict(
            engines=[engine],
            repeats=1,
            batching=asdict(config),
            vllm_source=None if engine == "original" else dict(source_sha256="vllm"),
        )
        for name, value in dict(
            inputs=inputs,
            manifest=manifest,
            validation=dict(status="behavior_checked"),
            runs=[run],
        ).items():
            (directory / f"{name}.json").write_text(json.dumps(value))
        (directory / "measurements.jsonl").write_text(json.dumps(row) + "\n")
    if changed:
        with pytest.raises(ValueError, match="Changed inputs, checkpoint, or code"):
            helpers["load_batching"](directories, ["a"])
        # Different implementation snapshots require an explicit identity map.
        snapshots = {
            ("Qwen", engine): {
                "tokengate": json.loads((directory / "inputs.json").read_text())["source_sha256"],
                "vllm": None if engine == "original" else "vllm",
            }
            for engine, directory in zip(("original", "vllm_tiled"), directories)
        }
        if changed == "source_sha256":
            table = helpers["load_batching"](directories, ["a"], source_snapshots=snapshots)
            assert table.source_snapshot.nunique() == 2
            snapshots[("Qwen", "vllm_tiled")]["tokengate"] = "unapproved"
            with pytest.raises(ValueError, match="Source snapshot differs"):
                helpers["load_batching"](directories, ["a"], source_snapshots=snapshots)
        else:
            with pytest.raises(ValueError, match="Changed inputs, checkpoint, or code"):
                helpers["load_batching"](directories, ["a"], source_snapshots=snapshots)
    else:
        table = helpers["load_batching"](directories, ["a"])
        assert table.accuracy.tolist() == [100, 100]
        assert table.requests_per_s.tolist() == pytest.approx([2.5, 2.5])
        runs = directories[0] / "runs.json"
        data = json.loads(runs.read_text())
        data[0]["requests_per_s"] *= 2
        runs.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="differs from request timelines"):
            helpers["load_batching"](directories, ["a"])


def test_batch_reader_accepts_targeted_budget_subset(helpers, tmp_path):
    """A targeted sweep can reuse shared budgets from a larger verified sweep."""
    from dataclasses import asdict
    from benchmarks.batching import BatchingConfig, run_record

    config = BatchingConfig(mode="static", size=2)
    directories = []
    for engine, budgets in [("original", [2048]), ("vllm_tiled", [64, 2048])]:
        directory = tmp_path / engine
        directory.mkdir()
        directories.append(directory)
        inputs = dict(
            config=dict(name="Qwen", budgets=budgets, k_values=budgets, max_new_tokens=64),
            samples=[dict(id="a")],
            sample_sha256="same-inputs",
            gate_sha256="gate",
            adapter_sha256="adapter",
            base_files={"weights": "base"},
            source_sha256={"inference.py": "code"},
        )
        rows, runs = [], []
        for budget in budgets:
            row = dict(
                model="Qwen", engine=engine, id="a", budget=budget, repeat=0,
                request_id=f"{budget}-0-0", token_ids=[7, 2], correct=True, score=1.0,
                submitted_s=1.0, returned_s=1.3, token_arrivals_s=[1.1, 1.2],
                ttft_s=0.1, final_s=0.2, wall_s=0.3,
            )
            rows.append(row)
            runs.append(dict(
                run_record([row], 1.0, 1.4, budget=budget, repeat=0, config=config),
                model="Qwen", engine=engine,
            ))
        for name, value in dict(
            inputs=inputs,
            manifest=dict(engines=[engine], repeats=1, batching=asdict(config), vllm_source=None),
            validation=dict(status="behavior_checked"), runs=runs,
        ).items():
            (directory / f"{name}.json").write_text(json.dumps(value))
        (directory / "measurements.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )

    table = helpers["load_batching"](directories, ["a"])
    assert len(table[table.budget.astype(str) == "2048"]) == 2
    assert len(table) == 3

    # Sweep membership is independent of generation settings and row coverage.
    path = directories[1] / "inputs.json"
    inputs = json.loads(path.read_text())
    inputs["config"]["max_new_tokens"] = 32
    path.write_text(json.dumps(inputs))
    with pytest.raises(ValueError, match="Changed inputs, checkpoint, or code"):
        helpers["load_batching"](directories, ["a"])
    inputs["config"]["max_new_tokens"] = 64
    path.write_text(json.dumps(inputs))
    path = directories[1] / "measurements.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError):
        helpers["load_batching"](directories, ["a"])


def test_batch_selection_checks_source_manifest_hash(helpers, tmp_path):
    import hashlib

    source = tmp_path / "source.json"
    source.write_text(json.dumps({"inference.py": "pinned"}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "sources": {"code": {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}},
        "implementations": [{"model": "Qwen", "engine": "original", "tokengate_source": "code", "vllm_source": None}],
        "runs": ["run"],
    }))
    directories, snapshots = helpers["batching_selection"](selection)
    assert directories == [tmp_path / "run"]
    assert snapshots[("Qwen", "original")]["tokengate"] == {"inference.py": "pinned"}
    source.write_text(json.dumps({"inference.py": "changed"}))
    with pytest.raises(ValueError, match="Source manifest hash differs"):
        helpers["batching_selection"](selection)


def test_existing_later_run_blocks_all_launches(helpers, launch, monkeypatch):
    later = launch["configs"][0].with_name("later.yaml")
    later.write_text("name: Later\n")
    launch["configs"].append(later)
    (launch["output"] / "later").mkdir(parents=True)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("Started a run")
    )
    with pytest.raises(FileExistsError):
        helpers["run_benchmarks"](**launch)
    assert not (launch["output"] / "qwen").exists()


def test_saved_replay_never_launches_or_requires_vllm(
    helpers, launch, tmp_path, monkeypatch
):
    run = tmp_path / "saved"
    run.mkdir()
    for name, content in {
        "manifest": {"engines": ["vllm_tiled"], "warmups": 1, "repeats": 1},
        "validation": {"status": "behavior_checked"},
        "inputs": {"config": {"budgets": [64]}, "samples": [{"id": "a"}]},
    }.items():
        (run / f"{name}.json").write_text(json.dumps(content))
    (run / "measurements.jsonl").write_text(
        json.dumps(
            {
                "model": "Qwen",
                "engine": "vllm_tiled",
                "budget": 64,
                "id": "a",
                "repeat": 0,
                "token_ids": [7, 2],
                "correct": True,
                "score": 1.0,
                "ttft_s": 0.1,
                "final_s": 0.2,
                "wall_s": 0.3,
            }
        )
        + "\n"
    )
    launch.update(saved_runs=[run], vllm_source=tmp_path / "unavailable")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("Started a run")
    )
    assert helpers["run_benchmarks"](**launch) == [run]
    assert not launch["output"].exists()


def test_failed_subprocess_preserves_log(helpers, tmp_path):
    log = tmp_path / "failed.log"
    with pytest.raises(RuntimeError, match="failed.log") as error:
        helpers["_run_logged"](
            [sys.executable, "-c", "print('failure details'); raise SystemExit(3)"],
            log,
            "Test failure",
        )
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)
    assert "failure details" in log.read_text()


def test_prepare_rejects_changed_pool_before_configs(helpers, launch, tmp_path):
    samples = tmp_path / "samples.jsonl"
    samples.write_text("changed data\n")
    with pytest.raises(ValueError, match="differs from the verified"):
        helpers["prepare_niah_configs"](launch["configs"], samples, launch["output"])
    assert not list(launch["output"].glob("*.yaml"))


def test_prepared_configs_select_only_reserved_ids(
    helpers, launch, tmp_path, monkeypatch
):
    split = helpers["NIAH_SPLIT"]
    partitions = json.loads(split.read_text())["niah_kamradt_32k"]
    samples = tmp_path / "samples.jsonl"
    examples = [
        {"id": key, "body": "text", "question": "question", "answer": "answer"}
        for key in partitions["test"] + partitions["val"]
    ]
    samples.write_text("".join(json.dumps(row) + "\n" for row in examples))
    published = tmp_path / "examples/data/niah_32k_samples.jsonl"
    published.parent.mkdir(parents=True)
    published.write_text(json.dumps(examples[0]) + "\n")
    state = helpers["prepare_niah_configs"].__globals__
    monkeypatch.setitem(state, "ROOT", tmp_path)
    monkeypatch.setitem(state, "NIAH_POOL_SHA256", file_hash(samples))
    launch["configs"][0].write_text("name: Qwen\ntask: niah_kamradt\n")

    configs = helpers["prepare_niah_configs"](
        launch["configs"], samples, launch["output"]
    )
    cfg = yaml.safe_load(configs[0].read_text())
    ids = [row["id"] for row in read_samples(cfg, "test", None)]
    assert ids == partitions["test"]
    assert len(ids) == 184 and not set(ids) & set(partitions["val"])
    with pytest.raises(FileExistsError):
        helpers["prepare_niah_configs"](launch["configs"], samples, launch["output"])


def test_remote_runtime_pins_source_and_isolates_imports(
    helpers, tmp_path, monkeypatch
):
    """Use real Git and a real venv, substituting only the dependency download."""
    remote = tmp_path / "remote"
    remote.mkdir()
    subprocess.run(["git", "init", "-b", "test-branch", str(remote)], check=True)
    package = remote / "vllm"
    (package / "v1/attention/ops").mkdir(parents=True)
    (package / "__init__.py").write_text("REVISION = 'published'\n")
    (package / "v1/attention/ops/token_gate_attention.py").write_text(
        "def token_gate_prefill(): pass\n"
    )
    subprocess.run(["git", "-C", str(remote), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(remote),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "Published code",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(remote), "rev-parse", "HEAD"], text=True
    ).strip()
    # These working-tree changes must never reach a clone of the published commit.
    (package / "__init__.py").write_text(
        "raise RuntimeError('Local source was used')\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(remote))
    monkeypatch.setattr(sys, "platform", "linux")
    state = helpers["prepare_vllm_runtime"].__globals__
    actual_run = state["_run_logged"]
    monkeypatch.setattr(state["shutil"], "which", lambda name: name)
    workspace = tmp_path / "temporary"
    workspace.mkdir()
    monkeypatch.setattr(state["tempfile"], "mkdtemp", lambda **kw: str(workspace))

    def run(command, log, description, **kwargs):
        if command[:2] == ["uv", "venv"]:
            # The real isolated interpreter has no access to the test process's packages.
            subprocess.run(
                [sys.executable, "-m", "venv", "--without-pip", command[-1]], check=True
            )
        elif command[:3] == ["uv", "pip", "install"]:
            python = command[command.index("--python") + 1]
            site = subprocess.check_output(
                [
                    python,
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('purelib'))",
                ],
                text=True,
            ).strip()
            Path(site, "fixture.pth").write_text(str(workspace / "vllm") + "\n")
            assert kwargs["env"]["VLLM_USE_PRECOMPILED"] == "1"
            assert "PYTHONPATH" not in kwargs["env"]
        else:
            actual_run(command, log, description, **kwargs)

    monkeypatch.setitem(state, "_run_logged", run)
    runtime = helpers["prepare_vllm_runtime"](str(remote), "test-branch", revision)
    source = runtime["vllm_source"]
    assert source != remote
    assert (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        == revision
    )
    assert (source / "vllm/__init__.py").read_text() == "REVISION = 'published'\n"
    assert str(source / "vllm/__init__.py") in (workspace / "verify.log").read_text()
    assert (
        "include-system-site-packages = false"
        in (workspace / "venv/pyvenv.cfg").read_text()
    )


def test_remote_runtime_rejects_unpinned_revision(helpers, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ValueError, match="full lowercase Git commit SHA"):
        helpers["prepare_vllm_runtime"]("https://example.com/repo.git", "main", "main")


def test_fresh_batching_sweep_covers_full_test_and_matched_protocol(
    helpers, launch, tmp_path, monkeypatch
):
    """Exercise preparation and CLI dispatch without loading GPU models."""
    partitions = json.loads(helpers["NIAH_SPLIT"].read_text())["niah_kamradt_32k"]
    samples = tmp_path / "samples.jsonl"
    examples = [
        dict(id=key, body="text", question="question", answer="answer")
        for key in partitions["test"] + partitions["val"]
    ]
    samples.write_text("".join(json.dumps(row) + "\n" for row in examples))
    published = tmp_path / "examples/data/niah_32k_samples.jsonl"
    published.parent.mkdir(parents=True)
    published.write_text(json.dumps(examples[0]) + "\n")
    state = helpers["run_batching_benchmarks"].__globals__
    monkeypatch.setitem(state, "ROOT", tmp_path)
    monkeypatch.setitem(state, "NIAH_POOL_SHA256", file_hash(samples))
    configs = []
    for model in ("qwen3", "qwen3_5"):
        config = Path(f"benchmarks/configs/vllm_niah_{model}.yaml")
        copied = tmp_path / config.name
        copied.write_text(config.read_text())
        configs.append(copied)
    observed = []

    def run(command, **kwargs):
        def arg(name):
            return command[command.index(name) + 1]
        cfg = yaml.safe_load(Path(arg("--config")).read_text())
        assert [r["id"] for r in read_samples(cfg, arg("--split"), None)] == partitions["test"]
        assert "--n" not in command
        assert arg("--k") == "2048"
        assert arg("--warmups") == "1" and arg("--repeats") == "3"
        assert arg("--base-settings") == "matched"
        assert arg("--gpu") == "2"
        assert arg("--vllm-python") == "vllm-python"
        engines = command[command.index("--engines") + 1:command.index("--mode")]
        observed.extend(
            (cfg["name"], arg("--mode"), int(arg("--batch-size")), int(arg("--duration")), engine)
            for engine in engines
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    runs = helpers["run_batching_benchmarks"](
        configs, samples, launch["output"], gpu="2",
        vllm_source=launch["vllm_source"], vllm_python="vllm-python",
    )
    expected = {
        (model, mode, size, duration, engine)
        for model in ("Qwen3-1.7B", "Qwen3.5-2B")
        for mode, duration in (("static", 0), ("continuous", 0), ("continuous", 60))
        for size in (1, 2, 4, 8, 16)
        for engine in (["original", "vllm_tiled", "vllm_base"] if mode == "static" else ["vllm_tiled", "vllm_base"])
    }
    assert len(observed) == 70 and set(observed) == expected
    assert len(runs) == len(set(runs)) == 30
    assert json.loads((launch["output"] / "runs.json").read_text()) == list(map(str, runs))
    observed.clear()
    with pytest.raises(FileExistsError):
        helpers["run_batching_benchmarks"](
            configs, samples, launch["output"], gpu="2",
            vllm_source=launch["vllm_source"], vllm_python="vllm-python",
        )
    assert not observed


def test_batching_replay_requires_explicit_available_selection(
    helpers, tmp_path, monkeypatch
):
    monkeypatch.delenv("TOKEN_GATE_BATCH_RUNS", raising=False)
    monkeypatch.delenv("TOKEN_GATE_BATCH_SELECTION", raising=False)
    assert helpers["saved_batching_runs"]() == ([], None)
    monkeypatch.setenv("TOKEN_GATE_BATCH_SELECTION", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        helpers["saved_batching_runs"]()
    monkeypatch.setenv("TOKEN_GATE_BATCH_RUNS", '["saved"]')
    with pytest.raises(ValueError, match="either"):
        helpers["saved_batching_runs"]()
    monkeypatch.delenv("TOKEN_GATE_BATCH_SELECTION")
    assert helpers["saved_batching_runs"]() == ([Path("saved").resolve()], None)


@pytest.mark.parametrize("replay_batching", [False, True])
def test_notebook_prepares_runtime_when_batching_needs_inference(
    helpers, monkeypatch, replay_batching
):
    import runpy

    notebook_path = Path("examples/inference-vllm.ipynb")
    if not notebook_path.is_file():
        pytest.skip("Notebook integration runs once the notebook PR is present")
    setup_calls = []
    monkeypatch.setenv("TOKEN_GATE_SAVED_RUNS", '["saved-single"]')
    monkeypatch.setenv("TOKEN_GATE_TEST_RUNS", '["saved-test"]')
    monkeypatch.delenv("TOKEN_GATE_BATCH_SELECTION", raising=False)
    monkeypatch.setenv("TOKEN_GATE_BATCH_RUNS", '["saved-batch"]' if replay_batching else '[]')
    monkeypatch.setitem(helpers, "prepare_vllm_runtime", lambda *args: setup_calls.append(args) or {})
    monkeypatch.setattr(runpy, "run_path", lambda path: helpers)
    notebook = json.loads(notebook_path.read_text())
    code = next(c for c in notebook["cells"] if c["cell_type"] == "code")
    namespace = {}
    exec("".join(code["source"]), namespace)
    assert bool(setup_calls) == (not replay_batching)
    assert bool(namespace["BATCH_RUNS"]) == replay_batching
