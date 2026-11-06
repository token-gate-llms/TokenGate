"""Verify the shared trainer's objectives, validation and exact-resume policy."""

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from tokengate.datasets.pack import epoch_order
from tokengate.e2e_gate.gate_module.lora import apply_lora_to_layers
from tokengate.e2e_gate.train.selection import averaged_parameters
from tokengate.e2e_gate.train.train_e2e_gate import answer_loss, sampled_budget


def test_baseline_answer_loss_and_gradients_match_full_logits():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).eval()
    model.requires_grad_(False)
    apply_lora_to_layers(model, [0, 1], rank=4, alpha=8, targets=["q_proj", "v_proj"])
    row = {"input_ids": list(range(1, 13)), "ctx_len": 8, "n_ans": 4}
    loss = answer_loss(model, row)
    loss.backward()
    actual = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad
    }
    assert any(g.abs().sum() > 0 for g in actual.values())
    model.zero_grad(set_to_none=True)
    logits = model(torch.tensor([row["input_ids"]]), use_cache=False).logits
    reference = F.cross_entropy(logits[0, 7:11], torch.tensor(row["input_ids"][8:]))
    reference.backward()
    torch.testing.assert_close(loss, reference)
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, actual[name])
        else:
            assert p.grad is None


def test_epoch_shards_cover_every_example_once():
    order = epoch_order(12000, seed=2, epoch=3)
    shards = [set(order[rank::4]) for rank in range(4)]
    assert all(len(shard) == 3000 for shard in shards)
    assert len(set.union(*shards)) == 12000
    assert (order == epoch_order(12000, seed=2, epoch=3)).all()


def test_budget_excludes_protected_tokens_and_is_reproducible():
    row = {"input_ids": [1] * 1000, "article_start": 60, "article_end": 940}
    total, budget = sampled_budget(row, seed=2, step=15)
    assert 128 <= total <= 500
    assert budget == max(1, total - 120)
    assert (total, budget) == sampled_budget(row, seed=2, step=15)


def test_swa_swap_restores_live_weights_on_failure():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    try:
        with averaged_parameters({"p": parameter}, {"p": torch.tensor([3.0])}):
            assert parameter.item() == 3.0
            raise RuntimeError("evaluation failed")
    except RuntimeError:
        pass
    assert parameter.item() == 1.0


def test_reproduction_plan_reuses_public_trainer_without_running_jobs(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    from tokengate.e2e_gate.train.selection import load_recipe
    from tokengate.trainer import TokenGateTrainer

    source = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "reproduce", source / "train-scripts/niah/reproduce.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    recipe = load_recipe()
    configs = list(driver.configurations(tmp_path, recipe, smoke=False))
    assert len(configs) == len({c["run_id"] for c in configs}) == 18
    assert {c["seed"] for c in configs} == {0, 1, 2}
    assert sum(c["gate_layer"] is None for c in configs) == 6
    assert len(list(driver.configurations(tmp_path, recipe, smoke=True))) == 6
    assert not list(tmp_path.iterdir())
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(configs[0]))
    command = TokenGateTrainer.reproduction_command(
        config_path, resume=True, stop_after=4
    )
    assert "tokengate.e2e_gate.train.train_e2e_gate" in command
    assert "--nproc-per-node=4" in command
    assert command[-3:] == ["--resume", "--stop-after", "4"]
    assert "--reproduction-config" in command


@pytest.fixture
def cpu_training(tmp_path, monkeypatch):
    """Exercise the real trainer with a tiny Qwen and real packed inputs.

    Only model/tokenizer downloads and CUDA placement are replaced. Optimizers,
    scheduling, sampling, SWA, validation, checkpoint IO and resume remain real.
    """
    import json
    import pickle
    import runpy
    from pathlib import Path
    from types import SimpleNamespace

    from tokengate.e2e_gate.artifacts import capture_source, file_hash

    from tokengate.datasets import pack
    from tokengate.e2e_gate.train import train_e2e_gate as runtime
    from tokengate.e2e_gate.train.selection import load_recipe

    source = Path(__file__).resolve().parents[3]
    driver = runpy.run_path(str(source / "train-scripts/niah/reproduce.py"))
    config = next(
        c
        for c in driver["configurations"](tmp_path, load_recipe(), smoke=True)
        if c["gate_layer"] is None
    )
    config.update(world_size=1, lora_rank=4, lora_alpha=8, max_gen_tokens=2)
    manifest = capture_source(source, tmp_path, training=True)
    config.update(code_manifest=str(manifest), code_manifest_sha256=file_hash(manifest))
    directory = (
        tmp_path / "data/prepared/niah_kamradt/niah_kamradt/tokenized/Qwen3-1.7B"
    )
    directory.mkdir(parents=True)
    row = dict(
        full_ids=list(range(1, 13)),
        ctx_len=8,
        n_ans=4,
        article_start=1,
        article_end=7,
        article="fixture article",
        target="fixture",
        prompt_version="niah_v1",
    )
    rows = [
        dict(
            row,
            id=str(i),
            full_ids=[1 + ((token + i) % 30) for token in row["full_ids"]],
        )
        for i in range(12000)
    ]
    (directory / "train.pkl").write_bytes(pickle.dumps(rows))
    pack.pack_one(directory / "train.pkl", directory, model="fixture")
    validation = [
        dict(row, id=str(i), full_ids=[1] * 8 + [9, 10, 11, 12]) for i in range(200)
    ]
    (directory / "val.pkl").write_bytes(pickle.dumps(validation))
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "niah_kamradt": dict(
                    val=[str(i) for i in range(50)],
                    test=[str(i) for i in range(50, 200)],
                )
            }
        )
    )
    config["eval_split_manifest"] = str(split)

    class TorchOnCPU:
        def __getattr__(self, name):
            return getattr(torch, name)

        @staticmethod
        def device(*args, **kwargs):
            return torch.device("cpu")

    monkeypatch.setattr(runtime, "torch", TorchOnCPU())
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda: torch.tensor([], dtype=torch.uint8)
    )
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *a: None)
    monkeypatch.setattr(runtime, "_HAVE_MPL", False)
    monkeypatch.setattr(
        runtime,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda *a, **k: SimpleNamespace(
                pad_token="pad",
                eos_token_id=31,
                pad_token_id=0,
                encode=lambda *a, **k: [1, 2],
                decode=lambda *a, **k: "fixture",
            )
        ),
    )
    monkeypatch.setattr(
        runtime,
        "AutoModelForCausalLM",
        SimpleNamespace(
            from_pretrained=lambda *a, **k: Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    eos_token_id=31,
                    pad_token_id=0,
                )
            ).eval()
        ),
    )
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.setenv(key, str(tmp_path / "cache"))
    for key in ("RANK", "LOCAL_RANK"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    deterministic = torch.are_deterministic_algorithms_enabled()
    yield runtime, config
    torch.use_deterministic_algorithms(deterministic)


def test_common_trainer_exact_resume_and_swa(cpu_training):
    import json
    from pathlib import Path

    from tokengate.e2e_gate.train.selection import verify_resume

    runtime, config = cpu_training
    runtime.train(runtime.reproduction_args(config, stop_after=2))
    runtime.train(runtime.reproduction_args(config, resume=True, stop_after=4))
    control = {
        **config,
        "run_id": "control",
        "output_dir": str(Path(config["root"]) / "control"),
    }
    runtime.train(runtime.reproduction_args(control, stop_after=4))
    run, reference = Path(config["output_dir"]), Path(control["output_dir"])
    verify_resume(run, reference)
    state = torch.load(run / "train_state.pt", weights_only=False)
    assert state["average_count"] == 2
    assert state["scheduler"]["last_epoch"] == 4
    records = [
        json.loads(line)
        for line in (run / "train-rank0.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in records] == [1, 2, 3, 4]
    expected_lr = [
        config["learning_rate"] * factor for factor in (1e-8, 0.500000005, 1, 1)
    ]
    assert [r["lr"] for r in records] == pytest.approx(expected_lr)
    assert all(r["budget"] is None for r in records)
    validation = [
        json.loads(line) for line in (run / "validation.jsonl").read_text().splitlines()
    ]
    assert {(r["step"], r["variant"]) for r in validation} == {
        (2, "raw"),
        (4, "raw"),
        (4, "swa"),
    }
    assert not json.loads((run / "status.json").read_text())["complete"]
    assert list((run / "checkpoints").glob("swa_lora*.pt"))
    # Reject a scheduler mismatch as strictly as a parameter mismatch.
    state["scheduler"]["last_epoch"] += 1
    torch.save(state, run / "train_state.pt")
    with pytest.raises(AssertionError, match="scheduler"):
        verify_resume(run, reference)


@pytest.mark.parametrize("failure", ["oom", "nonfinite"])
def test_common_trainer_never_skips_failed_reproduction_steps(
    cpu_training, monkeypatch, failure
):
    from pathlib import Path

    runtime, config = cpu_training
    original = runtime.answer_loss

    def fail(model, row):
        if failure == "oom":
            raise RuntimeError("CUDA out of memory")
        return original(model, row) * float("nan")

    monkeypatch.setattr(runtime, "answer_loss", fail)
    with pytest.raises(
        (RuntimeError, FloatingPointError), match="out of memory|Nonfinite"
    ):
        runtime.train(runtime.reproduction_args(config, stop_after=2))
    output = Path(config["output_dir"])
    assert not (output / "status.json").exists()
    assert not (output / "train_state.pt").exists()


def test_common_trainer_releases_gate_graph_on_failure(cpu_training, monkeypatch):
    import weakref

    from tokengate.e2e_gate.gate_module import gate_forward

    runtime, config = cpu_training
    config = {**config, "gate_layer": 0, "attn_backend": "flex"}
    observed = {}

    def fail(model, gate, *args, **kwargs):
        gate._sparsity_term = next(gate.parameters()).square().sum()
        gate._kept_counts = [1]
        gate._n_art_counts = [6]
        observed.update(gate=gate, graph=weakref.ref(gate._sparsity_term))
        raise RuntimeError("injected gated forward failure")

    monkeypatch.setattr(runtime, "warmup_flex_attention", lambda *a, **k: None)
    monkeypatch.setattr(gate_forward, "forward_backward_with_gate_multi_K_detach", fail)
    with pytest.raises(RuntimeError, match="injected gated forward failure"):
        runtime.train(runtime.reproduction_args(config, stop_after=2))
    assert observed["graph"]() is None
    assert observed["gate"]._sparsity_term is None
    assert observed["gate"]._kept_counts == []
    assert observed["gate"]._n_art_counts == []


def test_public_sft_training_saves_live_and_averaged_adapters(
    cpu_training, monkeypatch
):
    from pathlib import Path

    runtime, config = cpu_training
    args = runtime.reproduction_args(config, stop_after=4)
    args.reproduction = None  # Exercise the public trainer without the fixed protocol.
    args.save_swa_step_checkpoints = True
    evaluations = []

    def evaluate(model, gate, *args, **kwargs):
        assert gate is None
        evaluations.append(1)
        return 0.5, 0.0, 1.0, 1.0, 1.0, [], None, 8, 8, 6, 6, {}, 1.0, 0.0

    # Generic evaluation's text formatting is independent of the tiny tokenizer
    # fixture. Keep its metric aggregation, SWA swaps and checkpointing real.
    monkeypatch.setattr(runtime, "evaluate", evaluate)
    monkeypatch.setattr(
        runtime, "evaluate_no_compression", lambda *a, **k: (0.5, 0.0, 1.0, {}, {})
    )
    runtime.train(args)
    output = Path(config["output_dir"])
    assert len(evaluations) >= 4
    assert (output / "lora_final.pt").is_file()
    assert (output / "swa_lora_final.pt").is_file()
    assert (output / "swa_lora_step4.pt").is_file()
    assert not list(output.glob("*gate*.pt"))
