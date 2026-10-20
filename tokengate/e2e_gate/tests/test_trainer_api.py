"""Tests for the TRL-style Python training interface (tokengate.trainer).

CPU-only: validates that TokenGateConfig fails loudly on the same conditions
the shell driver does, that the dataset handoff (prepared store name vs raw
rows + task) resolves correctly, that the auto-carved eval split matches the
committed iso-split recipe (sorted ids, first N val), and that the CLI
TokenGateTrainer generates parses cleanly with the trainer's real argparse
parser (so the two interfaces cannot silently drift apart).

Run: python -m tokengate.e2e_gate.tests.test_trainer_api
"""
import dataclasses
import json
import os
import pickle
import tempfile

from tokengate import TokenGateConfig, TokenGateTrainer

_REQUIRED = dict(
    base_model="Qwen/Qwen3-1.7B",
    gate_layer=1,
    output_dir="e2e_gates/test-out",
    learning_rate=1e-4,
    epochs=20,
    k_eval=[1000],
    lora_rank=16, lora_alpha=32, lora_lr=1e-4,
    lora_targets=["q_proj", "v_proj"],
)

_ROWS = [{"article": f"document {i} " * 50, "answer": f"answer {i}"}
         for i in range(4)]


def _cfg(**overrides):
    return TokenGateConfig(**{**_REQUIRED, **overrides})


def test_cli_round_trip():
    """The generated trainer CLI must parse with the trainer's real parser."""
    from tokengate.e2e_gate.train.train_e2e_gate import build_parser
    config = _cfg(min_article_tokens=2048, max_article_tokens=16384,
                  eval_group_override="biodexr:biodex_reactions",
                  swa_freq=1, eval_freq=600, num_gpus=8,
                  wandb_project="token-gate")
    trainer = TokenGateTrainer(config, train_dataset="biodex/biodex_reactions")
    assert trainer._dataset == "biodex" and trainer._task == "biodex_reactions"
    args = build_parser().parse_args(trainer._trainer_args())
    assert args.base_model == config.base_model
    assert args.dataset == "biodex" and args.task == "biodex_reactions"
    assert args.gate_layer == 1 and args.k_eval == [1000]
    assert args.phase2_loguniform and not args.phase2_len_list
    # converged-recipe defaults: Muon run + gate optimizer, no decay on
    # scalars, shuffled epochs — none of them stated in the config above.
    assert args.optimizer == "muon" and args.muon_gate and args.no_decay_scalars
    assert args.shuffle_epochs
    assert args.weight_decay == 0.1 and args.lr_schedule == "constant"
    assert args.lora_rank == 16 and args.lora_targets == ["q_proj", "v_proj"]
    assert args.lora_layers == "all"
    assert args.checkpoint_freq == 600        # None → eval_freq
    assert args.attn_backends == "fa2"        # Qwen3 dense family default
    assert args.compile_phase2 and args.pack_phase2_max_tokens == 12288
    assert args.pretrain_packed and args.pretrain_data_root == "datasets"
    # no explicit eval_split → the auto-carved split in output_dir
    assert args.eval_split_manifest == os.path.join(
        config.output_dir, "eval_split_auto.json")
    assert args.eval_min_article_tokens == 0
    assert args.eval_max_article_tokens == 39000   # default window
    assert args.swa_freq == 1
    assert args.wandb_project == "token-gate" and args.wandb_run_name
    # torchrun invocation shape
    cmd = trainer.command
    assert "torch.distributed.run" in cmd[2] and "--nproc_per_node=8" in cmd
    # adamw run: gate optimizer must be demoted explicitly, then no muon flags
    cfg_adamw = _cfg(optim="adamw", gate_optimizer="adamw", num_gpus=1)
    args2 = build_parser().parse_args(
        TokenGateTrainer(cfg_adamw, train_dataset="biodex/biodex_reactions")
        ._trainer_args())
    assert args2.optimizer == "adamw" and not args2.muon_gate
    print("  [pass] CLI round-trips through the trainer's parser")


def test_dataset_handoff():
    """Store names vs raw rows resolve/refuse like the docstring says."""
    cases = [
        (dict(train_dataset="biodex"), "store name"),
        (dict(train_dataset="biodex/biodex_reactions", eval_dataset=_ROWS),
         "already holds its val split"),
        (dict(train_dataset="biodex/biodex_reactions", task="niah_kamradt"),
         "conflicts with the store name"),
        (dict(train_dataset=_ROWS), "needs task="),
        (dict(train_dataset=_ROWS, task="biodex_reactions"),
         "needs eval_dataset"),
        (dict(train_dataset=_ROWS, eval_dataset=_ROWS, task="not_a_task"),
         "Unknown task"),
        (dict(train_dataset=[{"answer": "x"}], eval_dataset=_ROWS,
              task="biodex_reactions"), "'article'"),
        (dict(train_dataset=[{"article": "doc"}], eval_dataset=_ROWS,
              task="biodex_reactions"), "'answer'"),
    ]
    for kwargs, expect in cases:
        try:
            TokenGateTrainer(_cfg(num_gpus=1), **kwargs)
        except (ValueError, FileNotFoundError) as e:
            assert expect in str(e), (kwargs, str(e))
        else:
            raise AssertionError(f"trainer accepted invalid kwargs {kwargs}")
    # the happy raw-rows path: rows normalized, "state" accepted as alias
    tr = TokenGateTrainer(
        _cfg(num_gpus=1),
        train_dataset=[{"state": "doc one " * 50, "answer": "a"}],
        eval_dataset=[{"article": "doc two " * 50, "answer": "b"}],
        task="biodex_reactions")
    assert tr._dataset == "biodex_reactions" and tr._task == "biodex_reactions"
    assert tr._train_rows[0]["article"].startswith("doc one")
    assert tr._train_rows[0]["id"] == "biodex_reactions_000000"
    print(f"  [pass] dataset handoff: {len(cases)} refusals + raw-row "
          "normalization")


def test_auto_eval_split():
    """The auto split is the committed iso recipe: sorted ids, first N val."""
    with tempfile.TemporaryDirectory() as td:
        store = os.path.join(td, "data", "biodex", "biodex_reactions",
                             "tokenized", "Qwen3-1.7B")
        os.makedirs(store)
        ids = [f"id_{i:03d}" for i in range(7)]
        with open(os.path.join(store, "val.pkl"), "wb") as f:
            pickle.dump([{"id": i} for i in reversed(ids)], f)
        config = _cfg(num_gpus=1, data_root=os.path.join(td, "data"),
                      output_dir=os.path.join(td, "out"), eval_val_samples=5)
        trainer = TokenGateTrainer(config,
                                   train_dataset="biodex/biodex_reactions")
        trainer._write_auto_split()
        split = json.load(open(trainer.eval_split_path))
        assert split["biodex_reactions"]["val"] == ids[:5]      # sorted, first N
        assert split["biodex_reactions"]["test"] == ids[5:]
        # explicit eval_split short-circuits the auto path
        config2 = _cfg(num_gpus=1, eval_split="my_split.json")
        t2 = TokenGateTrainer(config2, train_dataset="biodex/biodex_reactions")
        assert t2.eval_split_path == "my_split.json"
    print("  [pass] auto eval split carves sorted ids (first N val, rest test)")


def test_fail_loudly():
    """Construction must refuse the same configs the shell driver refuses."""
    cases = [
        (dict(lora_alpha=None), "lora_rank=16 requires"),
        (dict(optim="adamw"), "gate_optimizer='muon' requires"),
        (dict(gate_optimizer="sgd"), "gate_optimizer must be"),
        (dict(phase2_loguniform=False), "K source"),
        (dict(phase2_len_list=[4000]), "mutually exclusive"),
        (dict(k_eval=[]), "k_eval"),
        (dict(eval_val_samples=0), "eval_val_samples"),
        (dict(adapter_type="delta"), "rank/alpha-free"),
        (dict(base_model="meta-llama/Llama-3-8B"), "kernel defaults"),
        (dict(nnodes=2), "nnodes>1"),
        (dict(attn_backend="flex", pack_phase2=True), "pack_phase2 requires"),
    ]
    for overrides, expect in cases:
        try:
            _cfg(**overrides)
        except ValueError as e:
            assert expect in str(e), (overrides, str(e))
        else:
            raise AssertionError(f"config accepted invalid overrides {overrides}")
    print(f"  [pass] {len(cases)} invalid configs fail loudly")


def test_family_kernel_resolution():
    cfg35 = _cfg(base_model="Qwen/Qwen3.5-27B")
    assert cfg35.attn_backend == "flex"
    assert not cfg35.pack_phase2 and not cfg35.compile_phase2
    other = _cfg(base_model="meta-llama/Llama-3-8B", attn_backend="flex")
    assert not other.pack_phase2 and not other.compile_phase2
    print("  [pass] per-family kernel defaults resolve like the driver")


def test_asdict_round_trip():
    """A resolved config must reconstruct from its own asdict() unchanged."""
    config = _cfg()
    clone = TokenGateConfig(**dataclasses.asdict(config))
    assert dataclasses.asdict(clone) == dataclasses.asdict(config)
    print("  [pass] asdict() reconstruction is stable")


def main():
    print("[test_trainer_api]")
    test_cli_round_trip()
    test_dataset_handoff()
    test_auto_eval_split()
    test_fail_loudly()
    test_family_kernel_resolution()
    test_asdict_round_trip()
    print("[test_trainer_api] ALL PASS")


if __name__ == "__main__":
    main()
