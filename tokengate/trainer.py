"""TRL-style Python interface for training TokenGate.

`TokenGateConfig` + `TokenGateTrainer` mirror the TRL/PEFT shape — a config
object plus a trainer that takes the dataset — and are a thin, faithful wrapper
around the same launch the shell lane configs perform
(train-scripts/run_e2e_gate_pretrain.sh → e2e_gate_lib.sh →
`torchrun -m tokengate.e2e_gate.train.train_e2e_gate`): the config builds the
identical trainer CLI and the trainer launches it under torchrun, so a
Python-configured run and a lane-script run with the same values are the same
run.

    from tokengate import TokenGateConfig, TokenGateTrainer

    config = TokenGateConfig(
        base_model="Qwen/Qwen3-1.7B",
        gate_layer=1,
        output_dir="e2e_gates/biodexr-1p7b",
        learning_rate=1e-4,
        epochs=20,
        k_eval=[1000],
        lora_rank=16, lora_alpha=32, lora_lr=1e-4,
        lora_targets=["q_proj", "v_proj"],
    )
    trainer = TokenGateTrainer(
        config,
        train_dataset=train_rows,     # [{"article": ..., "answer": ...}, ...]
        eval_dataset=val_rows,        # same schema (an HF Dataset also works)
        task="biodex_reactions",      # registered prompt/metric (datasets/tasks)
    )
    trainer.train()

`train_dataset` takes either raw rows as above — the trainer tokenizes them
into the store and carves a deterministic eval split — or the name of an
already-prepared store, `"<dataset>/<task>"` under `data_root` (written by a
per-dataset prepare.py), in which case `eval_dataset` stays unset and the
store's own val split is used.

Config philosophy: run-defining hyperparameters are REQUIRED (no silent
defaults — construction fails loudly), while the converged recipe carries its
documented defaults (Muon for both run and gate, constant LR after warmup,
weight decay 0.1, no decay on gate scalars, shuffled epochs, SWA off, LoRA on
all layers).
"""
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

_TRAIN_MODULE = "tokengate.e2e_gate.train.train_e2e_gate"
_PACK_MODULE = "tokengate.datasets.pack"

_AUTO_SPLIT_NAME = "eval_split_auto.json"


@dataclass
class TokenGateConfig:
    """Full configuration of one TokenGate training run.

    Required fields are the run-defining hyperparameters; defaulted fields
    carry the converged-recipe values and are documented inline.
    """

    # ── Model / gate (required) ──────────────────────────────────────────────
    base_model: str                 # HF model id, e.g. "Qwen/Qwen3-1.7B"
    gate_layer: int                 # layer L whose hidden states the gate scores
    output_dir: str                 # where gate_final.pt / checkpoints / curves land
    # ── Schedule (required) ──────────────────────────────────────────────────
    learning_rate: float
    epochs: float
    k_eval: List[int]               # in-training eval K sweep (first = headline K)
    # ── Adapter (required rank; the backbone itself stays frozen) ────────────
    lora_rank: int                  # 0 disables the factored adapter
    # LoRA (required whenever lora_rank > 0)
    lora_alpha: Optional[float] = None
    lora_lr: Optional[float] = None
    lora_targets: Optional[List[str]] = None   # e.g. ["q_proj", "v_proj"]
    adapter_type: str = "lora"      # "lora" | "delta" (mutually exclusive with lora_rank>0)
    lora_layers: str = "all"
    # ── K source (converged recipe: loguniform) ──────────────────────────────
    phase2_loguniform: bool = True  # per step: L ~ LogUniform(128, seq/2) + full sentinel
    phase2_len_list: Optional[List[int]] = None  # fixed target lengths instead
    # ── Optimizer / schedule (converged-recipe defaults) ─────────────────────
    optim: str = "muon"             # run optimizer (TRL-style name)
    gate_optimizer: str = "muon"    # the gate's 2D matrices: "muon" | "adamw"
    no_decay_scalars: bool = True   # no weight decay on gate scalars/vectors
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    lr_schedule: str = "constant"
    shuffle_epochs: bool = True     # per-epoch shuffled pass vs i.i.d. with-replacement
    seed: int = 0
    batch_size: int = 1             # per-GPU micro-batch
    grad_accum: int = 1
    grad_checkpoint: bool = False
    # ── Eval / checkpoint cadence (driver defaults) ──────────────────────────
    eval_freq: int = 1800
    checkpoint_freq: Optional[int] = None      # None = eval_freq (driver default)
    eval_gen_batch_size: int = 4
    gate_full_eval_freq: int = 0    # gate@full-context diagnostic every Nth eval (0 = off)
    swa_freq: int = -1              # SWA off by default; 1 enables
    swa_start: int = 0
    swa_eval_freq: int = 1
    log_examples: int = 20
    # ── Eval split (auto-carved; see TokenGateTrainer) ───────────────────────
    eval_val_samples: int = 50      # eval rows used in-training; the rest are the
                                    # reserved test partition (sorted-id carve)
    eval_split: Optional[str] = None  # explicit split JSON overriding the auto carve
    eval_only_datasets: Optional[List[str]] = None  # extra eval-only stores, "ds/task"
    # ── Data windows (driver defaults) ───────────────────────────────────────
    data_root: str = "datasets"     # tokenized stores (prepare.py layout)
    packed: bool = True             # memory-mapped packed token store (built idempotently)
    min_article_tokens: int = 0     # train rows outside [min,max] are dropped, not truncated
    max_article_tokens: int = 40960
    eval_min_article_tokens: int = 0
    eval_max_article_tokens: int = 39000  # fits a 40,960-token window + generation headroom
    max_answer_tokens: int = 1024
    max_gen_tokens: int = 1024
    target_balanced_map: Optional[str] = None
    eval_group_override: Optional[str] = None
    eval_test_full_tasks: Optional[List[str]] = None
    # ── Kernels (None = resolved from the model family, like the driver) ─────
    attn_backend: Optional[str] = None     # "fa2" | "flash" | "flex"
    compile_phase2: Optional[bool] = None
    pack_phase2: Optional[bool] = None
    pack_phase2_max_tokens: int = 12288
    # ── Launch topology ──────────────────────────────────────────────────────
    num_gpus: Optional[int] = None  # None = torch.cuda.device_count()
    nnodes: int = 1
    node_rank: Optional[int] = None          # required when nnodes > 1
    master_addr: Optional[str] = None        # required when nnodes > 1
    master_port: int = 29500
    # ── Logging / lifecycle ──────────────────────────────────────────────────
    wandb_project: Optional[str] = None      # None = no W&B logging
    run_name: Optional[str] = None           # None = derived from model/dataset/lr/seed
    max_steps: Optional[int] = None          # cap the train loop (smoke runs)
    profile: bool = False                    # per-step timing; skips eval/checkpoint/SWA
    resume_from: Optional[str] = None        # run dir or train_state_step*.pt
    extra_args: List[str] = field(default_factory=list)  # verbatim extra trainer CLI args

    def __post_init__(self):
        if self.gate_optimizer not in ("muon", "adamw"):
            raise ValueError(f"gate_optimizer must be 'muon' or 'adamw', got "
                             f"{self.gate_optimizer!r}")
        if self.gate_optimizer == "muon" and self.optim != "muon":
            raise ValueError("gate_optimizer='muon' requires optim='muon' "
                             f"(got optim={self.optim!r})")
        if self.adapter_type == "delta":
            if self.lora_rank > 0 or self.lora_targets is not None:
                raise ValueError("adapter_type='delta' is rank/alpha-free with fixed "
                                 "targets; set lora_rank=0 and leave lora_targets unset")
        elif self.adapter_type != "lora":
            raise ValueError(f"adapter_type must be 'lora' or 'delta', got "
                             f"{self.adapter_type!r}")
        if self.adapter_type == "lora" and self.lora_rank > 0:
            missing = [n for n in ("lora_alpha", "lora_lr", "lora_targets")
                       if getattr(self, n) is None]
            if missing:
                raise ValueError(f"lora_rank={self.lora_rank} requires {missing} "
                                 "(no silent LoRA defaults)")
        if self.phase2_loguniform and self.phase2_len_list:
            raise ValueError("phase2_loguniform and phase2_len_list are mutually "
                             "exclusive K sources")
        if not self.phase2_loguniform and not self.phase2_len_list:
            raise ValueError("set a K source: phase2_loguniform=True or phase2_len_list")
        if not self.k_eval:
            raise ValueError("k_eval must list at least one K (the headline eval K)")
        if self.eval_val_samples < 1:
            raise ValueError("eval_val_samples must be >= 1")
        if self.nnodes > 1 and (self.node_rank is None or self.master_addr is None):
            raise ValueError("nnodes>1 requires node_rank (0..nnodes-1, per node) "
                             "and master_addr (the rank-0 node)")
        if self.attn_backend is None:
            # Same per-family kernel defaults as the shell driver; unknown
            # families must choose explicitly rather than inherit a wrong kernel.
            if self.base_model.startswith("Qwen/Qwen3.5"):
                self.attn_backend = "flex"       # fa2bias backward is head_dim-128 only
                self.compile_phase2 = False if self.compile_phase2 is None else self.compile_phase2
                self.pack_phase2 = False if self.pack_phase2 is None else self.pack_phase2
            elif self.base_model.startswith("Qwen/Qwen3-"):
                self.attn_backend = "fa2"        # FlashAttention-2 + key-bias fork
                self.compile_phase2 = True if self.compile_phase2 is None else self.compile_phase2
                self.pack_phase2 = True if self.pack_phase2 is None else self.pack_phase2
            else:
                raise ValueError(
                    f"no kernel defaults for base_model={self.base_model!r}: set "
                    "attn_backend ('fa2'|'flash'|'flex'), compile_phase2 and "
                    "pack_phase2 explicitly")
        else:
            self.compile_phase2 = bool(self.compile_phase2)
            self.pack_phase2 = bool(self.pack_phase2)
        if self.pack_phase2 and self.attn_backend != "fa2":
            raise ValueError("pack_phase2 requires attn_backend='fa2' "
                             "(the varlen key-bias kernel)")


def _parse_store_name(name: str) -> Tuple[str, str]:
    """'<dataset>/<task>' (or '<dataset>:<task>') → (dataset, task)."""
    sep = "/" if "/" in name else ":"
    parts = [p for p in name.split(sep) if p]
    if len(parts) != 2:
        raise ValueError(
            f"train_dataset {name!r} is not a '<dataset>/<task>' store name; pass "
            "either a prepared store under data_root or raw rows + task=...")
    return parts[0], parts[1]


def _normalize_rows(rows: Iterable[dict], dataset: str, task: str) -> List[dict]:
    """Coerce user rows to the canonical raw schema (fail loudly on gaps)."""
    out = []
    for i, r in enumerate(rows):
        article = r.get("article", r.get("state"))
        if not article:
            raise ValueError(
                f"row {i}: needs a non-empty 'article' (alias: 'state') — the "
                f"prunable document; got keys={sorted(r.keys())}")
        if "answer" not in r:
            raise ValueError(f"row {i}: needs an 'answer' string; got "
                             f"keys={sorted(r.keys())}")
        out.append({
            "id": str(r.get("id", f"{task}_{i:06d}")),
            "dataset": dataset,
            "task": task,
            "article": article,
            "answer": r["answer"],
            "metadata": r.get("metadata", {}),
        })
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row ids after normalization — pass unique "
                         "'id' values or omit them entirely")
    return out


class TokenGateTrainer:
    """Launches one TokenGate training run from a `TokenGateConfig`.

    Like TRL, the dataset goes to the trainer, not the config:

      * ``train_dataset=rows, eval_dataset=rows, task="<name>"`` — raw rows
        (each ``{"article": str, "answer": str}``; ``"state"`` is accepted as
        an alias for ``"article"``, ``"id"``/``"metadata"`` optional; any HF
        Dataset with those columns works).  ``task`` names a registered Task
        (tokengate/datasets/tasks) — the prompt template and metric.  The
        trainer tokenizes the rows into ``data_root/<task>/<task>/`` and
        carves a deterministic eval split (sorted ids, first
        ``eval_val_samples`` in-training, rest reserved as test).
      * ``train_dataset="<dataset>/<task>"`` — an already-prepared store under
        ``data_root`` (the per-dataset prepare.py layout); its own val split
        is used and ``eval_dataset`` stays unset.

    `train()` (idempotently) builds the packed token store and runs the
    trainer under torchrun — the identical launch the shell lane configs
    perform. Blocks until training finishes; the final gate is
    `<output_dir>/gate_final.pt` (plus the name-paired adapter file).
    """

    def __init__(self, config: TokenGateConfig, *,
                 train_dataset: Union[str, Iterable[dict]],
                 eval_dataset: Optional[Iterable[dict]] = None,
                 task: Optional[str] = None,
                 dataset_name: Optional[str] = None):
        self.config = config
        self._train_rows: Optional[List[dict]] = None
        self._eval_rows: Optional[List[dict]] = None

        if isinstance(train_dataset, str):
            ds, tk = _parse_store_name(train_dataset)
            if task is not None and task != tk:
                raise ValueError(
                    f"task={task!r} conflicts with the store name "
                    f"{train_dataset!r} (its task is {tk!r}); a prepared "
                    "store's task component IS the task")
            if eval_dataset is not None:
                raise ValueError(
                    "eval_dataset is only for raw rows — a prepared store "
                    "already holds its val split (val.pkl)")
            self._dataset, self._task = ds, tk
        else:
            if task is None:
                raise ValueError(
                    "raw-row train_dataset needs task=<registered task name> "
                    "(the prompt template + metric; see tokengate/datasets/tasks)")
            if eval_dataset is None:
                raise ValueError(
                    "raw-row train_dataset needs eval_dataset too — in-training "
                    "eval drives checkpoint selection (k_eval)")
            from tokengate.datasets.tasks import get_task
            get_task(task)  # fail loudly on unknown task names
            self._dataset = dataset_name or task
            self._task = task
            self._train_rows = _normalize_rows(train_dataset, self._dataset, task)
            self._eval_rows = _normalize_rows(eval_dataset, self._dataset, task)

        self._eval_only = [_parse_store_name(e)
                           for e in (config.eval_only_datasets or [])]
        if config.num_gpus is None:
            import torch
            config.num_gpus = torch.cuda.device_count()
            if config.num_gpus < 1:
                raise RuntimeError("no CUDA devices visible; set num_gpus explicitly "
                                   "on a GPU machine")

    # ── Eval split (sorted-id carve, same recipe as the committed iso splits) ─
    @property
    def eval_split_path(self) -> str:
        c = self.config
        return c.eval_split or str(Path(c.output_dir) / _AUTO_SPLIT_NAME)

    def _model_store_dir(self) -> str:
        return Path(self.config.base_model).name

    def _write_auto_split(self):
        c = self.config
        split = {}
        for ds, tk in [(self._dataset, self._task)] + self._eval_only:
            p = (Path(c.data_root) / ds / tk / "tokenized"
                 / self._model_store_dir() / "val.pkl")
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found — the store has no val split to carve the "
                    "eval set from (prepare the dataset first)")
            with open(p, "rb") as f:
                ids = [r["id"] for r in pickle.load(f)]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{p}: row ids are not unique")
            ids = sorted(ids)
            n = min(c.eval_val_samples, len(ids))
            split[tk] = {"group": tk, "val": ids[:n], "test": ids[n:]}
        out = Path(self.eval_split_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(split, f, indent=0)
        for tk, v in split.items():
            print(f"[tokengate.trainer] eval split {tk}: {len(v['val'])} val / "
                  f"{len(v['test'])} reserved test → {out}")

    # ── Raw-row preparation (tokenize into the standard store layout) ────────
    def _prepare_rows(self):
        from tokengate.datasets.prepare import prepare_dataset
        c = self.config
        print(f"[tokengate.trainer] tokenizing {len(self._train_rows)} train / "
              f"{len(self._eval_rows)} eval rows into "
              f"{c.data_root}/{self._dataset}/{self._task} …")
        prepare_dataset(
            dataset_name=self._dataset,
            task_name=self._task,
            load_train_fn=lambda: self._train_rows,
            load_val_fn=lambda: self._eval_rows,
            model=c.base_model,
            output_root=Path(c.data_root),
            max_article_tokens=c.max_article_tokens,
            min_article_tokens=c.min_article_tokens,
            run_sanity=False,
        )

    # ── CLI construction (mirrors e2e_gate_lib.sh:train_gate) ────────────────
    def _trainer_args(self) -> List[str]:
        c = self.config
        args = [
            "--base-model", c.base_model,
            "--dataset", self._dataset,
            "--task", self._task,
            "--gate-layer", str(c.gate_layer),
            "--epochs", str(c.epochs),
            "--lr", str(c.learning_rate),
            "--seed", str(c.seed),
            "--batch-size", str(c.batch_size),
            "--grad-accum", str(c.grad_accum),
            "--checkpoint-freq", str(c.checkpoint_freq if c.checkpoint_freq
                                     is not None else c.eval_freq),
            "--eval-freq", str(c.eval_freq),
            "--log-examples", str(c.log_examples),
            "--max-gen-tokens", str(c.max_gen_tokens),
            "--eval-gen-batch-size", str(c.eval_gen_batch_size),
            "--warmup-steps", str(c.warmup_steps),
            "--lr-schedule", c.lr_schedule,
            "--swa-freq", str(c.swa_freq),
            "--swa-start", str(c.swa_start),
            "--swa-eval-freq", str(c.swa_eval_freq),
            "--output-dir", c.output_dir,
            "--k-eval", *[str(k) for k in c.k_eval],
            "--scalar-lr-mult", "1",
            "--max-article-tokens", str(c.max_article_tokens),
            "--min-article-tokens", str(c.min_article_tokens),
            "--eval-max-article-tokens", str(c.eval_max_article_tokens),
            "--eval-min-article-tokens", str(c.eval_min_article_tokens),
            "--max-answer-tokens", str(c.max_answer_tokens),
            "--attn-backends", c.attn_backend,
            "--pretrain-data-root", c.data_root,
            "--eval-split-manifest", self.eval_split_path,
        ]
        if c.optim != "adamw":
            args += ["--optimizer", c.optim]
        args += ["--weight-decay", str(c.weight_decay)]
        if c.gate_optimizer == "muon":
            args += ["--muon-gate"]
        if c.no_decay_scalars:
            args += ["--no-decay-scalars"]
        if c.shuffle_epochs:
            args += ["--shuffle-epochs"]
        if c.phase2_loguniform:
            args += ["--phase2-loguniform"]
        if c.phase2_len_list:
            args += ["--phase2-len-list", *[str(x) for x in c.phase2_len_list]]
        if c.adapter_type != "lora":
            args += ["--adapter-type", c.adapter_type]
            if c.lora_lr is not None:
                args += ["--lora-lr", str(c.lora_lr)]
        elif c.lora_rank > 0:
            args += ["--lora-rank", str(c.lora_rank),
                     "--lora-alpha", str(c.lora_alpha),
                     "--lora-lr", str(c.lora_lr),
                     "--lora-targets", *c.lora_targets,
                     "--lora-layers", c.lora_layers]
        if c.grad_checkpoint:
            args += ["--grad-checkpoint"]
        if c.gate_full_eval_freq:
            args += ["--gate-full-eval-freq", str(c.gate_full_eval_freq)]
        if c.packed:
            args += ["--pretrain-packed"]
        if self._eval_only:
            args += ["--pretrain-eval-only-pairs",
                     ",".join(f"{d}:{t}" for d, t in self._eval_only)]
        if c.target_balanced_map:
            args += ["--target-balanced-map", c.target_balanced_map]
        if c.eval_group_override:
            args += ["--eval-group-override", c.eval_group_override]
        if c.eval_test_full_tasks:
            args += ["--eval-test-full-tasks", *c.eval_test_full_tasks]
        if c.compile_phase2:
            args += ["--compile-phase2"]
        if c.pack_phase2:
            args += ["--pack-phase2-max-tokens", str(c.pack_phase2_max_tokens)]
        if c.max_steps is not None:
            args += ["--max-steps", str(c.max_steps)]
        if c.profile:
            args += ["--profile"]
        if c.resume_from:
            args += ["--resume-from", c.resume_from]
        if c.wandb_project:
            run_name = c.run_name or (
                f"{Path(c.base_model).name}_{self._dataset}-{self._task}"
                f"_L{c.gate_layer}_lr={c.learning_rate}_seed={c.seed}")
            args += ["--wandb-project", c.wandb_project,
                     "--wandb-run-name", run_name]
        return args + list(c.extra_args)

    @property
    def command(self) -> List[str]:
        """The full torchrun command `train()` executes (for inspection)."""
        c = self.config
        cmd = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc_per_node={c.num_gpus}",
               f"--master_port={c.master_port}"]
        if c.nnodes > 1:
            cmd += [f"--nnodes={c.nnodes}", f"--node_rank={c.node_rank}",
                    f"--master_addr={c.master_addr}"]
        return cmd + ["-m", _TRAIN_MODULE] + self._trainer_args()

    def _env(self) -> dict:
        env = dict(os.environ)
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        env.setdefault("TRAIN_LOG_FREQ", "50")
        return env

    def _ensure_packed_store(self):
        """Idempotent packed-store build (node rank 0 only, like the driver)."""
        c = self.config
        if not c.packed or (c.nnodes > 1 and c.node_rank != 0):
            return
        pair = f"{self._dataset}:{self._task}"
        print(f"[tokengate.trainer] ensuring packed token store for "
              f"{pair} (idempotent)…")
        subprocess.run(
            [sys.executable, "-m", _PACK_MODULE, pair,
             "--root", c.data_root, "--model", self._model_store_dir()],
            check=True, env=self._env())

    def train(self) -> Path:
        """Run training to completion; returns the output directory."""
        c = self.config
        if c.attn_backend == "fa2":
            # Fail before any data work: the fa2 training step (and packed
            # Phase 2) has no runtime fallback, unlike inference.
            from kernels.fa2bias.keybias import available as fa2_bias_available
            if not fa2_bias_available():
                raise RuntimeError(
                    "attn_backend='fa2' (the Qwen3 default) needs the key-bias "
                    "FlashAttention-2 kernel, which is not built on this machine: "
                    "build it with `python kernels/fa2bias/build.py`, or train on "
                    "the no-build flex kernels with "
                    "TokenGateConfig(..., attn_backend='flex')")
        if self._train_rows is not None:
            self._prepare_rows()
        self._ensure_packed_store()
        if c.eval_split is None:
            self._write_auto_split()
        print(f"[tokengate.trainer] {self._dataset}/{self._task} — gate layer "
              f"{c.gate_layer}, {c.num_gpus} GPU(s) × {c.nnodes} node(s), "
              f"attn={c.attn_backend}")
        subprocess.run(self.command, check=True, env=self._env())
        out = Path(c.output_dir)
        gate = out / "gate_final.pt"
        if gate.exists():
            print(f"[tokengate.trainer] done — gate at {gate}")
        else:
            # profile / max_steps diagnostic runs skip checkpointing
            print(f"[tokengate.trainer] done — outputs in {out} "
                  "(no gate_final.pt: diagnostic run)")
        return out
