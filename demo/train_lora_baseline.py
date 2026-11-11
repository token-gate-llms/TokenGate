"""LoRA-only NIAH SFT baseline for the optional full-recipe demo.

Run with torchrun after TokenGateTrainer has prepared its packed NIAH store.
This shares the gate run's data, prompt/answer masking, LoRA shape, learning
rate, schedule, and validation IDs. The public paper does not provide the
baseline's exact stopping checkpoint, so max_steps is an explicit input.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


class PackedNiahDataset:
    def __init__(self, packed_dir: Path):
        import numpy as np

        self.tokens = np.load(packed_dir / "train_tokens.npy", mmap_mode="r")
        self.meta = np.load(packed_dir / "train_meta.npy", mmap_mode="r")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        import torch

        offset, length, ctx_len = (int(x) for x in self.meta[index, :3])
        ids = torch.tensor(self.tokens[offset:offset + length].astype("int64"))
        labels = ids.clone()
        labels[:ctx_len] = -100
        return {"input_ids": ids, "labels": labels}


class EvalNiahDataset:
    def __init__(self, val_path: Path, split_path: Path):
        split = json.loads(split_path.read_text())["niah_kamradt"]
        self.ids = set(split["val"])
        with val_path.open("rb") as f:
            self.rows = [row for row in pickle.load(f) if row["id"] in self.ids]
        if len(self.ids) != 50 or len(self.rows) != len(self.ids):
            raise ValueError("expected exactly the 50 frozen NIAH validation IDs")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        row = self.rows[index]
        ids = torch.tensor(row["full_ids"], dtype=torch.long)
        labels = ids.clone()
        labels[:row["ctx_len"]] = -100
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id: int):
    import torch

    width = max(len(row["input_ids"]) for row in batch)
    inputs = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    mask = torch.zeros_like(inputs)
    for i, row in enumerate(batch):
        n = len(row["input_ids"])
        inputs[i, :n] = row["input_ids"]
        labels[i, :n] = row["labels"]
        mask[i, :n] = 1
    return {"input_ids": inputs, "attention_mask": mask, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--eval-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, default=1)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from transformers.trainer_utils import get_last_checkpoint

    store = (args.data_root / "niah_kamradt/niah_kamradt/tokenized"
             / args.base_model.rsplit("/", 1)[-1])
    train_data = PackedNiahDataset(store / "packed")
    eval_data = EvalNiahDataset(store / "val.pkl", args.eval_split)
    if len(train_data) != 12000:
        raise ValueError(f"expected 12,000 NIAH training rows, found {len(train_data)}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), max_steps=args.max_steps,
        num_train_epochs=20, per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        learning_rate=1e-4, warmup_steps=1000,
        lr_scheduler_type="constant_with_warmup", optim="adamw_torch",
        weight_decay=0.1, max_grad_norm=1.0, bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=600, prediction_loss_only=True,
        save_strategy="steps", save_steps=600, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, logging_steps=50, report_to="none",
        remove_unused_columns=False, ddp_find_unused_parameters=False,
        seed=0, data_seed=0,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=lambda batch: collate(batch,
                                            tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(args.output_dir)))
    trainer.save_model(str(args.output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
