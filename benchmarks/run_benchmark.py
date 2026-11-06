#!/usr/bin/env python3
"""Benchmark a TokenGate model against standard full-context prefill.

Measures, on the same samples and the same base model:
  * accuracy  — the task's own metric on greedy generations;
  * TTFT      — prefill-to-first-token latency, per sample at batch size 1.

Three arms: `full` (the unmodified base model prefilling every token — the
TTFT reference), `tokengate@full` (the trained TokenGate model with nothing
pruned — the accuracy reference), and `tokengate@K` for each configured K
(the gate keeps K state tokens; the layers above the gate never see — or
cache — the rest).

Dataset, checkpoint and K values come from a YAML config:

    python benchmarks/run_benchmark.py --config benchmarks/configs/niah.yaml
    python benchmarks/run_benchmark.py --config ... --split test   # full held-out set
    python benchmarks/run_benchmark.py --config ... --n 4 --k 500 1000

Config keys: base_model, task, gate_layer, checkpoint (HF repo id or a local
run dir holding the gate + name-paired adapter), k_values, max_new_tokens,
gate_attn (fa2|flash|flex), and one sample source — either samples (jsonl of
{id, article, target} rows) or dataset + eval_split (a prepared tokenized
store + split manifest; --split picks the val partition or the full reserved
test partition).
"""
import argparse
import json
import os
import sys

import torch
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


import fcntl
from pathlib import Path

from tokengate.datasets.tasks.niah import NiahKamradtTask
from tokengate.datasets.dataset import load_rows
from tokengate.e2e_gate.artifacts import (
    atomic_json,
    capture_source,
    file_hash,
    runtime_evidence,
)
from tokengate.e2e_gate.bench.budget_selection import (
    evaluation_budgets,
    read_answers,
    validate_decision,
)
from tokengate.e2e_gate.train.selection import selected_checkpoint
from tokengate.generators import _strip_thinking
from tokengate.inference import _resolve_gate_path
from tokengate.e2e_gate.bench.runtime import (
    load_checkpoint,
    prefill_selection,
    serving_settings,
    validate_retained_positions,
)
from tokengate.e2e_gate.bench.export import verify_export


@torch.no_grad()
def _full_prefill_arm(base_model: str, metas, task, max_new_tokens: int):
    """Standard prefill baseline: unmodified model, every token cached."""
    from transformers import AutoModelForCausalLM
    from tokengate.e2e_gate.decode.gate_decode import generate_batch

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    def _run(meta):
        result = generate_batch(
            model, None, _TOK, [meta["prompt_ids"]], [meta["article_start"]],
            [meta["article_end"]], None, None, max_new_tokens,
        )
        return result.answers[0], result.prefill_s

    _run(metas[0])  # warmup (kernel autotune, allocator)
    rows = []
    for m in metas:
        ans, ttft = _run(m)
        rows.append({"id": m["sample"]["id"], "ttft_s": ttft,
                     "acc": task.evaluate(ans, m["sample"]["target"]),
                     "answer": ans})
    model = None
    torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def _tokengate_arm(base_model, gate_path, gate_layer, gate_attn, metas, task,
                   k_values, max_new_tokens):
    from tokengate.e2e_gate.bench.benchmark_gate_e2e import (
        gate_generate_batch,
        load_gate_model,
    )
    model, gate, _ = load_gate_model(
        base_model, gate_path, gate_layer, gate_attn=gate_attn)

    results = {}
    for k in ["full"] + list(k_values):
        k = 10_000_000 if k == "full" else k
        gate_generate_batch(model, gate, _TOK, metas[:1], gate_layer, k,
                            max_new_tokens, input_all_prev_layers=False)  # warmup
        rows = []
        for m in metas:  # batch size 1: honest per-sample TTFT
            answers, _, n_kept, prefill_s, _, _ = gate_generate_batch(
                model, gate, _TOK, [m], gate_layer, k,
                max_new_tokens, input_all_prev_layers=False)
            rows.append({"id": m["sample"]["id"], "ttft_s": prefill_s,
                         "acc": task.evaluate(answers[0], m["sample"]["target"]),
                         "n_kept": n_kept[0], "answer": answers[0]})
        results["full" if k == 10_000_000 else k] = rows
    del model, gate
    torch.cuda.empty_cache()
    return results


def _load_split_samples(cfg, split: str):
    """Samples from a prepared store's eval pool, filtered to one partition of
    the eval-split manifest (`val` = in-training eval rows, `test` = the
    reserved held-out partition)."""
    import pickle
    store = os.path.join(_REPO_ROOT, cfg.get("data_root", "datasets"),
                         cfg["dataset"], cfg["task"], "tokenized",
                         os.path.basename(cfg["base_model"]), "val.pkl")
    if not os.path.exists(store):
        raise FileNotFoundError(
            f"{store} not found — prepare the dataset first "
            f"(datasets/{cfg['dataset']}/{cfg['task']}/prepare.py)")
    split_path = cfg["eval_split"] if os.path.isabs(cfg["eval_split"]) else \
        os.path.join(_REPO_ROOT, cfg["eval_split"])
    ids = set(json.load(open(split_path))[cfg["task"]][split])
    rows = [r for r in pickle.load(open(store, "rb")) if r["id"] in ids]
    if len(rows) != len(ids):
        raise ValueError(f"{store}: found {len(rows)} of {len(ids)} "
                         f"'{split}' ids from {split_path}")
    return [{"id": r["id"], "article": r["article"], "target": r["target"]}
            for r in rows]


def main():
    if "--stage" in sys.argv:
        return reproduction_main(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="benchmark YAML config")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="eval-split partition to run (store-backed configs): "
                         "'val' = the in-training eval rows, 'test' = the full "
                         "reserved held-out partition (default: val)")
    ap.add_argument("--n", type=int, default=None, help="samples to run (default all)")
    ap.add_argument("--k", type=int, nargs="+", default=None,
                    help="override the config's k_values")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    required = ("base_model", "task", "gate_layer", "checkpoint",
                "k_values", "max_new_tokens", "gate_attn")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"benchmark config {args.config} is missing keys: {missing}")
    if ("samples" in cfg) == ("eval_split" in cfg):
        raise KeyError(f"benchmark config {args.config}: set exactly one sample "
                       "source — 'samples' (a jsonl of {id, article, target} "
                       "rows) or 'dataset' + 'eval_split' (a prepared store + "
                       "split manifest, run with --split val|test)")
    k_values = args.k or cfg["k_values"]

    from transformers import AutoTokenizer

    from tokengate.datasets.tasks import get_task
    from tokengate.e2e_gate.bench.benchmark_gate_e2e import build_metas

    global _TOK
    _TOK = AutoTokenizer.from_pretrained(cfg["base_model"])

    task = get_task(cfg["task"])
    if "samples" in cfg:
        samples_path = cfg["samples"] if os.path.isabs(cfg["samples"]) else \
            os.path.join(_REPO_ROOT, cfg["samples"])
        samples = [json.loads(l) for l in open(samples_path)]
        src = os.path.basename(samples_path)
    else:
        samples = _load_split_samples(cfg, args.split)
        src = f"{args.split} split ({os.path.basename(cfg['eval_split'])})"
    samples = samples[: args.n]
    print(f"[bench] samples: {src}")
    for s in samples:
        s["input"] = task.format_prompt(s["article"])
    metas = build_metas(_TOK, task, samples)
    n_ctx = [len(m["prompt_ids"]) for m in metas]
    print(f"[bench] {cfg['task']}: {len(metas)} samples, "
          f"{min(n_ctx)}-{max(n_ctx)} prompt tokens, arms: full + "
          f"tokengate@K={k_values}")

    full_rows = _full_prefill_arm(cfg["base_model"], metas, task,
                                  cfg["max_new_tokens"])
    gate_rows = _tokengate_arm(
        cfg["base_model"], _resolve_gate_path(cfg["checkpoint"]),
        cfg["gate_layer"], cfg["gate_attn"], metas, task, k_values,
        cfg["max_new_tokens"])

    def _mean(rows, key):
        return sum(r[key] for r in rows) / max(len(rows), 1)

    full_ttft = _mean(full_rows, "ttft_s")
    tag = task.acc_metric_name
    print(f"\n{'arm':<12} {'K':>6} {'acc(' + tag + ')':>10} "
          f"{'TTFT(ms)':>10} {'speedup':>8}")
    print("-" * 50)
    print(f"{'full':<12} {'-':>6} {_mean(full_rows, 'acc'):>10.3f} "
          f"{full_ttft * 1e3:>10.1f} {'1.00x':>8}")
    for k in ["full"] + list(k_values):
        rows = gate_rows[k]
        ttft = _mean(rows, "ttft_s")
        print(f"{'tokengate':<12} {str(k):>6} {_mean(rows, 'acc'):>10.3f} "
              f"{ttft * 1e3:>10.1f} {full_ttft / ttft:>7.2f}x")
    print()




def run_research_evaluation(args: argparse.Namespace) -> None:
    from tokengate.e2e_gate.train.train_e2e_gate import evaluate_reproduction

    config, checkpoint = selected_checkpoint(
        args.run, diagnostic=args.diagnostic, training_source=args.training_source
    )
    if args.partition == "validation" and args.tier != "16k":
        raise ValueError("32K reserved validation cannot select budgets/checkpoints")
    if args.diagnostic and args.partition != "validation":
        raise ValueError("Smoke diagnostics cannot evaluate held-out examples")
    decision_hash = None
    if args.partition != "validation":
        if args.decision is None:
            raise ValueError("Freeze a validation-only deployment decision first")
        decision_hash = validate_decision(args.decision, checkpoint)
    rows, data_evidence = load_rows(config, args.tier)
    if args.partition != "all":
        rows = [row for row in rows if row["partition"] == args.partition]
    if args.diagnostic:
        rows = rows[:4]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = capture_source(Path(__file__).resolve().parents[1], output)
        budgets = evaluation_budgets() if checkpoint["gate_path"] else (None,)
        manifest = {
            "engine": "research",
            "execution": "eager",
            "attention_backend": config["attn_backend"],
            "deterministic_algorithms": True,
            "checkpoint": checkpoint,
            "tier": args.tier,
            "partition": args.partition,
            "ids": [row["id"] for row in rows],
            "data": data_evidence,
            "code_manifest_sha256": file_hash(source),
            "training_source": str(args.training_source.resolve())
            if args.training_source
            else str(Path(__file__).resolve().parents[1]),
            "training_source_manifest_sha256": config["code_manifest_sha256"],
            "decision_sha256": decision_hash,
            "max_gen_tokens": config["max_gen_tokens"],
            "budgets": budgets,
        }
        manifest_path = output / "evaluation.json"
        # JSON normalizes tuples to lists; compare the serialized representation.
        manifest = json.loads(json.dumps(manifest))
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(
                    "Evaluation inputs changed; use a new output directory"
                )
        else:
            atomic_json(manifest, manifest_path)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model, gate, tokenizer = load_checkpoint(config, checkpoint)
        summaries = []
        for budget in budgets:
            directory = output / ("full" if budget is None else f"K{budget}")
            completed = directory / "summary.json"
            if completed.exists():
                summary = json.loads(completed.read_text())
                read_answers(directory, summary, manifest["ids"])
            else:
                # Incomplete directories are evidence of a failed attempt. Keep
                # them intact and require a new output path after diagnosis.
                directory.mkdir()
                metrics = evaluate_reproduction(
                    model,
                    gate,
                    tokenizer,
                    rows,
                    config,
                    budget=budget,
                    step=checkpoint["selection"]["step"],
                    variant=checkpoint["selection"]["variant"],
                    rank=0,
                    world_size=1,
                    output=directory,
                )
                paths = list(directory.glob("step*.jsonl"))
                if len(paths) != 1:
                    raise RuntimeError("Expected exactly one evaluation answer file")
                answers = [
                    json.loads(line) for line in paths[0].read_text().splitlines()
                ]
                if [item["id"] for item in answers] != manifest["ids"]:
                    raise RuntimeError("Evaluation answers do not match requested IDs")
                with (directory / "answers.jsonl").open("x") as stream:
                    for row, answer in zip(rows, answers, strict=True):
                        answer.update(
                            {
                                key: row[key]
                                for key in (
                                    "partition",
                                    "city",
                                    "depth_requested",
                                    "depth_actual",
                                    "article_start",
                                    "article_end",
                                )
                            }
                        )
                        stream.write(json.dumps(answer, allow_nan=False) + "\n")
                summary = {
                    "budget": budget,
                    **metrics,
                    "answers_sha256": file_hash(directory / "answers.jsonl"),
                }
                atomic_json(summary, completed)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        atomic_json({"complete": True, "results": summaries}, output / "status.json")


def evaluate_serving_budget(runner, rows: list[dict], *, budget, cap: int):
    """Yield scored serial responses and the selection used during each prefill."""
    tokenizer = runner.llm.get_tokenizer()
    task = NiahKamradtTask()
    for row in rows:
        responses = runner.generate(row, budget, cap)
        completion = next(responses)
        positions = (
            runner.llm.collective_rpc(prefill_selection)[0]["retained"].tolist()
            if runner.gate
            else list(range(row["ctx_len"]))
        )
        validate_retained_positions(positions, row, budget if runner.gate else None)
        for completion in responses:
            pass
        tokens = list(completion.token_ids)
        prediction = _strip_thinking(tokenizer.decode(tokens, skip_special_tokens=True))
        yield {
            **{
                key: row[key]
                for key in (
                    "id",
                    "target",
                    "partition",
                    "city",
                    "depth_requested",
                    "depth_actual",
                    "article_start",
                    "article_end",
                )
            },
            "prediction": prediction,
            "accuracy": task.evaluate(prediction, row["target"]),
            "token_ids": tokens,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "kept_positions": positions,
            "prompt_tokens": row["ctx_len"],
            "budget": budget,
        }


def run_serving_evaluation(args: argparse.Namespace) -> None:
    from tokengate.e2e_gate.bench.runtime import ServingRunner

    if args.partition == "validation" and args.tier != "16k":
        raise ValueError("32K reserved validation cannot select budgets/checkpoints")
    if args.diagnostic and args.partition != "validation":
        raise ValueError("Smoke diagnostics cannot evaluate held-out examples")
    if args.quick and not args.diagnostic:
        raise ValueError("Quick evaluation is restricted to smoke diagnostics")
    if not args.diagnostic and args.decision is None:
        raise ValueError(
            "Freeze the research deployment decision before vLLM evaluation"
        )
    config, checkpoint = selected_checkpoint(
        args.run,
        diagnostic=args.diagnostic,
        training_source=getattr(args, "training_source", None),
    )
    decision_hash = (
        validate_decision(args.decision, checkpoint) if args.decision else None
    )
    model_path, exported = verify_export(args.export, checkpoint)
    if exported["kind"] != "serving":
        raise ValueError("vLLM evaluation requires a serving export")
    rows, data_evidence = load_rows(config, args.tier)
    if args.partition != "all":
        rows = [row for row in rows if row["partition"] == args.partition]
    if args.diagnostic:
        rows = rows[:4]
    gate = checkpoint["gate_path"] is not None
    budgets = ((512, None) if args.quick else evaluation_budgets()) if gate else (None,)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = capture_source(Path(__file__).resolve().parents[1], output)
    settings = serving_settings(
        model_path, config, rows, gate=gate, execution=args.execution
    )
    manifest = {
        "engine": "vllm",
        "execution": args.execution,
        "checkpoint": checkpoint,
        "tier": args.tier,
        "partition": args.partition,
        "ids": [row["id"] for row in rows],
        "data": data_evidence,
        "code_manifest_sha256": file_hash(source),
        "decision_sha256": decision_hash,
        "export": str(args.export.resolve()),
        "export_status_sha256": file_hash(args.export / "status.json"),
        "max_gen_tokens": config["max_gen_tokens"],
        "budgets": budgets,
        "settings": settings,
    }
    atomic_json(manifest, output / "evaluation.json")
    runner = ServingRunner(settings, gate=gate)
    manifest["runtime"] = runtime_evidence(serving=True)
    atomic_json(manifest, output / "evaluation.json")
    summaries = []
    for budget in budgets:
        directory = output / ("full" if budget is None else f"K{budget}")
        directory.mkdir()
        answers_path = directory / "answers.jsonl"
        correct, ids = 0, []
        with answers_path.open("x") as stream:
            for answer in evaluate_serving_budget(
                runner,
                rows,
                budget=budget,
                cap=config["max_gen_tokens"],
            ):
                stream.write(json.dumps(answer, allow_nan=False) + "\n")
                stream.flush()
                ids.append(answer["id"])
                correct += answer["accuracy"]
        if ids != manifest["ids"]:
            raise RuntimeError("Evaluation answers do not match the requested IDs")
        summary = {
            "budget": budget,
            "accuracy": correct / len(ids),
            "count": len(ids),
            "answers_sha256": file_hash(answers_path),
        }
        atomic_json(summary, directory / "summary.json")
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    atomic_json(
        {
            "complete": True,
            "results": summaries,
            "resolved_backends": runner.settings["resolved_backends"],
            "resolved_compilation": runner.settings["resolved_compilation"],
            "evaluation_sha256": file_hash(output / "evaluation.json"),
            "numerical_agreement": "assessed separately",
        },
        output / "status.json",
    )


def reproduction_main(argv):
    """Research evaluation and serving comparisons using the fixed protocol."""
    from tokengate.e2e_gate.bench.benchmark_gate_e2e import run_serial_benchmark
    from tokengate.e2e_gate.bench.benchmark_sanity_check import (
        audit_quality,
        collect_decode,
        qualify_decode,
        run_comparison,
        run_serving_trace,
    )
    from tokengate.e2e_gate.bench.budget_selection import freeze_budget
    from tokengate.e2e_gate.bench.export import export_run

    handlers = {
        "quality": run_research_evaluation,
        "vllm-quality": run_serving_evaluation,
        "export": export_run,
        "benchmark": run_serial_benchmark,
        "vllm-trace": run_serving_trace,
        "compare": run_comparison,
    }
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--stage",
        required=True,
        choices=[*handlers, "select-budget", "qualify", "audit"],
    )
    selection, remaining = selector.parse_known_args(argv)
    stage = selection.stage
    parser = argparse.ArgumentParser(description=f"NIAH {stage}")
    parser.add_argument("--output", type=Path, required=True)
    if stage != "select-budget":
        parser.add_argument("--run", type=Path, required=True)
    if stage in handlers:
        parser.add_argument(
            "--training-source",
            type=Path,
            help="Original frozen training source when using a separate inference revision",
        )
        parser.add_argument("--diagnostic", action="store_true")
    if stage in {"quality", "vllm-quality", "benchmark", "vllm-trace", "qualify"}:
        parser.add_argument(
            "--tier",
            choices=["16k", "32k"],
            default="16k",
            required=stage != "qualify",
        )
    if stage in {"quality", "vllm-quality", "benchmark"}:
        parser.add_argument(
            "--partition",
            choices=["validation", "test", "all"],
            required=True,
        )
        parser.add_argument("--decision", type=Path)
    if stage in {"vllm-quality", "vllm-trace", "benchmark"}:
        parser.add_argument("--export", type=Path, required=stage != "benchmark")
    if stage in {"vllm-quality", "vllm-trace", "benchmark"}:
        parser.add_argument(
            "--execution", choices=["eager", "optimized"], default="eager"
        )
    if stage in {"vllm-quality", "vllm-trace"}:
        parser.add_argument("--quick", action="store_true")

    if stage == "export":
        parser.add_argument("--vllm-source", type=Path, help="TokenGate vLLM fork checkout; defaults to the installed editable checkout")
        parser.add_argument(
            "--kind", choices=["serving", "merged-reference"], required=True
        )
    elif stage == "select-budget":
        for name in ("validation", "baseline"):
            parser.add_argument(f"--{name}", type=Path, required=True)
    elif stage == "benchmark":
        parser.add_argument("--quality", type=Path)
        parser.add_argument(
            "--qualification",
            type=Path,
            help="Completed qualify (original) or compare (vLLM) output; required for measurements",
        )
        parser.add_argument(
            "--training-root",
            type=Path,
            help="Completed pipeline containing this run; defaults to the run's root",
        )
        parser.add_argument("--engine", choices=["research", "vllm"], required=True)
        parser.add_argument("--budget", required=True)
        parser.add_argument("--phase", choices=["prefill", "decode"], default="prefill")
    elif stage == "vllm-trace":
        parser.add_argument("--backend", choices=["auto", "triton"], default="auto")
    elif stage == "compare":
        for name in ("trace", "merged-export"):
            parser.add_argument(f"--{name}", type=Path, required=True)
    elif stage == "qualify":
        for name in ("reference-source", "inference-source", "training-root"):
            parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument("--gpu", required=True)
        parser.add_argument(
            "--mode",
            choices=("original", "revised", "callback"),
            help=argparse.SUPPRESS,
        )
    elif stage == "audit":
        for name in ("original", "revised", "evaluation", "qualification", "decision"):
            parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument("--compare-subset", type=Path, action="append", default=[])

    args = parser.parse_args(remaining)
    if stage == "select-budget":
        freeze_budget(args.validation, args.baseline, args.output)
    elif stage == "qualify":
        if not args.gpu.isdecimal():
            parser.error("Select one physical GPU index")
        for name in (
            "run",
            "reference_source",
            "inference_source",
            "training_root",
            "output",
        ):
            setattr(args, name, getattr(args, name).resolve())
        if args.mode:
            collect_decode(args, args.output / args.mode)
        else:
            qualify_decode(args)
    elif stage == "audit":
        record = audit_quality(args)
        with args.output.open("x") as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    else:
        handlers[stage](args)


if __name__ == "__main__":
    main()
