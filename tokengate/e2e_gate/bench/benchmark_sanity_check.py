"""Correctness sanity check: does our benchmark inference reproduce the gate's
*training-time* eval accuracy?

It reconstructs the EXACT pretraining eval set (the deterministic
``random.Random(seed).sample`` per task-category, drawn from each dataset's
first-N val.pkl rows in manifest order — see train_e2e_gate.py:2649-2665), runs
**our** benchmark gate inference (gate_generate_batch from benchmark_gate_e2e) at
the training K, and reports per-category + overall accuracy.  Compare the printed
numbers against the training run's ``eval_responses.csv`` (final step).

Match (within batching/numerical noise) ⇒ the gate inference path is correct.

    python -m tokengate.e2e_gate.bench.benchmark_sanity_check \
        --gpu 0 \
        --gate-path .../baseline/L0/gate_step137700.pt --gate-layer 0 \
        --manifest v3_accepted28 \
        --pretrain-data-root /future/.../datasets \
        --k 2000 --eval-samples-per-group 50 --seed 0 \
        --max-new-tokens 1024 --batch-size 4
"""

import argparse
import os
import random
import subprocess
import sys

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
for _i, _a in enumerate(sys.argv):
    if (
        _a == "--gpu"
        and _i + 1 < len(sys.argv)
        and "CUDA_VISIBLE_DEVICES" not in os.environ
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]

import torch  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import datetime
import fcntl
import gc
import json
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoModelForCausalLM, AutoTokenizer

from tokengate.datasets.tasks import get_task, get_task_category  # noqa: E402
from tokengate.datasets.tasks.niah import NiahKamradtTask
from tokengate.datasets.dataset import load_rows
from tokengate.e2e_gate.attn.qwen3_flex_attention import (
    patch_model_with_flex_attention,
    set_attention_backends_for_layers,
)
from tokengate.e2e_gate.bench.runtime import ServingRunner
from tokengate.e2e_gate.bench.benchmark_gate_e2e import (
    build_metas,
    gate_generate_batch,
    load_gate_model,
    load_val_samples,
)
from tokengate.e2e_gate.decode.gate_decode import generate_batch
from tokengate.e2e_gate.gate_module.e2e_gate_module import load_gate
from tokengate.e2e_gate.train.selection import assert_equal, selected_checkpoint
from tokengate.e2e_gate.artifacts import (
    atomic_json,
    atomic_save,
    capture_source,
    file_hash,
    runtime_evidence,
    verify_source,
)
from tokengate.e2e_gate.bench.budget_selection import (
    choose_budget,
    evaluation_budgets,
    read_answers,
    read_validation,
    validate_decision,
)
from tokengate.generators import _strip_thinking
from tokengate.e2e_gate.bench.runtime import (
    load_checkpoint,
    prefill_selection,
    serving_settings,
    validate_retained_positions,
)
from tokengate.e2e_gate.bench.export import verify_export


def manifest_pairs(manifest: str):
    def _q(field):
        return subprocess.check_output([
            sys.executable, "-m", "tokengate.datasets.manifest",
            manifest, field,
        ]).decode().strip().split(",")
    return list(zip(_q("dataset"), _q("task")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--gate-path", required=True)
    p.add_argument("--gate-layer", type=int, required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--pretrain-data-root", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--eval-samples-per-group", type=int, required=True)
    p.add_argument("--val-samples-per-dataset", type=int, default=50)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--max-article-tokens", type=int, default=30000)
    p.add_argument("--max-answer-tokens", type=int, default=1024)
    p.add_argument("--only-groups", default=None,
                   help="comma list to restrict groups, e.g. qa,code_completion")
    p.add_argument("--dump-csv", default=None,
                   help="optional path: write per-sample (group,dataset,target,prediction,acc,f1)")
    args = p.parse_args()

    import csv as _csv
    _dump_rows = []

    model_name = os.path.basename(args.base_model)
    pairs = manifest_pairs(args.manifest)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Reconstruct the exact pretraining eval set (train_e2e_gate.py:2649-2665) ─
    # Per-dataset first-N val rows (manifest order), filtered identically, then
    # pooled by task-category and sub-sampled with random.Random(seed).
    eval_by_dataset = {}
    cat_of_ds = {}
    for ds, tk in pairs:
        rows = load_val_samples(
            args.pretrain_data_root, ds, tk, model_name,
            args.val_samples_per_dataset, args.max_article_tokens, args.max_answer_tokens,
        )
        eval_by_dataset[ds] = rows
        cat_of_ds[ds] = get_task_category(tk)

    grp_pool = {}
    for ds, rows in eval_by_dataset.items():
        grp_pool.setdefault(cat_of_ds[ds], []).extend(rows)
    grng = random.Random(args.seed)
    eval_by_group = {}
    for g, pool in grp_pool.items():
        n = min(args.eval_samples_per_group, len(pool))
        eval_by_group[g] = grng.sample(pool, n)

    only = set(args.only_groups.split(",")) if args.only_groups else None
    print("[sanity] reconstructed eval set:")
    for g, s in eval_by_group.items():
        src = {}
        for r in s:
            src[r["dataset"]] = src.get(r["dataset"], 0) + 1
        print(f"  {g:16s}: n={len(s)}  from {src}")

    # ── Load gate model ──────────────────────────────────────────────────────
    gmodel, gate, iapl = load_gate_model(args.base_model, args.gate_path, args.gate_layer)

    # ── Run our gate inference per group at K, batched ───────────────────────
    print(f"\n[sanity] running gate inference at K={args.k}, "
          f"max_new_tokens={args.max_new_tokens}, batch_size={args.batch_size}\n")
    overall_acc = overall_n = 0
    results = {}
    for g, samples in eval_by_group.items():
        if only and g not in only:
            continue
        # one task per sample (mixed within a category) → group metas by task
        metas = []
        for idx, s in enumerate(samples):
            task = get_task(s["task"])
            m_list = build_metas(tokenizer, task, [s])
            if m_list:
                m = m_list[0]
                m["idx"] = idx
                m["task_obj"] = task
                metas.append(m)
        acc_sum = f1_sum = 0.0
        n = 0
        i = 0
        bs = args.batch_size
        while i < len(metas):
            batch = metas[i:i + bs]
            try:
                answers, _ngen, _nkept, *_ = gate_generate_batch(
                    gmodel, gate, tokenizer, batch, args.gate_layer, args.k,
                    args.max_new_tokens, iapl)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    i += 1
                    continue
                bs = max(1, bs // 2)
                continue
            for m, ans in zip(batch, answers):
                t = m["task_obj"]
                s = m["sample"]
                a_i = t.evaluate(ans, s["target"], sample=s)
                f_i = t.evaluate_f1(ans, s["target"], sample=s)
                acc_sum += a_i
                f1_sum += f_i
                n += 1
                if args.dump_csv:
                    _dump_rows.append({
                        "group": g, "dataset": s["dataset"], "target": s["target"][:300],
                        "prediction": ans[:300], "acc": f"{a_i:.6f}", "f1": f"{f_i:.6f}",
                    })
            i += bs
        acc = acc_sum / max(n, 1)
        f1 = f1_sum / max(n, 1)
        results[g] = (acc, f1, n)
        overall_acc += acc_sum
        overall_n += n
        print(f"  {g:16s}: acc={acc:.3f}  f1={f1:.3f}  n={n}", flush=True)

    if args.dump_csv and _dump_rows:
        with open(args.dump_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["group", "dataset", "target", "prediction", "acc", "f1"])
            w.writeheader()
            w.writerows(_dump_rows)
        print(f"[sanity] wrote {len(_dump_rows)} per-sample rows → {args.dump_csv}")

    print(f"\n  {'OVERALL':16s}: acc={overall_acc/max(overall_n,1):.3f}  n={overall_n}")
    print("\n[sanity] compare these against the training run's eval_responses.csv "
          "(final step, same K). A close match validates the gate inference path.")


ATOL = 0.02
RTOL = 0.02

@contextmanager
def forced_history(model, tokens: list[int]):
    """Record logits before argmax while supplying an identical decode history."""
    recorded = []

    def capture_and_force(module, inputs, output):
        index = len(recorded)
        if index >= len(tokens) or not 0 <= tokens[index] < output.shape[-1]:
            raise ValueError("Unexpected LM-head call or invalid forced token")
        logits = output.reshape(-1, output.shape[-1])[-1]
        recorded.append(logits.detach().float().cpu())
        forced = torch.full_like(output, float("-inf"))
        forced[..., tokens[index]] = 0
        return forced

    handle = model.lm_head.register_forward_hook(capture_and_force)
    try:
        yield recorded
        if len(recorded) != len(tokens):
            raise RuntimeError(
                "Research decode did not consume the entire supplied history"
            )
    finally:
        handle.remove()


def load_merged(model_path: Path, config: dict, checkpoint: dict):
    """Load a gate-free weight export and attach its original research gate."""
    model, info = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
        output_loading_info=True,
    )
    if info["missing_keys"] or info.get("mismatched_keys") or info["unexpected_keys"]:
        raise ValueError(f"Merged research weights did not load completely: {info}")
    if hasattr(model.config, "token_gate"):
        raise ValueError("Use the merged-reference export for research execution")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    gate = None
    if checkpoint["gate_path"]:
        gate, _ = load_gate(checkpoint["gate_path"])
        gate.to(device=model.device, dtype=torch.float32).eval().requires_grad_(False)
        patch_model_with_flex_attention(model)
        set_attention_backends_for_layers(model, config["attn_backend"])
    model.eval().requires_grad_(False)
    return model, gate, tokenizer


@torch.no_grad()
def scorer_trace(gate, probe: dict) -> dict:
    """Use the research branch, normalization, and selection on shared inputs."""
    hidden = probe["inputs"].to(next(gate.parameters()).device)
    normalized = []

    def capture(module, inputs, output):
        normalized.append(output.reshape(-1, output.shape[-1])[0].detach().cpu())

    handle = gate.b_norm.register_forward_hook(capture)
    try:
        bias = gate(gate.gate_branch(hidden).float())
        selected = gate.aggregate(bias).topk(probe["keep_tokens"]).indices.sort().values
    finally:
        handle.remove()
    if len(normalized) != 1:
        raise RuntimeError("Expected one research scorer normalization")
    return {"scores": normalized[0], "retained": selected.cpu()}


@torch.no_grad()
def trace(
    model, gate, tokenizer, config: dict, case: dict, *, capture_inputs=False
) -> dict:
    """Return logits, prefill selection, and scores on a vLLM-generated history."""
    prompt, tokens = case["prompt"], case["tokens"]
    if not prompt or not tokens:
        raise ValueError(
            "Numerical traces require a prompt and at least one output token"
        )
    scores, scorer_inputs, handles = [], [], []
    start, end = case["span"]
    if gate is not None:

        def capture_scores(module, inputs, output):
            scores.append(output.reshape(-1, output.shape[-1])[0].detach().cpu())

        handles.append(gate.b_norm.register_forward_hook(capture_scores))
        if capture_inputs:

            def capture_branch_input(module, inputs):
                hidden = inputs[0].reshape(-1, inputs[0].shape[-1])
                scorer_inputs.append(hidden[start:end].detach().cpu())

            handles.append(
                gate.gate_branch.register_forward_pre_hook(capture_branch_input)
            )
    try:
        with forced_history(model, tokens) as recorded:
            if gate is None:
                inputs = torch.tensor([prompt], device=model.device)
                cache = None
                for token in tokens:
                    result = model(
                        inputs, past_key_values=cache, use_cache=True, logits_to_keep=1
                    )
                    cache = result.past_key_values
                    inputs = torch.tensor([[token]], device=model.device)
                retained = torch.arange(len(prompt))
            else:
                # Fixed-length numerical histories include EOS like any other
                # token. Keep the real decoder/tokenizer and disable EOS stopping.
                decoding = SimpleNamespace(eos_token_id=-1, decode=tokenizer.decode)
                result = generate_batch(
                    model,
                    gate,
                    decoding,
                    [prompt],
                    [start],
                    [end],
                    config["gate_layer"],
                    case["budget"] or end - start,
                    len(tokens),
                )
                retained = result.kept_pos[0, : result.n_kepts[0]].cpu()
    finally:
        for handle in handles:
            handle.remove()
    if gate is not None and len(scores) != 1:
        raise RuntimeError("Expected one gate-normalization call during prefill")
    logits = torch.stack(recorded)
    return {
        "id": case["id"],
        "budget": case["budget"],
        "tokens": tokens,
        "logits": logits,
        "logprobs": logits.log_softmax(-1),
        "retained": retained,
        "scores": scores[0] if scores else None,
        "scorer_inputs": scorer_inputs[0] if scorer_inputs else None,
    }


def scorer_probe(worker):
    """Exercise the loaded serving scorer on fixed, shared BF16 inputs."""
    gate = worker.model_runner.model_state.history.gate
    generator = torch.Generator(device="cpu").manual_seed(0)
    hidden = torch.randn(
        1024, gate.input_gate.in_features, generator=generator, dtype=torch.float32
    ).to(device=gate.input_gate.weight.device, dtype=torch.bfloat16)
    scores = gate.score(hidden)
    return {
        "inputs": hidden.cpu(),
        "scores": scores.cpu(),
        "retained": gate.select(scores, 512).cpu(),
        "keep_tokens": 512,
    }


def run_serving_trace(args: argparse.Namespace) -> None:
    config, checkpoint = selected_checkpoint(
        args.run,
        diagnostic=args.diagnostic,
        training_source=getattr(args, "training_source", None),
    )
    model_path, exported = verify_export(args.export, checkpoint)
    if exported["kind"] != "serving":
        raise ValueError("vLLM numerical checks require a serving export")
    if args.quick and not args.diagnostic:
        raise ValueError("Quick checks are only permitted for smoke diagnostics")
    rows, data_evidence = load_rows(config, args.tier)
    rows = [row for row in rows if row["partition"] == "validation"][:2]
    budgets = (
        ((512, None) if args.quick else evaluation_budgets())
        if checkpoint["gate_path"]
        else (None,)
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = capture_source(Path(__file__).resolve().parents[3], output)
    gate = checkpoint["gate_path"] is not None
    settings = serving_settings(
        model_path,
        config,
        rows,
        gate=gate,
        execution=args.execution,
        max_new_tokens=16,
    )
    settings.update(
        max_logprobs=-1,
        attention_config={
            "backend": "TRITON_ATTN" if args.backend == "triton" else None
        },
    )
    manifest = {
        "checkpoint": checkpoint,
        "tier": args.tier,
        "data": data_evidence,
        "export": str(args.export.resolve()),
        "export_status_sha256": file_hash(args.export / "status.json"),
        "code_manifest_sha256": file_hash(source),
        "settings": settings,
        "backend": args.backend,
        "execution": args.execution,
        "ids": [row["id"] for row in rows],
        "budgets": budgets,
        "output_tokens": 2,
        "ignore_eos": True,
    }
    atomic_json(manifest, output / "trace.json")
    runner = ServingRunner(settings, gate=gate)
    llm = runner.llm
    manifest["runtime"] = runtime_evidence(serving=True)
    atomic_json(manifest, output / "trace.json")
    if checkpoint["gate_path"]:
        atomic_save(llm.collective_rpc(scorer_probe)[0], output / "scorer.pt")
    cases = []
    for row in rows:
        for budget in budgets:
            prompt = row["input_ids"][: row["ctx_len"]]
            responses = runner.generate(
                row,
                budget,
                2,
                ignore_eos=True,
                logprobs=-1,
            )
            completion = next(responses)
            selection = (
                llm.collective_rpc(prefill_selection)[0]
                if checkpoint["gate_path"]
                else {"retained": torch.arange(len(prompt)), "selected_scores": None}
            )
            for completion in responses:
                pass
            if len(completion.token_ids) != 2 or len(completion.logprobs) != 2:
                raise RuntimeError("Expected two numerical decode distributions")
            distributions = []
            for step in completion.logprobs:
                ids = sorted(step)
                if ids != list(range(len(ids))):
                    raise RuntimeError(
                        "Numerical comparison requires the entire vocabulary"
                    )
                distributions.append(
                    torch.tensor([step[index].logprob for index in ids])
                )
            case = {
                "id": row["id"],
                "budget": budget,
                "prompt": prompt,
                "span": (row["article_start"], row["article_end"]),
                "tokens": list(completion.token_ids),
                "logprobs": torch.stack(distributions),
                **selection,
            }
            cases.append(case)
            atomic_save({"cases": cases}, output / "cases.pt")
            print(
                json.dumps(
                    {"id": row["id"], "budget": budget, "tokens": case["tokens"]}
                ),
                flush=True,
            )
    atomic_json(
        {
            "complete": True,
            "cases_sha256": file_hash(output / "cases.pt"),
            "resolved_backends": runner.settings["resolved_backends"],
            "trace_sha256": file_hash(output / "trace.json"),
            "scorer_sha256": file_hash(output / "scorer.pt")
            if checkpoint["gate_path"]
            else None,
        },
        output / "status.json",
    )


def compare_logprobs(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Check every vocabulary entry; do not hide failures behind mean errors."""
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError("Expected matching [decode steps, vocabulary] distributions")
    error = (actual - expected).abs()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    violations = ~finite | (error > ATOL + RTOL * expected.abs())
    return {
        "close": not bool(violations.any()),
        "violating_entries": int(violations.sum()),
        "nonfinite_entries": int((~finite).sum()),
        "max_logprob_error": float(error.max()) if bool(finite.all()) else None,
        "argmax_equal": torch.equal(actual.argmax(-1), expected.argmax(-1)),
        "entries": actual.numel(),
    }


def compare_selection(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Report exact retained positions and overlap without relaxing the criterion."""
    actual_set, expected_set = set(actual.tolist()), set(expected.tolist())
    union = actual_set | expected_set
    return {
        "selected_rows_equal": torch.equal(actual, expected),
        "retained_overlap": len(actual_set & expected_set) / len(union)
        if union
        else 1.0,
        "actual_retained": len(actual_set),
        "expected_retained": len(expected_set),
    }


def load_trace(directory: Path, config: dict, checkpoint: dict):
    status = json.loads((directory / "status.json").read_text())
    if not status["complete"]:
        raise ValueError("Serving trace is incomplete")
    for name, key in (("trace.json", "trace_sha256"), ("cases.pt", "cases_sha256")):
        if file_hash(directory / name) != status[key]:
            raise ValueError(f"Serving trace changed: {name}")
    manifest = json.loads((directory / "trace.json").read_text())
    if manifest["checkpoint"] != checkpoint:
        raise ValueError("Serving trace belongs to a different checkpoint")
    rows, data_evidence = load_rows(config, manifest["tier"])
    if data_evidence != manifest["data"]:
        raise ValueError("Research and serving data differ")
    rows = {row["id"]: row for row in rows if row["partition"] == "validation"}
    cases = torch.load(directory / "cases.pt", weights_only=True)["cases"]
    expected = [
        (index, budget) for index in manifest["ids"] for budget in manifest["budgets"]
    ]
    if [(case["id"], case["budget"]) for case in cases] != expected:
        raise ValueError("Serving trace has missing, repeated, or reordered cases")
    for case in cases:
        row = rows[case["id"]]
        if (
            case["prompt"] != row["input_ids"][: row["ctx_len"]]
            or tuple(case["span"]) != (row["article_start"], row["article_end"])
            or len(case["tokens"]) != manifest["output_tokens"]
        ):
            raise ValueError("Serving trace does not match the declared input/history")
    probe = None
    if checkpoint["gate_path"]:
        if file_hash(directory / "scorer.pt") != status["scorer_sha256"]:
            raise ValueError("Shared scorer probe changed")
        probe = torch.load(directory / "scorer.pt", weights_only=True)
    return manifest, cases, probe


def compare_case(actual: dict, expected: dict) -> dict:
    if any(actual[name] != expected[name] for name in ("id", "budget", "tokens")):
        raise ValueError("Numerical cases must use the same prompt and decode history")
    result = {
        "id": actual["id"],
        "budget": actual["budget"],
        **compare_logprobs(actual["logprobs"], expected["logprobs"]),
        **compare_selection(actual["retained"], expected["retained"]),
    }
    if actual.get("scores") is not None and expected.get("scores") is not None:
        result["max_gate_score_error"] = float(
            (actual["scores"] - expected["scores"]).abs().max()
        )
    if "logits" in actual and "logits" in expected:
        result["max_logit_error"] = float(
            (actual["logits"] - expected["logits"]).abs().max()
        )
    result["qualified"] = result["close"] and result["selected_rows_equal"]
    return result


def run_comparison(args: argparse.Namespace) -> None:
    config, checkpoint = selected_checkpoint(
        args.run,
        diagnostic=args.diagnostic,
        training_source=getattr(args, "training_source", None),
    )
    manifest, cases, probe = load_trace(args.trace, config, checkpoint)
    model_path, exported = verify_export(args.merged_export, checkpoint)
    if checkpoint["gate_path"] and exported["kind"] != "merged-reference":
        raise ValueError("TokenGate needs a gate-free merged research export")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = capture_source(Path(__file__).resolve().parents[3], output)
    atomic_json(
        {
            "checkpoint": checkpoint,
            "trace": str(args.trace.resolve()),
            "trace_status_sha256": file_hash(args.trace / "status.json"),
            "merged_export": str(args.merged_export.resolve()),
            "export_status_sha256": file_hash(args.merged_export / "status.json"),
            "source_manifest_sha256": file_hash(source),
            "runtime": runtime_evidence(serving=False),
            "atol": ATOL,
            "rtol": RTOL,
            "exact_selection_required": True,
            "exact_shared_scorer_required": True,
        },
        output / "comparison.json",
    )
    collected, scorers = {}, {}
    for name in ("original", "merged"):
        model, gate, tokenizer = (
            load_checkpoint(config, checkpoint)
            if name == "original"
            else load_merged(model_path, config, checkpoint)
        )
        collected[name] = []
        if probe is not None:
            scorers[name] = scorer_trace(gate, probe)
        for case in cases:
            result = trace(model, gate, tokenizer, config, case)
            collected[name].append(result)
            atomic_save(collected[name], output / f"{name}.pt")
            print(
                json.dumps(
                    {"engine": name, "id": case["id"], "budget": case["budget"]}
                ),
                flush=True,
            )
        del model, gate, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    results = {
        "merge": [
            compare_case(merged, original)
            for merged, original in zip(
                collected["merged"], collected["original"], strict=True
            )
        ],
        "serving": [
            compare_case(serving, merged)
            for serving, merged in zip(cases, collected["merged"], strict=True)
        ],
        "shared_scorer": {},
    }
    for name, scorer in scorers.items():
        results["shared_scorer"][name] = {
            "scores_equal": bool(torch.isfinite(scorer["scores"]).all())
            and torch.equal(scorer["scores"], probe["scores"]),
            **compare_selection(scorer["retained"], probe["retained"]),
            "max_score_error": float((scorer["scores"] - probe["scores"]).abs().max()),
        }
    atomic_save(scorers, output / "scorers.pt")
    results["qualified"] = all(
        row["qualified"] for stage in ("merge", "serving") for row in results[stage]
    ) and all(
        row["scores_equal"] and row["selected_rows_equal"]
        for row in results["shared_scorer"].values()
    )
    results["tier"] = manifest["tier"]
    atomic_json(results, output / "results.json")
    atomic_json(
        {
            "complete": True,
            "qualified": results["qualified"],
            "files": {
                path.name: file_hash(path)
                for path in sorted(output.iterdir())
                if path.suffix in {".pt", ".json"}
            },
        },
        output / "status.json",
    )
    if not results["qualified"]:
        raise SystemExit("Strict numerical comparison failed; results preserved")


def compare_records(reference: list[dict], actual: list[dict]) -> None:
    """Reject any changed example, output, retained position or logit."""
    if not reference:
        raise AssertionError("Decode qualification requires nonempty records")
    assert_equal(actual, reference, "decode")


def collect_decode(args: argparse.Namespace, output: Path) -> None:
    """Run one isolated source revision; preserve full-vocabulary CPU logits."""
    import torch

    from tokengate.datasets.dataset import load_rows
    from tokengate.e2e_gate.decode.gate_decode import generate_batch
    from tokengate.e2e_gate.artifacts import (
        atomic_json,
        capture_source,
        file_hash,
        runtime_evidence,
    )
    from tokengate.e2e_gate.train.selection import selected_checkpoint
    from tokengate.e2e_gate.bench.runtime import load_checkpoint

    options = (
        {} if args.mode == "original" else {"training_source": args.reference_source}
    )
    config, checkpoint = selected_checkpoint(args.run, **options)
    rows, data = load_rows(config, args.tier)
    rows = sorted(
        (row for row in rows if row["partition"] == "validation"),
        key=lambda row: row["id"],
    )[:2]
    if len(rows) != 2:
        raise ValueError("Expected two fixed validation prompts")
    source = args.reference_source if args.mode == "original" else args.inference_source
    output.mkdir()
    source_manifest = capture_source(source, output)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, gate, tokenizer = load_checkpoint(config, checkpoint)
    records = []
    budgets = evaluation_budgets() if gate is not None else (None,)
    for row in rows:
        for budget in budgets:
            logits = []

            def record_logits(module, inputs, result, saved=logits):
                saved.append(result.detach().reshape(-1, result.shape[-1])[-1].cpu())

            delivery = None
            if args.mode == "callback":
                from tokengate.e2e_gate.decode.gate_decode import TokenDelivery

                delivery = TokenDelivery()
            prompt = row["input_ids"][: row["ctx_len"]]
            handle = model.lm_head.register_forward_hook(record_logits)
            try:
                with torch.no_grad():
                    kwargs = {"token_callback": delivery.put} if delivery else {}
                    result = generate_batch(
                        model,
                        gate,
                        tokenizer,
                        [prompt],
                        [row["article_start"]],
                        [row["article_end"]],
                        config["gate_layer"],
                        budget or row["article_end"] - row["article_start"],
                        config["max_gen_tokens"],
                        **kwargs,
                    )
                    generated = (
                        None
                        if args.mode == "original" and gate is not None
                        else result.token_ids[0]
                    )
                    answer = result.answers[0]
                    kept = result.kept_pos[0, : result.n_kepts[0]].cpu().tolist()
            finally:
                handle.remove()
            stacked = torch.stack(logits)
            if not torch.isfinite(stacked).all():
                raise FloatingPointError("Nonfinite decode logits")
            # HF generation may apply its configured logits processors. Gate
            # decoding selects the raw argmax, including in the original code.
            tokens = generated if gate is None else stacked.argmax(-1).tolist()
            if len(tokens) != len(logits):
                raise AssertionError("Expected one LM-head result per output token")
            eos = tokenizer.eos_token_id
            if eos in tokens[:-1] or not 1 <= len(tokens) <= config["max_gen_tokens"]:
                raise AssertionError("Invalid generation length or EOS stopping")
            if tokens[-1] != eos and len(tokens) != config["max_gen_tokens"]:
                raise AssertionError("Generation stopped before EOS or its cap")
            if gate is not None and generated is not None and generated != tokens:
                raise AssertionError("Returned tokens differ from LM-head logits")
            if delivery is not None and delivery.token_ids != tokens:
                raise AssertionError("Delivered tokens differ from LM-head logits")
            records.append(
                {
                    "id": row["id"],
                    "budget": budget,
                    "token_ids": tokens,
                    "answer": answer,
                    "kept_positions": kept,
                    "logits": stacked,
                }
            )
    torch.save(records, output / "records.pt")
    atomic_json(
        {
            "complete": True,
            "mode": args.mode,
            "checkpoint": checkpoint,
            "data": data,
            "ids": [row["id"] for row in rows],
            "budgets": budgets,
            "max_gen_tokens": config["max_gen_tokens"],
            "partition": "validation",
            "tier": args.tier,
            "source": str(source),
            "source_manifest_sha256": file_hash(source_manifest),
            "records_sha256": file_hash(output / "records.pt"),
            "runtime": runtime_evidence(serving=False),
            "benchmark_timings": None,
        },
        output / "status.json",
    )


def qualify_decode(args: argparse.Namespace) -> None:
    """Hold the experiment lock and refuse GPU work until all training finishes."""
    import torch

    from tokengate.e2e_gate.bench.benchmark_gate_e2e import verify_completed_training
    from tokengate.e2e_gate.artifacts import atomic_json, file_hash
    from tokengate.e2e_gate.train.selection import selected_checkpoint
    from tokengate.e2e_gate.resources import available_gpus

    config, _ = selected_checkpoint(args.run, training_source=args.reference_source)
    with (args.training_root / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        training = verify_completed_training(args.training_root, config)
        available_gpus(args.gpu)
        args.output.mkdir(parents=True, exist_ok=False)
        evidence = {}
        reference = None
        for mode in ("original", "revised", "callback"):
            source = (
                args.reference_source if mode == "original" else args.inference_source
            )
            environment = dict(
                os.environ,
                PYTHONPATH=str(source),
                CUDA_VISIBLE_DEVICES=args.gpu,
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
            )
            available_gpus(args.gpu)
            with (args.output / f"{mode}.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(source / "benchmarks/run_benchmark.py"),
                        "--stage",
                        "qualify",
                        "--run",
                        str(args.run),
                        "--reference-source",
                        str(args.reference_source),
                        "--inference-source",
                        str(args.inference_source),
                        "--training-root",
                        str(args.training_root),
                        "--output",
                        str(args.output),
                        "--gpu",
                        args.gpu,
                        "--tier",
                        args.tier,
                        "--mode",
                        mode,
                    ],
                    env=environment,
                    cwd=source,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            directory = args.output / mode
            record = json.loads((directory / "status.json").read_text())
            if (
                not record["complete"]
                or file_hash(directory / "records.pt") != record["records_sha256"]
            ):
                raise ValueError("Decode qualification evidence changed")
            records = torch.load(
                directory / "records.pt", map_location="cpu", weights_only=True
            )
            if reference is None:
                reference = records
            else:
                compare_records(reference, records)
                for key in (
                    "checkpoint",
                    "data",
                    "ids",
                    "budgets",
                    "max_gen_tokens",
                    "partition",
                    "tier",
                ):
                    if record[key] != evidence["original"][key]:
                        raise AssertionError(f"Changed qualification inputs: {key}")
            evidence[mode] = record
        atomic_json(
            {
                "complete": True,
                "exact_match": True,
                "rtol": 0,
                "atol": 0,
                "completed_training": training,
                "modes": evidence,
                "driver_sha256": file_hash(Path(__file__)),
                "benchmark_timings": None,
            },
            args.output / "status.json",
        )


def verify_answer(answer, row, budget, tokenizer, gated):
    """Reject changed metadata, stopping, decoding, scores or retained positions."""
    for key in (
        "id",
        "target",
        "partition",
        "city",
        "depth_requested",
        "depth_actual",
        "article_start",
        "article_end",
    ):
        assert answer[key] == row[key], (row["id"], key)
    assert answer["budget"] == budget and answer["prompt_tokens"] == row["ctx_len"]
    tokens = answer["token_ids"]
    assert 1 <= len(tokens) <= 1024 and tokenizer.eos_token_id not in tokens[:-1]
    assert tokens[-1] == tokenizer.eos_token_id or len(tokens) == 1024
    prediction = _strip_thinking(tokenizer.decode(tokens, skip_special_tokens=True))
    assert prediction == answer["prediction"]
    assert NiahKamradtTask().evaluate(prediction, row["target"]) == answer["accuracy"]
    assert math.isfinite(answer["loss"]) and answer["loss"] >= 0
    kept = answer["kept_positions"]
    length = row["ctx_len"]
    if not gated:
        assert budget is None
        # Older SFT artifacts used None to mean the complete prompt.
        kept = list(range(length)) if kept is None else kept
    validate_retained_positions(kept, row, budget)
    return {
        "id": row["id"],
        "city": row["city"],
        "depth_requested": row["depth_requested"],
        "accuracy": answer["accuracy"],
        "loss": answer["loss"],
        "prompt_tokens": length,
        "retained_tokens": len(kept),
        "generated_tokens": len(tokens),
    }


def audit_quality(args):
    config, checkpoint = selected_checkpoint(args.run, training_source=args.original)
    directory = args.evaluation
    manifest = json.loads((directory / "evaluation.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    assert status["complete"] and manifest["checkpoint"] == checkpoint
    assert not checkpoint["diagnostic"] and manifest["engine"] == "research"
    assert manifest["execution"] == "eager" and manifest["deterministic_algorithms"]
    assert manifest["max_gen_tokens"] == config["max_gen_tokens"] == 1024
    tier, partition = manifest["tier"], manifest["partition"]
    assert tier in ("16k", "32k") and partition in ("validation", "test", "all")
    assert partition != "validation" or tier == "16k"
    (source_manifest,) = directory.glob("source-*.json")
    verify_source(args.revised, source_manifest, manifest["code_manifest_sha256"])
    assert Path(manifest["training_source"]).resolve() == args.original.resolve()
    assert manifest["training_source_manifest_sha256"] == config["code_manifest_sha256"]
    qualified = json.loads(args.qualification.read_text())
    assert qualified["complete"] and qualified["exact_match"]
    assert qualified["atol"] == qualified["rtol"] == 0
    assert set(qualified["modes"]) == {"original", "revised", "callback"}
    assert all(
        m["checkpoint"] == checkpoint and m["tier"] == tier
        for m in qualified["modes"].values()
    )
    decision_hash = validate_decision(args.decision, checkpoint)
    decision = json.loads(args.decision.read_text())
    validation_inputs = [
        Path(p).parent for p in decision["inputs"] if Path(p).name == "evaluation.json"
    ]
    validated = [read_validation(p) for p in validation_inputs]
    target = [v for v in validated if v[0]["checkpoint"] == checkpoint]
    baseline = [
        v for v in validated if v[0]["checkpoint"] == decision["baseline_checkpoint"]
    ]
    assert len(target) == len(baseline) == 1
    assert target[0][0]["ids"] == baseline[0][0]["ids"]
    assert baseline[0][0]["checkpoint"]["gate_path"] is None
    assert (
        Path(config["base_model"]).resolve()
        == Path(baseline[0][1]["base_model"]).resolve()
    )
    assert config["seed"] == baseline[0][1]["seed"]
    assert decision["baseline_accuracy"] == baseline[0][2][0]["accuracy"]
    choice = (
        choose_budget(target[0][2], decision["baseline_accuracy"])
        if checkpoint["gate_path"]
        else {
            "matches_sft_accuracy": True,
            "budget": None,
            "accuracy": decision["baseline_accuracy"],
        }
    )
    assert all(decision[k] == value for k, value in choice.items())
    assert manifest["decision_sha256"] == (
        None if partition == "validation" else decision_hash
    )
    pool, data = load_rows(config, tier)
    rows = (
        pool if partition == "all" else [r for r in pool if r["partition"] == partition]
    )
    count = {
        ("16k", "validation"): 50,
        ("16k", "test"): 150,
        ("32k", "test"): 184,
        ("16k", "all"): 200,
        ("32k", "all"): 200,
    }[(tier, partition)]
    assert len(rows) == count and manifest["ids"] == [r["id"] for r in rows]
    assert manifest["data"] == data
    budgets = list(evaluation_budgets()) if checkpoint["gate_path"] else [None]
    assert manifest["budgets"] == budgets == [r["budget"] for r in status["results"]]
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"], local_files_only=True
    )
    files = {
        str(p): file_hash(p)
        for p in (
            directory / "evaluation.json",
            directory / "status.json",
            source_manifest,
            args.qualification,
            args.decision,
        )
    }
    results = []
    for summary in status["results"]:
        budget = summary["budget"]
        name = "full" if budget is None else f"K{budget}"
        folder = directory / name
        answers_path = folder / "answers.jsonl"
        assert json.loads((folder / "summary.json").read_text()) == summary
        answers = read_answers(folder, summary, manifest["ids"])
        metrics = [
            verify_answer(a, r, budget, tokenizer, bool(checkpoint["gate_path"]))
            for a, r in zip(answers, rows, strict=True)
        ]
        assert summary["count"] == count
        assert summary["accuracy"] == sum(a["accuracy"] for a in answers) / count
        assert math.isclose(
            summary["loss"],
            sum(a["loss"] for a in answers) / count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for earlier in args.compare_subset:
            path = earlier / name / "answers.jsonl"
            prior = [json.loads(line) for line in path.read_text().splitlines()]
            actual = {a["id"]: a for a in answers}
            assert prior and all(actual[a["id"]] == a for a in prior)
            files[str(path)] = file_hash(path)
        files[str(answers_path)] = file_hash(answers_path)
        files[str(folder / "summary.json")] = file_hash(folder / "summary.json")
        results.append(
            {
                **summary,
                "rows": metrics,
                "failures": [
                    {k: a[k] for k in ("id", "prediction", "target")}
                    for a in answers
                    if not a["accuracy"]
                ],
            }
        )
    return {
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "complete": True,
        "source_verified": True,
        "data_verified": True,
        "predictions_recomputed": True,
        "retained_positions_verified": True,
        "decode_qualified": True,
        "engine": "research",
        "run_id": args.run.name,
        "tier": tier,
        "partition": partition,
        "count": count,
        "checkpoint": checkpoint,
        "decision": decision,
        "decision_sha256": decision_hash,
        "results": results,
        "compared_subsets": [str(p) for p in args.compare_subset],
        "files": files,
        "timings_collected": False,
    }


if __name__ == "__main__":
    main()
