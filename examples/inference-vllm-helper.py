"""Run and present the comparisons in the vLLM inference notebook.

Use the existing benchmark runner for inference and the NIAH generator for the
held-out examples. Display hardware, answers, and aggregate measurements with
Rich, keeping timing repeats separate from accuracy sample counts. Export
latency figures with retrieval outcomes as PNG, PDF, and SVG.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
NIAH_SPLIT = (
    ROOT / "tokengate/datasets/eval_splits/longbench_eval_split_niah32k_iso_v1.json"
)
NIAH_POOL_SHA256 = "15699b25054205022652bdbea182e5a742210eb68ac08e19c501a57a85882b3e"
console = Console(width=140)

PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def saved_run_paths(variable: str) -> list[Path]:
    """Read an optional JSON list of saved run directories from the environment."""
    paths = json.loads(os.environ.get(variable, "[]"))
    if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
        raise ValueError(f"{variable} must be a JSON list of directory paths")
    return [Path(path).expanduser().resolve() for path in paths]


def show_hardware(saved_runs: list[Path], gpu: str) -> None:
    """Print recorded inference hardware, or inspect the current host for a new run."""
    from benchmarks.run_vllm_benchmark import hardware_info

    hosts = []
    for run in saved_runs:
        manifest = json.loads((run / "manifest.json").read_text())
        hosts.append(
            (manifest["config"]["name"], manifest["hardware"], manifest["selected_gpu"])
        )
    if not saved_runs:
        hosts.append(("Current inference host", hardware_info(), gpu))
    for label, hardware, selected_gpu in hosts:
        table = Table(title=f"Inference hardware: {label}", show_header=False)
        for name in (
            "cpu",
            "logical_cpus",
            "ram_bytes",
            "python",
            "pytorch_cuda_version",
        ):
            value = hardware.get(name, "unavailable")
            if name == "ram_bytes" and isinstance(value, int):
                value = f"{value / 2**30:.1f} GiB"
            table.add_row(name.replace("_", " ").capitalize(), str(value))
        for device in hardware["gpus"]:
            if device["index"] == selected_gpu:
                table.add_row(
                    f"GPU {selected_gpu}",
                    f"{device['name']}, {float(device['vram_mib']) / 1024:.1f} GiB",
                )
                table.add_row("GPU driver", device["driver"])
        console.print(table)


def show_checkpoints(configs: list[Path]) -> None:
    """Print the models, released checkpoints, gate layers, and configured budgets."""
    table = Table(
        "Model", "Gate layer", "Checkpoint", "Budgets", title="Published checkpoints"
    )
    for path in configs:
        cfg = yaml.safe_load(path.read_text())
        table.add_row(
            cfg["name"],
            str(cfg["gate_layer"]),
            cfg["checkpoint"],
            ", ".join([*(f"{k:,}" for k in cfg["k_values"]), "All"]),
        )
    console.print(table)


def show_example(path: Path, n: int) -> dict:
    """Preview the first example and check that the requested sample count exists.

    Returns the displayed example so its ID can select the answer table later.
    """
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    if not 1 <= n <= len(samples):
        raise ValueError(f"Choose between 1 and {len(samples)} examples")
    sample = samples[0]
    console.print(f"[bold]Question:[/bold] {sample['question']}")
    console.print(f"[bold]Expected answer:[/bold] {sample['answer']}")
    console.print(f"[bold]Text preview:[/bold] {sample['body'][:500]}...")
    return sample


def _run_logged(
    command: list[str],
    log: Path,
    description: str,
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Run a subprocess from the checkout and retain its output when it fails."""
    try:
        with console.status(description), log.open("w") as stream:
            subprocess.run(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
                check=True,
            )
    except subprocess.CalledProcessError as error:
        console.print(log.read_text()[-6000:], markup=False)
        raise RuntimeError(f"Run failed. Details: {log}") from error


def run_benchmarks(
    configs: list[Path],
    output: Path,
    *,
    saved_runs: list[Path],
    engines: list[str],
    vllm_source: Path,
    gpu: str,
    warmups: int,
    repeats: int,
    n: int | None = None,
    split: str = "val",
    profile: bool = False,
    original_python: str = sys.executable,
    vllm_python: str = sys.executable,
) -> list[Path]:
    """Reuse saved measurements or launch each model through the benchmark CLI.

    Args:
        configs: Runner YAML files, one per model.
        output: Parent directory for new runs; existing run folders are rejected.
        saved_runs: Completed runs to load instead of launching any subprocesses.
        engines: Implementations to measure, using the runner's engine names.
        vllm_source: Checkout containing the TokenGate vLLM exporter.
        gpu: CUDA device passed to each sequential run.
        warmups: Unmeasured repetitions before timing each setting.
        repeats: Measured repetitions per example and setting.
        n: Example limit, or None to use the entire selected split.
        split: Validation or test partition, as understood by the runner.
        profile: Whether to collect an additional, untimed trace.
        original_python: Python executable for the runner and original worker.
        vllm_python: Python executable for the vLLM worker.

    Returns:
        Directories containing the recorded measurements and their manifests.

    Raises:
        FileExistsError: A new run would overwrite an existing run directory.
        RuntimeError: A benchmark subprocess failed; its log is retained.
    """
    if saved_runs:
        for run in saved_runs:
            table, _ = load_results([run])
            manifest = json.loads((run / "manifest.json").read_text())
            console.print(
                f"{run.name}: loaded {len(table[table.engine.isin(engines)])} settings, "
                f"{manifest['warmups']} warmup(s), {manifest['repeats']} timing repetitions."
            )
        console.print("Using saved measurements; inference will not run again.")
        return list(saved_runs)
    if not (vllm_source / "tools/export_token_gate.py").is_file():
        raise FileNotFoundError("Set VLLM_SOURCE to the TokenGate vLLM checkout")
    runs = [output / config.stem for config in configs]
    if len(runs) != len(set(runs)):
        raise ValueError("Configuration filenames must have distinct stems")
    for run in runs:
        if run.exists():
            raise FileExistsError(f"Choose a new output directory: {run}")
    output.mkdir(parents=True, exist_ok=True)
    for config, run in zip(configs, runs, strict=True):
        console.rule(f"Run {yaml.safe_load(config.read_text())['name']}")
        command = [
            original_python,
            "-m",
            "benchmarks.run_vllm_benchmark",
            "--config",
            str(config),
            "--output",
            str(run),
            "--split",
            split,
            "--warmups",
            str(warmups),
            "--repeats",
            str(repeats),
            "--gpu",
            gpu,
            "--original-python",
            original_python,
            "--vllm-python",
            vllm_python,
            "--vllm-source",
            str(vllm_source),
            "--engines",
            *engines,
        ]
        if n is not None:
            command.extend(["--n", str(n)])
        if profile:
            command.append("--profile")
        _run_logged(command, output / f"{config.stem}.log", "Warm up and measure")
        console.print(f"Finished. Raw results: {run}")
    return runs


def show_answers(runs: list[Path], sample_id: str, engines: list[str]) -> None:
    """Print one example's first-repeat answers and recorded token agreement."""
    from benchmarks.run_vllm_benchmark import LABELS

    answers = Table(
        "Model",
        "Implementation",
        "K",
        "Answer",
        "Correct",
        "Cap reached",
        title="Answers to the same question",
    )
    for run in runs:
        with (run / "measurements.jsonl").open() as stream:
            for row in map(json.loads, stream):
                if (
                    row["engine"] in engines
                    and row["repeat"] == 0
                    and row["id"] == sample_id
                ):
                    answers.add_row(
                        row["model"],
                        LABELS[row["engine"]],
                        str(row["budget"]),
                        row["answer"],
                        "yes" if row["correct"] else "no",
                        "yes" if row["hit_output_cap"] else "no",
                    )
    console.print(answers)
    for run in runs:
        validation = json.loads((run / "validation.json").read_text())
        model = json.loads((run / "inputs.json").read_text())["config"]["name"]
        agreement = (
            "Generated tokens match at every tested budget."
            if validation["cross_engine_token_agreement"]
            else "Generated tokens differ for at least one tested budget."
        )
        console.print(f"{model}, original versus Triton: {agreement}")
        if validation.get("kernel_token_agreement") is not None:
            agreement = "match" if validation["kernel_token_agreement"] else "differ"
            console.print(
                f"{model}, Triton versus tiled prefill: generated tokens {agreement}."
            )


def niah_test_ids() -> list[str]:
    """Load the 184 reserved NIAH IDs and verify validation examples are excluded."""
    split = json.loads(NIAH_SPLIT.read_text())["niah_kamradt_32k"]
    ids = split["test"]
    if len(ids) != 184 or len(set(ids)) != 184 or set(ids) & set(split["val"]):
        raise ValueError("Expected 184 distinct test IDs disjoint from validation")
    console.print(
        f"Test split: {len(ids)} examples; {len(split['val'])} validation examples excluded."
    )
    return ids


def prepare_niah_configs(
    configs: list[Path],
    samples: Path,
    output: Path,
    *,
    python: str = sys.executable,
) -> list[Path]:
    """Prepare verified NIAH test data and write runner configs in a new directory.

    Generate the 32K pool if absent, require its recorded SHA-256, and check the
    published examples and reserved IDs before writing any model configurations.
    Existing output directories are rejected so failed runs remain inspectable.

    Args:
        configs: Model YAML files to copy with the test dataset and split paths.
        samples: JSONL evaluation pool; generated here if it does not exist.
        output: New directory for configs and the generation log.
        python: Python executable with the NIAH generator's dependencies.

    Returns:
        Paths to the generated model configs for the benchmark runner.
    """
    from benchmarks.run_vllm_benchmark import file_hash, read_samples

    test_ids = niah_test_ids()
    output.mkdir(parents=True, exist_ok=False)
    if not samples.exists():
        samples.parent.mkdir(parents=True, exist_ok=True)
        _run_logged(
            [
                python,
                str(ROOT / "datasets/niah_kamradt/generate_niah_kamradt.py"),
                "--output-file",
                str(samples),
                "--n-train",
                "0",
                "--seed",
                "0",
                "--test-target",
                "32000",
                "--test-jit",
                "300",
            ],
            output / "prepare.log",
            "Prepare the 32K evaluation pool",
            env=dict(os.environ, NIAH_DATA_DIR=str(samples.parent)),
        )
    if file_hash(samples) != NIAH_POOL_SHA256:
        raise ValueError("The evaluation pool differs from the verified 32K source.")
    pool = {row["id"]: row for row in map(json.loads, samples.read_text().splitlines())}
    examples = ROOT / "examples/data/niah_32k_samples.jsonl"
    for example in map(json.loads, examples.read_text().splitlines()):
        if any(
            pool[example["id"]][key] != example[key]
            for key in ("body", "question", "answer")
        ):
            raise ValueError(
                f"Published example differs from the pool: {example['id']}"
            )
    prepared = []
    for path in configs:
        cfg = yaml.safe_load(path.read_text())
        cfg.update(
            samples=str(samples),
            eval_split=str(NIAH_SPLIT),
            split_task="niah_kamradt_32k",
        )
        if [row["id"] for row in read_samples(cfg, "test", None)] != test_ids:
            raise ValueError("Prepared examples do not match the reserved test IDs")
        config = output / path.name
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        prepared.append(config)
    return prepared


def show_test_results(
    table, configs: list[Path], engines: list[str], output: Path
) -> None:
    """Check model/engine coverage, print test statistics, and save their JSON table.

    Args:
        table: Example-level test statistics returned by load_tradeoff.
        configs: YAML model configs defining the expected model names.
        engines: Complete set of expected implementations for each model.
        output: Directory in which to write tradeoff-summary.json.
    """
    from benchmarks.run_vllm_benchmark import LABELS

    models = {yaml.safe_load(path.read_text())["name"] for path in configs}
    if set(table.model) != models:
        raise ValueError("Test results do not cover the configured models")
    for model, rows in table.groupby("model"):
        if set(rows.engine) != set(engines):
            raise ValueError(
                f"Test results do not cover the implementations for {model}"
            )
    results = Table(
        "Model",
        "Implementation",
        "K",
        "Correct",
        "Accuracy",
        "TTFT (ms)",
        title="Held-out retrieval and first-token latency (95% intervals)",
    )
    for row in table.itertuples():
        results.add_row(
            row.model,
            LABELS[row.engine],
            row.budget,
            f"{row.correct}/{row.examples}",
            f"{row.accuracy:.1f}% [{row.accuracy_low:.1f}, {row.accuracy_high:.1f}]",
            f"{row.ttft_ms:.1f} [{row.ttft_ms_low:.1f}, {row.ttft_ms_high:.1f}]",
        )
    console.print(results)
    if table.inconsistent_examples.any():
        console.print(
            "Some examples changed their generated tokens across repeats; "
            "the scores above require every repeat to be correct."
        )
    table.to_json(output / "tradeoff-summary.json", orient="records", indent=2)


def show_comparison(table, output: Path) -> None:
    """Export and display the latency figure at notebook resolution, then close it."""
    import matplotlib.pyplot as plt
    from IPython.display import display
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("retina", "svg")
    figure = plot_comparison(table, output)
    try:
        display(figure)
    finally:
        plt.close(figure)


def load_results(directories: list[Path]):
    """Load complete runs with their recorded validation status.

    Args:
        directories: Benchmark output directories, one per model and sample set.

    Returns:
        A table of aggregate measurements and whether every run is qualified.

    Raises:
        ValueError: A run failed or duplicates a model/implementation/budget cell.
    """
    import pandas as pd

    from benchmarks.run_vllm_benchmark import (
        TOKEN_GATE_ENGINES,
        summarize,
        validate_rows,
    )

    rows, qualified = [], True
    for directory in directories:
        inputs = json.loads((directory / "inputs.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        validation = json.loads((directory / "validation.json").read_text())
        if validation["status"] == "failed":
            raise ValueError(f"Failed run: {directory}")
        qualified &= validation.get("qualified_speedups") is True
        measurements = [
            json.loads(line)
            for line in (directory / "measurements.jsonl").read_text().splitlines()
        ]
        for engine in manifest["engines"]:
            budgets = (
                inputs["config"]["budgets"] if engine in TOKEN_GATE_ENGINES else ["All"]
            )
            validate_rows(
                [r for r in measurements if r["engine"] == engine],
                inputs,
                budgets,
                manifest["repeats"],
            )
        rows.extend(summarize(measurements))
    table = pd.DataFrame(rows)
    if table.empty or table.duplicated(["model", "engine", "budget"]).any():
        raise ValueError(
            "Provide nonempty runs with distinct model/engine/budget cells"
        )
    return table, qualified


def plot_comparison(table, output: Path):
    """Plot latency and per-bar retrieval outcomes against base-model baselines.

    One-example runs show correctness symbols; larger sets put accuracy in a row
    below the bars. Pruned budgets show speed-up above the bar relative to the
    faster base backend.

    Args:
        table: Aggregate measurements from load_results or load_tradeoff.
        output: File stem; PNG, SVG, and PDF are written beside it.

    Returns:
        The Matplotlib figure, also usable in a notebook.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    from benchmarks.run_vllm_benchmark import LABELS, TOKEN_GATE_ENGINES

    data = table[
        table.engine.isin(TOKEN_GATE_ENGINES)
        | (
            table.engine.isin(["vllm_base", "vllm_base_triton"])
            & table.budget.eq("All")
        )
    ].copy()
    single_example = data["examples"].eq(1).all()
    models = list(dict.fromkeys(data["model"]))
    budgets = sorted(set(data["budget"]) - {"All"}, key=int) + ["All"]
    configurations = [
        (m, k)
        for m in models
        for k in budgets
        if ((data.model == m) & (data.budget == k)).any()
    ]
    categories = [f"{model}\nK={budget}" for model, budget in configurations]
    budget_labels = [
        "K=All" if budget == "All" else f"K={int(budget):,}"
        for _, budget in configurations
    ]
    data["Configuration"] = data["model"] + "\nK=" + data["budget"]
    data["Implementation"] = data["engine"].map(LABELS)
    centers = dict(zip(categories, range(len(categories)), strict=True))
    group_widths = dict.fromkeys(categories, 0.9)
    if not single_example:
        cursor = 0.0
        for category in categories:
            width = 0.24 * data.Configuration.eq(category).sum()
            centers[category] = cursor + width / 2
            group_widths[category] = width
            cursor += width + 0.28
    model_spans = {}
    for model in models:
        model_categories = [
            category
            for category, (name, _) in zip(categories, configurations, strict=True)
            if name == model
        ]
        first, last = model_categories[0], model_categories[-1]
        model_spans[model] = (
            centers[first] - group_widths[first] / 2,
            centers[last] + group_widths[last] / 2,
        )
    model_centers = [(left + right) / 2 for left, right in model_spans.values()]
    implementations = [LABELS[e] for e in LABELS if e in set(data.engine)]
    palette = dict(
        zip(
            implementations,
            sns.color_palette("colorblind", len(implementations)),
            strict=True,
        )
    )
    palette[LABELS["vllm_base"]] = "#a6a6a6"
    palette[LABELS["vllm_base_triton"]] = "#a6a6a6"
    palette[LABELS["vllm_tiled"]] = "#FFD700"
    style = dict(PLOT_STYLE)
    if not single_example:
        style.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})
    with (
        sns.axes_style("whitegrid"),
        plt.rc_context(style),
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(18, 5.2) if single_example else (22, 7.1),
            layout="constrained",
        )
        for ax, metric, title, label in zip(
            axes,
            ["ttft_ms", "final_ms"],
            [
                "(a) Time to first token",
                "(b) Time to final token",
            ],
            ["Latency (ms)", "Latency (ms)"],
            strict=True,
        ):
            accuracy_labels = []
            sns.barplot(
                data=data,
                x="Configuration",
                y=metric,
                hue="Implementation",
                order=categories,
                hue_order=implementations,
                palette=palette,
                errorbar=None,
                saturation=1,
                ax=ax,
            )
            baselines = (
                data[data.engine.isin(["vllm_base", "vllm_base_triton"])]
                .groupby("model")[metric]
                .min()
            )
            for model, baseline in baselines.items():
                ax.hlines(
                    baseline,
                    *model_spans[model],
                    colors="#555555",
                    linestyles="--",
                    linewidth=1,
                    zorder=3,
                )
            for container, implementation in zip(
                ax.containers, implementations, strict=True
            ):
                if implementation in (
                    LABELS["vllm_base_triton"],
                    LABELS["vllm_tiled"],
                ):
                    for bar in container.patches:
                        bar.set_hatch("xx")
                        bar.set_edgecolor("black")
                        bar.set_linewidth(0.5)
                rows = (
                    data[data.Implementation == implementation]
                    .set_index("Configuration")
                    .reindex(categories)
                    .dropna(subset=[metric])
                )
                for bar, row in zip(container.patches, rows.itertuples(), strict=True):
                    present = data.loc[
                        data.Configuration == row.Index, "Implementation"
                    ].tolist()
                    present.sort(key=implementations.index)
                    if not single_example:
                        bar.set_width(0.24)
                    center = (
                        centers[row.Index]
                        + (present.index(implementation) - (len(present) - 1) / 2)
                        * bar.get_width()
                    )
                    bar.set_x(center - bar.get_width() / 2)
                    latency = getattr(row, metric)
                    speedup = (
                        f"{baselines[row.model] / latency:.1f}×"
                        if row.budget != "All"
                        and row.model in baselines
                        and latency > 0
                        else ""
                    )
                    if single_example:
                        outcome = "✓" if row.correct else "✗"
                        color = "#176b47" if row.correct else "#b3261e"
                    else:
                        accuracy = 100 * row.correct / row.examples
                        outcome = f"{accuracy:.1f}".rstrip("0").rstrip(".") + "%"
                        color = "#333333"
                    position = (bar.get_x() + bar.get_width() / 2, bar.get_height())
                    outcome_label = ax.annotate(
                        outcome,
                        position if single_example else (center, 0),
                        xycoords="data" if single_example else ax.get_xaxis_transform(),
                        xytext=(0, 3 if single_example else -6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if single_example else "top",
                        rotation=0 if single_example else 90,
                        fontfamily="DejaVu Sans" if single_example else "serif",
                        fontsize=9 if single_example else 10,
                        fontweight="normal" if single_example else "bold",
                        color=color,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
                        annotation_clip=False,
                    )
                    if not single_example:
                        accuracy_labels.append(outcome_label)
                    if speedup:
                        ax.annotate(
                            speedup,
                            position,
                            xytext=(0, 14 if single_example else 4),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            rotation=90,
                            fontsize=7 if single_example else 9,
                            color=plt.rcParams["text.color"]
                            if single_example
                            else "#555555",
                            bbox={
                                "facecolor": "white",
                                "edgecolor": "none",
                                "pad": 0.4,
                            },
                        )
            ax.set(title=title, xlabel="", ylabel=label)
            ax.set_xticks(list(centers.values()), budget_labels)
            model_axis = ax.secondary_xaxis("bottom")
            model_axis.set_xticks(model_centers, models)
            accuracy_height = 0
            if not single_example:
                renderer = fig.canvas.get_renderer()
                accuracy_height = (
                    max(
                        text.get_window_extent(renderer).height
                        for text in accuracy_labels
                    )
                    * 72
                    / fig.dpi
                )
                ax.annotate(
                    "Accuracy",
                    (0, 0),
                    xycoords="axes fraction",
                    xytext=(-10, -6 - accuracy_height / 2),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=10,
                    color="#444444",
                    annotation_clip=False,
                )
                ax.tick_params(axis="x", pad=accuracy_height + 15, length=0)
            model_axis.tick_params(
                length=0,
                pad=20 if single_example else accuracy_height + 37,
                labelsize=8 if single_example else 11,
            )
            model_axis.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", labelsize=7 if single_example else 10)
            if not single_example:
                ax.set_xlim(-0.14, cursor - 0.14)
                for (_, right), (left, _) in pairwise(model_spans.values()):
                    ax.axvline((left + right) / 2, color="#dddddd", linewidth=0.8)
            ax.grid(axis="x", visible=False)
            ax.set_axisbelow(True)
            sns.despine(ax=ax)
            ax.margins(y=0.14)
            ax.set_ylim(bottom=0)
            ax.get_legend().remove()
        handles = [container.patches[0] for container in axes[0].containers]
        legend_labels = list(implementations)
        if single_example:
            handles.extend(
                [
                    Line2D([], [], marker="$✓$", color="#176b47", linestyle="none"),
                    Line2D([], [], marker="$✗$", color="#b3261e", linestyle="none"),
                ]
            )
            legend_labels.extend(["Correct", "Incorrect"])
        if not baselines.empty:
            handles.append(Line2D([], [], color="#555555", linestyle="--"))
            legend_labels.append("Fastest Base Qwen")
        fig.legend(
            handles,
            legend_labels,
            loc="outside upper center",
            ncol=(len(legend_labels) + 1) // 2,
            frameon=False,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output.with_suffix("." + suffix), bbox_inches="tight", dpi=600)
    return fig


def load_tradeoff(directories: list[Path], expected_ids: list[str]):
    """Summarize a complete test split, allowing disjoint runs of the same model.

    Each example contributes one median latency and one retrieval outcome per
    setting. A retrieval counts as correct only if every timing repeat is
    correct, matching the benchmark runner. Confidence intervals use Wilson's
    method for accuracy and 2,000 paired example resamples for median latency.

    Args:
        directories: Complete benchmark runs, optionally split into sample shards.
        expected_ids: Exact IDs in the reserved test partition.

    Returns:
        A DataFrame with accuracy, latency, and their 95% confidence intervals.

    Raises:
        ValueError: Runs overlap, omit test examples, or use different weights,
            code, token budgets, implementations, or timing settings for a model.
    """
    import math

    import numpy as np
    import pandas as pd

    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Expected test IDs must be nonempty and unique")
    frames, identities = [], {}
    for directory in directories:
        load_results([directory])  # Validate each run's timing and repeat coverage.
        inputs = json.loads((directory / "inputs.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        ids = [sample["id"] for sample in inputs["samples"]]
        if len(ids) != len(set(ids)) or not set(ids) <= set(expected_ids):
            raise ValueError(
                f"Run contains unexpected or duplicate test IDs: {directory}"
            )
        model = inputs["config"].get("name", inputs["config"]["base_model"])
        identity = {
            "weights": [
                inputs["gate_sha256"],
                inputs["adapter_sha256"],
                inputs["base_files"],
            ],
            "source": manifest["source_sha256"],
            "vllm_source": manifest["vllm_source"]["source_sha256"],
            "config": {
                k: v
                for k, v in inputs["config"].items()
                if k not in {"samples", "eval_split"}
            },
            "engines": manifest["engines"],
            "repeats": manifest["repeats"],
            "warmups": manifest["warmups"],
        }
        if model in identities and identities[model] != identity:
            raise ValueError(f"Inconsistent run configuration for {model}")
        identities[model] = identity
        columns = (
            "model",
            "engine",
            "budget",
            "id",
            "repeat",
            "correct",
            "token_ids",
            "ttft_s",
            "final_s",
        )
        with (directory / "measurements.jsonl").open() as stream:
            frames.append(
                pd.DataFrame(
                    {key: row[key] for key in columns}
                    for row in map(json.loads, stream)
                )
            )
    if not frames:
        raise ValueError("Provide at least one complete test run")
    data = pd.concat(frames, ignore_index=True)
    data["budget"] = data["budget"].astype(str)
    if data.duplicated(["model", "engine", "budget", "id", "repeat"]).any():
        raise ValueError("Overlapping test runs duplicate measurements")

    result = []
    # Reuse these ID resamples for every setting, preserving paired comparisons.
    draws = np.random.default_rng(0).integers(
        len(expected_ids), size=(2000, len(expected_ids))
    )
    for (model, engine, budget), rows in data.groupby(
        ["model", "engine", "budget"], sort=False
    ):
        if set(rows.id) != set(expected_ids):
            raise ValueError(f"Incomplete test split for {model}/{engine}/{budget}")
        examples = rows.groupby("id", sort=False)
        correct = examples.correct.all().reindex(expected_ids)
        n, successes = len(correct), int(correct.sum())
        p, z = successes / n, 1.959963984540054
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
        item = {
            "model": model,
            "engine": engine,
            "budget": budget,
            "examples": n,
            "correct": successes,
            "accuracy": 100 * p,
            "accuracy_low": 0.0 if successes == 0 else 100 * max(0, center - radius),
            "accuracy_high": 100.0 if successes == n else 100 * min(1, center + radius),
            "inconsistent_examples": sum(
                len({tuple(tokens) for tokens in group.token_ids}) != 1
                for _, group in examples
            ),
        }
        for source, target in [("ttft_s", "ttft_ms"), ("final_s", "final_ms")]:
            values = 1000 * examples[source].median().reindex(expected_ids).to_numpy()
            low, high = np.quantile(np.median(values[draws], axis=1), [0.025, 0.975])
            item.update(
                {
                    target: float(np.median(values)),
                    target + "_low": float(low),
                    target + "_high": float(high),
                }
            )
        result.append(item)
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table, _ = load_results(args.runs)
    plot_comparison(table, args.output)
    from rich.console import Console

    Console().print(f"Saved PNG, PDF, and SVG to {args.output.parent}")


if __name__ == "__main__":
    main()
