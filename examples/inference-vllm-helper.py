"""Load benchmark results and draw figures for the vLLM inference notebook.

Read the runner's saved measurements, keep timing repeats separate from accuracy
sample counts, and export latency figures with retrieval outcomes as PNG, PDF,
and SVG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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

    Args:
        table: Aggregate table returned by load_results.
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
    model_centers = []
    for model in models:
        positions = [i for i, (name, _) in enumerate(configurations) if name == model]
        model_centers.append((positions[0] + positions[-1]) / 2)
    data["Configuration"] = data["model"] + "\nK=" + data["budget"]
    data["Implementation"] = data["engine"].map(LABELS)
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
    with (
        sns.axes_style("whitegrid"),
        plt.rc_context(
            {
                "font.family": "serif",
                "font.size": 9,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "figure.dpi": 180,
                "savefig.dpi": 600,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        ),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(18, 5.2), layout="constrained")
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
                positions = [
                    i for i, (name, _) in enumerate(configurations) if name == model
                ]
                ax.hlines(
                    baseline,
                    positions[0] - 0.45,
                    positions[-1] + 0.45,
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
                    center = (
                        categories.index(row.Index)
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
                        outcome = f"{int(row.correct)}/{int(row.examples)}"
                        if speedup:
                            outcome += f"  {speedup}"
                        color = "#333333"
                    position = (bar.get_x() + bar.get_width() / 2, bar.get_height())
                    ax.annotate(
                        outcome,
                        position,
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        rotation=0 if single_example else 90,
                        fontfamily="DejaVu Sans",
                        fontsize=9 if single_example else 7,
                        color=color,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
                    )
                    if single_example and speedup:
                        ax.annotate(
                            speedup,
                            position,
                            xytext=(0, 14),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            rotation=90,
                            fontsize=7,
                            bbox={
                                "facecolor": "white",
                                "edgecolor": "none",
                                "pad": 0.4,
                            },
                        )
            ax.set(title=title, xlabel="", ylabel=label)
            ax.set_xticks(range(len(categories)), budget_labels)
            model_axis = ax.secondary_xaxis("bottom")
            model_axis.set_xticks(model_centers, models)
            model_axis.tick_params(length=0, pad=20, labelsize=8)
            model_axis.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", labelsize=7)
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
