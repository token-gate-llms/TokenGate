"""Schedule benchmark requests and measure throughput over elapsed run windows.

Static batching waits for a complete batch before submitting the next. Continuous
batching replaces completed requests up to a client concurrency limit. These
controls affect submission; the inference engine retains its own scheduler.
"""

from __future__ import annotations

import asyncio
import math
import random
import statistics
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BatchingConfig:
    """Explicit request-submission controls, independent of token-retention K."""

    mode: str = "sequential"
    size: int = 1
    token_budget: int | None = None
    duration_s: float = 0.0
    seed: int = 0
    base_settings: str = "matched"
    ramp_up_s: float = 5.0

    def __post_init__(self) -> None:
        if self.mode not in {"sequential", "static", "continuous"}:
            raise ValueError("mode must be sequential, static, or continuous")
        if self.size < 1 or (self.mode == "sequential" and self.size != 1):
            raise ValueError(
                "size must be positive and sequential mode requires size=1"
            )
        if self.token_budget is not None and self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("duration_s must be finite and nonnegative")
        if self.duration_s and self.mode != "continuous":
            raise ValueError("A sustained measurement window requires continuous mode")
        if not math.isfinite(self.ramp_up_s) or self.ramp_up_s < 0:
            raise ValueError("ramp_up_s must be finite and nonnegative")
        if self.base_settings not in {"matched", "normal"}:
            raise ValueError("base_settings must be matched or normal")

    def ordered(self, samples: list[dict], repeat: int) -> list[dict]:
        """Use the same reproducible order for every engine at a given repeat."""
        ordered = list(samples)
        random.Random(self.seed + repeat).shuffle(ordered)
        return ordered

    def batches(self, samples: list[dict], repeat: int) -> Iterator[list[dict]]:
        """Yield actual batches, including a final batch smaller than size."""
        ordered = self.ordered(samples, repeat)
        for start in range(0, len(ordered), self.size):
            yield ordered[start : start + self.size]


def run_record(
    rows: list[dict],
    start: float,
    end: float,
    *,
    budget: int | str,
    repeat: int,
    config: BatchingConfig,
    drained: float | None = None,
) -> dict:
    """Count completions in the window, without summing overlapping latencies."""
    if end <= start:
        raise ValueError("The measurement window must have positive duration")
    completed = [
        r for r in rows if start <= r["returned_s"] <= end and not r.get("error")
    ]
    failed = [r for r in rows if start <= r["returned_s"] <= end and r.get("error")]
    delivered = sum(
        start <= arrival <= end
        for row in rows
        if not row.get("error")
        for arrival in row["token_arrivals_s"]
    )
    latencies = {
        f"{metric}_{stat}_ms": (
            1000
            * (
                statistics.median(values)
                if stat == "median"
                else quantile(values, 0.95)
            )
            if (values := [r[f"{metric}_s"] for r in completed])
            else None
        )
        for metric in ("ttft", "final")
        for stat in ("median", "p95")
    }
    return dict(
        budget=budget,
        repeat=repeat,
        mode=config.mode,
        size=config.size,
        start_s=start,
        end_s=end,
        drain_end_s=end if drained is None else drained,
        elapsed_s=end - start,
        completed=len(completed),
        failed=len(failed),
        submitted=len(rows),
        output_tokens=delivered,
        requests_per_s=len(completed) / (end - start),
        output_tokens_per_s=delivered / (end - start),
        duration_s=config.duration_s,
        ramp_up_s=config.ramp_up_s if config.duration_s else 0,
        valid=not any(r.get("error") for r in rows),
        **latencies,
    )


async def measure_async(
    request: Callable[[dict, str], Awaitable[dict]],
    samples: list[dict],
    config: BatchingConfig,
    *,
    budget: int | str,
    repeat: int,
    before_batch: Callable[[int], Awaitable[None]] | None = None,
) -> tuple[list[dict], dict]:
    """Submit bounded concurrent work and retain every successful or failed row.

    A finite pass visits each sample once. A sustained run cycles the same input
    order until its deadline, then drains outstanding work. Replayed inputs are
    load requests, not additional independent accuracy examples.
    """
    if not samples:
        raise ValueError("A measurement needs at least one sample")
    ordered = config.ordered(samples, repeat)
    launch = time.perf_counter()
    start = launch + config.ramp_up_s if config.duration_s else launch
    deadline = start + config.duration_s if config.duration_s else math.inf
    index, rows = 0, []

    async def submit(sample: dict, request_index: int) -> dict:
        request_id = f"{budget}-{repeat}-{request_index}"
        queued = time.perf_counter()
        try:
            result = await request(sample, request_id)
        except Exception as exc:
            result = dict(
                id=sample["id"],
                token_ids=[],
                returned_s=time.perf_counter(),
                error=f"{type(exc).__name__}: {exc}",
            )
        return dict(
            result,
            request_id=request_id,
            request_index=request_index,
            budget=budget,
            repeat=repeat,
            queued_s=queued,
            batch_id=request_index // config.size if config.mode == "static" else None,
            batch_size=(
                min(
                    config.size,
                    len(ordered) - (request_index // config.size) * config.size,
                )
                if config.mode == "static"
                else None
            ),
            concurrency=config.size if config.mode == "continuous" else None,
        )

    pending = set()
    try:
        while pending or index < len(ordered) or config.duration_s:
            if config.mode == "static" and not pending and before_batch is not None:
                await before_batch(min(config.size, len(ordered) - index))
            while len(pending) < config.size and time.perf_counter() < deadline:
                if not config.duration_s and index == len(ordered):
                    break
                pending.add(
                    asyncio.create_task(submit(ordered[index % len(ordered)], index))
                )
                index += 1
            if not pending:
                break
            done, pending = await asyncio.wait(
                pending,
                return_when=(
                    asyncio.ALL_COMPLETED
                    if config.mode == "static"
                    else asyncio.FIRST_COMPLETED
                ),
            )
            rows.extend(task.result() for task in done)
            # Preserve failure records and stop admitting work to a broken engine.
            if any(row.get("error") for row in rows):
                if pending:
                    rows.extend(await asyncio.gather(*pending))
                    pending.clear()
                break
            if config.duration_s and time.perf_counter() >= deadline:
                if pending:
                    rows.extend(await asyncio.gather(*pending))
                    pending.clear()
                break
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    end = time.perf_counter()
    window_end = min(deadline, end)
    # Failed runs retain their actual attempted interval and cannot be summarized.
    if any(row.get("error") for row in rows):
        start, window_end = launch, end
    rows.sort(key=lambda row: row["request_index"])
    return rows, run_record(
        rows,
        start,
        window_end,
        budget=budget,
        repeat=repeat,
        config=config,
        drained=end,
    )


def quantile(values: list[float], probability: float) -> float:
    """Return the linearly interpolated empirical quantile."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean_interval(values: list[float]) -> dict:
    """Bootstrap the mean over independent runs, not concurrent requests."""
    mean = statistics.mean(values)
    if len(values) == 1:
        return dict(mean=mean, low=None, high=None, runs=1)
    rng = random.Random(0)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000)]
    return dict(
        mean=mean,
        low=quantile(means, 0.025),
        high=quantile(means, 0.975),
        runs=len(values),
    )


def validate_runs(
    rows: list[dict],
    runs: list[dict],
    inputs: dict,
    budgets: list[int | str],
    repeats: int,
    config: BatchingConfig,
) -> None:
    """Require complete run windows and internally consistent request timelines."""
    expected = {
        (str(budget), repeat) for budget in budgets for repeat in range(repeats)
    }
    actual = [(str(run["budget"]), run["repeat"]) for run in runs]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Incomplete or duplicated run coverage")
    allowed = {sample["id"] for sample in inputs["samples"]}
    request_ids = [row["request_id"] for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Repeated request identity")
    for row in rows:
        if row["id"] not in allowed or row.get("error"):
            raise ValueError("Failed or unknown benchmark request")
        if (str(row["budget"]), row["repeat"]) not in expected:
            raise ValueError("Request has no matching measurement window")
        arrivals = row["token_arrivals_s"]
        if not row["token_ids"] or len(arrivals) != len(row["token_ids"]):
            raise ValueError("Incomplete token arrival timeline")
        if any(type(token) is not int for token in row["token_ids"]):
            raise ValueError("Invalid generated token stream")
        times = [row["submitted_s"], *arrivals, row["returned_s"]]
        if not all(math.isfinite(t) for t in times) or times != sorted(times):
            raise ValueError("Invalid request timeline")
        for metric, expected_value in (
            ("ttft_s", arrivals[0] - row["submitted_s"]),
            ("final_s", arrivals[-1] - row["submitted_s"]),
            ("wall_s", row["returned_s"] - row["submitted_s"]),
        ):
            if not math.isclose(
                row[metric], expected_value, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError("Latency differs from its recorded timeline")
    for run in runs:
        if not run["valid"] or run["completed"] < 1:
            raise ValueError("No successful completed measurement window")
        if run["mode"] != config.mode or run["size"] != config.size:
            raise ValueError("Run configuration changed")
        if config.duration_s and not math.isclose(run["elapsed_s"], config.duration_s):
            raise ValueError("Incomplete sustained measurement window")
        selected = [
            row
            for row in rows
            if (str(row["budget"]), row["repeat"])
            == (str(run["budget"]), run["repeat"])
        ]
        if any(row["returned_s"] > run["drain_end_s"] for row in selected):
            raise ValueError("Request completed outside the recorded drain interval")
        recalculated = run_record(
            selected,
            run["start_s"],
            run["end_s"],
            budget=run["budget"],
            repeat=run["repeat"],
            config=config,
        )
        for key in (
            "completed",
            "failed",
            "submitted",
            "output_tokens",
            "elapsed_s",
            "requests_per_s",
            "output_tokens_per_s",
            "ttft_median_ms",
            "ttft_p95_ms",
            "final_median_ms",
            "final_p95_ms",
        ):
            if not math.isclose(
                run[key], recalculated[key], rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(f"Run {key} differs from request timelines")


@dataclass
class BatchTokenCollector:
    """Collect per-row token arrival times, ignoring tokens after the first EOS."""

    size: int
    eos_token_id: int
    tokens: list[list[int]] = field(init=False)
    arrivals: list[list[float]] = field(init=False)
    ended: bool = False

    def __post_init__(self) -> None:
        self.tokens = [[] for _ in range(self.size)]
        self.arrivals = [[] for _ in range(self.size)]

    def put(self, value) -> None:
        if value.numel() != self.size:
            raise ValueError("The stream must contain one token per batch row")
        now = time.perf_counter()
        for tokens, arrivals, token in zip(
            self.tokens, self.arrivals, value.reshape(-1).tolist(), strict=True
        ):
            if tokens and tokens[-1] == self.eos_token_id:
                continue
            tokens.append(token)
            arrivals.append(now)

    def end(self) -> None:
        self.ended = True
