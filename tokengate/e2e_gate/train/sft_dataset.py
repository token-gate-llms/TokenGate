"""SFT dataset construction utilities for e2e gate training.

Wraps existing task + dataset infrastructure to produce
  {"input": formatted_prompt, "target": ground_truth_answer}
pairs suitable for supervised fine-tuning.
"""

import os
import random
import sys
from typing import Callable, Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokengate.datasets.dataset import DatasetConfig as ExperimentConfig
from tokengate.datasets.dataset import load_samples, get_article_text_for_dataset
from tokengate.datasets.tasks.base import Task


def build_sft_samples(
    task: Task,
    dataset_name: str,
    n: int = 10000,
    split: str = "train",
    seed: int = 42,
    max_article_tokens: Optional[int] = None,
    min_article_tokens: Optional[int] = None,
    tokenizer=None,
    article_text_fn: Optional[Callable[[Dict], str]] = None,
) -> List[Dict[str, str]]:
    """Build SFT samples for the given task.

    Parameters
    ----------
    task : Task
        Instantiated task object (provides format_prompt, get_ground_truth, filter_sample).
    dataset_name : str
        Dataset name (e.g. "biodex", "longbench_2wikimqa").
    n : int
        Maximum number of SFT samples to return.
    split : str
        Dataset split to load (default "train").
    seed : int
        Random seed for sampling.
    max_article_tokens : int, optional
        Truncate articles to this many tokens before building the prompt.
        Only the compressible prefix is truncated; the non-compressible suffix
        (e.g. embedded question or MCQ options) is always preserved.
    min_article_tokens : int, optional
        Drop samples whose tokenized compressible region is shorter than this.
        Used in multitask training to filter out trivially short articles that
        would give weak compression signal.  Requires `tokenizer` to be set.
    tokenizer : optional
        HuggingFace tokenizer used for article truncation and length filtering
        (required if max_article_tokens or min_article_tokens is set).

    Returns
    -------
    list of dicts with keys:
        "input"   : full prompt string (formatted with format_prompt)
        "target"  : ground-truth answer string
        "article" : raw article text (before prompt formatting)
    """
    # n < 0 means "all available samples" — use a large cap to fetch everything.
    _all_samples = n < 0
    _n_fetch = (2 ** 30) if _all_samples else n * 4
    config_kwargs = dict(
        dataset=dataset_name,
        num_samples=_n_fetch,
        seed=seed,
        dataset_split=split,
    )
    config = ExperimentConfig(**config_kwargs)
    samples = load_samples(config)

    rng = random.Random(seed)
    valid = [s for s in samples if task.filter_sample(s)]
    rng.shuffle(valid)

    sft_samples = []
    for sample in valid:
        if not _all_samples and len(sft_samples) >= n:
            break

        article = (
            article_text_fn(sample)
            if article_text_fn is not None
            else get_article_text_for_dataset(sample, dataset_name)
        )

        # Optional truncation and min-length filter on the compressible region.
        # The non-compressible suffix (embedded question / MCQ options) is always
        # preserved.  When either bound is set, the tokenizer is required.
        if max_article_tokens is not None or min_article_tokens is not None:
            if tokenizer is None:
                raise ValueError(
                    "build_sft_samples: tokenizer is required when "
                    "max_article_tokens or min_article_tokens is set."
                )
            nc_suffix = task.get_non_compressible_context(article)
            compressible = article[: len(article) - len(nc_suffix)] if nc_suffix else article
            ids = tokenizer.encode(
                compressible,
                add_special_tokens=False,
                truncation=(max_article_tokens is not None),
                max_length=(max_article_tokens if max_article_tokens is not None else None),
            )
            if min_article_tokens is not None and len(ids) < min_article_tokens:
                continue
            if max_article_tokens is not None:
                article = tokenizer.decode(ids) + nc_suffix

        gt = task.get_sft_target(sample)
        if not gt or gt.strip().lower() == "unknown":
            continue

        sft_samples.append(
            {
                "input": task.format_prompt(article),
                "target": gt,
                "article": article,
                # Identity tags consumed by the multitask trainer / eval loop.
                # The trainer uses `task` to look up the right Task object at
                # eval time; `dataset` partitions per-dataset W&B metrics.
                "dataset": dataset_name,
                "task": task.name,
                # Pass through any fields the judge may need (e.g. "question" for
                # QasperQATask) and fields used by alternative-article baselines
                # (e.g. RepoBench gold-context baseline needs the raw context
                # list and gold_snippet_index).  Unknown keys are silently
                # ignored by other tasks.
                **{
                    k: sample[k]
                    for k in (
                        "question",
                        # RepoBench gold-context baseline fields:
                        "context", "gold_snippet_index", "file_path",
                        "repo_name", "import_statement", "cropped_code",
                    )
                    if k in sample
                },
            }
        )

    return sft_samples
