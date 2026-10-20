"""Dataset loading and utilities for BioDEX and QASPER."""

import json
import random
from typing import List, Dict, Any, Optional

from datasets import load_dataset

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Parameters consumed by the sample loaders below.

    (Replaces the legacy compressor-era ExperimentConfig — this is purely
    dataset-loading config.)"""
    dataset: str = "biodex"
    dataset_name: str = "BioDEX/raw_dataset"
    dataset_split: str = "train"
    num_samples: int = 200
    seed: int = 42


# Backwards-compatible alias (old loader signatures referenced ExperimentConfig).
ExperimentConfig = DatasetConfig


# ── BioDEX ────────────────────────────────────────────────────────────────────

def _load_biodex_dataset(dataset_name: str, split: str):
    """Load BioDEX dataset, with a direct-from-arrow fallback for offline envs.

    datasets 2.21+ requires the config name to include the hash suffix
    (e.g. "default-4857aa8790d0e403") when the hub is unreachable.  We try
    the normal load first; on failure we locate the arrow files in the HF
    cache and load them directly, which always works offline.
    """
    import glob
    import os
    from datasets import Dataset, concatenate_datasets

    # First attempt: standard load (works when hub is reachable)
    try:
        return load_dataset(dataset_name, "default", split=split)
    except (ValueError, Exception):
        pass

    # Fallback: load directly from cached arrow files
    hf_cache = os.environ.get(
        "HF_DATASETS_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets"),
    )
    # Cache dir uses triple-underscore for slashes: "BioDEX/raw_dataset" → "BioDEX___raw_dataset"
    cache_key = dataset_name.replace("/", "___")
    pattern = os.path.join(hf_cache, cache_key, "**", f"*{split}*.arrow")
    arrow_files = sorted(glob.glob(pattern, recursive=True))
    if not arrow_files:
        raise FileNotFoundError(
            f"Could not load {dataset_name} from HF hub or local cache. "
            f"Searched: {pattern}"
        )
    print(f"[dataset] Loading {dataset_name} from {len(arrow_files)} cached arrow file(s).")
    shards = [Dataset.from_file(f) for f in arrow_files]
    return concatenate_datasets(shards)


def load_biodex_samples(config: ExperimentConfig):
    """Load and subsample BioDEX dataset.

    Only keeps samples that have fulltext articles. Returns a list of
    dataset examples (dicts).
    """
    print("Loading BioDEX dataset from HuggingFace...")
    dataset = _load_biodex_dataset(config.dataset_name, config.dataset_split)
    print(f"Full dataset size: {len(dataset)}")

    # Filter to samples with fulltext, patient sex, and seriousness labels
    valid_indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        # Must have fulltext
        if not sample["article"].get("fulltext"):
            continue
        # Must have patient sex
        reports = sample.get("reports", [])
        if not reports:
            continue
        patient = reports[0].get("patient") or {}
        sex = patient.get("patientsex")
        if sex is None or str(sex) not in ("1", "2"):
            continue
        # Must have seriousness
        report = reports[0]
        serious = report.get("serious")
        fec = report.get("fulfillexpeditecriteria")
        if serious is None and fec is None:
            continue
        valid_indices.append(i)

    print(f"Samples with fulltext + sex/seriousness fields: "
          f"{len(valid_indices)} ({len(valid_indices) * 100 / len(dataset):.1f}%)")

    random.seed(config.seed)
    n = min(config.num_samples, len(valid_indices))
    selected = random.sample(valid_indices, n)
    samples = dataset.select(selected)
    print(f"Selected {len(samples)} samples (seed={config.seed})")
    return samples


def get_biodex_article_text(sample: Dict[str, Any]) -> str:
    """Extract the best available article text from a BioDEX sample.

    Prefers fulltext; falls back to abstract.  Prepends the title when available.
    """
    article = sample["article"]
    body = article.get("fulltext") or article.get("abstract") or ""
    title = article.get("title") or ""
    if title and body:
        return f"{title}\n\n{body}"
    return body or title or ""


# Keep old name as alias for backwards compatibility
get_article_text = get_biodex_article_text


# ── Dispatchers ───────────────────────────────────────────────────────────────

def load_prebuilt_eval_samples(path: str) -> List[Dict[str, Any]]:
    """Load pre-built eval samples saved by train_e2e_gate.py.

    Returns a list of synthetic sample dicts recognised by run_experiment.py
    via the ``_prebuilt`` sentinel key.  Each entry carries:
      _prebuilt         : True  — signals that filter_sample should be skipped
      _prebuilt_target  : str   — ground-truth answer (bypasses task.get_ground_truth)
      _prebuilt_article : str   — article text (bypasses get_article_text_for_dataset)
    """
    with open(path) as f:
        entries = json.load(f)
    return [
        {
            "_prebuilt": True,
            "_prebuilt_target": e["target"],
            "_prebuilt_article": e["article"],
        }
        for e in entries
    ]


def load_longbench_samples(config: ExperimentConfig):
    """Load a LongBench English subset (HF: THUDM/LongBench).

    `config.dataset` must be of the form 'longbench:<subset>' where <subset> is
    one of: multifieldqa_en, hotpotqa, 2wikimqa, trec, samsum, passage_count,
    passage_retrieval_en.  Returns a list of normalized sample dicts with
    `article` (= context + "\\n\\nQuestion: " + input), `answer` (first answer
    string), `answers` (full list), `all_classes`, `dataset`, `task`.
    """
    from datasets import load_dataset
    name = getattr(config, "dataset", "")
    subset = name.split(":", 1)[1] if ":" in name else name.replace("longbench_", "", 1)
    split = getattr(config, "dataset_split", None) or "test"
    raw = load_dataset("THUDM/LongBench", subset, split=split)
    n = getattr(config, "num_samples", None) or len(raw)
    out = []
    for i, r in enumerate(raw):
        if i >= n:
            break
        ctx = r.get("context", "") or ""
        inp = (r.get("input", "") or "").strip()
        article = f"{ctx}\n\nQuestion: {inp}" if inp else ctx
        ans = r.get("answers") or []
        out.append({
            "article": article,
            "answer": ans[0] if ans else "",
            "answers": list(ans),
            "all_classes": r.get("all_classes"),
            "dataset": f"longbench_{subset}",
            "task": f"longbench_{subset}",
            "language": r.get("language", "en"),
            "_id": r.get("_id", str(i)),
        })
    return out


def get_longbench_article_text(sample: Dict[str, Any]) -> str:
    return sample.get("article", "") or ""


def load_samples(config: ExperimentConfig):
    """Load dataset samples based on config.dataset."""
    dataset = getattr(config, "dataset", "biodex")
    if dataset.startswith("longbench"):
        return load_longbench_samples(config)
    if dataset == "biodex":
        return load_biodex_samples(config)
    raise ValueError(f"Unknown dataset '{dataset}' — load_samples handles "
                     f"'biodex' and 'longbench_*' (everything else is read "
                     f"from the pre-prepped tokenized store).")


def get_article_text_for_dataset(sample: Dict[str, Any], dataset_name: str, doc_truncate_len: int = None) -> str:
    """Get article text using the appropriate extractor for the dataset."""
    if dataset_name.startswith("longbench"):
        return get_longbench_article_text(sample)
    return get_biodex_article_text(sample)
