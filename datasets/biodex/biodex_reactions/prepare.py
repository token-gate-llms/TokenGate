"""biodex_reactions — the OFFICIAL BioDEX-Reactions task, iso-run prep (run 16).

Exactness contract (vs our biodex_drugs slice, which is a repo construction):
  * dataset  = BioDEX/BioDEX-Reactions (D'Oosterlinck et al., 2023), official
    train/validation/test splits VERBATIM (11,543 / 2,886 / 4,249; read via
    pyarrow — the installed `datasets` cannot parse the cache metadata);
  * document = the official `fulltext_processed` input construction VERBATIM
    (median ~3.5k tokens; NOT our title+fulltext concat);
  * target   = the official `reactions` string VERBATIM (", "-joined MedDRA
    reaction terms, mean ~4.8 terms);
  * metric   = the official one: set precision/recall/F1 with EXACT string
    match over the terms (BiodexReactionsTask.evaluate; lowercased/stripped,
    punctuation preserved — MedDRA terms contain hyphens etc.).
The only repo-side addition is the decoder-style instruction+question
wrapper (their baselines were seq2seq without an instruction), documented in
the task class.

Length windows are TRAINING CONFIGURATION, not dataset construction: the
train loader does not length-filter.  The train window ([2048, 16384] for the
paper runs) is set in the train config (the lane script's MIN/MAX_ARTICLE_TOKENS,
i.e. TokenGateConfig.min/max_article_tokens) and applied at train time; the
eval-time window likewise (EVAL_MIN/MAX_ARTICLE_TOKENS).  The store only
enforces the structural cap (max_article_tokens=39000: fits the 40,960-token
context; rows over it are dropped, never truncated).

Eval draw (FROZEN benchmark construction — do not change): seeded shuffle of
the OFFICIAL VALIDATION rows within the eval protocol window [2048, 39000]
-> 500 rows, ids "{i:04d}_biodexr_test"; iso split val_n=50 (50 in-training
val / 450 reported heldout, biodex_drugs parity).  The drawn ids are frozen
in longbench_eval_split_biodexr_iso_v1.json and scored in the paper; changing
the pool filter or seed would silently remap those ids to different articles.
Official test stays untouched in reserve.

Usage:
    python datasets/biodex/biodex_reactions/prepare.py
"""
import glob
import os
import random
from pathlib import Path

import pyarrow as pa
from transformers import AutoTokenizer

from tokengate.datasets.prepare import prepare_dataset

DATASET = "biodex"
TASK = "biodex_reactions"
_HERE = Path(__file__).resolve()
# PREPARE_MODEL selects the tokenizer/store (Qwen3-1.7B default; the Qwen3.5-2B
# hybrid benchmark tokenizes the same official rows into tokenized/Qwen3.5-2B/).
MODEL = os.environ.get("PREPARE_MODEL", "Qwen/Qwen3-1.7B")
OUTPUT_ROOT = Path(os.environ.get("PRETRAIN_DATA_ROOT", _HERE.parents[2]))
# Eval-pool window [EVAL_MIN, EVAL_MAX]: part of the frozen eval draw (see
# module docstring).  EVAL_MAX doubles as the store's structural cap.  The
# train-time length window lives in the train config, not here.
EVAL_MIN, EVAL_MAX = 2048, 39000
EVAL_DRAW = 500
SEED = 0
QUESTION = "What adverse drug reactions did the patient experience?"

_tok = AutoTokenizer.from_pretrained(MODEL)


def _ntok(text):
    return len(_tok(text, add_special_tokens=False)["input_ids"])


def _read_split(split):
    cache = os.path.join(os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")), "BioDEX___bio_dex-reactions")
    files = sorted(glob.glob(f"{cache}/**/*{split}*.arrow", recursive=True))
    assert files, f"no cached BioDEX-Reactions arrow files for split {split!r}"

    def read(f):
        with pa.memory_map(f) as src:
            try:
                return pa.ipc.open_stream(src).read_all()
            except pa.ArrowInvalid:
                return pa.ipc.open_file(src).read_all()

    return pa.concat_tables([read(f) for f in files])


def _rows(split, lo=None, hi=None):
    """lo/hi are set ONLY by the frozen eval draw (pool window); the train
    loader passes none — train-time length windows come from the train config."""
    t = _read_split(split)
    rows, dropped_len, dropped_empty = [], 0, 0
    for i in range(t.num_rows):
        doc = t["fulltext_processed"][i].as_py() or ""
        reactions = (t["reactions"][i].as_py() or "").strip()
        if not doc.strip() or not reactions:
            dropped_empty += 1
            continue
        article = f"{doc}\n\nQuestion: {QUESTION}"
        if lo is not None:
            n = _ntok(article)
            if not (lo <= n <= hi):
                dropped_len += 1
                continue
        terms = [x.strip() for x in reactions.split(",") if x.strip()]
        rows.append({
            "dataset": DATASET, "task": TASK,
            "article": article, "answer": reactions,
            "metadata": {"answers": terms, "pmid": str(t["pmid"][i].as_py()),
                         "n_reactions": len(terms), "all_classes": None},
        })
    window = f"in [{lo}, {hi}] " if lo is not None else ""
    print(f"[biodex_reactions] {split}: {t.num_rows} official rows -> {len(rows)} kept "
          f"{window}({dropped_len} length-dropped, {dropped_empty} empty)", flush=True)
    return rows


def load_train():
    rows = _rows("train")
    rows.sort(key=lambda r: r["metadata"]["pmid"])
    for i, r in enumerate(rows):
        r["id"] = f"biodexr_train_{i:05d}"
    return rows


def load_val():
    rows = _rows("validation", EVAL_MIN, EVAL_MAX)
    rows.sort(key=lambda r: r["metadata"]["pmid"])
    rng = random.Random(SEED)
    rng.shuffle(rows)
    assert len(rows) >= EVAL_DRAW, f"validation pool {len(rows)} < {EVAL_DRAW}"
    rows = rows[:EVAL_DRAW]
    for i, r in enumerate(rows):
        r["id"] = f"{i:04d}_biodexr_test"
    return rows


if __name__ == "__main__":
    prepare_dataset(dataset_name=DATASET, task_name=TASK,
                    load_train_fn=load_train, load_val_fn=load_val,
                    model=MODEL, output_root=OUTPUT_ROOT, run_sanity=False,
                    max_article_tokens=EVAL_MAX)
