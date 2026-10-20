"""Packed, memory-mappable token store for pretraining-mixture training.

Motivation
----------
The per-(dataset,task) ``tokenized/<model>/train.pkl`` files store a
``List[Dict]`` where every row carries ``full_ids`` as a Python ``int`` list
plus the raw ``article`` / ``target`` strings.  This representation is:

  * ~9x larger in RAM than the packed token bytes (Python ``int`` objects +
    list overhead), so the 50 GB on-disk pool unpickles to ~290 GB; and
  * slow to load (pickle rebuilds millions of Python objects, ~10 s/GB); and
  * replicated per rank (each of the 8 ranks unpickles its own copy).

During training the step only needs ``full_ids`` + four integers
(``ctx_len``, ``article_start``, ``article_end``, ``n_ans``) + the dataset
name.  The raw ``article`` / ``target`` / ``input`` text is used *only* by the
(tiny, capped) eval set, never by a training step.

This module converts each ``train.pkl`` into a compact, memory-mappable pair:

    <tokenized>/<model>/packed/
        train_tokens.npy   # 1-D int32, every row's full_ids concatenated
        train_meta.npy     # 2-D int64 (n_rows, 7), one row of metadata each
        info.json          # provenance + counts for cheap validation

``train_tokens.npy`` is loaded with ``mmap_mode='r'`` so it is *never* read
into the process heap: the OS pages it in on demand and shares one physical
copy across every rank on the node (page cache).  ``train_meta.npy`` is tiny
(~56 bytes/row) and loaded fully.

Meta columns (all int64)::

    0 OFF   token offset into train_tokens.npy
    1 LEN   number of tokens (== len(full_ids))
    2 CTX   ctx_len (prompt length, answer excluded)
    3 AS    article_start
    4 AE    article_end
    5 NANS  n_ans
    6 HASH  stable 63-bit hash of the article token sub-array, for the
            train/eval dedup safety net (see art_token_hash)

The format is intentionally defined in one place and consumed by both the
trainer (``train_e2e_gate.py``) and the round-trip test.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Format constants ─────────────────────────────────────────────────────────
TOKEN_DTYPE = np.int32          # Qwen3 vocab ~152k < 2**31, fits int32
META_DTYPE = np.int64
N_META_COLS = 7
OFF, LEN, CTX, AS, AE, NANS, HASH = range(N_META_COLS)

_TOKENS_FNAME = "train_tokens.npy"
_META_FNAME = "train_meta.npy"
_INFO_FNAME = "info.json"


def packed_dir(tokenized_model_dir: str | Path) -> Path:
    """Return the packed/ subdir for a ``tokenized/<model>`` directory."""
    return Path(tokenized_model_dir) / "packed"


def packed_paths(tokenized_model_dir: str | Path) -> Tuple[Path, Path, Path]:
    d = packed_dir(tokenized_model_dir)
    return d / _TOKENS_FNAME, d / _META_FNAME, d / _INFO_FNAME


def art_token_hash(article_ids) -> int:
    """Stable 63-bit hash of an article's token sub-array.

    Used as the train/eval dedup key.  Must be deterministic across processes
    (Python's built-in ``hash`` is salted, so we use blake2b over the raw
    int32 bytes).  Computed identically in the converter (over a train row's
    ``full_ids[article_start:article_end]``) and in the trainer's loader (over
    an eval row's matching sub-array) so the two are directly comparable.
    """
    arr = np.ascontiguousarray(np.asarray(article_ids, dtype=TOKEN_DTYPE))
    h = hashlib.blake2b(arr.tobytes(), digest_size=8).digest()
    # Mask to 63 bits so the value always fits a signed int64 meta column.
    return int.from_bytes(h, "little") & 0x7FFFFFFFFFFFFFFF


def is_fresh(tokenized_model_dir: str | Path, src_pkl: str | Path) -> bool:
    """True if a packed store exists and is newer than its source train.pkl."""
    _tok, _meta, _info = packed_paths(tokenized_model_dir)
    if not (_tok.exists() and _meta.exists() and _info.exists()):
        return False
    try:
        info = json.loads(Path(_info).read_text())
    except Exception:
        return False
    src = Path(src_pkl)
    if not src.exists():
        # Source removed but packed present + complete: treat as fresh.
        return True
    return (
        info.get("source_pkl_size") == src.stat().st_size
        and abs(info.get("source_pkl_mtime", 0) - src.stat().st_mtime) < 1.0
    )


def pack_one(
    src_pkl: str | Path,
    tokenized_model_dir: str | Path,
    *,
    model: str,
    force: bool = False,
    verbose: bool = True,
) -> Dict:
    """Convert one ``train.pkl`` into the packed store under ``packed/``.

    Returns the info dict.  Idempotent: skips when a fresh store already exists
    (unless ``force``).  Writes atomically (``*.tmp`` then ``os.replace``).
    """
    src_pkl = Path(src_pkl)
    out_dir = packed_dir(tokenized_model_dir)
    tok_path, meta_path, info_path = packed_paths(tokenized_model_dir)

    if not force and is_fresh(tokenized_model_dir, src_pkl):
        if verbose:
            print(f"[pack] fresh, skipping: {out_dir}")
        return json.loads(info_path.read_text())

    if not src_pkl.exists():
        raise FileNotFoundError(f"[pack] source not found: {src_pkl}")

    t0 = time.time()
    with src_pkl.open("rb") as f:
        rows = pickle.load(f)
    if not isinstance(rows, list):
        raise TypeError(
            f"[pack] expected a list of dicts in {src_pkl}, got {type(rows)}"
        )
    n_rows = len(rows)
    if n_rows == 0:
        raise RuntimeError(f"[pack] {src_pkl} contains 0 rows")

    # Two-pass would let us preallocate exactly, but full_ids lengths are cheap
    # to read first.  Preallocate tokens to total length; fill meta as we go.
    lengths = np.fromiter((len(r["full_ids"]) for r in rows), dtype=np.int64,
                          count=n_rows)
    total_tokens = int(lengths.sum())
    tokens = np.empty(total_tokens, dtype=TOKEN_DTYPE)
    meta = np.empty((n_rows, N_META_COLS), dtype=META_DTYPE)

    prompt_versions = set()
    cursor = 0
    for i, r in enumerate(rows):
        ids = r["full_ids"]
        ln = len(ids)
        a_s = int(r["article_start"])
        a_e = int(r["article_end"])
        # np.asarray on a Python int list is the fast path into the int32 buffer.
        ids_arr = np.asarray(ids, dtype=TOKEN_DTYPE)
        tokens[cursor:cursor + ln] = ids_arr
        meta[i, OFF] = cursor
        meta[i, LEN] = ln
        meta[i, CTX] = int(r["ctx_len"])
        meta[i, AS] = a_s
        meta[i, AE] = a_e
        meta[i, NANS] = int(r["n_ans"])
        meta[i, HASH] = art_token_hash(ids_arr[a_s:a_e])
        cursor += ln
        pv = r.get("prompt_version")
        if pv is not None:
            prompt_versions.add(pv)

    if cursor != total_tokens:
        raise RuntimeError(
            f"[pack] token cursor {cursor} != total {total_tokens} for {src_pkl}"
        )
    if len(prompt_versions) > 1:
        raise RuntimeError(
            f"[pack] {src_pkl} mixes prompt_versions {prompt_versions}; refusing "
            f"to pack (would make the trainer's prompt_version check ambiguous)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Atomic writes.  Temp names must keep the .npy extension, otherwise
    # np.save() appends a second ".npy" and the rename target won't exist.
    tmp_tok = out_dir / "train_tokens.tmp.npy"
    tmp_meta = out_dir / "train_meta.tmp.npy"
    np.save(tmp_tok, tokens)
    np.save(tmp_meta, meta)
    os.replace(tmp_tok, tok_path)
    os.replace(tmp_meta, meta_path)

    info = {
        "n_rows": n_rows,
        "n_tokens": total_tokens,
        "token_dtype": np.dtype(TOKEN_DTYPE).name,
        "meta_cols": ["off", "len", "ctx", "article_start", "article_end",
                      "n_ans", "art_hash"],
        "model": model,
        "prompt_version": next(iter(prompt_versions)) if prompt_versions else None,
        "source_pkl": str(src_pkl),
        "source_pkl_size": src_pkl.stat().st_size,
        "source_pkl_mtime": src_pkl.stat().st_mtime,
        "packed_at": time.time(),
    }
    tmp_info = info_path.with_suffix(".json.tmp")
    tmp_info.write_text(json.dumps(info, indent=2))
    os.replace(tmp_info, info_path)

    if verbose:
        dt = time.time() - t0
        print(f"[pack] {src_pkl} → {out_dir}  "
              f"({n_rows} rows, {total_tokens/1e6:.1f}M tok, "
              f"{tokens.nbytes/1e9:.2f}GB tokens, {dt:.1f}s)")
    # Drop big arrays before returning.
    del tokens, meta, rows
    return info


def load_packed(
    tokenized_model_dir: str | Path,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load a packed store.  Tokens are memory-mapped (read-only); meta is
    loaded fully into RAM (tiny)."""
    tok_path, meta_path, info_path = packed_paths(tokenized_model_dir)
    for p in (tok_path, meta_path, info_path):
        if not p.exists():
            raise FileNotFoundError(f"[pack] missing packed file: {p}")
    tokens = np.load(tok_path, mmap_mode="r")
    meta = np.load(meta_path)
    info = json.loads(info_path.read_text())
    return tokens, meta, info


# ── CLI ──────────────────────────────────────────────────────────────────────
def _resolve_pairs(manifest_or_csv: str) -> List[Tuple[str, str]]:
    """Resolve (dataset, task) pairs from a manifest version name, or from an
    explicit 'ds1:tk1,ds2:tk2' CSV."""
    if ":" in manifest_or_csv and "/" not in manifest_or_csv:
        pairs = []
        for tok in manifest_or_csv.split(","):
            tok = tok.strip()
            if not tok:
                continue
            ds, tk = tok.split(":", 1)
            pairs.append((ds.strip(), tk.strip()))
        return pairs
    from tokengate.datasets.manifest import load_manifest
    m = load_manifest(manifest_or_csv)
    return [(r["dataset"], r["task"]) for r in m["datasets"]]


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Pack tokenized train.pkl files into mmap-able token stores."
    )
    ap.add_argument("manifest", help="manifest version name, or 'ds:tk,ds:tk' CSV")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]),
                    help="pretraining_data root (default: this package's parent)")
    ap.add_argument("--model", default="Qwen3-8B",
                    help="model subdir under tokenized/ (default: Qwen3-8B)")
    ap.add_argument("--force", action="store_true",
                    help="repack even if a fresh store exists")
    ap.add_argument("--only", default="",
                    help="comma-separated dataset names to restrict to")
    args = ap.parse_args()

    pairs = _resolve_pairs(args.manifest)
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        pairs = [(d, t) for d, t in pairs if d in keep]

    root = Path(args.root)
    print(f"[pack] root={root}  model={args.model}  pairs={len(pairs)}  force={args.force}")
    total_tok = 0
    total_rows = 0
    t0 = time.time()
    for i, (ds, tk) in enumerate(pairs):
        td = root / ds / tk / "tokenized" / args.model
        src = td / "train.pkl"
        print(f"[pack] ({i+1}/{len(pairs)}) {ds}/{tk}")
        info = pack_one(src, td, model=args.model, force=args.force)
        total_tok += info["n_tokens"]
        total_rows += info["n_rows"]
    print(f"[pack] DONE  {total_rows} rows  {total_tok/1e6:.1f}M tokens  "
          f"({total_tok * np.dtype(TOKEN_DTYPE).itemsize / 1e9:.2f}GB packed)  "
          f"in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
