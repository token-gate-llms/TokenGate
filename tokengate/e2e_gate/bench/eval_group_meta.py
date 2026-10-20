"""Shared helper for annotating benchmark plots with train/eval distribution info.

Given a pretraining manifest, resolve which LongBench subsets were **in
distribution** (trained on) vs held-out, and each trained subset's mean training
context length (`ctx_mean`, recorded in the manifest).  Eval context length comes
from the benchmark CSV (`n_ctx_tokens`), so plotters can show train-vs-eval ctx
side by side and mark which datasets a panel's numbers were trained on.
"""
from tokengate.datasets.manifest import load_manifest


def manifest_task_ctx(manifest):
    """-> (trained_ctx {task: ctx_mean float}, trained_tasks set[str]).

    Trained subsets come from manifest['datasets'] (each has a ctx_mean); held-out
    (eval_only) subsets are absent from trained_ctx (they have no training data).
    """
    m = load_manifest(manifest)
    # Some trained rows are style-twins (mfqa_style / multihop_style) that are NOT
    # LongBench eval subsets and carry no ctx_mean — skip them (they never key an
    # eval group anyway).  Keeps plotting robust across manifest versions.
    trained_ctx = {r["task"]: float(r["ctx_mean"])
                   for r in m["datasets"] if r.get("ctx_mean") is not None}
    return trained_ctx, set(trained_ctx)


def manifest_train_breakdown(manifest):
    """-> list of {task, n_train, ctx_mean} for the trained (in-distribution)
    subsets, in manifest order.  Held-out subsets have no training data and are
    excluded.  Used to draw the pretraining-data-size histograms."""
    m = load_manifest(manifest)
    return [{"task": r["task"],
             "n_train": int(r["n_train_post_filter"]),
             "ctx_mean": float(r.get("ctx_mean") or 0.0)}
            for r in m["datasets"]]


def manifest_eval_only_tasks(manifest):
    """-> list of held-out (eval-only) task names, in manifest order.  These are
    evaluated but never trained on, so they have no n_train / ctx_mean."""
    m = load_manifest(manifest)
    return [r["task"] for r in (m.get("eval_only_datasets") or [])]


def _short(t):
    return t.replace("longbench_", "") if isinstance(t, str) else t


def panel_annotation(tasks, trained_ctx, trained_tasks, eval_ctx):
    """Annotation for a panel spanning `tasks` (a list of task names).

    Returns (paren, ctx_line):
      paren    — 'in-dist: a, b' when the panel has trained subsets, else
                 'held-out: a, b' (by design each group is homogeneous, but a
                 mixed panel lists its in-dist members).
      ctx_line — 'train ctx≈X · eval ctx≈Y', with train ctx = mean ctx_mean over
                 the panel's in-dist subsets, or 'n/a (held-out)' when none.
    """
    indist = [t for t in tasks if t in trained_tasks]
    held = [t for t in tasks if t not in trained_tasks]
    ectx = f"eval ctx≈{eval_ctx:.0f}" if eval_ctx == eval_ctx else "eval ctx≈?"
    if indist:
        paren = "in-dist: " + ", ".join(_short(t) for t in indist)
        tctx = sum(trained_ctx[t] for t in indist) / len(indist)
        return paren, f"train ctx≈{tctx:.0f} · {ectx}"
    paren = "held-out: " + ", ".join(_short(t) for t in held)
    return paren, f"train ctx: n/a (held-out) · {ectx}"


def overall_annotation(tasks, trained_ctx, trained_tasks, eval_ctx):
    """Annotation for an aggregate 'overall' panel over all `tasks`."""
    indist = [t for t in tasks if t in trained_tasks]
    held = [t for t in tasks if t not in trained_tasks]
    paren = f"{len(indist)} in-dist, {len(held)} held-out"
    ectx = f"eval ctx≈{eval_ctx:.0f}" if eval_ctx == eval_ctx else "eval ctx≈?"
    if indist:
        tctx = sum(trained_ctx[t] for t in indist) / len(indist)
        return paren, f"train ctx≈{tctx:.0f} (in-dist) · {ectx}"
    return paren, f"train ctx: n/a · {ectx}"
