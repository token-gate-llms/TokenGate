"""Answer-type taxonomy for QA eval rows (qasper-style).

Splits eval samples by what the gold answer REQUIRES, which turns out to dominate
accuracy-at-budget far more than the selection rule does:

  extractive    the gold answer appears VERBATIM in the article → a tiny contiguous
                window suffices (measured on qasper: median span = 0.2% of the article)
  abstractive   the answer does not appear verbatim → needs synthesis across the doc,
                so no single window contains it
  yes_no        the answer is just "yes"/"no" → a judgement, not a location
  unanswerable  the answer asserts the paper does not answer the question → requires
                verifying ABSENCE across the whole document, i.e. actively
                anti-compression

Measured shares on the qasper LongBench test split (n=200): extractive 27%,
abstractive 54%, yes_no 11%, unanswerable 8%. The full-context ceiling differs
sharply by type (0.40 / 0.32 / 0.28 / 0.24), so an aggregate number mixes four very
different problems — hence the per-type panels in plot_accuracy_vs_k.py.
"""

import re

ANSWER_TYPES = ("extractive", "abstractive", "yes_no", "unanswerable")


def _norm(x) -> str:
    return re.sub(r"\s+", " ", str(x).lower()).strip()


def classify_answer(target, article) -> str:
    """Return one of ANSWER_TYPES for a (gold answer, article) pair.

    ``target`` may be a str or a list/tuple of references (the first is used).
    Order matters: unanswerable and yes/no are checked before the verbatim test,
    since e.g. "no" would trivially substring-match almost any article.
    """
    if isinstance(target, (list, tuple)):
        target = target[0] if target else ""
    t = _norm(target)
    if not t:
        return "abstractive"
    if "unanswerable" in t:
        return "unanswerable"
    if t in ("yes", "no", "yes.", "no."):
        return "yes_no"
    if t in _norm(article):
        return "extractive"
    return "abstractive"
