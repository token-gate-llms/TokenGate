"""Needle-in-a-Haystack tasks: Kamradt (2023) and RULER NIAH-8 (Hsieh et al., 2024).

Both datasets are OUR generated train/test pools (see
datasets/niah_kamradt/generate_niah_kamradt.py /
the retired ruler generator) that faithfully reproduce the published task forms:
PG-essay haystacks, the canonical needle/query templates, and RULER's
string-containment scoring.  Sample geometry follows the repo convention
(article = haystack + "\n\nQuestion: " + question; the question is the
non-compressible suffix and is rendered BEFORE the haystack so article tokens
are query-aware at the gate layer).
"""
from typing import Optional

from tokengate.datasets.tasks.longbench import _LongBenchTaskBase, _normalize


class NiahKamradtTask(_LongBenchTaskBase):
    """Kamradt NIAH — retrieve the planted 'best thing to do in {city}' activity.

    Scoring: normalized containment of the gold activity phrase in the
    prediction (the canonical 'did it retrieve the needle' read, binary acc).
    """

    prompt_version = "niah_v1"

    @property
    def name(self) -> str: return "niah_kamradt"

    @property
    def acc_metric_name(self) -> str: return "acc"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        intro = ("Answer the question using only the document below. "
                 "Keep your answer short and direct.")
        if question_line:
            return f"{intro}\n\n{question_line}\n\nDocument:\n{body_display}\n\nAnswer:"
        return f"{intro}\n\nDocument:\n{body_display}\n\nAnswer:"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        gt = _normalize(ground_truth)
        if not gt:
            return 0.0
        return 1.0 if gt in _normalize(prediction) else 0.0


class RulerNiah8Task(_LongBenchTaskBase):
    """RULER NIAH-8 — the 8 niah_* subtasks in one pool (subtask in the row id).

    Scoring: RULER's containment recall — fraction of gold values (split from
    the comma-joined target; values are numbers/uuids, never containing commas)
    present verbatim (case-insensitive) in the prediction.  Single-needle
    subtasks reduce to 0/1.
    """

    prompt_version = "ruler_v1"

    @property
    def name(self) -> str: return "ruler_niah8"

    @property
    def acc_metric_name(self) -> str: return "acc"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        intro = ("Special magic values (numbers and uuids) are hidden within the text "
                 "below. Answer the question about them; report every requested value.")
        if question_line:
            return f"{intro}\n\n{question_line}\n\nText:\n{body_display}\n\nAnswer:"
        return f"{intro}\n\nText:\n{body_display}\n\nAnswer:"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        golds = [g.strip() for g in (ground_truth or "").split(",") if g.strip()]
        if not golds:
            return 0.0
        pl = (prediction or "").lower()
        return sum(1.0 for g in golds if g.lower() in pl) / len(golds)
