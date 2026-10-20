"""LongBench (THUDM/LongBench) English subsets matching FastKV's eval.

Subsets included (English, non-overlapping with our existing eval/heldout):
  * multifieldqa_en   — single-doc QA over multi-field documents (F1)
  * hotpotqa          — multi-doc Wikipedia QA (F1)
  * 2wikimqa          — multi-doc Wikipedia QA (F1)
  * trec              — 50-class question type (accuracy)
  * samsum            — dialogue summarization (ROUGE-L)
  * passage_count     — synthetic distinct-paragraph counting (accuracy)
  * passage_retrieval_en — synthetic passage-identification (accuracy)

LongBench rows are normalized to our pipeline's sample format:
  sample = {
    "article": context + "\\n\\nQuestion: " + input,
    "answer":  answers[0]   (first reference; QA tasks use SQuAD-style F1
                             so multiple answers can be handled in evaluate)
    "answers": answers     (full list for token-F1 over alternatives)
    "all_classes": optional list (TREC etc.)
  }

This matches the convention used by trivia_qa / qasper_qa / narrative_qa:
the per-sample question is appended to article_text with the literal
"\\n\\nQuestion: " separator, and format_prompt() splits on that.

Notes
-----
* FastKV (arxiv 2502.01068) evaluates on LongBench with these (and Chinese)
  subsets at TSP layer 15, retention 10%; their full-context retention budget
  is "context_max // 10". We expose K-sweep via our usual bench_grouped_eval
  --k-values flag and let the harness decide the budget grid.
* These are eval-only tasks (held-out / OOD). Do not add to v3 pretrain.
"""

import re
import string
from typing import Any, Dict, List, Optional

from tokengate.datasets.tasks.base import Task


# ── shared text-normalisation + token-F1 (SQuAD style) ────────────────────────

_PUNCT_MAP = str.maketrans(string.punctuation, " " * len(string.punctuation))


def _normalize(text: str) -> str:
    """Lower, strip punctuation, drop articles, collapse spaces (SQuAD)."""
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(_PUNCT_MAP)
    return " ".join(text.split())


def _f1(pred: str, gt: str) -> float:
    p = _normalize(pred).split(); g = _normalize(gt).split()
    if not g:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    common = sum(min(p.count(t), g.count(t)) for t in set(p) & set(g))
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def _max_f1(pred: str, gts: List[str]) -> float:
    return max((_f1(pred, g) for g in gts), default=0.0)


def _levenshtein(a: str, b: str) -> int:
    """Character-level Levenshtein distance (two-row DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1,
                          prev[j - 1] + (0 if ca == cb else 1))
        prev = curr
    return prev[len(b)]


def _edit_sim(pred: str, gt: str) -> float:
    """Edit similarity = 1 - Levenshtein/max(len). LongBench code_sim metric.

    LongBench officially uses fuzzywuzzy `fuzz.ratio` on the first predicted
    line; char-level edit similarity over the first line is equivalent up to
    fuzzywuzzy's internal normalization and keeps us dependency-free (matches
    repobench_completion.py / lcc_python_completion.py in this repo).
    """
    pred = (pred or "").strip().split("\n")[0].strip()
    gt = (gt or "").strip().split("\n")[0].strip()
    denom = max(len(pred), len(gt))
    if denom == 0:
        return 1.0
    return 1.0 - _levenshtein(pred, gt) / denom


def _rougel(pred: str, gt: str) -> float:
    """Cheap ROUGE-L (token LCS / harmonic mean). Works under no rouge_score dep."""
    p = _normalize(pred).split(); g = _normalize(gt).split()
    if not p or not g:
        return 0.0
    m, n = len(p), len(g)
    # 1D rolling table for memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if p[i - 1] == g[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    pr = lcs / m; rc = lcs / n
    return 2 * pr * rc / (pr + rc)


# ── base class with per-sample question encoded into article ──────────────────

class _LongBenchTaskBase(Task):
    """Shared scaffolding for LongBench subsets.

    Subclasses must override:
      * name                — short id used across harness
      * subset              — exact LongBench config name
      * acc_metric_name     — "F1" / "ROUGE-L" / "acc"
      * format_prompt(article_text)
      * evaluate(prediction, ground_truth, **kwargs)

    They inherit:
      * get_question() — fixed instruction (the *per-sample* question lives in
        article_text after the literal "\\n\\nQuestion: " marker; this base
        method returns the generic instruction).
      * get_ground_truth(sample) — first answer string (subclasses may extend).
      * filter_sample(sample) — has-non-empty-answer guard.
      * get_non_compressible_context(article_text) — the trailing
        "\\n\\nQuestion: ..." block.
    """

    prompt_version = "longbench_v1"
    subset: str = ""

    @property
    def name(self) -> str:  # subclasses MUST override; safe default for tooling.
        return "longbench"

    @property
    def acc_metric_name(self) -> str:
        return "F1"

    @property
    def secondary_metric_name(self) -> str:
        return "F1"

    def get_question(self) -> str:
        return "Answer using the context below."

    def get_ground_truth(self, sample: Dict[str, Any]) -> str:
        ans = sample.get("answer", "")
        if isinstance(ans, list):
            return ans[0] if ans else ""
        return ans.strip() if isinstance(ans, str) else ""

    def get_sft_target(self, sample: Dict[str, Any]) -> str:
        return self.get_ground_truth(sample)

    def filter_sample(self, sample: Dict[str, Any]) -> bool:
        return bool(self.get_ground_truth(sample))

    def _question_suffix(self, article_text: str):
        idx = article_text.rfind("\n\nQuestion:")
        if idx == -1:
            return article_text, ""
        return article_text[:idx], article_text[idx:].strip()

    def get_non_compressible_context(self, article_text: str) -> str:
        idx = article_text.rfind("\n\nQuestion:")
        return article_text[idx:] if idx != -1 else ""

    def _clean_query_cue(self, question_line: str, cue: str) -> str:
        """Strip the trailing cue (e.g. 'Type:') and a doubled leading 'Question:'
        from the query block so format_prompt's own cue isn't repeated (a LongBench
        render quirk: the peeled query already carries the cue).  Query TEXT is
        unchanged — this only removes the redundant scaffolding tokens."""
        q = question_line.strip()
        if q.endswith(cue):
            q = q[: -len(cue)].rstrip()
        q = re.sub(r"^Question:\s*(Question:\s*)", r"\1", q)  # 'Question: Question:' -> 'Question:'
        return q


# ── concrete subsets ──────────────────────────────────────────────────────────

class LongBenchMultifieldQAEnTask(_LongBenchTaskBase):
    """LongBench multifieldqa_en — single-document QA, SQuAD-style token F1."""
    subset = "multifieldqa_en"

    @property
    def name(self) -> str: return "longbench_multifieldqa_en"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        return (
            "Answer the question below based on the document.\n\n"
            f"{question_line}\n\n" if question_line else
            "Answer the question below based on the document.\n\n"
        ) + (
            "Document:\n"
            f"{body_display}"
            "\n\nAnswer in a short phrase.\nAnswer:"
        )

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        sample = kwargs.get("sample") or {}
        gts = sample.get("answers") or [ground_truth]
        return _max_f1(prediction, gts)


class LongBenchHotpotQATask(_LongBenchTaskBase):
    """LongBench hotpotqa — multi-doc Wikipedia QA (F1)."""
    subset = "hotpotqa"

    @property
    def name(self) -> str: return "longbench_hotpotqa"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        intro = "Answer the multi-hop question using the Wikipedia passages below."
        if question_line:
            return f"{intro}\n\n{question_line}\n\nPassages:\n{body_display}\n\nAnswer in a few words.\nAnswer:"
        return f"{intro}\n\nPassages:\n{body_display}\n\nAnswer in a few words.\nAnswer:"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        sample = kwargs.get("sample") or {}
        gts = sample.get("answers") or [ground_truth]
        return _max_f1(prediction, gts)


class LongBench2WikiMQATask(_LongBenchTaskBase):
    """LongBench 2wikimqa — multi-doc Wikipedia QA (F1)."""
    subset = "2wikimqa"

    @property
    def name(self) -> str: return "longbench_2wikimqa"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        intro = "Answer the question using the Wikipedia passages below."
        if question_line:
            return f"{intro}\n\n{question_line}\n\nPassages:\n{body_display}\n\nAnswer in a few words.\nAnswer:"
        return f"{intro}\n\nPassages:\n{body_display}\n\nAnswer in a few words.\nAnswer:"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        sample = kwargs.get("sample") or {}
        gts = sample.get("answers") or [ground_truth]
        return _max_f1(prediction, gts)


class LongBenchTRECTask(_LongBenchTaskBase):
    """LongBench trec — few-shot 50-class question-type classification (acc)."""
    subset = "trec"

    @property
    def name(self) -> str: return "longbench_trec"

    @property
    def acc_metric_name(self) -> str: return "acc"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        intro = "Classify the question's type. Use the examples as a key. Reply with the type only."
        if question_line:
            ql = self._clean_query_cue(question_line, "Type:")
            # QUERY-FIRST (query before the prunable demos), matching the extractive
            # LongBench tasks + the few-shot training packs (qt_squad/trec_style) +
            # the v10 decision.  Under causal attention this lets the gate
            # score demos with the query visible → query-conditioned demo selection.
            # (Was query-last, which left every trec-trained policy inert at eval.)
            return f"{intro}\n\n{ql}\n\nExamples:\n{body_display}\n\nType:"
        return f"{intro}\n\nExamples:\n{body_display}\n\nType:"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        p = (prediction or "").lower().strip()
        gt = (ground_truth or "").lower().strip()
        if not gt: return 0.0
        # Substring match — LongBench convention for classification.
        return 1.0 if gt in p else 0.0


class LongBenchSAMSumTask(_LongBenchTaskBase):
    """LongBench samsum — dialogue summarization (ROUGE-L)."""
    subset = "samsum"

    @property
    def name(self) -> str: return "longbench_samsum"

    @property
    def acc_metric_name(self) -> str: return "ROUGE-L"

    @property
    def secondary_metric_name(self) -> str: return "ROUGE-L"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        intro = "Summarize the final dialogue in 1-2 sentences. Use the examples as a style guide."
        if question_line:
            ql = self._clean_query_cue(question_line, "Summary:")
            return f"{intro}\n\nExamples:\n{body_display}\n\n{ql}\n\nSummary:"
        return f"{intro}\n\nExamples:\n{body_display}\n\nSummary:"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        return _rougel(prediction, ground_truth)


class LongBenchPassageCountTask(_LongBenchTaskBase):
    """LongBench passage_count — count distinct paragraphs (acc)."""
    subset = "passage_count"

    @property
    def name(self) -> str: return "longbench_passage_count"

    @property
    def acc_metric_name(self) -> str: return "acc"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        intro = "Some of the following paragraphs are repeated. Count how many DISTINCT paragraphs there are. Respond with the count as an integer only."
        if question_line:
            return f"{intro}\n\n{question_line}\n\nParagraphs:\n{body_display}\n\nCount:"
        return f"{intro}\n\nParagraphs:\n{body_display}\n\nCount:"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        # Exact-integer match on first integer in prediction vs ground truth.
        gt = (ground_truth or "").strip()
        m = re.search(r"-?\d+", prediction or "")
        return 1.0 if (m and m.group(0) == gt) else 0.0


class LongBenchPassageRetrievalEnTask(_LongBenchTaskBase):
    """LongBench passage_retrieval_en — identify the source paragraph (acc)."""
    subset = "passage_retrieval_en"

    @property
    def name(self) -> str: return "longbench_passage_retrieval_en"

    @property
    def acc_metric_name(self) -> str: return "acc"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        intro = "Identify which paragraph the abstract is from. Reply with the paragraph number only (e.g. 'Paragraph 3')."
        if question_line:
            return f"{intro}\n\n{question_line}\n\n{body_display}\n\nAnswer:"
        return f"{intro}\n\n{body_display}\n\nAnswer:"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        # LongBench ground_truth is e.g. "Paragraph 3" — match the integer.
        def _num(s):
            m = re.search(r"\d+", s or "")
            return m.group(0) if m else None
        return 1.0 if _num(prediction) == _num(ground_truth) else 0.0


class PassageRetrievalSynthTask(LongBenchPassageRetrievalEnTask):
    """In-distribution TWIN of passage_retrieval_en — LLM-generated abstractive summaries
    over 30 Wikipedia paragraphs (synthetic passage-retrieval pool).

    Inherits the IDENTICAL prompt + integer-match scoring from the held-out task, but carries a
    DISTINCT task id so the twin's own val logs as its own **in-distribution** eval group
    ("synthetic") alongside the held-out real "synthetic_heldout" group — letting us compare
    in-dist learning vs held-out transfer."""

    @property
    def name(self) -> str: return "passage_retrieval_synth"


# ── additional subsets (added for the full LongBench-matched eval) ─────────────
# These complete the 16-subset English LongBench eval. The 13 with a clean
# external original are also trained on;
# the eval tasks here are what the matched train data is evaluated against.


class _LongBenchF1QATask(_LongBenchTaskBase):
    """Shared QA scaffold: short-phrase answer, SQuAD token-F1 over all refs."""

    intro = "Answer the question below based on the context."

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        if question_line:
            return f"{self.intro}\n\n{question_line}\n\nContext:\n{body_display}\n\nAnswer in a short phrase.\nAnswer:"
        return f"{self.intro}\n\nContext:\n{body_display}\n\nAnswer in a short phrase.\nAnswer:"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        sample = kwargs.get("sample") or {}
        gts = sample.get("answers") or [ground_truth]
        return _max_f1(prediction, gts)


class LongBenchNarrativeQATask(_LongBenchF1QATask):
    """LongBench narrativeqa — QA over full stories/scripts (F1)."""
    subset = "narrativeqa"
    intro = "Answer the question below based on the story."

    @property
    def name(self) -> str: return "longbench_narrativeqa"


class LongBenchQasperTask(_LongBenchF1QATask):
    """LongBench qasper — QA over NLP papers (F1)."""
    subset = "qasper"
    intro = ("Answer the question below based on the scientific paper. "
             "If the question is not answerable from the paper, reply 'unanswerable'.")

    @property
    def name(self) -> str: return "longbench_qasper"


class LongBenchMusiqueTask(_LongBenchF1QATask):
    """LongBench musique — multi-hop QA (F1)."""
    subset = "musique"
    intro = "Answer the multi-hop question using the passages below."

    @property
    def name(self) -> str: return "longbench_musique"


class LongBenchTriviaQATask(_LongBenchF1QATask):
    """LongBench triviaqa — few-shot reading-comprehension QA (F1)."""
    subset = "triviaqa"
    intro = "Answer the final question. Use the examples as a format guide."

    @property
    def name(self) -> str: return "longbench_triviaqa"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        if question_line:
            ql = self._clean_query_cue(question_line, "Answer:")
            return f"{self.intro}\n\nExamples:\n{body_display}\n\n{ql}\n\nAnswer:"
        return f"{self.intro}\n\nExamples:\n{body_display}\n\nAnswer:"


class _LongBenchSummTask(_LongBenchTaskBase):
    """Shared summarization scaffold (ROUGE-L)."""

    intro = "Write a summary of the document below."

    @property
    def acc_metric_name(self) -> str: return "ROUGE-L"

    @property
    def secondary_metric_name(self) -> str: return "ROUGE-L"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        if question_line:
            return f"{self.intro}\n\n{question_line}\n\nDocument:\n{body_display}\n\nSummary:"
        return f"{self.intro}\n\nDocument:\n{body_display}\n\nSummary:"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        return _rougel(prediction, ground_truth)


class LongBenchGovReportTask(_LongBenchSummTask):
    """LongBench gov_report — long government-report summarization (ROUGE-L)."""
    subset = "gov_report"
    intro = "Write a one-page summary of the government report below."

    @property
    def name(self) -> str: return "longbench_gov_report"


class LongBenchQMSumTask(_LongBenchSummTask):
    """LongBench qmsum — query-based meeting summarization (ROUGE-L)."""
    subset = "qmsum"
    intro = "Answer the query by summarizing the relevant parts of the meeting transcript below."

    @property
    def name(self) -> str: return "longbench_qmsum"


class LongBenchMultiNewsTask(_LongBenchSummTask):
    """LongBench multi_news — multi-document news summarization (ROUGE-L)."""
    subset = "multi_news"
    intro = "Write a summary of the news articles below."

    @property
    def name(self) -> str: return "longbench_multi_news"


class _LongBenchCodeTask(_LongBenchTaskBase):
    """Shared next-line code-completion scaffold (edit similarity)."""

    intro = ("Output only the single next line of code that continues the file "
             "below. Do not explain or use markdown; output exactly one line.")

    @property
    def acc_metric_name(self) -> str: return "ES"

    @property
    def secondary_metric_name(self) -> str: return "ES"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        if question_line:
            return f"{self.intro}\n\n{body_display}\n\n{question_line}\n"
        return f"{self.intro}\n\n{body_display}\n"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        return _edit_sim(prediction, ground_truth)


class LongBenchLCCTask(_LongBenchCodeTask):
    """LongBench lcc — long-code next-line completion (edit similarity)."""
    subset = "lcc"

    @property
    def name(self) -> str: return "longbench_lcc"


class LongBenchRepoBenchPTask(_LongBenchCodeTask):
    """LongBench repobench-p — cross-file next-line completion (edit similarity)."""
    subset = "repobench-p"

    @property
    def name(self) -> str: return "longbench_repobench_p"
