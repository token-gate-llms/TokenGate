"""v8 coverage tasks: extractive QA (Natural Questions), synthetic
retrieval (passkey), synthetic counting (passage_count).

Layout follows the design in extractive/retrieval tasks put the query
FIRST (so the gate can select query-relevant tokens); passage_count has no
per-item query (the paragraph pool is the prunable region).  Article encodes the
query as a trailing "\\n\\nQuestion: ..." block (QUESTION_MARKER), peeled off as
non-compressible; format_prompt re-renders it query-first.
"""
import re
from collections import Counter
from typing import Any, Dict, Optional

from tokengate.datasets.tasks.base import Task

QUESTION_MARKER = "\n\nQuestion:"


def _norm_txt(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _token_f1(pred: str, gold: str) -> float:
    p, g = _norm_txt(pred).split(), _norm_txt(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


class _QuestionFirstTask(Task):
    """Extractive/retrieval base: query FIRST, prunable document AFTER."""
    prompt_version = "v0"
    intro = "Answer the question using the document below."
    doc_label = "Document"
    answer_cue = "Answer:"

    @property
    def name(self) -> str:
        raise NotImplementedError

    def get_question(self) -> str:
        return "Answer the question using the document below."

    def _split(self, article_text: str):
        idx = article_text.rfind(QUESTION_MARKER)
        if idx == -1:
            return article_text, ""
        return article_text[:idx], article_text[idx:].strip()

    def get_non_compressible_context(self, article_text: str) -> str:
        idx = article_text.rfind(QUESTION_MARKER)
        return article_text[idx:] if idx != -1 else ""

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body, query_line = self._split(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        if query_line:
            return f"{self.intro}\n\n{query_line}\n\n{self.doc_label}:\n{body_display}\n\n{self.answer_cue}"
        return f"{self.intro}\n\n{self.doc_label}:\n{body_display}\n\n{self.answer_cue}"

    def get_ground_truth(self, sample: Dict[str, Any]) -> str:
        a = sample.get("answer", "")
        return a.strip() if isinstance(a, str) else ""

    def get_sft_target(self, sample: Dict[str, Any]) -> str:
        return self.get_ground_truth(sample)

    def filter_sample(self, sample: Dict[str, Any]) -> bool:
        return bool(self.get_ground_truth(sample))


class NaturalQuestionsQATask(_QuestionFirstTask):
    """Single-document extractive QA over a Wikipedia article (token F1)."""
    prompt_version = "v0"
    intro = "Answer the question using the Wikipedia article below. Answer with a short span from the article."
    doc_label = "Article"

    @property
    def name(self) -> str:
        return "natural_questions_qa"

    @property
    def acc_metric_name(self) -> str:
        return "F1"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        sample = kwargs.get("sample") or {}
        golds = sample.get("answers") or [ground_truth]
        return max(_token_f1(prediction, g) for g in golds)


class MFQAStyleTask(NaturalQuestionsQATask):
    """Diverse-domain single-doc QA twin for held-out multifieldqa_en.
    Same query-first single-doc layout + token-F1 as NQ, but renders the EXACT
    multifieldqa eval template (intro/label/cue) so train==eval, and consumes the
    distractor-padded, de-centered, multi-domain packs (mfqa_<source> datasets)."""
    intro = "Answer the question below based on the document."
    doc_label = "Document"
    answer_cue = "Answer in a short phrase.\nAnswer:"

    @property
    def name(self) -> str:
        return "mfqa_style"


class MultiHopQAStyleTask(NaturalQuestionsQATask):
    """Multi-hop QA twin trained at EVAL length: real gold
    supporting + real distractor passages padded to ~13-16k, gold de-centered.
    Renders the LongBench hotpotqa/2wikimqa/musique template so train==eval."""
    intro = "Answer the multi-hop question using the Wikipedia passages below."
    doc_label = "Passages"
    answer_cue = "Answer in a few words.\nAnswer:"

    @property
    def name(self) -> str:
        return "multihop_style"


class PasskeyRetrievalTask(_QuestionFirstTask):
    """Needle-in-haystack: find a hidden passphrase (substring accuracy)."""
    prompt_version = "v0"
    intro = ("Find the passphrase hidden somewhere in the text and answer the "
             "question. Reply with the passphrase only.")
    doc_label = "Text"

    @property
    def name(self) -> str:
        return "passkey_retrieval"

    @property
    def acc_metric_name(self) -> str:
        return "acc"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        p = _norm_txt(prediction)
        gt = _norm_txt(ground_truth)
        return 1.0 if gt and gt in p else 0.0


class SyntheticPassageCountTask(Task):
    """Count DISTINCT paragraphs in a pool where some are duplicated (integer
    match).  No per-item query — the paragraph pool is the prunable region."""
    prompt_version = "v0"

    @property
    def name(self) -> str:
        return "passage_count_synth"

    @property
    def acc_metric_name(self) -> str:
        return "acc"

    def get_question(self) -> str:
        return "Count how many distinct paragraphs there are."

    def get_pre_article_prompt(self) -> str:
        return ("Some of the following paragraphs are repeated. Count how many "
                "DISTINCT paragraphs there are. Respond with the count as an "
                "integer only.\n\nParagraphs:\n")

    def get_post_article_prompt(self) -> str:
        return "\n\nCount:"

    def format_prompt(self, article_text: str, truncate_len: Optional[int] = None) -> str:
        body = article_text if truncate_len is None else article_text[:truncate_len]
        return f"{self.get_pre_article_prompt()}{body.lstrip(chr(10)) or '.'}{self.get_post_article_prompt()}"

    def get_ground_truth(self, sample: Dict[str, Any]) -> str:
        a = sample.get("answer", "")
        return a.strip() if isinstance(a, str) else ""

    def get_sft_target(self, sample: Dict[str, Any]) -> str:
        return self.get_ground_truth(sample)

    def filter_sample(self, sample: Dict[str, Any]) -> bool:
        return self.get_ground_truth(sample).lstrip("-").isdigit()

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        gt = (ground_truth or "").strip()
        m = re.search(r"-?\d+", prediction or "")
        return 1.0 if (m and m.group(0) == gt) else 0.0
