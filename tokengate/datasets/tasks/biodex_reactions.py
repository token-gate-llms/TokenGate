"""BioDEX-Reactions — the OFFICIAL task (run 16).

Primary metric = the official evaluation: per-sample set precision/recall/F1
with EXACT string match over the predicted vs gold MedDRA reaction terms
(D'Oosterlinck et al., 2023).  Normalization is lowercase+strip only —
punctuation is PRESERVED (MedDRA terms contain hyphens/apostrophes), unlike
the drug-extraction normalizer.  Secondary "fuzzyF1" = the same sets scored
with bidirectional substring containment, for read-across to biodex_drugs.

Prompt (biodexr_v1, repo question-first geometry — the paper's seq2seq
baselines used no instruction; wording is ours, data/target/metric are
official):

    Answer the question using only the biomedical article below. Answer with
    the adverse drug reaction terms only, separated by commas.

    Question: What adverse drug reactions did the patient experience?

    Article:
    {fulltext_processed}

    Answer:

SFT target = the official ", "-joined reactions string verbatim.
"""
import re

from tokengate.datasets.tasks.longbench import _LongBenchTaskBase


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _parse_terms(text: str):
    return {_norm(p) for p in re.split(r"[,;\n]|\d+\.\s*", text) if _norm(p)}


class BiodexReactionsTask(_LongBenchTaskBase):
    """Predict the paper's FAERS adverse-reaction terms (official BioDEX-Reactions)."""

    prompt_version = "biodexr_v1"

    @property
    def name(self) -> str: return "biodex_reactions"

    @property
    def acc_metric_name(self) -> str: return "F1"

    @property
    def secondary_metric_name(self) -> str: return "fuzzyF1"

    def format_prompt(self, article_text: str, truncate_len=None) -> str:
        body, question_line = self._question_suffix(article_text)
        if truncate_len is not None:
            body = body[:truncate_len]
        body_display = body.lstrip("\n") or "."
        intro = ("Answer the question using only the biomedical article below. "
                 "Answer with the adverse drug reaction terms only, separated by commas.")
        if question_line:
            return f"{intro}\n\n{question_line}\n\nArticle:\n{body_display}\n\nAnswer:"
        return f"{intro}\n\nArticle:\n{body_display}\n\nAnswer:"

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        gt = {_norm(x) for x in ground_truth.split(",") if _norm(x)}
        pred = _parse_terms(prediction)
        if not gt:
            return 1.0 if not pred else 0.0
        if not pred:
            return 0.0
        tp = len(pred & gt)
        if tp == 0:
            return 0.0
        prec, rec = tp / len(pred), tp / len(gt)
        return 2 * prec * rec / (prec + rec)

    def evaluate_f1(self, prediction: str, ground_truth: str, **kwargs) -> float:
        gt = {_norm(x) for x in ground_truth.split(",") if _norm(x)}
        pred = _parse_terms(prediction)
        if not gt:
            return 1.0 if not pred else 0.0
        if not pred:
            return 0.0
        tp, matched = 0, set()
        for p in pred:
            for g in gt:
                if (p in g or g in p) and g not in matched:
                    tp += 1
                    matched.add(g)
                    break
        if tp == 0:
            return 0.0
        prec, rec = tp / len(pred), tp / len(gt)
        return 2 * prec * rec / (prec + rec)
