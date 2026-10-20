"""Abstract base class for evaluation tasks."""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


# ── answer extraction ──────────────────────────────────────────────────────────

_FINAL_ANSWER_RE = re.compile(
    r"FINAL\s*ANSWER\s*:\s*(.+)",
    re.IGNORECASE,
)


def extract_final_answer(text: str) -> Optional[str]:
    """Extract the content after ``FINAL ANSWER:`` from model output.

    Returns None if the marker is not found.
    """
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


class Task(ABC):
    """A single evaluation task over dataset articles (BioDEX, QASPER, etc.).

    To add a new task, subclass this and register it in ``tasks/__init__.py``.
    """

    # Subclasses MUST set this to a non-empty string (e.g., "v0").  Bump
    # whenever format_prompt / get_non_compressible_context changes so any
    # cached tokenization built against the prior prompt fails loudly at
    # load time rather than silently training on stale tokens.
    prompt_version: Optional[str] = None

    def get_prompt_version(self) -> str:
        """Return this task's prompt_version, raising loudly if unset."""
        pv = type(self).prompt_version
        if not pv or not isinstance(pv, str):
            raise RuntimeError(
                f"Task class {type(self).__name__} must set a class-level "
                "`prompt_version` string (e.g., `prompt_version = \"v0\"`). "
                "Bump it whenever format_prompt or get_non_compressible_context "
                "changes so cached tokenizations fail loudly instead of silently "
                "training on stale tokens."
            )
        return pv

    # ── identity ──────────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique, snake_case identifier for this task."""
        ...

    # ── prompting ─────────────────────────────────────────────────────────────
    @abstractmethod
    def get_question(self) -> str:
        """Return the task-specific question (without the article)."""
        ...

    def get_pre_article_prompt(self) -> str:
        """Text placed *before* the article in the LLM prompt."""
        return (
            "Read the following medical article and answer the question.\n\n"
            "Article:\n"
        )

    def get_post_article_prompt(self) -> str:
        """Text placed *after* the article (includes the question)."""
        return f"\n\nQuestion: {self.get_question()}\nAnswer:"

    def format_prompt(self, article_text: str, truncate_len: int = None) -> str:
        """Build the full prompt (used by text-based compressors)."""
        return (
            f"{self.get_pre_article_prompt()}"
            f"{article_text[:truncate_len]}"
            f"{self.get_post_article_prompt()}"
        )

    # ── ground truth & evaluation ─────────────────────────────────────────────
    @abstractmethod
    def get_ground_truth(self, sample: Dict[str, Any]) -> str:
        """Extract the ground-truth answer from a BioDEX sample."""
        ...

    @abstractmethod
    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """Score a single prediction against the ground truth.

        Parameters
        ----------
        prediction : str
        ground_truth : str
        sample : dict, optional
            The original dataset sample (passed via kwargs).  Tasks that
            need extra context (e.g. the question) can read it from here.

        Returns a float in [0, 1].
        """
        ...

    @abstractmethod
    def filter_sample(self, sample: Dict[str, Any]) -> bool:
        """Return True if this sample has valid ground truth for this task."""
        ...

    def evaluate_f1(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """Token-level F1 score (always, regardless of API availability).

        Most tasks already use F1 or exact-match in ``evaluate``, so the
        default delegates there.  Override in tasks that use a more expensive
        evaluator (e.g. LLM judge) so that F1 can always be logged alongside
        the primary metric.
        """
        return self.evaluate(prediction, ground_truth, **kwargs)

    def get_sft_target(self, sample: Dict[str, Any]) -> str:
        """Return the target string used for SFT training loss.

        Defaults to get_ground_truth.  Override when the ideal training target
        differs from the evaluation ground truth — e.g. when the prompt instructs
        a specific output format but evaluation uses a substring check on just
        the answer keyword.
        """
        return self.get_ground_truth(sample)

    # ── optional hooks ────────────────────────────────────────────────────────
    def prepare(self, samples, config=None):
        """Optional pre-computation step (e.g. oracle answers)."""
        pass

    def get_non_compressible_context(self, article_text: str) -> str:
        """Return the non-compressible trailing portion of article_text.

        Used by the layer-pruning compressor to exclude task-specific content
        (e.g. MCQ options, questions) from the scored/prunable article region.

        The compressor finds the article boundary by comparing
        ``format_prompt(article_text)`` against a baseline prompt.  By default
        the baseline is ``format_prompt("")`` — so the entire article_text is
        treated as the compressible context.

        When this method returns a non-empty string *suffix* of article_text,
        the compressor instead uses ``format_prompt(suffix)`` as the baseline.
        The common suffix of the two prompts then includes the suffix tokens,
        moving them outside the scored article region.

        Override in tasks where article_text bundles both compressible content
        (e.g. patient documents) and non-compressible content (e.g. MCQ
        question + options) into a single string.
        """
        return ""

    @property
    def acc_metric_name(self) -> str:
        """Short tag for the primary accuracy metric (used in W&B key names).

        Defaults to ``"F1"`` (appropriate for token-overlap tasks like BioDEX).
        Override in tasks where ``evaluate()`` returns something other than F1
        (e.g. ``"acc"`` for exact-match / substring tasks like LongHealth).
        The training loop uses this to name ``eval/accuracy_{tag}_K{k}`` so that
        per-K curves and flat baseline lines share the same W&B chart.
        """
        return "F1"

    @property
    def secondary_metric_name(self) -> str:
        """Short tag for the secondary metric returned by ``evaluate_f1()``.

        Defaults to ``"F1"``.  Override when ``evaluate_f1()`` returns something
        other than token-F1 (e.g. ``"EM"`` for exact-match in RepoBench).
        Used to label W&B keys and terminal prints for the secondary metric.
        """
        return "F1"

    @property
    def requires_api(self) -> bool:
        """Return True if the task needs an external API (e.g. OpenAI)."""
        return False

    @property
    def max_new_tokens(self) -> int:
        """Per-task generation cap used by the benchmark/grouped-eval harnesses.

        Tasks with **constrained-format answers** (single token / single letter /
        short phrase) should override this to a small value (e.g. ``1`` for
        binary classification, ``8`` for short-phrase QA, ``32`` for spans)
        so that:
          (a) per-baseline timing reflects realistic deployment cost rather than
              occasional runaway 1024-token rambles,
          (b) accuracy isn't polluted by tail samples that hit the 1024-token cap
              and emit nonsense.

        Default ``1024`` is the legacy "open-ended generation" value (matches
        what training used for free-form QA / summarization). Override per-task.
        """
        return 1024

    @property
    def use_chat_template(self) -> bool:
        """If False, tokenize the prompt as raw text and skip the model's
        chat template (no user/assistant wrapping, no `<think>` priming).

        Defaults to True (chat-style Q&A tasks). Override to False for tasks
        whose prompt is a literal continuation that the model should complete
        verbatim — e.g. next-line code completion on RepoBench when the prompt
        contains no explicit instruction.
        """
        return True
