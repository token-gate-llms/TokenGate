"""Task registry – import all concrete tasks here."""

from typing import List, Optional
from tokengate.datasets.tasks.base import Task
from tokengate.datasets.tasks.v8_coverage_tasks import (
    NaturalQuestionsQATask, MFQAStyleTask, MultiHopQAStyleTask, PasskeyRetrievalTask, SyntheticPassageCountTask,
)
from tokengate.datasets.tasks.longbench import (
    LongBenchMultifieldQAEnTask,
    LongBenchHotpotQATask,
    LongBench2WikiMQATask,
    LongBenchTRECTask,
    LongBenchSAMSumTask,
    LongBenchPassageCountTask,
    LongBenchPassageRetrievalEnTask,
    PassageRetrievalSynthTask,
    LongBenchNarrativeQATask,
    LongBenchQasperTask,
    LongBenchMusiqueTask,
    LongBenchTriviaQATask,
    LongBenchGovReportTask,
    LongBenchQMSumTask,
    LongBenchMultiNewsTask,
    LongBenchLCCTask,
    LongBenchRepoBenchPTask,
)
from tokengate.datasets.tasks.niah import NiahKamradtTask, RulerNiah8Task
from tokengate.datasets.tasks.biodex_reactions import BiodexReactionsTask

# Canonical registry ----------------------------------------------------------
TASK_REGISTRY = {
    # Trained release targets.
    "biodex_reactions":                 BiodexReactionsTask,
    "niah_kamradt":                     NiahKamradtTask,
    "niah_kamradt_32k":                 NiahKamradtTask,
    "niah_kamradt_64k":                 NiahKamradtTask,
    "niah_kamradt_128k":                NiahKamradtTask,
    "ruler_niah8":                      RulerNiah8Task,
    # 2WikiMQA full-Wikipedia multi-hop twin (the iso_2wiki trained pair).
    "natural_questions_qa":             NaturalQuestionsQATask,
    "mfqa_style":                       MFQAStyleTask,
    "multihop_style":                   MultiHopQAStyleTask,
    "passkey_retrieval":                PasskeyRetrievalTask,
    "passage_count_synth":              SyntheticPassageCountTask,
    # LongBench: the multinews/repobench/2wikimqa trained pairs
    # (longbench-datasets) + the OOD eval set (matches FastKV's evaluation set).
    "longbench_multifieldqa_en":        LongBenchMultifieldQAEnTask,
    "longbench_hotpotqa":               LongBenchHotpotQATask,
    "longbench_2wikimqa":               LongBench2WikiMQATask,
    "longbench_trec":                   LongBenchTRECTask,
    "longbench_samsum":                 LongBenchSAMSumTask,
    "longbench_passage_count":          LongBenchPassageCountTask,
    "longbench_passage_retrieval_en":   LongBenchPassageRetrievalEnTask,
    "passage_retrieval_synth":          PassageRetrievalSynthTask,
    "longbench_narrativeqa":            LongBenchNarrativeQATask,
    "longbench_qasper":                 LongBenchQasperTask,
    "longbench_musique":                LongBenchMusiqueTask,
    "longbench_triviaqa":               LongBenchTriviaQATask,
    "longbench_gov_report":             LongBenchGovReportTask,
    "longbench_qmsum":                  LongBenchQMSumTask,
    "longbench_multi_news":             LongBenchMultiNewsTask,
    "longbench_lcc":                    LongBenchLCCTask,
    "longbench_repobench_p":            LongBenchRepoBenchPTask,
}


# Task → coarse category, used to group eval into a few val sets (one per
# category) rather than one curve per dataset.  PRETRAIN_VAL_CATEGORIES are the
# three tracked families; everything else is reported but is NOT one of those
# three val sets.  This is the single source of truth — every entry in
# TASK_REGISTRY must have a category here (validated at import below) so adding a
# new task fails loudly until its category is declared.
PRETRAIN_VAL_CATEGORIES = ("qa", "code_completion", "summarization")

TASK_CATEGORY = {
    "biodex_reactions":               "extraction",
    "niah_kamradt":                   "synthetic",
    "niah_kamradt_32k":               "synthetic",
    "niah_kamradt_64k":               "synthetic",
    "niah_kamradt_128k":              "synthetic",
    "ruler_niah8":                    "synthetic",
    "natural_questions_qa":           "qa",
    "mfqa_style":                     "qa",
    "multihop_style":                 "qa",
    "passkey_retrieval":              "synthetic",
    "passage_count_synth":            "synthetic",
    "longbench_multifieldqa_en":      "qa",
    "longbench_hotpotqa":             "qa",
    "longbench_2wikimqa":             "qa",
    "longbench_trec":                 "classification",
    "longbench_samsum":               "summarization",
    "longbench_passage_count":        "synthetic",
    "longbench_passage_retrieval_en": "synthetic",
    "passage_retrieval_synth":        "synthetic",
    "longbench_narrativeqa":          "qa",
    "longbench_qasper":               "qa",
    "longbench_musique":              "qa",
    "longbench_triviaqa":             "qa",
    "longbench_gov_report":           "summarization",
    "longbench_qmsum":                "summarization",
    "longbench_multi_news":           "summarization",
    "longbench_lcc":                  "code_completion",
    "longbench_repobench_p":          "code_completion",
}

# Fail loudly if a registered task has no declared category (or vice-versa).
_missing_category = set(TASK_REGISTRY) - set(TASK_CATEGORY)
if _missing_category:
    raise RuntimeError(
        f"TASK_CATEGORY is missing entries for registered tasks: "
        f"{sorted(_missing_category)}.  Declare a category (one of "
        f"{sorted(set(TASK_CATEGORY.values()))} or a new family) in "
        f"tokengate/datasets/tasks/__init__.py."
    )
_unknown_category = set(TASK_CATEGORY) - set(TASK_REGISTRY)
if _unknown_category:
    raise RuntimeError(
        f"TASK_CATEGORY has entries for unregistered tasks: "
        f"{sorted(_unknown_category)}.  Remove them or register the task."
    )


def get_task_category(name: str) -> str:
    """Return the coarse category for a task name, raising if unknown."""
    if name not in TASK_CATEGORY:
        raise ValueError(
            f"No category declared for task '{name}'. "
            f"Add it to TASK_CATEGORY in tokengate/datasets/tasks/__init__.py."
        )
    return TASK_CATEGORY[name]


def get_task(name: str, **kwargs) -> Task:
    """Instantiate a task by name."""
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name](**kwargs)


def get_all_tasks(exclude_llm_judge: bool = False) -> List[Task]:
    """Return instances of every registered task.

    Set *exclude_llm_judge=True* to skip tasks that require an OpenAI API key.
    """
    tasks = []
    for name, cls in TASK_REGISTRY.items():
        t = cls()
        if exclude_llm_judge and t.requires_api:
            continue
        tasks.append(t)
    return tasks
