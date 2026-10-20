"""mfqa_2wiki_wiki — 2WikiMultihopQA multi-hop reconstructed from FULL Wikipedia
passages. Eval-faithful: `Passage N:\\n<title>\\n<body>` with the
question's own ~10 gold+distractor titles as full Wikipedia articles, gold
de-centered. LOW priority: 2wikimqa already scores ~0.5 because its eval is short
(~6k tok → K=2000 keeps 1:3), so expect small gains — built for consistency.

PRE-REQ: build_wiki_dict.py with SOURCES including `2wiki` wrote wiki_dict.pkl +
wiki_questions.pkl (set WIKI_SP). Decontam in build_wiki_multihop:
question-level exact vs LongBench 2wikimqa + gold-passage 13-gram containment.

    WIKI_SP=<dir> python datasets/mfqa_2wiki_wiki/multihop_style/prepare.py
"""
import os

from tokengate.datasets.prepare import prepare_dataset
from tokengate.datasets.wiki_multihop_reconstruct import build_wiki_multihop

DATASET = "mfqa_2wiki_wiki"
TASK = "multihop_style"
SP = os.environ["WIKI_SP"]
WIKI, Q = f"{SP}/wiki_dict.pkl", f"{SP}/wiki_questions.pkl"
TRAIN = int(os.environ.get("MFQA_2WIKI_WIKI_TRAIN", "20000"))


def load_train():
    return build_wiki_multihop("2wiki", "train", 0, TRAIN, WIKI, Q)


def load_val():
    return build_wiki_multihop("2wiki", "val", 1, 200, WIKI, Q)


if __name__ == "__main__":
    prepare_dataset(dataset_name=DATASET, task_name=TASK,
                    load_train_fn=load_train, load_val_fn=load_val, run_sanity=False)
