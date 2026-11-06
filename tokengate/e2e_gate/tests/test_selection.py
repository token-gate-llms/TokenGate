"""Checkpoint selection must not let extra SWA evaluations consume patience."""

import pytest

from tokengate.e2e_gate.train.selection import CheckpointSelection


def test_raw_and_swa_share_one_patience_cycle():
    selection = CheckpointSelection(patience=2, warmup_steps=1000)
    assert selection.consider(600, "raw", 0.9, 0.2)
    assert selection.consider(600, "swa", 0.9, 0.1)
    assert not selection.finish_cycle(600)
    selection.consider(1200, "raw", 0.8, 0.05)
    selection.consider(1200, "swa", 0.9, 0.1)
    assert not selection.finish_cycle(1200)
    assert selection.best["step"] == 600
    restored = CheckpointSelection(**selection.state_dict())
    restored.consider(1800, "raw", 0.9, 0.1)
    assert restored.finish_cycle(1800)


def test_accuracy_precedes_loss_and_loss_breaks_ties():
    selection = CheckpointSelection()
    selection.consider(600, "raw", 0.9, 0.2)
    assert not selection.consider(600, "swa", 0.8, 0.1)
    assert selection.consider(1200, "raw", 0.9, 0.1)
    assert selection.consider(1200, "swa", 0.92, 0.3)
    assert not selection.consider(1800, "raw", 0.92, 0.3)


@pytest.mark.parametrize("accuracy,loss", [(float("nan"), 1), (1, float("inf"))])
def test_nonfinite_metrics_fail(accuracy, loss):
    with pytest.raises(ValueError, match="nonfinite"):
        CheckpointSelection().consider(600, "raw", accuracy, loss)
