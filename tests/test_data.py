"""The Sleep-EDF reading code, tested on files written in Sleep-EDF's layout."""
import numpy as np
import pytest

from sleepnet.data import (IGNORE, labels_from_annotations, list_recordings, load_recording,
                           trim_wake, write_synthetic_edf_pair)
from sleepnet.metrics import summarize


def test_labels_merge_n3_and_ignore_unknown_and_movement():
    notes = [(0, 60, "Sleep stage W"), (60, 30, "Sleep stage 1"), (90, 30, "Sleep stage 2"),
             (120, 30, "Sleep stage 3"), (150, 30, "Sleep stage 4"), (180, 30, "Sleep stage R"),
             (210, 30, "Sleep stage ?"), (240, 30, "Movement time")]
    labels = labels_from_annotations(notes, 9)
    assert labels.tolist() == [0, 0, 1, 2, 3, 3, 4, IGNORE, IGNORE]


def test_labels_clip_annotations_that_run_past_the_end():
    assert labels_from_annotations([(0, 300, "Sleep stage 2")], 4).tolist() == [2, 2, 2, 2]


def test_trim_wake_keeps_a_margin_around_sleep():
    labels = np.array([0] * 10 + [2] * 5 + [0] * 10, dtype=np.int8)
    assert trim_wake(labels, margin=3) == slice(7, 18)
    assert trim_wake(np.zeros(20, dtype=np.int8), margin=3) == slice(0, 0)


def test_psg_and_hypnogram_are_paired_and_nights_share_a_subject(tmp_path):
    for prefix, seed in (("SC4001E", 1), ("SC4002E", 2), ("SC4011E", 3)):
        write_synthetic_edf_pair(tmp_path / "sleep-cassette", prefix, 40, seed)
    found = list_recordings(tmp_path)
    assert [r["name"] for r in found] == ["SC4001", "SC4002", "SC4011"]
    assert [r["subject"] for r in found] == ["SC400", "SC400", "SC401"]
    assert all(r["hypnogram"].name.endswith("Hypnogram.edf") for r in found)


def test_a_psg_without_a_hypnogram_is_skipped_with_a_warning(tmp_path):
    psg, hyp, _ = write_synthetic_edf_pair(tmp_path, "SC4001E", 40, 1)
    hyp.unlink()
    with pytest.warns(UserWarning, match="no hypnogram"):
        assert list_recordings(tmp_path) == []


def test_load_recording_reads_signal_and_labels_correctly(tmp_path):
    psg, hyp, stages = write_synthetic_edf_pair(tmp_path, "SC4001E", 80, 7)
    epochs, labels = load_recording(psg, hyp, margin=5)
    window = trim_wake(stages.astype(np.int8), 5)
    assert epochs.shape == (len(stages[window]), 3000) and epochs.dtype == np.float32
    assert np.array_equal(labels, stages[window])
    assert 5 < epochs.std() < 100  # volts were converted to microvolts


def test_metrics_on_a_known_example():
    names = ("W", "N1", "N2", "N3", "REM")
    y_true = [0, 0, 1, 2, 2, 2, 3, 4]
    y_pred = [0, 1, 1, 2, 2, 0, 3, 4]
    scores = summarize(y_true, y_pred, names)
    assert scores["accuracy"] == 0.75
    assert scores["per_stage"]["W"]["recall"] == 0.5 and scores["per_stage"]["W"]["precision"] == 0.5
    assert scores["per_stage"]["N2"]["recall"] == pytest.approx(0.6667, abs=1e-4)
    assert 0.65 < scores["kappa"] < 0.7
    assert np.trace(scores["confusion_matrix"]) == 6
