"""Preprocessing is shared by training and serving, so these protect both."""
import numpy as np
import pytest

from sleepnet.preprocessing import (CLIP_UV, EPOCH_SAMPLES, normalize, split_epochs,
                                    to_epochs)


def test_whole_epochs_only_and_remainder_is_dropped():
    signal = np.random.default_rng(0).normal(0, 20, 100 * 95)  # 95 s -> 3 epochs + 5 s
    assert to_epochs(signal, 100).shape == (3, EPOCH_SAMPLES)


@pytest.mark.parametrize("rate", [50, 125, 128, 200, 256, 500])
def test_resampling_keeps_the_epoch_length(rate):
    signal = np.random.default_rng(1).normal(0, 20, rate * 65)  # 65 s -> 2 epochs
    assert to_epochs(signal, rate).shape == (2, EPOCH_SAMPLES)


def test_a_slow_wave_survives_and_mains_noise_is_removed():
    t = np.arange(100 * 30) / 100
    slow = 50 * np.sin(2 * np.pi * 1.0 * t)      # 1 Hz delta wave: must be kept
    hum = 50 * np.sin(2 * np.pi * 48.0 * t)      # above the 35 Hz band: must be removed
    kept = to_epochs(slow, 100)[0]
    removed = to_epochs(hum, 100)[0]
    assert kept.std() > 0.9 * slow.std()
    assert removed.std() < 0.05 * hum.std()


@pytest.mark.parametrize("bad, message", [
    (np.zeros((2, 3000)), "1-D"),
    (np.array([1.0, np.nan] + [0.0] * 3000), "NaN"),
    (np.zeros(100 * 10), "at least 30 s"),
])
def test_bad_signals_are_rejected_with_a_reason(bad, message):
    with pytest.raises(ValueError, match=message):
        to_epochs(bad, 100)


def test_normalize_removes_offset_clips_artefacts_and_is_per_epoch():
    epochs = np.zeros((2, EPOCH_SAMPLES), dtype=np.float32)
    epochs[0] += 1000.0               # a big DC offset
    epochs[1, 0] = 10 * CLIP_UV       # a huge spike
    out = normalize(epochs, scale=25.0)
    assert np.allclose(out[0], 0.0)                   # offset gone
    assert out[1].max() <= CLIP_UV / 25.0 + 1e-6      # spike clipped
    # one epoch alone is normalised exactly like the same epoch inside a night
    assert np.array_equal(normalize(epochs[1:], 25.0)[0], out[1])


def test_split_epochs_shape():
    assert split_epochs(np.zeros(3 * EPOCH_SAMPLES + 17)).shape == (3, EPOCH_SAMPLES)
