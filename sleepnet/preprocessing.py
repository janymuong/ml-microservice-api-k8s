"""Turn a raw single-channel EEG signal into model-ready 30-second epochs.

Training and serving both call these functions, so the network sees identically
prepared signals in both places (no train/serve skew).
"""
from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

SAMPLE_RATE = 100                              # Hz: the rate the model works at
EPOCH_SECONDS = 30                             # sleep is scored in 30 s epochs
EPOCH_SAMPLES = SAMPLE_RATE * EPOCH_SECONDS    # 3000 samples per epoch
STAGES = ("W", "N1", "N2", "N3", "REM")

CLIP_UV = 500.0                                # microvolts; larger swings are artefacts
_BANDPASS_HZ = (0.3, 35.0)                     # the band sleep scoring cares about
_BANDPASS = butter(4, _BANDPASS_HZ, btype="bandpass", fs=SAMPLE_RATE, output="sos")


def clean_signal(signal, sampling_rate):
    """Validate, resample to 100 Hz and band-pass filter a 1-D EEG signal in microvolts."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("signal contains NaN or infinite values")
    if not sampling_rate > 0:
        raise ValueError("sampling_rate must be positive")

    if sampling_rate != SAMPLE_RATE:
        ratio = (Fraction(SAMPLE_RATE) / Fraction(sampling_rate)).limit_denominator(1000)
        signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    if len(signal) < EPOCH_SAMPLES:
        raise ValueError(f"need at least {EPOCH_SECONDS} s of signal for one epoch")
    return sosfiltfilt(_BANDPASS, signal).astype(np.float32)


def split_epochs(clean):
    """Cut a cleaned 100 Hz signal into whole 30 s epochs; a trailing remainder is dropped."""
    n_epochs = len(clean) // EPOCH_SAMPLES
    return clean[: n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)


def to_epochs(signal, sampling_rate):
    """Raw signal (microvolts, any sampling rate) -> array of shape (n_epochs, 3000)."""
    return split_epochs(clean_signal(signal, sampling_rate))


def normalize(epochs, scale):
    """Remove each epoch's DC offset, clip artefacts and divide by the training-set scale.

    The scale is a single number fixed at training time and stored in the checkpoint,
    so a lone epoch is normalised exactly like one from a whole night.
    """
    epochs = np.asarray(epochs, dtype=np.float32)
    centred = epochs - epochs.mean(axis=-1, keepdims=True)
    return np.clip(centred, -CLIP_UV, CLIP_UV) / np.float32(scale)
