"""Reading Sleep-EDF recordings, and generating synthetic nights for tests and the demo model.

Sleep-EDF Expanded (PhysioNet, Open Data Commons Attribution License) stores every night
as two EDF files that share their first six characters:

    SC4001E0-PSG.edf          the signals (EEG Fpz-Cz and Pz-Oz at 100 Hz, EOG, EMG, ...)
    SC4001EC-Hypnogram.edf    the expert-scored stage of every 30 s, as EDF+ annotations
"""
import warnings
from pathlib import Path

import numpy as np

from .preprocessing import EPOCH_SAMPLES, EPOCH_SECONDS, SAMPLE_RATE, STAGES, to_epochs

CHANNEL = "EEG Fpz-Cz"
# Rechtschaffen & Kales stages 3 and 4 are merged into N3, as in today's AASM scheme.
# Movement time, "?" (not scored) and anything else get label -1 and are ignored.
STAGE_OF = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}
IGNORE = -1
# The SC recordings run ~20 h but the subject is awake most of the first and last hours,
# which would swamp the classes. Keep 30 minutes (60 epochs) of wake around the sleep.
WAKE_MARGIN_EPOCHS = 60


def list_recordings(root):
    """Pair every *-PSG.edf under root with its hypnogram; returns a list of dicts."""
    recordings = []
    for psg in sorted(Path(root).rglob("*-PSG.edf")):
        prefix = psg.name[:6]
        hypnograms = sorted(psg.parent.glob(f"{prefix}*-Hypnogram.edf"))
        if not hypnograms:
            warnings.warn(f"no hypnogram found for {psg.name}, skipping it")
            continue
        recordings.append({
            "name": prefix,
            "subject": prefix[:5],  # SC4ssN: both nights of a subject share SC4ss
            "psg": psg,
            "hypnogram": hypnograms[0],
        })
    return recordings


def labels_from_annotations(annotations, n_epochs):
    """Hypnogram annotations [(onset_s, duration_s, text), ...] -> one label per 30 s epoch."""
    labels = np.full(n_epochs, IGNORE, dtype=np.int8)
    for onset, duration, text in annotations:
        stage = STAGE_OF.get(str(text).strip())
        if stage is None:
            continue
        first = int(round(onset / EPOCH_SECONDS))
        count = int(round(duration / EPOCH_SECONDS))
        labels[first:first + count] = stage  # slicing clips anything past the end
    return labels


def trim_wake(labels, margin=WAKE_MARGIN_EPOCHS):
    """Slice that keeps `margin` epochs of wake before the first and after the last sleep."""
    asleep = np.flatnonzero(labels > 0)
    if len(asleep) == 0:
        return slice(0, 0)
    return slice(max(0, asleep[0] - margin), asleep[-1] + 1 + margin)


def load_recording(psg, hypnogram, channel=CHANNEL, margin=WAKE_MARGIN_EPOCHS):
    """Read one night: returns (epochs float32 (n, 3000) in microvolts, labels int8 (n,))."""
    import mne  # training-only dependency, deliberately not imported by the API

    raw = mne.io.read_raw_edf(psg, preload=False, verbose="error")  # header only for now
    if channel not in raw.ch_names:
        raise ValueError(f"No channel {channel!r} in {Path(psg).name}. "
                         f"Channels in the file: {raw.ch_names}. Pick one with --channel.")
    raw.pick([channel]).load_data(verbose="error")
    epochs = to_epochs(raw.get_data()[0] * 1e6, raw.info["sfreq"])  # volts -> microvolts
    notes = mne.read_annotations(hypnogram)
    labels = labels_from_annotations(
        zip(notes.onset, notes.duration, notes.description), len(epochs))
    keep = trim_wake(labels, margin)
    return epochs[keep], labels[keep]


# ----------------------------------------------------------------------------------------
# Synthetic data. NOT real EEG: just enough stage-dependent structure to exercise the code.
# ----------------------------------------------------------------------------------------

# (low Hz, high Hz, microvolt amplitude) of the oscillations mixed into each stage
_BANDS = {
    0: [(8.5, 12.0, 14.0), (15.0, 30.0, 5.0)],                 # W: alpha + beta
    1: [(4.0, 7.0, 13.0), (8.0, 11.0, 5.0)],                   # N1: theta, fading alpha
    2: [(4.0, 7.0, 10.0), (11.5, 15.0, 16.0), (0.5, 1.5, 12.0)],  # N2: spindles + slow
    3: [(0.5, 2.0, 55.0), (4.0, 7.0, 6.0)],                    # N3: big slow waves
    4: [(4.0, 8.0, 9.0), (16.0, 28.0, 3.0)],                   # REM: low-amplitude theta
}
# Row = current stage, column = next stage (W, N1, N2, N3, REM). Sticky, like real sleep.
_TRANSITIONS = np.array([
    [0.80, 0.15, 0.05, 0.00, 0.00],
    [0.10, 0.60, 0.28, 0.00, 0.02],
    [0.03, 0.05, 0.80, 0.09, 0.03],
    [0.01, 0.00, 0.10, 0.89, 0.00],
    [0.04, 0.03, 0.08, 0.00, 0.85],
])


def synthetic_stages(n_epochs, rng):
    """A plausible-looking hypnogram: starts awake, sleeps in sticky runs."""
    stages = np.empty(n_epochs, dtype=np.int8)
    stages[0] = 0
    for i in range(1, n_epochs):
        stages[i] = rng.choice(5, p=_TRANSITIONS[stages[i - 1]])
    return stages


def synthetic_signal(stages, rng):
    """100 Hz 'EEG' (microvolts) for a hypnogram: stage-specific rhythms plus noise."""
    t = np.arange(EPOCH_SAMPLES) / SAMPLE_RATE
    out = np.empty(len(stages) * EPOCH_SAMPLES)
    for i, stage in enumerate(stages):
        epoch = rng.normal(0.0, 9.0, EPOCH_SAMPLES)
        for low, high, amp in _BANDS[int(stage)]:
            freq, phase = rng.uniform(low, high), rng.uniform(0, 2 * np.pi)
            epoch += amp * rng.uniform(0.6, 1.4) * np.sin(2 * np.pi * freq * t + phase)
        out[i * EPOCH_SAMPLES:(i + 1) * EPOCH_SAMPLES] = epoch
    return out


def synthetic_night(n_epochs, seed):
    """(epochs (n, 3000) float32, labels (n,) int8) for one made-up night."""
    rng = np.random.default_rng(seed)
    stages = synthetic_stages(n_epochs, rng)
    return to_epochs(synthetic_signal(stages, rng), SAMPLE_RATE), stages


def write_synthetic_edf_pair(directory, prefix, n_epochs, seed):
    """Write a fake PSG + hypnogram pair that mimics Sleep-EDF's file layout.

    Lets the loader be tested end to end without the real dataset. Runs of N3 alternate
    between the old "Sleep stage 3" and "Sleep stage 4" labels to test the merge.
    """
    from edfio import Edf, EdfAnnotation, EdfSignal

    rng = np.random.default_rng(seed)
    stages = synthetic_stages(n_epochs, rng)
    signal = synthetic_signal(stages, rng)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    psg_path = directory / f"{prefix}0-PSG.edf"
    hyp_path = directory / f"{prefix}C-Hypnogram.edf"
    Edf([EdfSignal(signal, SAMPLE_RATE, label=CHANNEL, physical_dimension="uV")]).write(psg_path)

    names = {0: "W", 1: "1", 2: "2", 4: "R"}
    notes, start, n3_runs = [], 0, 0
    for i in range(1, n_epochs + 1):
        if i == n_epochs or stages[i] != stages[start]:
            stage = int(stages[start])
            if stage == 3:
                text = "Sleep stage 3" if n3_runs % 2 == 0 else "Sleep stage 4"
                n3_runs += 1
            else:
                text = f"Sleep stage {names[stage]}"
            notes.append(EdfAnnotation(start * EPOCH_SECONDS, (i - start) * EPOCH_SECONDS, text))
            start = i
    Edf([EdfSignal(np.zeros(1), 1, label="Event marker")], annotations=notes).write(hyp_path)
    return psg_path, hyp_path, stages


__all__ = [
    "CHANNEL", "IGNORE", "STAGES", "STAGE_OF", "WAKE_MARGIN_EPOCHS", "labels_from_annotations",
    "list_recordings", "load_recording", "synthetic_night", "trim_wake",
    "write_synthetic_edf_pair",
]
