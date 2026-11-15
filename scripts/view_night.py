"""Look inside one Sleep-EDF night: what the files contain, the expert hypnogram, and what each
sleep stage looks like in the EEG. Optionally overlay the model's staging and score it.

    python scripts/view_night.py SC4001E0-PSG.edf
    python scripts/view_night.py SC4001E0-PSG.edf SC4001EC-Hypnogram.edf --model model_data/sleepnet.pt

The hypnogram path is optional: it is found next to the PSG file by its first six characters.
Writes a PNG next to the PSG file (use --show for a window, --full to include the whole ~20 h).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `sleepnet` imports
# pylint: disable=wrong-import-position
from sleepnet.data import CHANNEL, IGNORE, load_recording  # noqa: E402
from sleepnet.metrics import summarize  # noqa: E402
from sleepnet.model import load_checkpoint, predict_proba  # noqa: E402
from sleepnet.preprocessing import EPOCH_SECONDS, SAMPLE_RATE, STAGES, normalize  # noqa: E402

# conventional hypnogram order, top to bottom: awake, REM, then light to deep sleep
ROW = {"W": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}
LOOK_FOR = {
    "W": "awake: fast, low-amplitude,\nalpha (8-12 Hz) when eyes closed",
    "N1": "drowsy: slow, low-amplitude,\nalpha fades into theta (4-7 Hz)",
    "N2": "light sleep: bursts of 12-15 Hz\n'spindles' on a slow background",
    "N3": "deep sleep: large, slow waves\n(under 2 Hz, high amplitude)",
    "REM": "dreaming: low-amplitude, mixed\nfast activity, like wake",
}


def describe_files(psg):
    """Print every channel in the PSG file with its sampling rate and unit."""
    import edfio  # pylint: disable=import-outside-toplevel

    edf = edfio.read_edf(psg)
    print(f"\n{Path(psg).name}: {len(edf.signals)} channels, "
          f"{edf.duration / 3600:.1f} hours recorded")
    for sig in edf.signals:
        print(f"  {sig.label:22s} {sig.sampling_frequency:6.1f} Hz  [{sig.physical_dimension}]")


def middle_of_longest_run(labels, stage):
    """Index of an epoch in the middle of the longest uninterrupted run of `stage`."""
    best_len, best_mid, run_start = 0, None, None
    padded = np.append(labels, IGNORE)
    for i, value in enumerate(padded):
        if value == stage and run_start is None:
            run_start = i
        elif value != stage and run_start is not None:
            if i - run_start > best_len:
                best_len, best_mid = i - run_start, (run_start + i) // 2
            run_start = None
    return best_mid


def as_rows(labels):
    """Stage ids -> hypnogram row heights (NaN where unscored, so the line has a gap)."""
    heights = np.array([ROW[s] for s in STAGES], dtype=float)
    return np.where(labels >= 0, heights[np.clip(labels, 0, None)], np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("psg", type=Path)
    parser.add_argument("hypnogram", type=Path, nargs="?")
    parser.add_argument("--channel", default=CHANNEL)
    parser.add_argument("--model", type=Path, help="checkpoint: overlay its staging and score it")
    parser.add_argument("--full", action="store_true", help="whole recording, not just the sleep")
    parser.add_argument("--show", action="store_true", help="open a window instead of only saving")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    if not args.psg.is_file():
        raise SystemExit(f"File not found: {args.psg}\n(the path is relative to the folder you run "
                         "the command from, e.g. model_data/sleep-edf/SC4001E0-PSG.edf)")
    hypnogram = args.hypnogram
    if hypnogram is None:
        found = sorted(args.psg.parent.glob(f"{args.psg.name[:6]}*-Hypnogram.edf"))
        if not found:
            raise SystemExit(f"No hypnogram next to {args.psg.name}; pass its path as 2nd argument.")
        hypnogram = found[0]

    describe_files(args.psg)
    epochs, labels = load_recording(args.psg, hypnogram, args.channel,
                                    margin=10 ** 9 if args.full else 60)
    hours = len(epochs) * EPOCH_SECONDS / 3600
    print(f"\nShowing {len(epochs)} epochs of 30 s = {hours:.1f} h "
          f"(channel '{args.channel}', band-passed 0.3-35 Hz)")
    print("Expert stage totals:")
    for i, stage in enumerate(STAGES):
        count = int((labels == i).sum())
        print(f"  {stage:4s} {count * EPOCH_SECONDS / 60:6.0f} min  ({100 * count / len(labels):4.1f}%)")

    predicted = None
    if args.model:
        model, meta = load_checkpoint(args.model)
        probs = predict_proba(model, normalize(epochs, meta["signal_scale"]))
        predicted = probs.argmax(1)
        known = labels != IGNORE
        scores = summarize(labels[known], predicted[known], STAGES)
        print(f"\nModel ({args.model.name}, trained on {meta.get('trained_on', '?')}) "
              f"vs expert on this night: accuracy {scores['accuracy']:.3f}, "
              f"kappa {scores['kappa']:.3f}")
        for stage, row in scores["per_stage"].items():
            print(f"  {stage:4s} precision {row['precision']:.2f}  recall {row['recall']:.2f}  "
                  f"n={row['support']}")

    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(3, 5, height_ratios=[1.3, 1.3 if predicted is not None else 0.01, 1.4],
                            hspace=0.55)
    t_hours = np.arange(len(epochs)) * EPOCH_SECONDS / 3600
    for slot, (name, values, color) in enumerate(
            (("Expert (from the hypnogram file)", labels, "#1f6f3f"),
             ("Model", predicted, "#b3401f"))):
        if values is None:
            continue
        ax = fig.add_subplot(grid[slot, :])
        ax.step(t_hours, as_rows(values), where="post", color=color, lw=1.2)
        ax.set_yticks(list(ROW.values()), list(ROW.keys()))
        ax.set_ylim(-0.5, 4.5)
        ax.set_xlim(0, t_hours[-1])
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel("hours" if slot == 1 or predicted is None else "")
        ax.grid(alpha=0.25)

    for col, stage in enumerate(STAGES):
        ax = fig.add_subplot(grid[2, col])
        idx = middle_of_longest_run(labels, STAGES.index(stage))
        if idx is None:
            ax.set_axis_off()
            ax.set_title(f"{stage}: not in this night", fontsize=9)
            continue
        ax.plot(np.arange(epochs.shape[1]) / SAMPLE_RATE, epochs[idx], lw=0.6, color="#333")
        ax.set_ylim(-150, 150)  # same scale everywhere, so amplitude differences are visible
        ax.set_xlim(0, EPOCH_SECONDS)
        ax.set_xlabel("seconds")
        if col == 0:
            ax.set_ylabel("microvolts")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{stage}  (at {t_hours[idx]:.1f} h)\n{LOOK_FOR[stage]}", fontsize=8)
    fig.suptitle(f"{args.psg.name}: expert hypnogram and one 30 s EEG example of each stage",
                 fontsize=11)

    out = args.out or args.psg.with_suffix(".png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
