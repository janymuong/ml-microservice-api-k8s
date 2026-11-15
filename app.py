"""Flask microservice that scores a night of sleep from one EEG channel (PyTorch).

This is a research demo, not a medical device.
"""
import io
import logging
import os
import time

import edfio
import numpy as np
import torch
from flask import Flask, jsonify, request

from sleepnet.model import load_checkpoint, predict_proba
from sleepnet.preprocessing import EPOCH_SECONDS, SAMPLE_RATE, normalize, to_epochs

MODEL_PATH = os.environ.get("MODEL_PATH", "./model_data/sleepnet.pt")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
MAX_HOURS = float(os.environ.get("MAX_HOURS", "24"))

# One inference thread per gunicorn worker: scale out with workers or pods instead.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
LOG = app.logger
LOG.setLevel(logging.INFO)

# Loaded at import time so it also works under gunicorn / `flask run`,
# not just `python app.py`.
MODEL, META = load_checkpoint(MODEL_PATH)
STAGES = META["stages"]
SCALE = META["signal_scale"]
TRAINED_ON = META.get("trained_on", "unknown")
with torch.inference_mode():  # warm-up, so the first real request is not the slow one
    MODEL(torch.zeros(1, 4, EPOCH_SECONDS * SAMPLE_RATE))
LOG.info("Loaded SleepNet from %s (trained on: %s)", MODEL_PATH, TRAINED_ON)
if TRAINED_ON != "sleep-edf":
    LOG.warning("This checkpoint was trained on %r, not on real Sleep-EDF recordings. "
                "Predictions are meaningless; run train.py on real data.", TRAINED_ON)

UNIT_TO_MICROVOLT = {"uv": 1.0, "mv": 1e3, "v": 1e6}

INDEX_HTML = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sleep staging</title>
<style>
 body{font:16px system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem}
 canvas{width:100%;height:180px;border:1px solid #8884;border-radius:6px;margin-top:1rem}
 code,pre{background:#8882;padding:.1rem .3rem;border-radius:4px}
 .warn{border-left:4px solid #c60;padding:.2rem .8rem;background:#c601}
 label{margin-right:1rem}
</style>
<h2>Sleep staging from one EEG channel</h2>
<p class="warn" id="banner"></p>
<p>Upload one night of a single EEG channel (e.g. Fpz-Cz) as a <code>.npy</code> array of
microvolts or an <code>.edf</code> file, and get a stage (Wake, N1, N2, N3, REM) for every
30-second epoch.</p>
<p><label>file <input type="file" id="file" accept=".npy,.edf"></label>
<label>sampling rate (Hz, .npy only) <input type="number" id="rate" value="100" min="1" style="width:5rem"></label></p>
<div id="out"></div><canvas id="plot" width="900" height="180"></canvas>
<script>
const STAGES = ["W", "N1", "N2", "N3", "REM"];
fetch("/model").then(r => r.json()).then(m => {
  banner.textContent = m.trained_on === "sleep-edf"
    ? "Research demo: not a medical device."
    : "This server runs a checkpoint trained on " + m.trained_on.toUpperCase() +
      " data. Its predictions are meaningless until it is retrained on real recordings.";
});
file.onchange = async () => {
  const f = file.files[0]; if (!f) return;
  out.textContent = "Scoring...";
  const body = new FormData(); body.append("file", f); body.append("sampling_rate", rate.value);
  const res = await fetch("/predict", {method: "POST", body});
  const data = await res.json();
  if (!res.ok) { out.textContent = data.error; return; }
  const s = data.summary;
  out.textContent = data.n_epochs + " epochs. Time asleep " + s.total_sleep_minutes +
    " min, efficiency " + (s.sleep_efficiency * 100).toFixed(0) + "%. Scored in " + data.latency_ms + " ms.";
  draw(data.stages);
};
function draw(stages) {
  const ctx = plot.getContext("2d"), w = plot.width, h = plot.height;
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = ctx.strokeStyle = "#888"; ctx.font = "12px sans-serif";
  const row = name => (STAGES.indexOf(name) + .5) * h / STAGES.length;
  STAGES.forEach(n => { ctx.fillText(n, 2, row(n) + 4); });
  ctx.strokeStyle = "#2d7d46"; ctx.lineWidth = 2; ctx.beginPath();
  const x = i => 36 + i * (w - 40) / Math.max(1, stages.length - 1);
  stages.forEach((st, i) => { i ? ctx.lineTo(x(i), row(st)) : ctx.moveTo(x(i), row(st)); });
  ctx.stroke();
}
</script>
"""


class BadRequest(Exception):
    """A client mistake: shown to the caller as a 400 with the message."""


def edf_to_signal(data, channel):
    """Pick one channel out of an EDF file; returns (microvolts, sampling rate)."""
    try:
        edf = edfio.read_edf(io.BytesIO(data))
    except Exception as err:  # edfio raises several different types for malformed files
        raise BadRequest(f"Could not read that file as EDF: {err}") from err
    labels = [s.label for s in edf.signals]
    if channel is None:
        eeg = [label for label in labels if "EEG" in label.upper()]
        preferred = [label for label in eeg if "FPZ" in label.upper()]
        if not eeg:
            raise BadRequest(f"No channel with 'EEG' in its name. Channels: {labels}. "
                             "Pass the one to use as form field 'channel'.")
        channel = (preferred or eeg)[0]
    elif channel not in labels:
        raise BadRequest(f"No channel {channel!r}. Channels: {labels}")
    sig = edf.get_signal(channel)
    factor = UNIT_TO_MICROVOLT.get(sig.physical_dimension.strip().lower().replace("µ", "u"), 1.0)
    return np.asarray(sig.data) * factor, float(sig.sampling_frequency)


def read_request():
    """The signal and sampling rate from either a file upload or a JSON body."""
    upload = request.files.get("file")
    if upload is not None:
        raw = upload.read()
        try:
            rate = float(request.form.get("sampling_rate", SAMPLE_RATE))
        except ValueError as err:
            raise BadRequest("sampling_rate must be a number") from err
        name = (upload.filename or "").lower()
        if name.endswith(".edf"):
            return edf_to_signal(raw, request.form.get("channel"))
        try:
            return np.load(io.BytesIO(raw), allow_pickle=False), rate
        except (ValueError, OSError, EOFError) as err:
            raise BadRequest("Upload a .npy array (1-D, microvolts) or an .edf file") from err

    body = request.get_json(silent=True)
    if not isinstance(body, dict) or "signal" not in body:
        raise BadRequest('Send multipart field "file" (.npy or .edf), or JSON '
                         '{"signal": [...microvolts...], "sampling_rate": 100}')
    try:
        return np.asarray(body["signal"], dtype=np.float64), float(body.get("sampling_rate", SAMPLE_RATE))
    except (TypeError, ValueError) as err:
        raise BadRequest("'signal' must be a flat list of numbers") from err


def summarize_night(stage_ids):
    """Standard sleep-report numbers from one stage per 30 s epoch."""
    minutes_per_epoch = EPOCH_SECONDS / 60
    counts = np.bincount(stage_ids, minlength=len(STAGES))
    asleep = np.flatnonzero(stage_ids != 0)
    rem = np.flatnonzero(stage_ids == len(STAGES) - 1)
    total_sleep = float(len(asleep) * minutes_per_epoch)
    return {
        "minutes_per_stage": {s: round(float(c * minutes_per_epoch), 1) for s, c in zip(STAGES, counts)},
        "total_sleep_minutes": round(total_sleep, 1),
        "time_in_bed_minutes": round(len(stage_ids) * minutes_per_epoch, 1),
        "sleep_efficiency": round(total_sleep / (len(stage_ids) * minutes_per_epoch), 3),
        "sleep_latency_minutes": round(float(asleep[0] * minutes_per_epoch), 1) if len(asleep) else None,
        "rem_latency_minutes": round(float((rem[0] - asleep[0]) * minutes_per_epoch), 1)
        if len(rem) and len(asleep) else None,
    }


@app.route("/")
def home():
    return INDEX_HTML


@app.route("/healthz")
def healthz():
    """Liveness / readiness probe: the model is loaded before the app can answer at all."""
    return jsonify({"status": "ok"})


@app.route("/model")
def model_info():
    """What is being served and how it scored on held-out subjects."""
    return jsonify({
        "name": "SleepNet", "stages": STAGES, "trained_on": TRAINED_ON,
        "parameters": META.get("parameters"), "test_metrics": META.get("metrics"),
        "input": {"channel": "single EEG, e.g. Fpz-Cz", "unit": "microvolts",
                  "model_sample_rate_hz": SAMPLE_RATE, "epoch_seconds": EPOCH_SECONDS},
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Stage every 30 s epoch of a recording.

    curl -F "file=@samples/synthetic_night.npy" -F sampling_rate=100 localhost:8000/predict
    curl -F "file=@SC4001E0-PSG.edf" localhost:8000/predict
    curl -H "Content-Type: application/json" -d '{"signal": [...], "sampling_rate": 100}' ...

    result looks like:
    {"n_epochs": 120, "epoch_seconds": 30, "stages": ["W", "N1", ...],
     "probabilities": [[...5 numbers per epoch...], ...], "summary": {...}, "latency_ms": 41.2}
    """
    try:
        signal, rate = read_request()
        if len(signal) / rate > MAX_HOURS * 3600:
            raise BadRequest(f"Recording is longer than {MAX_HOURS:g} hours")
        started = time.perf_counter()
        epochs = to_epochs(signal, rate)
    except (BadRequest, ValueError) as err:
        LOG.warning("Rejected request: %s", err)
        return jsonify({"error": str(err)}), 400

    probs = predict_proba(MODEL, normalize(epochs, SCALE))
    stage_ids = probs.argmax(1)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    LOG.info("Scored %d epochs in %s ms", len(stage_ids), latency_ms)
    return jsonify({
        "n_epochs": int(len(stage_ids)),
        "epoch_seconds": EPOCH_SECONDS,
        "stages": [STAGES[i] for i in stage_ids],
        "probabilities": np.round(probs, 4).tolist(),
        "summary": summarize_night(stage_ids),
        "trained_on": TRAINED_ON,
        "latency_ms": latency_ms,
    })


@app.errorhandler(413)
def too_large(_err):
    return jsonify({"error": f"Upload is larger than {MAX_UPLOAD_MB} MB"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
