import logging
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask.logging import create_logger

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

FEATURES = ["CHAS", "RM", "TAX", "PTRATIO", "B", "LSTAT"]
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "./model_data/boston_housing_prediction.joblib"
)

# Loaded at import time so it also works under gunicorn / `flask run`,
# not just `python app.py`.
clf = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return "<h3>Sklearn Prediction Home</h3>"


@app.route("/predict", methods=["POST"])
def predict():
    """Performs an sklearn prediction.

    input looks like:
    {
        "CHAS":    {"0": 0},
        "RM":      {"0": 6.575},
        "TAX":     {"0": 296.0},
        "PTRATIO": {"0": 15.3},
        "B":       {"0": 396.9},
        "LSTAT":   {"0": 4.98}
    }

    result looks like:
    { "prediction": [ <val> ] }
    """
    json_payload = request.get_json(silent=True)
    if not json_payload:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    LOG.info(f"JSON payload: \n{json_payload}")

    inference_payload = pd.DataFrame(json_payload)

    missing = [f for f in FEATURES if f not in inference_payload.columns]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    # Enforce column order and numeric types so JSON key order doesn't matter
    inference_payload = inference_payload[FEATURES].astype(float)
    LOG.info(f"Inference payload DataFrame: \n{inference_payload}")

    # The pipeline scales with statistics learned during training
    prediction = clf.predict(inference_payload).tolist()
    LOG.info(f"Log prediction to stdout: \n{prediction}")

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
