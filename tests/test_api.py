"""API tests. They run against whatever checkpoint is in model_data/ (the demo one is fine)."""
import io

import numpy as np
import pytest
from edfio import Edf, EdfSignal

from app import STAGES, app


@pytest.fixture(name="client")
def client_fixture():
    app.config["TESTING"] = True
    return app.test_client()


def night(minutes=10, rate=100, seed=0):
    return np.random.default_rng(seed).normal(0, 25, int(minutes * 60 * rate))


def npy_file(signal, dtype=np.float32):
    buf = io.BytesIO()
    np.save(buf, signal.astype(dtype))
    buf.seek(0)
    return buf


def post_file(client, data, name, **fields):
    return client.post("/predict", data={"file": (data, name), **fields},
                       content_type="multipart/form-data")


def test_healthz_and_model(client):
    assert client.get("/healthz").get_json() == {"status": "ok"}
    info = client.get("/model").get_json()
    assert info["stages"] == STAGES == ["W", "N1", "N2", "N3", "REM"]
    assert info["trained_on"] in ("sleep-edf", "synthetic")


def test_the_home_page_loads(client):
    assert b"Sleep staging" in client.get("/").data


def test_npy_upload_returns_one_stage_per_epoch(client):
    resp = post_file(client, npy_file(night(10)), "night.npy", sampling_rate="100")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["n_epochs"] == 20 == len(body["stages"]) == len(body["probabilities"])
    assert set(body["stages"]) <= set(STAGES)
    assert np.allclose(np.sum(body["probabilities"], axis=1), 1.0, atol=1e-3)
    summary = body["summary"]
    assert summary["time_in_bed_minutes"] == 10.0
    assert sum(summary["minutes_per_stage"].values()) == pytest.approx(10.0)
    assert 0.0 <= summary["sleep_efficiency"] <= 1.0


def test_json_body_and_other_sampling_rates(client):
    resp = client.post("/predict", json={"signal": night(3, rate=128).tolist(), "sampling_rate": 128})
    assert resp.status_code == 200
    assert resp.get_json()["n_epochs"] == 6


def test_a_long_recording_is_scored_in_overlapping_windows(client):
    resp = post_file(client, npy_file(night(120)), "long.npy")  # 240 epochs > one window
    assert resp.status_code == 200 and resp.get_json()["n_epochs"] == 240


def test_edf_upload_picks_the_eeg_channel_and_converts_units(client, tmp_path):
    eeg = night(5)
    path = tmp_path / "x.edf"
    Edf([EdfSignal(np.zeros(100 * 300), 100, label="EOG horizontal", physical_dimension="uV"),
         EdfSignal(eeg / 1000, 100, label="EEG Fpz-Cz", physical_dimension="mV")]).write(path)
    with open(path, "rb") as fh:
        resp = post_file(client, io.BytesIO(fh.read()), "x.edf")
    assert resp.status_code == 200 and resp.get_json()["n_epochs"] == 10


def test_edf_with_no_eeg_channel_lists_what_it_has(client, tmp_path):
    path = tmp_path / "y.edf"
    Edf([EdfSignal(np.zeros(100 * 60), 100, label="Resp airflow", physical_dimension="uV")]).write(path)
    with open(path, "rb") as fh:
        resp = post_file(client, io.BytesIO(fh.read()), "y.edf")
    assert resp.status_code == 400 and "Resp airflow" in resp.get_json()["error"]


@pytest.mark.parametrize("make, expected", [
    (lambda: post_file_args(io.BytesIO(b"nonsense"), "a.npy"), "npy"),
    (lambda: post_file_args(npy_file(night(0.2)), "short.npy"), "30 s"),
    (lambda: post_file_args(npy_file(np.full(100 * 60, np.nan)), "nan.npy"), "NaN"),
    (lambda: post_file_args(npy_file(np.zeros((10, 10))), "2d.npy"), "1-D"),
    (lambda: post_file_args(io.BytesIO(b"not an edf"), "b.edf"), "EDF"),
])
def test_bad_uploads_are_a_400_with_a_reason(client, make, expected):
    data, name = make()
    resp = post_file(client, data, name)
    assert resp.status_code == 400
    assert expected in resp.get_json()["error"]


def post_file_args(data, name):
    return data, name


def test_pickled_npy_is_refused_not_executed(client):
    buf = io.BytesIO()
    np.save(buf, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    buf.seek(0)
    resp = post_file(client, buf, "evil.npy")
    assert resp.status_code == 400


def test_empty_request_is_a_400(client):
    assert client.post("/predict").status_code == 400
    assert client.post("/predict", json={"nope": 1}).status_code == 400


def test_over_long_recording_is_refused(client, monkeypatch):
    monkeypatch.setattr("app.MAX_HOURS", 0.1)
    resp = post_file(client, npy_file(night(10)), "n.npy")
    assert resp.status_code == 400 and "longer than" in resp.get_json()["error"]


def test_oversized_upload_is_a_413(client, monkeypatch):
    monkeypatch.setitem(app.config, "MAX_CONTENT_LENGTH", 1000)
    resp = post_file(client, npy_file(night(1)), "big.npy")
    assert resp.status_code == 413 and "error" in resp.get_json()
