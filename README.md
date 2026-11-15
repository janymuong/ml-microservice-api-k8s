# EEG ML Microservice: PyTorch,  Docker and  Kubernetes

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/janymuong/ml-microservice-api-k8s/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/janymuong/ml-microservice-api-k8s/tree/main)

```Cloud DevOps Engineering```, ```Sleep Staging```

Operationalizing a machine learning microservice: a PyTorch model behind a Flask API, containerized with Docker and deployed on Kubernetes.

The model reads **one channel of EEG from a night's sleep** and labels every 30-second epoch as Wake, N1, N2, N3 or REM. That gives you a hypnogram (the standard sleep chart) plus the usual report numbers: time asleep, sleep efficiency, sleep latency, REM latency.

> **NOTE: not a medical device.** Do not use it to diagnose or treat anything.

The model, `model_data/sleepnet.pt`, was trained on real recordings from [Sleep-EDF](#the-problem-and-the-data): 39 nights from 20 subjects. On 4 held-out subjects it reaches a Cohen's kappa of **0.78** (see [Results](#results)). `/model`, every `/predict` response and the web page report `trained_on: sleep-edf`. Training on more subjects is the most direct way to improve it ([Train on real data](#train-on-real-data)).

The first version of this project served a scikit-learn model that predicted Boston house prices. That version, unchanged, lives in [legacy/boston-sklearn](./legacy/boston-sklearn). The Docker, Makefile, CI and Kubernetes plumbing was kept and adapted.

---
## The Problem and the Data

Sleep is scored by experts who look at 30-second pages of EEG and assign a stage. It is slow, and experts disagree with each other, so automating it is a well-studied machine learning problem.

**Data:** [Sleep-EDF Expanded](https://www.physionet.org/content/sleep-edfx/1.0.0/) on PhysioNet: 197 whole-night recordings (153 from the sleep-cassette study of healthy adults aged 25-101, 44 from the sleep-telemetry study), with stages scored by hand. EEG is sampled at 100 Hz. The full download is 8.1 GB. License: [Open Data Commons Attribution v1.0](https://opendatacommons.org/licenses/by/1-0/), so you must credit the source (see [Credits](#credits)).

The scoring in the files follows the 1968 Rechtschaffen & Kales manual, which has two deep-sleep stages (3 and 4). They are merged into one, N3, as in the current AASM scheme. Movement time and unscored epochs are ignored.

## The Model

`SleepNet` ([sleepnet/model.py](./sleepnet/model.py)), 293k parameters:

1. A **1-D CNN** turns each 30-second epoch (3,000 samples) into a feature vector.
2. A **bidirectional GRU** runs over a window of 20 consecutive epochs, because the neighbours of an ambiguous epoch (N1, REM) are the best clue to what it is.

At serving time a longer recording is cut into 50%-overlapping windows and the overlapping predictions are averaged. The same function scores the test set in `train.py` and answers `/predict`, so the reported metrics describe what the API really does.

**Preprocessing** ([sleepnet/preprocessing.py](./sleepnet/preprocessing.py)) is shared by training and serving: resample to 100 Hz, 0.3-35 Hz band-pass, cut into whole 30 s epochs, remove each epoch's DC offset, clip artefacts at 500 uV and divide by a scale learned from the training data.

**Evaluation:** recordings are split **by subject** (60/20/20). Both nights of a person always land in the same split, since splitting by night would leak a person's EEG signature into the test set and inflate the scores. The best epoch is picked on validation macro-F1 and the test subjects are scored once. `train.py` reports accuracy, macro-F1, Cohen's kappa (the number sleep-staging papers quote) and a per-stage breakdown.

### Results

Trained on sleep-cassette subjects 0-19 (39 nights; subject 13 has only one night in the dataset), 25 epochs, about 17 minutes on the author's machine.

| split | subjects | recordings | epochs |
|---|---|---|---|
| train | 12 | 24 | 25,344 |
| validation | 4 | 7 | 7,894 |
| **test** | **4** | **8** | **9,130** |

Test-set scores (best epoch 15, chosen on validation; 9,118 scored epochs):

| accuracy | macro-F1 | Cohen's kappa |
|---|---|---|
| 0.838 | 0.789 | **0.778** |

| stage | precision | recall | F1 | test epochs |
|---|---|---|---|---|
| W | 0.93 | 0.77 | 0.84 | 1,747 |
| N1 | 0.45 | 0.51 | 0.47 | 707 |
| N2 | 0.96 | 0.84 | 0.90 | 4,024 |
| N3 | 0.83 | 0.97 | 0.90 | 862 |
| REM | 0.73 | 0.95 | 0.83 | 1,778 |

The confusion matrix and the rest of the run are in [model_data/sleepnet.metrics.json](./model_data/sleepnet.metrics.json).

How to read it:

- A kappa of 0.61-0.80 is conventionally called *substantial* agreement with the human scorers.
- **N2 and N3 are strong** (F1 0.90). **N1 is the weak stage** (F1 0.47): it is rare (6.6% of training epochs) and even human scorers disagree on it.
- Recall is very high for N3 and REM but lower for W and N2, so the model leans towards calling things N3, REM or N1 and under-calls W and N2. Likely causes are the extra loss weight on rarer classes and having only one EEG channel: REM, wake and N1 all look like low-amplitude mixed-frequency EEG, and scorers separate them with eye and muscle signals the model never sees.
- Validation kappa plateaued around 0.81 from about epoch 8, while the test kappa is 0.78. Part of that gap is picking the best epoch on validation, and part is noise: the test set is only 4 subjects, so treat the score as having wide error bars. Repeating the run with different seeds would give a steadier estimate.
- Training for more epochs on the same data will not help much; more subjects (`./fetch_data.sh model_data/sleep-edf "SC4*"` fetches all 153 recordings) and a second EEG channel are the obvious next steps.

---
## Setup the Environment

- Create a siloed virtual environment with Python and **activate** it:
```bash
python -m venv .devops
source .devops/bin/activate
```
> Alternatively, `make setup` ([Makefile](./Makefile)).

- Run `make install` to install the dependencies (PyTorch's CPU-only build, ~200 MB, plus MNE, pylint and pytest) and `hadolint`.
- `make test` runs the 37 tests, `make lint` runs `pylint` and `hadolint`.

### Train on real data

```bash
./fetch_data.sh                 # subjects 0-19 of the sleep-cassette study into model_data/sleep-edf/
python train.py                 # writes model_data/sleepnet.pt and model_data/sleepnet.metrics.json
```

`fetch_data.sh` uses the AWS CLI (no account needed, and much faster than the `wget` fallback); pass a different glob as the second argument to fetch more or fewer subjects (`./fetch_data.sh model_data/sleep-edf "SC4*"` for all 153 recordings). Training time depends on how many subjects you use, so start small with `python train.py --max-subjects 8 --epochs 10`.

Useful flags: `--channel` (default `EEG Fpz-Cz`), `--epochs`, `--lr`, `--threads`. `python train.py --synthetic 12 --epochs 8 --out /tmp/x.pt` runs the whole pipeline on made-up data in about a minute (a smoke test, not a model; the checkpoint it writes says `trained_on: synthetic`).

`trained_on` records the data *source*, not how much of it was used, so a run with `--max-subjects 3` is still labelled `sleep-edf`. Check `sleepnet.metrics.json` for the split sizes and scores.

The checkpoint is loaded when the server starts, so after training, (re)start the server or (re)build the image so the container serves the new checkpoint.

### Running `app.py`

1. Standalone:  `PORT=8000 python app.py` (the default port is 80)
2. Run in Docker:  `./run_docker.sh`
3. Run in Kubernetes:  `./run_kubernetes.sh`

The scripts build and run the image `janymuong/eeg-sleep-staging` and expose the API on `localhost:8000`.

## Using the API

Open `http://localhost:8000/` for an upload page that draws the hypnogram, or call the API directly. Send one EEG channel as microvolts, in any of three ways:

```bash
curl -F "file=@samples/synthetic_night.npy" -F sampling_rate=100 localhost:8000/predict   # .npy array
curl -F "file=@SC4001E0-PSG.edf" localhost:8000/predict                                    # .edf file
curl -H "Content-Type: application/json" \
     -d '{"signal": [12.1, -3.4, ...], "sampling_rate": 100}' localhost:8000/predict       # JSON
# or: ./make_prediction.sh [file] [sampling_rate]
```

For an `.edf` upload the service picks the channel whose name contains `EEG` (preferring `Fpz`), converts V/mV/uV to microvolts, and uses the file's own sampling rate. Pass `-F channel="EEG Pz-Oz"` to choose another. Any sampling rate is resampled to 100 Hz; at least 30 s of signal is needed.

The shape of the response (the values here are illustrative):

```json
{
  "n_epochs": 120, "epoch_seconds": 30,
  "stages": ["W", "W", "N1", "N2", "N2", "..."],
  "probabilities": [[0.91, 0.06, 0.02, 0.0, 0.01], "..."],
  "summary": {
    "minutes_per_stage": {"W": 10.5, "N1": 5.0, "N2": 18.0, "N3": 26.5, "REM": 0.0},
    "total_sleep_minutes": 49.5, "time_in_bed_minutes": 60.0, "sleep_efficiency": 0.825,
    "sleep_latency_minutes": 3.5, "rem_latency_minutes": null
  },
  "trained_on": "sleep-edf", "latency_ms": 76.2
}
```

| endpoint | purpose |
|---|---|
| `POST /predict` | stage a recording (`400` with a reason for bad input, `413` if over the upload limit) |
| `GET /healthz` | liveness / readiness probe |
| `GET /model` | what is served, what it was trained on, its test scores |
| `GET /` | upload page |

Configuration by environment variable: `MODEL_PATH`, `MAX_UPLOAD_MB` (default 100), `MAX_HOURS` (default 24), `TORCH_NUM_THREADS` (default 1) and, in the container, `WEB_CONCURRENCY` (gunicorn workers, default 2). Uploads are parsed with `allow_pickle=False`, so a malicious `.npy` cannot run code.

`samples/synthetic_night.npy` is an hour of *synthetic* signal (with its made-up stages in `synthetic_night.stages.json`). It is only for checking that the API accepts a file and returns the right shape of response, e.g. in `./make_prediction.sh`. The model was trained on real EEG, so its stages for this synthetic signal mean nothing. To see real behaviour, upload a Sleep-EDF `*-PSG.edf` file, ideally from a subject that was not in the training split.

### Performance notes (measured on 2 CPU cores)

- A full 8-hour night (960 epochs, a 12 MB upload) is scored in about 0.5 s.
- Each gunicorn worker held about 700 MB of RAM with the default CUDA PyTorch wheel. The CPU-only wheel in the Dockerfile should be smaller but has not been measured, so check `kubectl top pod`. The Kubernetes manifest runs one worker per pod and scales by replicas.

## Kubernetes

[k8s/deployment.yaml](./k8s/deployment.yaml) runs 2 replicas with readiness and liveness probes on `/healthz` and starting CPU/memory requests and limits. [k8s/service.yaml](./k8s/service.yaml) exposes them inside the cluster. `./run_kubernetes.sh` builds the image, loads it into minikube, applies both files, restarts the pods to pick up the new image and port-forwards the service to `localhost:8000`.

---
## Appendix
### `appendix a` - File Info:
- [sleepnet/](./sleepnet) - the ML code: `preprocessing.py`, `model.py`, `data.py` (Sleep-EDF reader and synthetic data), `metrics.py`.
- [app.py](./app.py) - the Flask application that serves the model.
- [train.py](./train.py) - trains the model and writes the checkpoint and its metrics.
- [model_data](./model_data) - the checkpoint `sleepnet.pt` and `sleepnet.metrics.json`; downloaded recordings go in `model_data/sleep-edf/` (git-ignored).
- [tests](./tests) - API, preprocessing and data-loading tests. The EDF reader is tested on EDF files generated in Sleep-EDF's layout.
- [samples](./samples) - a synthetic recording for trying the API.
- [k8s](./k8s) - Kubernetes manifests.
- [Dockerfile](./Dockerfile), [Makefile](./Makefile), [.circleci/config](./.circleci/config.yml) - container, lint/test commands and CI.
- [requirements.txt](./requirements.txt) (what the container installs) and [requirements-dev.txt](./requirements-dev.txt) (adds MNE, pylint, pytest).
- [legacy/boston-sklearn](./legacy/boston-sklearn) - the original scikit-learn project.

### `appendix b` - Helper Scripts:
`fetch_data.sh` `make_prediction.sh` `run_docker.sh` `run_kubernetes.sh` `upload_docker.sh`
These are bash scripts that you can use in place of multiple/chained shell commands.

`resize.sh` - A bash script to resize an `AWS Cloud9` environment, if one opts to use a virtual machine for a flexible setup.
> usage: run this in the terminal: `bash resize.sh intended_cloud9_volume_size` - size in Gigabytes.
>
> running `bash resize.sh` uses 100GB by [default](./resize.sh).

### `appendix c` - Limitations
- One EEG channel only. Real scorers also use EOG and EMG, which helps most with REM.
- The model was trained on 20 of the 78 sleep-cassette subjects and scored on only 4 held-out subjects, so the results above are a rough estimate, not a benchmark.
- Training trims each night to about 30 minutes of wake before and after sleep, but the API scores whatever you send it. A recording that includes many hours of daytime wake (the raw sleep-cassette files run ~20 hours) may have some of that wake labelled as sleep, so judge it on the sleep period.
- Sleep-EDF has healthy adults (plus a small temazepam study) recorded in the 1980s-90s with older equipment. Expect worse results on other populations, devices and sleep disorders.
- The output of this is for decision support at most, never a diagnosis.

---
## Credits
Data: B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. *Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG.* IEEE-BME 47(9):1185-1194 (2000), distributed through [PhysioNet](https://www.physionet.org/content/sleep-edfx/1.0.0/) under the Open Data Commons Attribution License v1.0. and PhysioNet publication given on that page.
