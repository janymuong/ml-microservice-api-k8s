#!/usr/bin/env bash
# Stage a night of EEG with the running API.
# usage: ./make_prediction.sh [recording.npy|recording.edf] [sampling_rate_hz]
set -euo pipefail

PORT=${PORT:-8000}
FILE=${1:-samples/synthetic_night.npy}
RATE=${2:-100}
echo "Port: $PORT"
echo "File: $FILE (sampling rate $RATE Hz; only used for .npy)"

# POST method predict: the recording goes in as a multipart form field called "file"
curl -sS -F "file=@${FILE}" -F "sampling_rate=${RATE}" -X POST "http://localhost:${PORT}/predict"
echo
