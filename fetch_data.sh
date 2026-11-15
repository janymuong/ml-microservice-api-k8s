#!/usr/bin/env bash
# Download recordings from Sleep-EDF Expanded (PhysioNet, Open Data Commons Attribution
# License v1.0) into model_data/sleep-edf/, where train.py looks for them.
#
# usage: ./fetch_data.sh [destination] [file-glob-prefix]
#
# The full dataset is 8.1 GB. Files are named SC4<subject:2 digits><night>E0-PSG.edf, so the
# default glob SC4[0-1]* selects subjects 00-19 of the sleep-cassette study (both nights each,
# roughly 40 recordings). Use "SC4*" for all of them, or "SC40*" for a quick 10-subject run.
set -euo pipefail

dest="${1:-model_data/sleep-edf}"
glob="${2:-SC4[0-1]*}"
mkdir -p "$dest"

if command -v aws >/dev/null 2>&1; then
	# public bucket: no AWS account or credentials needed
	aws s3 sync --no-sign-request \
		s3://physionet-open/sleep-edfx/1.0.0/sleep-cassette/ "$dest" \
		--exclude "*" --include "${glob}-PSG.edf" --include "${glob}Hypnogram.edf"
else
	wget -r -N -c -np -nd -P "$dest" \
		-A "${glob}-PSG.edf,${glob}Hypnogram.edf" \
		https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
fi

echo "PSG files:        $(find "$dest" -name '*-PSG.edf' | wc -l)"
echo "Hypnogram files:  $(find "$dest" -name '*Hypnogram.edf' | wc -l)"
echo "Next: pip install -r requirements-dev.txt && python train.py"
