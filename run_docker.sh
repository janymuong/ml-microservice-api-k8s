#!/usr/bin/env bash
set -euo pipefail

## Complete the following steps to get Docker running locally

# Build image and add a descriptive tag
docker build --tag=janymuong/sleep-staging:latest .

# list docker images for logging
docker image ls janymuong/sleep-staging

# Run flask app (host 8000 -> container 80)
docker run --rm -p 8000:80 janymuong/sleep-staging:latest
