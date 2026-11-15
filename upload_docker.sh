#!/usr/bin/env bash
# This file uploads the image built by run_docker.sh to Docker Hub
set -euo pipefail

# Docker Hub username and image path (already tagged this way by run_docker.sh)
dockeruser=janymuong
dockerpath=$dockeruser/EEG-ML-sleep-staging

# Fail early if the image hasn't been built
docker image inspect "$dockerpath:latest" >/dev/null 2>&1 || {
	echo "Image $dockerpath:latest not found. Run ./run_docker.sh first." >&2
	exit 1
}

# Authenticate from the CLI (no browser flow)
echo "Docker ID and Image: $dockerpath"
if [ -n "${DOCKER_TOKEN:-}" ]; then
	# Non-interactive: token supplied via environment variable
	echo "$DOCKER_TOKEN" | docker login -u "$dockeruser" --password-stdin
else
	# Interactive: prompts for your password / personal access token
	docker login -u "$dockeruser"
fi

# Push image to a docker repository
docker push "$dockerpath:latest"

# this bash script simulates these manual steps:
# docker login -u janymuong
# docker push janymuong/sleep-staging:latest
