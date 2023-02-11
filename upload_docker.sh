#!/usr/bin/env bash
# This file tags and uploads an image to Docker Hub

# assumes that an image is built via `run_docker.sh`

# Create dockerpath
dockerpath=janymuong/mlapiscikit

# Authenticate & tag
echo "Docker ID and Image: $dockerpath"
docker login
docker tag mlapiscikit $dockerpath:latest

# Push image to a docker repository
docker push $dockerpath:latest

# this bash script simulates this manual steps:
# docker login
# docker tag mlapiscikit janymuong/mlapiscikit:latest
# docker push janymuong/mlapiscikit:latest