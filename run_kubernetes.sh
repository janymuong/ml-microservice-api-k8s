#!/usr/bin/env bash

# This tags and uploads an image to Docker Hub

# This is your Docker ID/path
dockerpath=janymuong/mlapiscikit

# Run the Docker Hub container with kubernetes
kubectl run mlapiscikit \
	--image=$dockerpath:latest --port=80

# List kubernetes pods
kubectl get pods

# Forward the container port to a host
kubectl port-forward mlapiscikit 8000:80

## [this] bash script simulates these manual steps:
# kubectl run mlapiscikit --image=janymuong/mlapiscikit:latest --port=80

# kubectl get pods
# kubectl port-forward mlapiscikit 8000:80