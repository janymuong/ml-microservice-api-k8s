#!/usr/bin/env bash
set -euo pipefail

image=janymuong/mlapiscikit:latest

# start minikube if it isn't running
minikube status >/dev/null 2>&1 || minikube start

# build locally and load into minikube
docker build --tag="$image" .
minikube image rm "$image" >/dev/null 2>&1 || true
minikube image load "$image"

# recreate the pod; pull policy Never because :latest would otherwise
# default to Always and fetch the old image from Docker Hub
kubectl delete pod mlapiscikit --ignore-not-found
kubectl run mlapiscikit \
	--image="$image" --image-pull-policy=Never --port=80

# wait until ready, then list pods
kubectl wait --for=condition=Ready pod/mlapiscikit --timeout=120s
kubectl get pods

# forward the container port to the host
kubectl port-forward pod/mlapiscikit 8000:80
