#!/usr/bin/env bash
set -euo pipefail

image=janymuong/sleep-staging:latest

# start minikube if it isn't running
minikube status >/dev/null 2>&1 || minikube start

# build locally and load into minikube
docker build --tag="$image" .
minikube image rm "$image" >/dev/null 2>&1 || true
minikube image load "$image"

# create or update the Deployment and Service (manifests live in k8s/)
kubectl apply -f k8s/

# the image tag is always :latest, so applying alone would not replace running pods
# that already use an older build; restart them to pick up the image just loaded
kubectl rollout restart deployment/sleep-staging
kubectl rollout status deployment/sleep-staging --timeout=180s
kubectl get pods,svc -l app=sleep-staging

# forward the service port to the host
kubectl port-forward service/sleep-staging 8000:80
