# Makefile
## The Makefile includes instructions on environment setup and lint tests
# Create and activate a virtual environment
# Install dependencies in requirements.txt
# Dockerfile should pass hadolint
# app.py, train.py and the sleepnet package should pass pylint
# pytest runs the API, preprocessing and data-loading tests

setup:
	# Create python virtualenv & source it
	# source ~/.devops/bin/activate
	python3 -m venv ~/.devops

install:
	# This should be run from inside a virtualenv
	# the CPU-only torch wheel is ~200 MB instead of several GB with CUDA
	pip install --upgrade pip &&\
		pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu

	# install hadolint for linting in sourced env (.devops)
	wget -O ./hadolint https://github.com/hadolint/hadolint/releases/download/v1.16.3/hadolint-Linux-x86_64 &&\
	chmod +x ./hadolint

test:
	# API tests, preprocessing checks, and the Sleep-EDF loader tested on generated EDF files
	# This should be run from inside a virtualenv
	python -m pytest -vv tests

train:
	# Train on the recordings in model_data/sleep-edf (get them with ./fetch_data.sh)
	python train.py

lint:
	# See local hadolint install instructions:   https://github.com/hadolint/hadolint
	# This is linter for Dockerfiles
	./hadolint Dockerfile
	# This is a linter for Python source code linter: https://www.pylint.org/
	# This should be run from inside a virtualenv
	pylint --disable=R,C,W1203,W1202 app.py train.py sleepnet

all: install lint test
