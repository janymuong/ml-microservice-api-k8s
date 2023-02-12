# Operationalize a Machine Learning Microservice API - Kubernetes

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/janymuong/ml-microservice-api-k8s/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/janymuong/ml-microservice-api-k8s/tree/main)


## Project Overview

```Cloud DevOps Engineering```

The premise and first instruction is - operationalizing a Machine Learning Microservice API with `Kubernetes`. 

The entire project is based on a pre-trained, `sklearn` model that has been trained to predict housing prices in Boston according to several features, such as average rooms in a home and data about highway access, teacher-to-pupil ratios, and so on. You can read more about the data, which was initially taken from Kaggle, on [the data source site](https://www.kaggle.com/c/boston-housing).

This is a `Python and Flask Implementation`. The application `app.py` serves out predictions (inference) about housing prices through API calls. This project could be extended to any pre-trained machine learning model, such as those for image recognition and data labeling.


### Project Tasks

The project goal is to operationalize this working, machine learning microservice using [kubernetes](https://kubernetes.io/), which is an open-source system for automating the management of containerized applications. 

The project tasks are listed out below:
- Test project code using linting
- Complete a Dockerfile to containerize this application
- Deploy containerized application using Docker and make a prediction
- Improve the log statements in the source code for this application
- Configure Kubernetes and create a Kubernetes cluster
- Deploy a container using Kubernetes and make a prediction
- Upload a complete Github repo with CircleCI to indicate that code has been tested

You can find a detailed [project rubric, here](https://review.udacity.com/#!/rubrics/2576/view).


> **Note**: The final implementation of the project will showcase abilities to operationalize production microservices.



---
## Setup the Environment

`Virtual Environments` - It's recommended to leverage a virtual environment whenever using Python for projects. This keeps your dependencies for each project separate and organized. Instructions for setting up a virtual environment for your platform can be found in the [python docs](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

- Create a siloed virtual environment with Python 3.7 and **activate** it. You should have Python 3.7 available in your host/local machine. 
Check the Python path using `which python3`
```bash
python3 -m pip install --user virtualenv 
# use a command similar to this one to create environment:
python3 -m virtualenv --python=<path-to-python3.7> .devops
source .devops/bin/activate
```
> Alternatively, you could setup the virtualenv via `make setup`. [this](./Makefile) is from a directive in `Makefile`.

- Run `make install` to install the necessary dependencies. This will install all relevant pip packages for the project.

### Running `app.py`

1. Standalone:  `python app.py`
2. Run in Docker:  `./run_docker.sh`
3. Run in Kubernetes:  `./run_kubernetes.sh`

### Kubernetes Steps

- Setup and Configure Docker locally
- Setup and Configure Kubernetes locally
- Create Flask app in Container
- Run via kubectl


---
## Appendix
This holds information for some highlighted regular files & directory files in the root project repository.

### `appendix a` - File Info:
- [.circleci/config](./.circleci/config.yml) - Configuration file for CircleCI CI/CD tooling.
- [model_data](./model_data) - Scikit Learn Dataset for the ML model.
- [output_txt_files](./output_txt_files) - Text output from CLI/shell commnds and/or scripts run against containers and clusters.
- [app.py](./app.py) - The Python - Flask application that serves out API calls to the model.
- [Dockerfile](./Dockerfile) - Dockerfile with Python base image and commnds to run app in-container.
- [Makefile](./Makefile) - Makefile file for environment setup and linting in `hadolint` and `pylint`.
- [requirements.txt](./requirements.txt) - File for `pip` packages/dependencies.


### `appendix b` - Helper Scripts:
`make_prediction.sh` `run_docker.sh` `run_kubernetes.sh` `upload_docker.sh`

These are bash scripts that you can use in place of multiple/chained shell commands. The scripts do exactly what you think they do.

`resize.sh` - A bash script to resize an `AWS Cloud9` environment, if one opts to use a virtual machine for a flexible setup. 
> usage: run this in the terminal: `bash resize.sh intended_cloud9_volume_size` - size in Gigabytes.
>
> running `bash resize.sh` uses 100GB by default.