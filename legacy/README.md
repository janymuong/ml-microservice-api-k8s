# Legacy projects

[boston-sklearn](./boston-sklearn) is the first version of this repository: a scikit-learn model that predicts Boston house prices, served by Flask, with its own `Dockerfile`, `requirements.txt`, training script and captured Docker/Kubernetes output. It is kept unchanged for reference and is no longer built by the root `Makefile`, CI or scripts.

To run it, `cd boston-sklearn` and build from that directory (`docker build -t mlapiscikit .`).
