FROM python:3.14-slim

WORKDIR /app

# PIP_EXTRA_INDEX_URL makes pip prefer PyTorch's CPU-only wheels (~200 MB)
# over the default CUDA build (several GB), which a CPU-only pod can't use anyway.
# TORCH_NUM_THREADS=1: one inference thread per gunicorn worker.
# WEB_CONCURRENCY: number of gunicorn workers (each holds its own copy of torch + the model);
# lower it for small pods:  docker run -e WEB_CONCURRENCY=1 ...
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    TORCH_NUM_THREADS=1 \
    WEB_CONCURRENCY=2

# install dependencies (this layer is cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY sleepnet/ sleepnet/
COPY model_data/sleepnet.pt model_data/

EXPOSE 80

# a long night is scored in a few seconds, but give slow pods some room
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--timeout", "120", "app:app"]
