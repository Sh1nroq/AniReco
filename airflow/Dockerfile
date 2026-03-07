# syntax=docker/dockerfile:1
FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y build-essential libpq-dev libgomp1 && apt-get clean

USER airflow

# Layer 1: torch (rarely changes, cached between builds)
RUN --mount=type=cache,target=/home/airflow/.cache/pip,uid=50000 \
    pip install torch torchvision torchaudio

# Layer 2: other deps (changes more often, but torch layer stays cached)
COPY airflow/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/home/airflow/.cache/pip,uid=50000 \
    pip install -r /requirements.txt
