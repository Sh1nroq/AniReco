# syntax=docker/dockerfile:1
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev build-essential && rm -rf /var/lib/apt/lists/*

# Layer 1: torch (rarely changes, cached between builds)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Layer 2: other deps
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY ./src ./src
COPY ./backend ./backend
COPY ./scripts ./scripts
COPY ./data/embeddings/anime_recommender_MiniLM-L6_v1.pt ./data/embeddings
COPY ./data/processed/parsed_anime_data.parquet ./data/processed/

ENV PYTHONPATH=/app

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
