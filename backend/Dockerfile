FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./backend ./backend
COPY ./scripts ./scripts
COPY ./data/embeddings/anime_recommender_MiniLM-L6_v1.pt ./data/embeddings
COPY ./data/processed/parsed_anime_data.parquet ./data/processed/

ENV PYTHONPATH=/app

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]