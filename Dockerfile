FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости, необходимые для faiss и других библиотек
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip и копируем зависимости
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .

# Теперь установка должна пройти успешно
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./scripts ./scripts
COPY ./data/processed ./data/processed
COPY ./data/embeddings ./data/embeddings

ENV PYTHONUNBUFFERED=1

# Команда для запуска (замени main.py на твой входной файл)
ENV PYTHONPATH=/app
CMD ["python", "src/model/inference.py"]

