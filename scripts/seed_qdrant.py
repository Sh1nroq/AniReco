import re
import os
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from backend.app.config import settings


def extract_year(date_str):
    if not date_str:
        return None
    match = re.search(r"\d{4}", str(date_str))
    if match:
        return int(match.group(0))
    return None


def clean_value(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, list):
        return v
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if np.isscalar(v):
        if pd.isna(v):
            return None
        if isinstance(v, (np.int64, np.int32, np.integer)):
            return int(v)
        if isinstance(v, (np.float64, np.float32, np.floating)):
            return float(v)
    return v


def migrate_to_qdrant(ds, **kwargs):
    BASE_DATA_DIR = "/app/data"

    ti = kwargs.get('ti')
    DF_PATH = None
    EMB_PATH = None

    if ti is not None:
        DF_PATH = ti.xcom_pull(task_id='parse_anime_to_parquet')
        EMB_PATH = ti.xcom_pull(task_id='get_embeddings')

    if not DF_PATH or not os.path.exists(DF_PATH):
        DF_PATH = f"{BASE_DATA_DIR}/processed/parsed_anime_data_{ds}.parquet"
    if not EMB_PATH or not os.path.exists(EMB_PATH):
        EMB_PATH = f"{BASE_DATA_DIR}/embeddings/embedding_of_all_anime_MiniLM_{ds}.npy"

    if not os.path.exists(DF_PATH):
        DF_PATH = f"{BASE_DATA_DIR}/processed/parsed_anime_data.parquet"
    if not os.path.exists(EMB_PATH):
        EMB_PATH = f"{BASE_DATA_DIR}/embeddings/embedding_of_all_anime_MiniLM.npy"

    print(f"Загрузка данных из: {DF_PATH}")

    df_anime = pd.read_parquet(DF_PATH).drop("image_url", axis=1, errors='ignore')
    embedding_anime = np.load(EMB_PATH)

    if "aired" in df_anime.columns:
        df_anime["start_year"] = df_anime["aired"].apply(extract_year)
    else:
        print("Колонка 'aired' не найдена, используем None")
        df_anime["start_year"] = None

    actual_dim = embedding_anime.shape[1]
    print(f"Обнаружен размер эмбеддинга: {actual_dim}")

    client = QdrantClient(url=settings.QDRANT_URL)
    COLLECTION_NAME = settings.COLLECTION_NAME

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Коллекция {COLLECTION_NAME} не найдена. Создаем новую...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=actual_dim, distance=Distance.COSINE),
        )
        print("Коллекция успешно создана.")
    else:
        print(f"Коллекция {COLLECTION_NAME} уже существует. Будем добавлять новые данные.")

    points = []
    df_anime = df_anime.reset_index(drop=True)

    for idx, row in df_anime.iterrows():
        if idx >= len(embedding_anime):
            print(f"Предупреждение: для строки {idx} нет эмбеддинга. Пропуск.")
            continue

        vector = embedding_anime[idx].astype(float).tolist()

        if len(vector) != actual_dim:
            print(f"Ошибка в строке {idx}: ожидалось {actual_dim}, пришло {len(vector)}")
            continue

        raw_payload = row.drop(["mal_id", "aired"], errors='ignore').to_dict()
        clean_payload = {k: clean_value(v) for k, v in raw_payload.items()}

        points.append(
            PointStruct(
                id=int(row["mal_id"]),
                vector=vector,
                payload=clean_payload
            )
        )

    batch_size = 100
    points_added = 0
    points_skipped = 0

    print(f"Всего подготовлено точек: {len(points)}. Начинаем загрузку...")

    for i in range(0, len(points), batch_size):
        batch_points = points[i: i + batch_size]
        batch_ids = [p.id for p in batch_points]

        existing_records = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=batch_ids,
            with_payload=False,
            with_vectors=False
        )

        existing_ids = {record.id for record in existing_records}

        new_points_batch = [p for p in batch_points if p.id not in existing_ids]

        points_skipped += (len(batch_points) - len(new_points_batch))

        if new_points_batch:
            client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=new_points_batch,
            )
            points_added += len(new_points_batch)

    print(f"Миграция завершена.")
    print(f"Добавлено новых записей: {points_added}")
    print(f"Пропущено существующих: {points_skipped}")

    return points_added