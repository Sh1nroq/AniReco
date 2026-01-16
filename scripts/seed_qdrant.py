from pathlib import Path

import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def migrate_data_to_qdrant():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DF_DIR = BASE_DIR / "data" / "processed" / "parsed_anime_data_to_qdrant.parquet"
    EMB_DIR = BASE_DIR / "data" / "embeddings" / "embedding_of_all_anime_MiniLM.npy"

    df_anime = pd.read_parquet(DF_DIR)
    embedding_anime = np.load(EMB_DIR)

    df_anime['emb'] = embedding_anime.tolist()
    return df_anime

client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "Embeddings_of_all_anime"

data = migrate_data_to_qdrant()

first_vector = data['emb'].iloc[0]
if hasattr(first_vector, "tolist"):
    actual_dim = len(first_vector.tolist())
else:
    actual_dim = len(first_vector)

print(f"Обнаружен размер эмбеддинга: {actual_dim}")

if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"Старая коллекция удалена.")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=actual_dim, distance=Distance.COSINE),
)
print(f"Коллекция создана с размером вектора {actual_dim}.")

def clean_value(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, list):
        return v
    if np.isscalar(v):
        if pd.isna(v): return None
        if isinstance(v, (np.int64, np.int32)): return int(v)
        if isinstance(v, (np.float64, np.float32)): return float(v)
    return v

points = []
for idx, row in data.iterrows():
    vector = clean_value(row['emb'])

    if len(vector) != actual_dim:
        print(f"Ошибка в строке {idx}: ожидалось {actual_dim}, пришло {len(vector)}")
        continue

    raw_payload = row.drop(['emb', 'mal_id']).to_dict()
    clean_payload = {k: clean_value(v) for k, v in raw_payload.items()}

    points.append(
        PointStruct(
            id=int(row['mal_id']),
            vector=vector,
            payload=clean_payload
        )
    )

batch_size = 100
for i in range(0, len(points), batch_size):
    client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points[i: i + batch_size],
    )

print(f"Успешно загружено {len(points)} точек в Qdrant.")