import random
import os
import re

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import torch

from src.model.architecture import AnimeRecommender


class SimpleTextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


def clean_synopsis(text):
    if text is None or not isinstance(text, str):
        return ""

    text = re.sub(r'\(Source:.*?\)', '', text)
    text = re.sub(r'\[Written by.*?\]', '', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def format_as_story(title, genres, synopsis):
    synopsis = clean_synopsis(synopsis)

    if isinstance(genres, (list, np.ndarray)):
        genres_list = list(genres)
    else:
        genres_clean = str(genres).translate(str.maketrans('', '', "[]'"))
        genres_list = [g.strip() for g in genres_clean.split(',') if g.strip()]

    if len(genres_list) > 1:
        genres_str = ", ".join(genres_list[:-1]) + " and " + genres_list[-1]
    elif len(genres_list) == 1:
        genres_str = genres_list[0]
    else:
        genres_str = "various themes"

    return f"This is an {genres_str} anime titled {title}. The story follows: {synopsis}"


def preprocessing_triplets(filepath: str, num_triplets=5000):
    df = pd.read_parquet(filepath)
    df = df[df["genres"].notna() & df["synopsis"].notna()].reset_index(drop=True)

    text_list = []
    print("Подготовка текстов...")
    for i in range(len(df)):
        synopsis = clean_synopsis(df['synopsis'].iloc[i])
        title = df['title'].iloc[i]
        genres = df['genres'].iloc[i]

        if random.random() > 0.5:
            text = format_as_story(title, genres, synopsis)
        else:
            text = f"{title}. {synopsis}"

        text_list.append(text)

    n = len(df)
    triplets = []
    print("Начинаю поиск троек...")

    while len(triplets) < num_triplets:
        idx_a = random.randint(0, n - 1)
        genres_a = set(df["genres"].iloc[idx_a])

        candidate_indices = random.sample(range(n), 200)

        pos_idx = None
        neg_idx = None

        for idx_c in candidate_indices:
            if idx_a == idx_c: continue

            genres_c = set(df["genres"].iloc[idx_c])
            intersection = len(genres_a & genres_c)
            union = len(genres_a | genres_c)
            jaccard = intersection / union if union > 0 else 0

            if jaccard >= 0.6 and pos_idx is None:
                pos_idx = idx_c

            elif jaccard < 0.2 and neg_idx is None:
                neg_idx = idx_c

            if pos_idx is not None and neg_idx is not None:
                triplets.append((text_list[idx_a], text_list[pos_idx], text_list[neg_idx]))
                break

    save_path = os.path.join(os.path.dirname(__file__), "../../data/processed/anime_triplets.parquet")
    pd.DataFrame(triplets, columns=["anchor", "positive", "negative"]).to_parquet(save_path, index=False)
    print(f"Готово! Сохранено {len(triplets)} троек.")


def move_to_device(anchor, pos, neg, device):
    def prepare_batch(batch, dev):
        return {k: v.to(dev) for k, v in batch.items()}

    return prepare_batch(anchor, device), prepare_batch(pos, device), prepare_batch(neg, device)


def save_embedding_of_all_anime():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(BASE_DIR, "../../data/embeddings/anime_recommender_MiniLM-L6_v1.pt")
    data_path = os.path.join(BASE_DIR, "../../data/processed/parsed_anime_data.parquet")
    save_path = os.path.join(BASE_DIR, "../../data/embeddings/embedding_of_all_anime_MiniLM.npy")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AnimeRecommender().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    df = pd.read_parquet(data_path)
    texts = []
    for i, row in df.iterrows():
        genres_str = ", ".join(row["genres"]) if isinstance(row["genres"], list) else str(row["genres"])
        text = f"{row['title']}. {genres_str}. {row['synopsis']}"
        texts.append(text)

    dataset = SimpleTextDataset(texts)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_embeddings = []

    print(f"Генерация эмбеддингов для {len(texts)} аниме на {device}...")
    with torch.inference_mode():
        for batch_texts in loader:
            inputs = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            embeddings = model(**inputs)
            all_embeddings.append(embeddings.cpu().numpy())

    embeddings_matrix = np.vstack(all_embeddings).astype("float32")
    np.save(save_path, embeddings_matrix)
    print(f"Готово! Матрица {embeddings_matrix.shape} сохранена в {save_path}")

