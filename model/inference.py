import faiss
import pandas as pd
import torch
from transformers import AutoTokenizer, BertModel
import torch.nn.functional as F
import numpy as np

class PredictionBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 256)

    def forward(self, **x):
        outputs = self.bert_model(**x)
        x = self.dropout(outputs.pooler_output)  # [batch, 768]
        x = self.linear(x)                       # [batch, 256]
        x = F.normalize(x, p=2, dim=1)           # нормализация (единичная длина вектора)
        return x

filepath = "anime_recommender.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = PredictionBert().to(device)
model.load_state_dict(torch.load(filepath, map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

df = pd.read_parquet("../data/faiss_anime_search.parquet")

anime_texts = [f"{t}. {s}" for t, s in zip(df['title'], df['synopsis'])]
anime_titles = df['title'].tolist()

embeddings_list = []

with torch.no_grad():
    for text in anime_texts:
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        ).to(device)

        emb = model(**tokenized)
        embeddings_list.append(emb.cpu().numpy())

embeddings_matrix = np.vstack(embeddings_list).astype('float32')

print(f"Матрица эмбеддингов: {embeddings_matrix.shape}")  # например (1000, 256)

dim = embeddings_matrix.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_matrix)
print(f"Добавлено {index.ntotal} аниме в FAISS индекс")

query = "In the year 2010, the Holy Empire of Britannia is establishing itself as a dominant military nation, starting with the conquest of Japan. Renamed to Area 11 after its swift defeat, Japan has seen significant resistance against these tyrants in an attempt to regain independence. Lelouch Lamperouge, a Britannian student, unfortunately finds himself caught in a crossfire between the Britannian and the Area 11 rebel armed forces. He is able to escape, however, thanks to the timely appearance of a mysterious girl named C.C., who bestows upon him Geass, the Power of Kings. Realizing the vast potential of his newfound power of absolute obedience, Lelouch embarks upon a perilous journey as the masked vigilante known as Zero, leading a merciless onslaught against Britannia in order to get revenge once and for all."
tokens = tokenizer(
    query,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=128
).to(device)

with torch.no_grad():
    query_emb = model(**tokens).cpu().numpy()

distances, indices = index.search(query_emb, k=10)

print("\nРекомендации по запросу:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    title = anime_titles[idx]
    print(f"{rank}. {title} (similarity={dist:.4f})")
