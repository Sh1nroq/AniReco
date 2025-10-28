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
        x = self.linear(x)  # [batch, 256]
        x = F.normalize(x, p=2, dim=1)  # нормализация (единичная длина вектора)
        return x


filepath = "../../data/embeddings/anime_recommender.pt"
filepath_anime = "../../data/processed/faiss_anime_search.parquet"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = PredictionBert().to(device)
model.load_state_dict(torch.load(filepath, map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

embeddings_matrix = np.load("../../data/embeddings/embedding_of_all_anime.npy")

anime = pd.read_parquet(filepath_anime)
anime_titles = anime['title']

dim = embeddings_matrix.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_matrix)
print(f"Добавлено {index.ntotal} аниме в FAISS индекс")

query = "For the agent known as Twilight, no order is too tall if it is for the sake of peace. Operating as Westalis' master spy, Twilight works tirelessly to prevent extremists from sparking a war with neighboring country Ostania. For his latest mission, he must investigate Ostanian politician Donovan Desmond by infiltrating his son's school: the prestigious Eden Academy. Thus, the agent faces the most difficult task of his career: get married, have a child, and play family. Twilight, or Loid Forger, quickly adopts the unassuming orphan Anya to play the role of a six-year-old daughter and prospective Eden Academy student. For a wife, he comes across Yor Briar, an absent-minded office worker who needs a pretend partner of her own to impress her friends. However, Loid is not the only one with a hidden nature. Yor moonlights as the lethal assassin Thorn Princess. For her, marrying Loid creates the perfect cover. Meanwhile, Anya is not the ordinary girl she appears to be; she is an esper, the product of secret experiments that allow her to read minds. Although she uncovers their true identities, Anya is thrilled that her new parents are cool secret agents! She would never tell them, of course. That would ruin the fun. Under the guise of The Forgers, the spy, the assassin, and the esper must act as a family while carrying out their own agendas. Although these liars and misfits are only playing parts, they soon find that family is about far more than blood relations."

tokens = tokenizer(
    query, return_tensors="pt", truncation=True, padding="max_length", max_length=128
).to(device)

with torch.no_grad():
    query_emb = model(**tokens).cpu().numpy()

distances, indices = index.search(query_emb, k=5)

print("\nРекомендации по запросу:")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    title = anime_titles[idx]
    print(f"{rank}. {title} (similarity={dist:.4f})")
