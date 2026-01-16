from qdrant_client import QdrantClient

from backend.app.config import settings
import torch
from torch import nn
from transformers import AutoTokenizer
import torch.nn.functional as F

from src.model.architecture import AnimeRecommender


class RecommenderService:

    def __init__(self):
        self.device = settings.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.client = QdrantClient(url="http://localhost:6333")

        self.model = AnimeRecommender().to(self.device)
        self.model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=self.device))
        self.model.eval()
        print("Model loaded successfully")

    def get_embedding(self, text:str):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            query_emb = self.model(**tokens).cpu().numpy()[0].tolist()

        return query_emb