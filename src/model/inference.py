from qdrant_client import QdrantClient

from backend.app.config import settings
import torch
from torch import nn
from transformers import AutoTokenizer, BertModel
import torch.nn.functional as F


class PredictionBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256)
        )

    def forward(self, **x):
        outputs = self.bert_model(**x)
        x = outputs.pooler_output
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class RecommenderService:

    def __init__(self):
        self.device = settings.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.client = QdrantClient(url="http://localhost:6333")

        self.model = PredictionBert().to(self.device)
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