from torch import nn
import torch
from transformers import AutoModel
import torch.nn.functional as F

class AnimeRecommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 256)
        )

    def forward(self, **x):
        outputs = self.bert_model(**x)

        last_hidden_state = outputs.last_hidden_state
        attention_mask = x['attention_mask']

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        x = self.head(embeddings)

        x = F.normalize(x, p=2, dim=1)
        return x