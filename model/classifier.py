import torch
from torch import nn
from torch.nn.functional import embedding
from transformers import AutoTokenizer, BertModel
from database.from_bd import get_info_from_bd
from sklearn.model_selection import train_test_split
from datasets.my_anime_dataset import MyAnimeDataset
from utils import preprocessing_data
from utils import similarity_anime
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

filename = '../data/anime_pairs.parquet'

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class AnimeRecommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        # self.lin_1 = nn.Linear(768, 1024)


    def forward(self, x):
        outputs = self.bert_model(**x)
        x = outputs
        # x = self.lin_1(x)
        return x

model = AnimeRecommender().to(device)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

dataset = MyAnimeDataset(filename, tokenizer)

train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

margin = 1.0
learning_rate = 1e-03
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
def fine_tuning(model, dataloader, optimizer, margin):
    model.train()

    for item_1, item_2, y in tqdm(dataloader):
        item_1, item_2, y = item_1.to(device), item_2.to(device), y.to(device)

        token_text_1 = model(**item_1).pooler_output
        token_text_2 = model(**item_2).pooler_output
        print(token_text_1.shape, token_text_2.shape)

        cosine_similarity = F.cosine_similarity(token_text_1, token_text_2, dim=-1)
        loss = y * (1 - cosine_similarity)**2 + (1 - y) * (torch.clamp(cosine_similarity - margin, min=0))**2

        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

        tqdm.write(f"Loss: {loss.item():.4f}")
