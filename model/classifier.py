import torch
from torch import nn
from transformers import AutoTokenizer, BertModel

class AnimeRecommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        ## хз надо придумать какая структура нейронки должна быть ##

    def forward(self, x):
        ## хз мда.... ##
        x = self.bert_model(x)
        return x


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
model = AnimeRecommender().to(device)