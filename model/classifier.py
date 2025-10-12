import torch
from torch import nn


class AnimeRecommender(nn.Module):
    def __init__(self):
        super().__init__()
        ## хз надо придумать какая структура нейронки должна быть ##

    def forward(self, x):
        ## хз мда.... ##
        return x


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
model = AnimeRecommender().to(device)