import torch

class MyAnimeDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.full_text = data
        self.data_len = len(data)
        anchor = self.full_text["anchor"].tolist()
        positive = self.full_text["positive"].tolist()
        negative = self.full_text["negative"].tolist()

        self.tokenized_anchor = tokenizer(
            anchor, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )
        self.tokenized_positive = tokenizer(
            positive, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )
        self.tokenized_negative = tokenizer(
            negative, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
        )

    def __getitem__(self, index):
        item_anchor = {k: v[index] for k, v in self.tokenized_anchor.items()}
        item_positive = {k: v[index] for k, v in self.tokenized_positive.items()}
        item_negative = {k: v[index] for k, v in self.tokenized_negative.items()}
        return item_anchor, item_positive, item_negative

    def __len__(self):
        return self.data_len