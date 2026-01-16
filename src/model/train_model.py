import os.path
from torch.amp import autocast, GradScaler
import torch
from torch import nn
from transformers import AutoTokenizer

from src.model.architecture import AnimeRecommender
from src.model.my_anime_dataset import MyAnimeDataset
from src.utils.utils import move_to_device
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(BASE_DIR, "../../data/processed/anime_triplets.parquet")

df = pd.read_parquet(filename)

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


model = AnimeRecommender().to(device)

for name, param in model.bert_model.named_parameters():
    if "encoder.layer" in name:
        param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

train_dataset = MyAnimeDataset(train_df, tokenizer)
val_dataset = MyAnimeDataset(val_df, tokenizer)

batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

weight_decay = 0.05
num_epochs = 15
margin = 0.5
max_length = 256
optimizer = torch.optim.AdamW([
    {'params': model.bert_model.parameters(), 'lr': 1e-6},
    {'params': model.head.parameters(), 'lr': 1e-4}
], weight_decay=0.01)

total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(0.2 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)


def fine_tuning(model, train_dataloader, optimizer, scheduler, device, margin=1.0, accum_steps=4, epoch=0):
    num_layers = len(model.bert_model.encoder.layer)
    layers_to_unfreeze = min(2 * (epoch + 1), num_layers)
    for i, layer in enumerate(model.bert_model.encoder.layer):
        requires_grad = i >= num_layers - layers_to_unfreeze
        for param in layer.parameters():
            param.requires_grad = requires_grad

    criterion = nn.TripletMarginLoss(margin=margin)
    model.train()
    scaler = GradScaler()
    optimizer.zero_grad()
    total_loss = 0.0
    num_batches = 0

    for i, (anchor, positive, negative) in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
        anchor, positive, negative = move_to_device(anchor, positive, negative, device)

        device_type = "cuda" if "cuda" in str(device) else "cpu"

        with autocast(device_type=device_type):
            emb_a = model(**anchor)
            emb_p = model(**positive)
            emb_n = model(**negative)
            loss = criterion(emb_a, emb_p, emb_n)

        loss = loss / accum_steps
        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(train_dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * accum_steps
        num_batches += 1

    return total_loss / num_batches


def validate(model, val_dataloader, device, margin=1.0):
    model.eval()
    total_loss = 0
    correct_rankings = 0
    total_samples = 0

    criterion = nn.TripletMarginLoss(margin=margin)

    with torch.no_grad():
        for anchor, positive, negative in tqdm(val_dataloader, desc="Validating"):
            anchor, positive, negative = move_to_device(anchor, positive, negative, device)

            emb_a = model(**anchor)
            emb_p = model(**positive)
            emb_n = model(**negative)

            loss = criterion(emb_a, emb_p, emb_n)
            total_loss += loss.item()

            d_pos = torch.norm(emb_a - emb_p, p=2, dim=1)
            d_neg = torch.norm(emb_a - emb_n, p=2, dim=1)
            correct_rankings += (d_pos < d_neg).sum().item()
            total_samples += d_pos.size(0)

    avg_loss = total_loss / len(val_dataloader)
    accuracy = (correct_rankings / total_samples) * 100
    print(f"Validation Accuracy (Correct Ranking): {accuracy:.2f}%")

    return avg_loss
for epoch in range(num_epochs):
    if epoch % 2 == 0 and epoch > 0:
        num_to_unfreeze = epoch // 2 * 2
        print(f"Размораживаем верхние {num_to_unfreeze} слоя(ев)")

    train_loss = fine_tuning(model, train_dataloader, optimizer, scheduler, device, margin, epoch=epoch)
    val_loss = validate(model, val_dataloader, device,margin)
    print(
        f"\nEpoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}"
    )

torch.save(model.state_dict(), os.path.join(BASE_DIR, "../../data/embeddings/anime_recommender_MiniLM-L6_v1.pt"))
