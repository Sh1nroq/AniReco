import random
from transformers import AutoTokenizer
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

def tokenizer_for_nn(object):
    tokenized = tokenizer(
        object,
        truncation=True,
        padding="max_length",
        return_tensors="pt"  # если хочешь сразу тензоры PyTorch
    )
    return {k: v.squeeze(0) for k, v in tokenized.items()}

def similarity_anime(genres1, genres2):
    set1 = set([g.strip().lower() for g in genres1.split(",") if g.strip()])
    set2 = set([g.strip().lower() for g in genres2.split(",") if g.strip()])

    if not set1 or not set2:
        return 0

    jaccard = len(set1 & set2) / len(set1 | set2)

    return 1 if jaccard >= 0.4 else 0

def preprocessing_data(titles, genres, synopsis, num_pairs = 5000):
    text = [f"{titles}.{synopsis}" for titles, synopsis in zip(titles, synopsis)]
    n = len(titles)
    pairs = []
    while len(pairs) < num_pairs:

            anime_1, anime_2 = random.sample(range(n), 2)
            print(f"id1:{anime_1}, id2:{anime_2}")
            label = similarity_anime(genres[anime_1], genres[anime_2])
            print(f"label:{label}, genre_1:{genres[anime_1]}, genre_2: {genres[anime_2]}")

            if label == 1 and sum(l == 1 for _, _, l in pairs) >= num_pairs // 2:
                continue
            if label == 0 and sum(l == 0 for _, _, l in pairs) >= num_pairs // 2:
                continue

            pairs.append((text[anime_1], text[anime_2], label))
            print(f"Overall:{pairs}")
    df = pd.DataFrame(pairs, columns=["text1", "text2", "label"])
    df.to_parquet("data/anime_pairs.parquet", index=False, compression="gzip")

    print(df['label'].value_counts(normalize=True))

def move_to_device(item_1, item_2, y, device):
    item_1 = {k: v.to(device) for k, v in item_1.items()}
    item_2 = {k: v.to(device) for k, v in item_2.items()}
    y = y.to(device)
    return item_1, item_2, y


