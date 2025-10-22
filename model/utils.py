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

def similarity_anime(anime_genre_1, anime_genre_2):
    if len(set(anime_genre_1) & set(anime_genre_2)):
        return 1
    else: return 0

def preprocessing_data(titles, genres, synopsis, num_pairs = 5000):
    text = [f"{titles}.{synopsis}" for titles, synopsis in zip(titles, synopsis)]
    n = len(titles)
    pairs = []
    for _ in range(num_pairs):

            anime_1, anime_2 = random.sample(range(n), 2)
            print(f"id1:{anime_1}, id2:{anime_2}")
            label = similarity_anime(genres[anime_1], genres[anime_2])
            print(f"label:{label}, genre_1:{genres[anime_1]}, genre_2: {genres[anime_2]}")
            pairs.append((text[anime_1], text[anime_2], label))
            print(f"Overall:{pairs}")
    df = pd.DataFrame(pairs, columns=["text1", "text2", "label"])
    df.to_parquet("../data/anime_pairs.parquet", index=False, compression="gzip")

