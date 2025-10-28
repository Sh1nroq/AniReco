import pandas as pd

from src.db.db_utils import get_info_from_bd

titles, genres, synopsis = get_info_from_bd()
# preprocessing_data(titles, genres, synopsis)
# get_anime_search_table(titles, synopsis)

df = pd.read_parquet("data/faiss_anime_search.parquet")
print(df['title'])
