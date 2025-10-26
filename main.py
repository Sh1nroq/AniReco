import pandas as pd

from database.from_bd import get_info_from_bd
from model.utils import preprocessing_data
from model.utils import get_anime_search_table

titles, genres, synopsis = get_info_from_bd()
# preprocessing_data(titles, genres, synopsis)
# get_anime_search_table(titles, synopsis)

df = pd.read_parquet("data/faiss_anime_search.parquet")
print(df['title'])
