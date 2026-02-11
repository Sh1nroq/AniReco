from backend.app.db.postgres import init_db
from src.utils.utils import preprocessing_triplets
from src.utils.utils import save_embedding_of_all_anime
from src.utils.json_utils import json_parser

def main():
    json_parser("../data/raw/anime.json")
    # preprocessing_triplets("../data/processed/parsed_anime_data.parquet", num_triplets= 10000)
    save_embedding_of_all_anime()

if __name__ == "__main__":
    main()
