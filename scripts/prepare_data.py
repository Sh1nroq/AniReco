from src.utils.utils import preprocessing_data, get_augmentation
from src.utils.utils import save_embedding_of_all_anime
from src.utils.json_utils import json_parser

def main():

    json_parser("../data/raw/anime.json")
    # preprocessing_data("../data/processed/parsed_anime_data.parquet")
    # save_embedding_of_all_anime()
    # get_augmentation()

if __name__ == "__main__":
    main()
