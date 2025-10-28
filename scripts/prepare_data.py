from src.utils.utils import preprocessing_data, get_anime_search_table
from src.db.db_utils import get_info_from_bd
from src.utils.utils import save_embedding_of_all_anime

def main():
    # titles, genres, synopsis = get_info_from_bd()
    # preprocessing_data(titles, genres, synopsis)
    # get_anime_search_table(titles, synopsis)
    save_embedding_of_all_anime()

if __name__ == "__main__":
    main()