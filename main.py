from database.from_bd import get_info_from_bd
from model.utils import preprocessing_data

titles, genres, synopsis = get_info_from_bd()
preprocessing_data(titles, genres, synopsis)


