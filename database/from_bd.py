from sqlalchemy import create_engine
from sqlalchemy import Table, MetaData

from scripts.json_parser import json_parser
from urllib.parse import quote_plus
from sqlalchemy import select


def get_info_from_bd():
    user = quote_plus('postgres')
    password = quote_plus('123321')  # если есть спецсимволы, они будут закодированы
    host = '127.0.0.1'
    port = 5432
    database = 'anime_information'

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        echo=True
    )

    metadata = MetaData()
    table = Table('anime', metadata, autoload_with=engine)

    stmt = select(
        # table.c.id.label("id"),
        table.c.data["title"].astext.label("title"),
        table.c.data["genres"].astext.label("genres"),
        table.c.data["synopsis"].astext.label("synopsis")
    )

    titles,genres, synopsis, id = [], [], [], []

    with engine.connect() as conn:
        result = conn.execute(stmt)
        for row in result:
            if row.synopsis and row.title:
                titles.append(row.title.strip())
                genres.append(row.genres.strip())
                synopsis.append(row.synopsis.strip())
                # id.append(id)

    return titles,genres, synopsis
