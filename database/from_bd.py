from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import Table, MetaData, Column, Integer, String, TIMESTAMP
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from torch.utils.hipify.hipify_python import meta_data

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
        table.c.data["title"].astext.label("title"),
        table.c.data["synopsis"].astext.label("synopsis")
    )

    X,y = [], []

    with engine.connect() as conn:
        result = conn.execute(stmt)
        for row in result:
            if row.synopsis and row.title:
                X.append(row.synopsis.strip())
                y.append(row.title.strip())

    return X,y
