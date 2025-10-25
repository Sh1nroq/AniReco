from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import Table, MetaData, Column, Integer, String, TIMESTAMP
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from scripts.json_parser import json_parser
from urllib.parse import quote_plus
from sqlalchemy import Text
from sqlalchemy.dialects.postgresql import JSONB

def info_to_postgres():
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

    anime_table = Table(
        "anime",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("data", JSONB),
        Column("last_updated", TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")),
    )

    metadata.create_all(engine)

    anime_info = json_parser("../data/anime.json")

    with Session(engine) as session:
        for mal_id, info in anime_info.items():
            insert_anime_to_bd = insert(anime_table).values(
                id=mal_id, data=info).on_conflict_do_nothing()
            session.execute(insert_anime_to_bd)
        session.commit()
