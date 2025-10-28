from sqlalchemy.orm import Session
from sqlalchemy import (
    Table,
    MetaData,
    Column,
    Integer,
    TIMESTAMP,
    create_engine,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from src.utils.json_utils import json_parser
from urllib.parse import quote_plus
from sqlalchemy.dialects.postgresql import JSONB


def get_info_from_bd():
    user = quote_plus("postgres")
    password = quote_plus("123321")
    host = "127.0.0.1"
    port = 5432
    database = "anime_information"

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}", echo=True
    )

    metadata = MetaData()
    table = Table("anime", metadata, autoload_with=engine)

    stmt = select(
        # table.c.id.label("id"),
        table.c.data["title"].astext.label("title"),
        table.c.data["genres"].astext.label("genres"),
        table.c.data["synopsis"].astext.label("synopsis"),
    )

    titles, genres, synopsis, id = [], [], [], []

    with engine.connect() as conn:
        result = conn.execute(stmt)
        for row in result:
            if row.synopsis and row.title:
                titles.append(row.title.strip())
                genres.append(row.genres.strip())
                synopsis.append(row.synopsis.strip())
                # id.append(id)

    return titles, genres, synopsis


def info_to_postgres():
    user = quote_plus("postgres")
    password = quote_plus("123321")
    host = "127.0.0.1"
    port = 5432
    database = "anime_information"

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}", echo=True
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
            insert_anime_to_bd = (
                insert(anime_table)
                .values(id=mal_id, data=info)
                .on_conflict_do_nothing()
            )
            session.execute(insert_anime_to_bd)
        session.commit()
