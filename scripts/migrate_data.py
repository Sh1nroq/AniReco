import orjson
from sqlalchemy import create_engine, URL, insert
from backend.app.db.postgres import AnimeInformation
from sqlalchemy import MetaData

url_object = URL.create(
    "postgresql+psycopg2",
    username="postgres",
    password="123321",
    host="localhost",
    database="AnirecoDB",
)

engine  = create_engine(url_object)

with engine.begin() as conn:
    with open('../data/raw/anime.json', "rb") as file:
        data = orjson.loads(file.read())
        for record in data:
            title = record.get("title", "")
            type = record.get("type", "")
            description = record.get("synopsis", "")
            mal_id = record.get("mal_id", "")
            score = record.get("score", "")
            image_url = record.get("images", {}).get("jpg", {}).get("image_url")
            if score is not None and score > 6.5 and type in ("TV", "Movie") and description is not None:
                metadata_obj = MetaData()
                description = description.replace("\n\n[Written by MAL Rewrite]", "")
                stmt = insert(AnimeInformation).values(title=title, description=description, mal_id = mal_id, score = score, image_url = image_url, status="ok")
                result = conn.execute(stmt)

