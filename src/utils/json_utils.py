import os.path

import orjson
import pandas as pd


def json_parser(filepath: str):
    with open(filepath, "r", encoding="utf8") as f:
        data = orjson.loads(f.read())

    # print("mal_id:", data[0]["mal_id"])
    parsed_info = []

    for record in data:
        mal_id = record.get("mal_id", "")
        title = record.get("title", "")
        synopsis = record.get("synopsis", "")
        type = record.get("type", "")
        score = record.get("score", [])
        date_str = record.get("year")
        popularity = record.get("popularity", "")
        image_url = record.get("images", {}).get("jpg", {}).get("large_image_url")

        raw_genres = record.get("genres") or []
        raw_themes = record.get("themes") or []
        raw_demographics = record.get("demographics") or []

        genres_list = [g.get("name") for g in raw_genres if g.get("name")]
        themes_list = [t.get("name") for t in raw_themes if t.get("name")]
        demo_list = [d.get("name") for d in raw_demographics if d.get("name")]

        final_themes = themes_list + demo_list

        start_year = None
        if date_str:
            start_year = int(date_str[:4])

        if synopsis is not None:
            synopsis = synopsis.replace("\n\n[Written by MAL Rewrite]", "")

        if score is not None and score > 6.5 and type in ("TV", "Movie"):
            parsed_info.append(
                (
                    mal_id,
                    title,
                    genres_list,
                    synopsis,
                    score,
                    type,
                    start_year,
                    final_themes,
                    popularity,
                    image_url
                )
            )
    DIR_BASE = os.path.dirname(os.path.abspath(__file__))
    df = pd.DataFrame(parsed_info, columns=["mal_id","title", "genres", "synopsis", "score", "type", "aired", "themes", "popularity", "image_url"])
    df.to_parquet(
        os.path.join(DIR_BASE, "../../data/processed/parsed_anime_data.parquet"),
        index=False,
        compression="gzip",
    )





