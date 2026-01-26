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
        genres = record.get("genres", [])
        synopsis = record.get("synopsis", "")
        type = record.get("type", "")
        score = record.get("score", [])
        date_str = record.get("aired", {}).get("from")

        start_year = None
        if date_str:
            start_year = int(date_str[:4])

        if synopsis is not None:
            synopsis = synopsis.replace("\n\n[Written by MAL Rewrite]", "")

        if score is not None and score > 6.5 and type in ("TV","Movie"):
            parsed_info.append(
                (
                    mal_id,
                    title,
                    [g["name"] for g in genres],
                    synopsis,
                    score,
                    type,
                    start_year
                )
            )
    DIR_BASE = os.path.dirname(os.path.abspath(__file__))
    df = pd.DataFrame(parsed_info, columns=["mal_id","title", "genres", "synopsis", "score", "type", "aired"])
    df.to_parquet(
        os.path.join(DIR_BASE, "../../data/processed/parsed_anime_data.parquet"),
        index=False,
        compression="gzip",
    )





