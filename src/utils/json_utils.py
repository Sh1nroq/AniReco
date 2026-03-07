import os
import re

import orjson
import pandas as pd

import os
import re
import orjson
import pandas as pd
from airflow.models import TaskInstance


def json_parser(ds, **kwargs):
    current_date = ds

    ti: TaskInstance = kwargs.get('ti')
    input_filepath = None

    if ti is not None:
        input_filepath = ti.xcom_pull(task_id='parse_anime')

    if not input_filepath or not os.path.exists(input_filepath):
        input_filepath = f"/app/data/raw/anime_{current_date}.json"

    if not os.path.exists(input_filepath):
        input_filepath = "/app/data/raw/anime.json"

    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"JSON файл не найден: {input_filepath}")

    with open(input_filepath, "r", encoding="utf8") as f:
        data = orjson.loads(f.read())

    print(f"Всего записей в JSON: {len(data)}")

    mal_ids = [record.get("mal_id") for record in data if record.get("mal_id")]
    unique_mal_ids = set(mal_ids)
    if len(mal_ids) != len(unique_mal_ids):
        print("В исходном JSON найдены дубликаты mal_id!")
        print(f"Всего mal_id: {len(mal_ids)}")
        print(f"Уникальных: {len(unique_mal_ids)}")

    parsed_info = []
    seen_mal_ids = set()

    for record in data:
        mal_id = record.get("mal_id", "")

        if mal_id in seen_mal_ids:
            continue

        title = record.get("title", "")
        synopsis = record.get("synopsis", "")
        type_ = record.get("type", "")
        score = record.get("score", None)
        popularity = record.get("popularity", "")

        images = record.get("images") or {}
        jpg = images.get("jpg") or {}
        image_url = jpg.get("large_image_url")

        aired = record.get("aired") or {}
        date_str = aired.get("from")

        start_year = None
        if date_str:
            try:
                start_year = int(str(date_str)[:4])
            except (ValueError, TypeError, IndexError):
                start_year = None

        raw_genres = record.get("genres") or []
        raw_themes = record.get("themes") or []
        raw_demographics = record.get("demographics") or []

        genres_list = [g.get("name") for g in raw_genres if g.get("name")]
        themes_list = [t.get("name") for t in raw_themes if t.get("name")]
        demo_list = [d.get("name") for d in raw_demographics if d.get("name")]

        final_themes = themes_list + demo_list
        is_adult = "Hentai" in genres_list or "Erotica" in genres_list

        duration_str = record.get("duration", "0 min")
        duration_match = re.search(r"\d+", str(duration_str))  # str() на случай, если там число
        duration_minutes = int(duration_match.group(0)) if duration_match else 0

        if type_ == "OVA" and duration_minutes < 15:
            continue

        if synopsis is not None:
            synopsis = synopsis.replace("\n\n[Written by MAL Rewrite]", "")

        if score is not None and score > 6.5 and type_ in ("TV", "Movie", "OVA", "ONA"):
            parsed_info.append(
                (
                    mal_id,
                    title,
                    genres_list,
                    synopsis,
                    score,
                    type_,
                    start_year,
                    final_themes,
                    popularity,
                    image_url,
                    is_adult,
                )
            )
            seen_mal_ids.add(mal_id)

    print(f"После фильтрации и удаления дубликатов: {len(parsed_info)} записей")

    df = pd.DataFrame(
        parsed_info,
        columns=[
            "mal_id",
            "title",
            "genres",
            "synopsis",
            "score",
            "type",
            "aired",
            "themes",
            "popularity",
            "image_url",
            "is_adult",
        ],
    )

    duplicates_in_df = df.duplicated(subset=["mal_id"], keep=False).sum()
    if duplicates_in_df > 0:
        print(f"Найдено дубликатов в DataFrame: {duplicates_in_df}")
        df = df.drop_duplicates(subset=["mal_id"], keep="first")
        print(f"После удаления: {len(df)} записей")

    output_path = f"/app/data/processed/parsed_anime_data_{current_date}.parquet"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_parquet(
        output_path,
        index=False,
        compression="gzip",
    )

    print(f"Parquet сохранён: {output_path}")
    print(f"Итого уникальных записей: {len(df)}")

    return output_path