import os.path
import re

import orjson
import pandas as pd


def json_parser(filepath: str):
    with open(filepath, "r", encoding="utf8") as f:
        data = orjson.loads(f.read())

    print(f"Всего записей в JSON: {len(data)}")

    mal_ids = [record.get("mal_id") for record in data if record.get("mal_id")]
    unique_mal_ids = set(mal_ids)
    if len(mal_ids) != len(unique_mal_ids):
        print(f"В исходном JSON найдены дубликаты mal_id!")
        print(f"Всего mal_id: {len(mal_ids)}")
        print(f"Уникальных: {len(unique_mal_ids)}")
        print(f"Дубликатов: {len(mal_ids) - len(unique_mal_ids)}")

    parsed_info = []
    seen_mal_ids = set()  # Отслеживаем уже обработанные mal_id

    for record in data:
        mal_id = record.get("mal_id", "")

        # Пропускаем записи с дубликатами mal_id
        if mal_id in seen_mal_ids:
            continue

        title = record.get("title", "")
        synopsis = record.get("synopsis", "")
        type_ = record.get("type", "")
        score = record.get("score", None)
        date_str = record.get("aired", {}).get("from")
        popularity = record.get("popularity", "")
        image_url = record.get("images", {}).get("jpg", {}).get("large_image_url")

        raw_genres = record.get("genres") or []
        raw_themes = record.get("themes") or []
        raw_demographics = record.get("demographics") or []

        genres_list = [g.get("name") for g in raw_genres if g.get("name")]
        themes_list = [t.get("name") for t in raw_themes if t.get("name")]
        demo_list = [d.get("name") for d in raw_demographics if d.get("name")]

        final_themes = themes_list + demo_list

        is_adult = "Hentai" in genres_list or "Erotica" in genres_list

        duration_str = record.get("duration", "0 min")
        duration_match = re.search(r'\d+', duration_str)
        duration_minutes = int(duration_match.group(0)) if duration_match else 0

        if type_ == "OVA" and duration_minutes < 15:
            continue

        start_year = None
        if date_str:
            start_year = int(date_str[:4])

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
                    is_adult
                )
            )
            seen_mal_ids.add(mal_id)

    print(f"После фильтрации и удаления дубликатов: {len(parsed_info)} записей")

    DIR_BASE = os.path.dirname(os.path.abspath(__file__))
    df = pd.DataFrame(
        parsed_info,
        columns=["mal_id", "title", "genres", "synopsis", "score", "type", "aired", "themes", "popularity", "image_url", "is_adult"]
    )

    duplicates_in_df = df.duplicated(subset=['mal_id'], keep=False).sum()
    if duplicates_in_df > 0:
        print(f"Найдено дубликатов в DataFrame: {duplicates_in_df}")
        df = df.drop_duplicates(subset=['mal_id'], keep='first')
        print(f"После удаления: {len(df)} записей")

    output_path = os.path.join(DIR_BASE, "../../data/processed/parsed_anime_data.parquet")
    df.to_parquet(
        output_path,
        index=False,
        compression="gzip",
    )

    print(f"Parquet сохранён: {output_path}")
    print(f"Итого уникальных записей: {len(df)}")
    print(f"Уникальных mal_id: {df['mal_id'].nunique()}")