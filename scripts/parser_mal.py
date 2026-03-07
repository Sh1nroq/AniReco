import os
import json
import requests
from time import sleep
from airflow.models import Variable


def parse_mal(ds, **kwargs):
    # pages 1165 by date 10.10.2025
    RAW_DATA_DIR = "/app/data/raw"
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    page = int(Variable.get("mal_last_page", default_var=1))

    current_date = ds
    results = []

    try:
        while True:
            url = f"https://api.jikan.moe/v4/anime?page={page}"
            response = requests.get(url)

            if response.status_code == 429:
                print("Слишком много запросов, спим 30 сек...")
                sleep(30)
                continue

            if response.status_code != 200:
                print(f"Ошибка на странице {page}: {response.status_code}")
                break

            info = response.json()
            anime_list = info.get("data", [])

            if not anime_list:
                print("Парсинг завершен, новых данных нет.")
                break

            results.extend(anime_list)
            print(f"Страница {page} загружена, всего в текущей сессии: {len(results)}")

            if page % 10 == 0:
                Variable.set("mal_last_page", page)

            page += 1
            sleep(1)

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        Variable.set("mal_last_page", page)
        raise e

    Variable.set("mal_last_page", page)

    filename = f"anime_{current_date}.json"
    filepath = os.path.join(RAW_DATA_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Файл сохранен: {filepath}")
    return filepath
