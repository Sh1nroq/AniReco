import requests
from time import sleep
import json
import os

os.makedirs("../data", exist_ok=True)

page = 1
results = []
while True:
    url = f"https://api.jikan.moe/v4/anime?page={page}"
    response = requests.get(url)
    print(response.text)
    if response.status_code != 200:
        print(f"Ошибка на странице {page}: {response.status_code}")
        break

    info = response.json()
    print(info)
    anime_list = info.get("data", [])

    if not anime_list:
        print("Аниме успешно получены!")
        break

    results.extend(anime_list)
    print(f"Страница {page} загружена, всего аниме: {len(results)}")

    page += 1
    sleep(1)

with open("../data/anime.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
