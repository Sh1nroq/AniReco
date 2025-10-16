import orjson
import re

def json_parser(filepath: str):
    with open(filepath, "r", encoding="utf8") as f:
        data = orjson.loads(f.read())

    # print("mal_id:", data[0]["mal_id"])
    parsed_info = {}

    for record in data:
        title = record.get("title_english", "")
        genres = record.get("genres", [])
        mal_id = record.get("mal_id")
        synopsis = record.get("synopsis", "")
        parsed_info[mal_id] = {"title": title,"genres": [g["name"] for g in genres], "synopsis": synopsis}
    #print(parsed_info, "record")
    return parsed_info