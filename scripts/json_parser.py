import orjson

def json_parser(filepath: str):
    with open(filepath, "r", encoding="utf8") as f:
        data = orjson.loads(f.read())

    # print("mal_id:", data[0]["mal_id"])
    parsed_info = {}

    for record in data:
        genres = record.get("genres", [])
        mal_id = record.get("mal_id")
        synopsis = record.get("synopsis", "")
        parsed_info[mal_id] = {"genres": [g["name"] for g in genres], "synopsis": synopsis}
    #print(parsed_info, "record")
    return parsed_info