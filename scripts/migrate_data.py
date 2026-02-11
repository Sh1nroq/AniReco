import asyncio
import os
import pandas as pd
import numpy as np
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import create_async_engine

from backend.app.config import settings
from backend.app.db.postgres import AnimeInformation

engine = create_async_engine(settings.POSTGRES_URL)


async def upload_data():
    DIR_BASE = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(DIR_BASE, "../data/processed/parsed_anime_data.parquet")

    if not os.path.exists(parquet_path):
        print(f"Ошибка: Файл не найден {parquet_path}")
        return

    data = pd.read_parquet(parquet_path)

    data = data[
        (data["score"].notna()) &
        (data["score"] > 6.5) &
        (data["type"].isin(["TV", "Movie", "OVA", "ONA"]))
        ].copy()

    print(f"После фильтрации осталось {len(data)} записей (Score > 6.5, TV/Movie/OVA/ONA)")

    def to_native_list(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    data["genres"] = data["genres"].apply(to_native_list)
    data["themes"] = data["themes"].apply(to_native_list)

    data["synopsis"] = data["synopsis"].str.replace("\n\n[Written by MAL Rewrite]", "", regex=False).fillna("")

    data_to_insert = data.rename(columns={
        "synopsis": "description",
        "aired": "start_year"
    }).to_dict(orient="records")

    for item in data_to_insert:
        item["status"] = "ok"

    batch_size = 500
    total = len(data_to_insert)

    async with engine.begin() as conn:
        for i in range(0, total, batch_size):
            batch = data_to_insert[i: i + batch_size]
            stmt = insert(AnimeInformation)
            await conn.execute(stmt, batch)
            print(f"Загружено: {min(i + batch_size, total)} / {total}")

        print("Все данные успешно загружены в PostgreSQL!")


if __name__ == "__main__":
    asyncio.run(upload_data())