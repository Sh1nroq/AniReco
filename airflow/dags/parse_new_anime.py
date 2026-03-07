from datetime import datetime, timedelta
import asyncio

from airflow.operators.python import PythonOperator
from airflow import DAG


def init_db_sync(**context):
    from sqlalchemy.ext.asyncio import create_async_engine
    from backend.app.config import settings
    from backend.app.db.postgres import Base

    async def _init():
        engine = create_async_engine(
            settings.POSTGRES_URL,
            pool_pre_ping=True,
            pool_timeout=30,
        )
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all, checkfirst=True)
            print("Таблицы созданы")
        finally:
            await engine.dispose()

    asyncio.run(_init())

def run_parse_mal(**context):
    from scripts.parser_mal import parse_mal
    ds = context.pop('ds')
    return parse_mal(ds, **context)

def run_json_parser(**context):
    from src.utils.json_utils import json_parser
    ds = context.pop('ds')
    return json_parser(ds, **context)

def run_save_embeddings(**context):
    from src.utils.utils import save_embedding_of_all_anime
    ds = context.pop('ds')
    return save_embedding_of_all_anime(ds, **context)

def run_upload_data(**context):
    from scripts.migrate_data import upload_data
    ds = context.pop('ds')
    asyncio.run(upload_data(ds, **context))

def run_migrate_to_qdrant(**context):
    from scripts.seed_qdrant import migrate_to_qdrant
    ds = context.pop('ds')
    migrate_to_qdrant(ds, **context)


with DAG(
    "anime_full_pipeline",
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    schedule="@quarterly",
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    create_postgres_table = PythonOperator(
        task_id="create_postgres_table",
        python_callable=init_db_sync,
    )
    parse_task = PythonOperator(
        task_id="parse_anime",
        python_callable=run_parse_mal,
    )
    parse_to_parquet = PythonOperator(
        task_id="parse_anime_to_parquet",
        python_callable=run_json_parser,
    )
    get_embeddings = PythonOperator(
        task_id="get_embeddings",
        python_callable=run_save_embeddings,
    )
    migrate_to_postgres = PythonOperator(
        task_id="migrate_to_postgres",
        python_callable=run_upload_data,
    )
    migrate_qdrant = PythonOperator(
        task_id="migrate_to_qdrant",
        python_callable=run_migrate_to_qdrant,
    )

    create_postgres_table >> parse_task
    parse_task >> parse_to_parquet >> get_embeddings
    get_embeddings >> [migrate_to_postgres, migrate_qdrant]