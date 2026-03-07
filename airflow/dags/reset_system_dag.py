from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
from qdrant_client import QdrantClient

try:
    from backend.app.config import settings
except ImportError:
    class Settings:
        QDRANT_URL = "http://anireco_qdrant:6333"
        COLLECTION_NAME = "anime_collection"


    settings = Settings()


def reset_sql_database():
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')

    tables = ["AnimeInformation"]
    for table in tables:
        try:
            sql = f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;"
            pg_hook.run(sql)
            print(f"Таблица {table} успешно очищена.")
        except Exception as e:
            print(f"Ошибка при очистке таблицы {table}: {e}")


def reset_qdrant_collection():
    client = QdrantClient(url=settings.QDRANT_URL)
    collection_name = settings.COLLECTION_NAME

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Коллекция Qdrant '{collection_name}' удалена.")
    else:
        print(f"Коллекция '{collection_name}' не найдена, удалять нечего.")


def reset_airflow_variables():
    variable_name = "mal_last_page"
    Variable.set(variable_name, 1)
    print(f"Переменная Airflow '{variable_name}' установлена в 1.")


with DAG(
        dag_id='system_full_reset',
        start_date=datetime(2023, 1, 1),
        schedule_interval=None,
        catchup=False,
        tags=['service', 'cleanup'],
) as dag:
    t1 = PythonOperator(
        task_id='reset_postgres',
        python_callable=reset_sql_database
    )

    t2 = PythonOperator(
        task_id='reset_qdrant',
        python_callable=reset_qdrant_collection
    )

    t3 = PythonOperator(
        task_id='reset_variables',
        python_callable=reset_airflow_variables
    )

    [t1, t2, t3]