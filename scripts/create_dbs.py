import os
import time
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASS", "postgres")
DB_HOST = "postgres"
DB_PORT = "5432"

DEFAULT_DB = os.getenv("DB_NAME", "postgres")

AIRFLOW_DB_NAME = "airflow_db"


def create_airflow_db():
    print(f"⏳ Подключаемся к Postgres через базу '{DEFAULT_DB}'...")

    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DEFAULT_DB
    )

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (AIRFLOW_DB_NAME,))
    exists = cursor.fetchone()

    if not exists:
        print(f"База '{AIRFLOW_DB_NAME}' не найдена. Создаем...")
        cursor.execute(f"CREATE DATABASE {AIRFLOW_DB_NAME}")
        print(f"База '{AIRFLOW_DB_NAME}' успешно создана!")

        cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {AIRFLOW_DB_NAME} TO {DB_USER}")
    else:
        print(f"База '{AIRFLOW_DB_NAME}' уже существует. Ничего делать не надо.")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    time.sleep(2)
    try:
        create_airflow_db()
    except Exception as e:
        print(f"Ошибка: {e}")