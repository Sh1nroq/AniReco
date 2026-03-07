from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# 1. Функция, которую будет выполнять наш шаг
def say_hello():
    print("Привет! Мой первый DAG в AnimeRecomendation работает!")

# 2. Описание самого DAG (его расписания и параметров)
with DAG(
    dag_id='my_first_dag',        # Имя, которое появится в интерфейсе
    start_date=datetime(2023, 1, 1),
    schedule='@daily',    #  Как часто запускать
    catchup=False                  # Не запускать за все прошлые даты
) as dag:

    # 3. Определение задачи (Task)
    hello_task = PythonOperator(
        task_id='hello_task',
        python_callable=say_hello
    )