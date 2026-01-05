import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.db.postgres import init_db

async def main():
    print("Начинаю инициализацию базы данных...")
    await init_db()
    print("База данных успешно инициализирована!")

if __name__ == "__main__":
    asyncio.run(main())