from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.routes import router
from src.model.inference import RecommenderService

@asynccontextmanager
async def lifespan(app:FastAPI):
    try:
        app.state.recommender = RecommenderService()
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    else:
        print("Загрузка произошла успешно!")
    yield
    print("Завершение!")
app = FastAPI(lifespan=lifespan)
app.include_router(router)