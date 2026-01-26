from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# 3. Добавляем само Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
