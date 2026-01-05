from datetime import datetime

from sqlalchemy import Float, DateTime, URL, BigInteger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncAttrs, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped
from sqlalchemy.testing.schema import mapped_column
from sqlalchemy.util.concurrency import asyncio

url_object = URL.create(
    "postgresql+asyncpg",
    username="postgres",
    password="123321",
    host="localhost",
    database="AnirecoDB",
)

engine = create_async_engine(url_object)

async_session = async_sessionmaker(engine, expire_on_commit=False)

class Base(AsyncAttrs, DeclarativeBase):
    pass

class AnimeInformation(Base):
    __tablename__ = "AnimeInformation"

    id: Mapped[int] = mapped_column(primary_key = True)
    title: Mapped[str]
    description: Mapped[str]
    mal_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    score: Mapped[float] = mapped_column(Float, nullable = True)
    image_url: Mapped[str] = mapped_column(nullable = True)
    status: Mapped[str] = mapped_column(nullable = True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Таблицы созданы")
