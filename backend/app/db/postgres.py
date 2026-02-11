from datetime import datetime

from sqlalchemy import Float, DateTime, URL, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncAttrs, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped
from sqlalchemy.orm import mapped_column

from backend.app.config import settings

from sqlalchemy import text

DATABASE_URL = settings.POSTGRES_URL
engine = create_async_engine(DATABASE_URL)

async_session = async_sessionmaker(engine, expire_on_commit=False)

class Base(AsyncAttrs, DeclarativeBase):
    pass

class AnimeInformation(Base):
    __tablename__ = "AnimeInformation"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    type: Mapped[str]
    description: Mapped[str]
    mal_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=True)
    image_url: Mapped[str] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(nullable=True)

    start_year: Mapped[int] = mapped_column(nullable=True)

    popularity: Mapped[int] = mapped_column(nullable=True)

    genres: Mapped[list[str]] = mapped_column(JSONB, default=[])
    themes: Mapped[list[str]] = mapped_column(JSONB, default=[])

    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Таблицы созданы")
