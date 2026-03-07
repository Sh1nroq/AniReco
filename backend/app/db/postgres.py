from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.app.config import settings

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

DATABASE_URL = settings.POSTGRES_URL
engine = create_async_engine(DATABASE_URL)

async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


class AnimeInformation(Base):
    __tablename__ = "AnimeInformation"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    type = Column(String, nullable=False)
    description = Column(String, nullable=False)
    mal_id = Column(BigInteger, unique=True, index=True, nullable=False)
    score = Column(Float, nullable=True)
    image_url = Column(String, nullable=True)
    status = Column(String, nullable=True)

    start_year = Column(Integer, nullable=True)

    popularity = Column(Integer, nullable=True)

    genres = Column(JSONB, default=[])
    themes = Column(JSONB, default=[])
    is_adult = Column(Boolean, default=False)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)
    print("Таблицы созданы")
