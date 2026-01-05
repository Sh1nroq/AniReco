from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

_BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    BASE_DIR: Path = _BASE_DIR

    POSTGRES_URL: str = Field(default=...)

    QDRANT_URL: str = "http://localhost:6333"
    COLLECTION_NAME: str = "Embeddings_of_all_anime"
    DEVICE: str = "cuda"

    @property
    def MODEL_PATH(self) -> Path:
        return self.BASE_DIR / "data" / "embeddings" / "anime_recommender_alpha.pt"

    model_config = SettingsConfigDict(
        env_file=_BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()


