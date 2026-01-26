from typing import Optional

from pydantic import BaseModel, ConfigDict


class AnimeSchema(BaseModel):
    mal_id: int
    title: str
    description: str
    score: Optional[float]
    image_url: Optional[str]
    status: Optional[str]

    model_config = ConfigDict(from_attributes=True)

class RecommendationRequest(BaseModel):
    text_query: str
    genre: Optional[str] = None
    type: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None

class RecommendationResponse(BaseModel):
    model_response: list[AnimeSchema]