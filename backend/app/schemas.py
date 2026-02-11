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
    genres: Optional[list[str]] = None
    type: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    min_score: Optional[float] = None
    popularity: Optional[int] = None
    themes: Optional[list[str]] = None
    include_adult: bool = False
    sort_by: Optional[str] = "relevance"

class RecommendationResponse(BaseModel):
    model_response: list[AnimeSchema]