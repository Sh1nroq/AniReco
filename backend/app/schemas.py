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

class RecommendationResponse(BaseModel):
    model_response: list[AnimeSchema]