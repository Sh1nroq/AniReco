from typing import Optional

from pydantic import BaseModel

class AnimeSchema(BaseModel):
    title: str
    description: str
    score: Optional[float]
    image_url: Optional[str]
    status: Optional[str]

    class Config:
        from_attributes = True

class RecommendationRequest(BaseModel):
    text_query: str

class RecommendationResponse(BaseModel):
    model_response: list[AnimeSchema]