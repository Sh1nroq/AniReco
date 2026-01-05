from fastapi import APIRouter, HTTPException, Depends
from backend.app.schemas import RecommendationRequest
from backend.app.services import get_recommendation
from backend.app.api.deps import get_recommender
from src.model.inference import RecommenderService

router = APIRouter()


@router.post("/recommend")  # Меняем на POST
async def user_request(
        data: RecommendationRequest,
        recommender: RecommenderService = Depends(get_recommender)
):
    result = await get_recommendation(data.text_query, recommender)

    if not result:
        raise HTTPException(status_code=404, detail="Nothing found!")
    return result