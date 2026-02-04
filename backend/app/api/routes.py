from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import select, func


from backend.app.db.postgres import AnimeInformation, async_session
from backend.app.schemas import RecommendationRequest, RecommendationResponse
from backend.app.services import get_recommendation
from backend.app.api.deps import get_recommender
from src.model.inference import RecommenderService

router = APIRouter()


@router.post("/recommend", response_model= RecommendationResponse)  # Меняем на POST
async def user_request(
        data: RecommendationRequest,
        recommender: RecommenderService = Depends(get_recommender)
):
    result = await get_recommendation(data, recommender)

    if not result:
        raise HTTPException(status_code=404, detail="Nothing found!")
    return {"model_response": result}
@router.get("/filters")
async def get_filters():
    async with async_session() as session:
        try:
            genres_query = select(func.distinct(func.jsonb_array_elements_text(AnimeInformation.genres)))
            themes_query = select(func.distinct(func.jsonb_array_elements_text(AnimeInformation.themes)))

            genres_res = await session.execute(genres_query)
            themes_res = await session.execute(themes_query)

            genres = sorted([r[0] for r in genres_res if r[0]])
            themes = sorted([r[0] for r in themes_res if r[0]])

            print(f"DEBUG: Found {len(genres)} genres and {len(themes)} themes")

            return {
                "genres": genres,
                "themes": themes
            }
        except Exception as e:
            print(f"SQL Error in /filters: {e}")
            return {"genres": [], "themes": []}