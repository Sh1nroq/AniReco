from fastapi import Request
from src.model.inference import RecommenderService

def get_recommender(request: Request) -> RecommenderService:
    return request.app.state.recommender