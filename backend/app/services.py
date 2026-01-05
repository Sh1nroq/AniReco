from sqlalchemy import select
from backend.app.db.postgres import async_session, AnimeInformation
from backend.app.db.qdrant import get_similar_emb
from src.model.inference import RecommenderService



async def get_recommendation(text_query: str, recommender: RecommenderService):
    async with async_session() as session:
        emb = recommender.get_embedding(text_query)

        anime_ids = get_similar_emb(emb, recommender.client)

        query = select(AnimeInformation).where(AnimeInformation.mal_id.in_(anime_ids))
        result = await session.execute(query)
        anime_list = result.scalars().all()

        anime_dict = {anime.mal_id: anime for anime in anime_list}
        sorted_anime = [anime_dict[m_id] for m_id in anime_ids if m_id in anime_dict]
        return sorted_anime