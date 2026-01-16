import re
from sqlalchemy import select, or_
from backend.app.db.postgres import async_session, AnimeInformation
from backend.app.db.qdrant import get_similar_emb

def extract_keywords(text: str):
    clean_text = re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower())
    words = clean_text.split()
    return [w for w in words if len(w) >= 3]


async def get_keyword_results(session, keywords):
    if not keywords:
        return []

    conditions = []
    for word in keywords:
        conditions.append(AnimeInformation.title.ilike(f"%{word}%"))
        conditions.append(AnimeInformation.description.ilike(f"%{word}%"))

    query = select(AnimeInformation).where(or_(*conditions)).limit(50)
    result = await session.execute(query)
    return result.scalars().all()


async def get_recommendation(data, recommender):
    text = data.text_query
    keywords = extract_keywords(text)

    async with async_session() as session:
        emb = recommender.get_embedding(text)
        qdrant_ids = get_similar_emb(emb, recommender.client)

        sql_keywords = keywords[:15]
        keyword_anime = await get_keyword_results(session, sql_keywords)

        scores = {}

        for i, m_id in enumerate(qdrant_ids):
            scores[m_id] = scores.get(m_id, 0) + (20 - i)

        for anime in keyword_anime:
            m_id = anime.mal_id
            title_lower = anime.title.lower()
            desc_lower = anime.description.lower()

            bonus = 0
            for kw in sql_keywords:
                if kw in title_lower:
                    bonus += 5
                elif kw in desc_lower:
                    bonus += 1

            scores[m_id] = scores.get(m_id, 0) + bonus

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        top_ids = sorted_ids[:10]

        query = select(AnimeInformation).where(AnimeInformation.mal_id.in_(top_ids))
        result = await session.execute(query)
        anime_list = result.scalars().all()

        anime_dict = {anime.mal_id: anime for anime in anime_list}
        final_results = [anime_dict[m_id] for m_id in top_ids if m_id in anime_dict]

        return final_results