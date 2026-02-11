import re
from sqlalchemy import select, or_
from backend.app.db.postgres import async_session, AnimeInformation
from backend.app.db.qdrant import get_similar_emb


def extract_keywords(text: str):
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower())
    STOP_WORDS = {
        'anime', 'story', 'follows', 'characters', 'plot', 'centers',
        'world', 'life', 'finds', 'series', 'everything', 'things',
        'years', 'high', 'school', 'striving', 'intense', 'each',
        'both', 'their', 'driven', 'sense', 'testing', 'lives',
        'want', 'find', 'mood', 'about', 'with'
    }
    words = text.split()
    return [w for w in words if len(w) > 3 and w not in STOP_WORDS]


async def get_keyword_results(session, keywords):
    if not keywords: return []
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

    user_filters = {
        "genres": data.genres,
        "themes": data.themes,
        "type": data.type,
        "year_min": data.year_min,
        "year_max": data.year_max,
        "min_score": data.min_score,
        "include_adult": data.include_adult
    }

    async with async_session() as session:
        emb = recommender.get_embedding(text)

        qdrant_results = get_similar_emb(emb, recommender.client, filters=user_filters)

        sql_keywords = keywords[:5]
        keyword_anime = []
        if len(keywords) < 7:
            keyword_anime = await get_keyword_results(session, sql_keywords)

        scores = {}

        THRESHOLD = 0.40
        for i, (m_id, q_score) in enumerate(qdrant_results):
            if q_score < THRESHOLD:
                continue
            scores[m_id] = scores.get(m_id, 0) + (1.0 / (i + 60))

        for i, anime in enumerate(keyword_anime):
            if anime.mal_id in scores or len(keywords) < 3:
                scores[anime.mal_id] = scores.get(anime.mal_id, 0) + (1.0 / (i + 60))

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        if not sorted_ids and qdrant_results:
            sorted_ids = [res[0] for res in qdrant_results]

        if not sorted_ids:
            return []

        query = select(AnimeInformation).where(AnimeInformation.mal_id.in_(sorted_ids))
        result = await session.execute(query)
        anime_list = result.scalars().all()

        filtered_list = []
        for a in anime_list:
            if data.year_min and (a.start_year or 0) < data.year_min: continue
            if data.year_max and (a.start_year or 0) > data.year_max: continue

            if data.min_score and (a.score or 0) < data.min_score: continue

            if data.type:
                if a.type not in data.type: continue

            if data.genres:
                if not all(g in a.genres for g in data.genres): continue

            if data.themes:
                if not all(t in a.themes for t in data.themes): continue

            filtered_list.append(a)

        anime_dict = {a.mal_id: a for a in filtered_list}

        final_results = []
        seen_titles = set()

        for m_id in sorted_ids:
            if m_id not in anime_dict: continue
            anime = anime_dict[m_id]

            title_stub = anime.title[:7].lower()
            if title_stub not in seen_titles:
                final_results.append(anime)
                seen_titles.add(title_stub)

            if len(final_results) >= 50:
                break

            if data.sort_by == "rating":
                final_results.sort(key=lambda x: x.score or 0, reverse=True)
            elif data.sort_by == "popularity":
                final_results.sort(key=lambda x: x.popularity if x.popularity is not None else 999999)

        return final_results