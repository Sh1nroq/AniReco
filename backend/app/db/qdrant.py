from qdrant_client import QdrantClient, models
from backend.app.config import settings


def get_similar_emb(query_vector, client: QdrantClient, filters: dict = None, limit=100):
    qdrant_filter = None

    if filters:
        conditions = []
        if filters.get("genre"):

            genre_val = filters["genre"]
            if genre_val.islower():
                if genre_val == "slice of life":
                    genre_val = "Slice of Life"
                elif genre_val == "sci-fi":
                    genre_val = "Sci-Fi"
                else:
                    genre_val = genre_val.title()

            conditions.append(
                models.FieldCondition(key="genres", match=models.MatchValue(value=genre_val))
            )

        if filters.get("type"):
            type_map = {"tv": "TV", "movie": "Movie", "ova": "OVA", "special": "Special"}
            type_input = filters["type"].lower()
            type_val = type_map.get(type_input, filters["type"].title())

            conditions.append(
                models.FieldCondition(key="type", match=models.MatchValue(value=type_val))
            )

        year_min = filters.get("year_min")
        year_max = filters.get("year_max")

        if year_min is not None or year_max is not None:
            kwargs = {}
            if year_min is not None: kwargs['gte'] = int(year_min)
            if year_max is not None: kwargs['lte'] = int(year_max)

            conditions.append(
                models.FieldCondition(key="start_year", range=models.Range(**kwargs))
            )

        if conditions:
            qdrant_filter = models.Filter(must=conditions)
        # print(f"DEBUG: Применены фильтры: {conditions}")

    search_result = client.query_points(
        collection_name="Embeddings_of_all_anime",
        query=query_vector,
        query_filter=qdrant_filter,
        with_payload=False,
        limit=limit
    )

    return [(hit.id, hit.score) for hit in search_result.points]