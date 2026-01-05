from qdrant_client import QdrantClient

def get_similar_emb(query, client):

    search_result = client.query_points(
        collection_name="Embeddings_of_all_anime",
        query=query,
        with_payload=False,
        limit=5
    ).points

    ids = [hit.id for hit in search_result]
    return ids

