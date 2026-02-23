# force update
import pytest
from httpx import AsyncClient
from backend.app.main import app


@pytest.mark.asyncio
async def test_get_filters():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/filters")

    assert response.status_code == 200
    data = response.json()
    assert "genres" in data
    assert "themes" in data
    assert isinstance(data["genres"], list)


@pytest.mark.asyncio
async def test_recommend_empty_query():
    payload = {
        "text_query": "naruto",
        "min_score": 8.0
    }

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/recommend", json=payload)

    assert response.status_code in (200, 404)