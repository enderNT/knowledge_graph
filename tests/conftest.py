from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.ai_provider import StubAIProvider
from app.config import Settings
from app.main import create_app
from app.store import InMemoryKnowledgeStore


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        embedding_dimensions=16,
    )


@pytest.fixture
def store(settings: Settings) -> InMemoryKnowledgeStore:
    return InMemoryKnowledgeStore(settings)


@pytest.fixture
def client(settings: Settings, store: InMemoryKnowledgeStore) -> TestClient:
    app = create_app(
        settings=settings,
        store=store,
        ai_provider=StubAIProvider(settings),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(settings: Settings) -> dict[str, str]:
    return {"X-API-Key": settings.api_key}
