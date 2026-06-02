from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.mcp_backend_client import MCPBackendError
from app.mcp_server import create_app, create_mcp_server


class FakeBackendClient:
    def __init__(self) -> None:
        self.ready = (True, {"status": "ready"})
        self.fragment_result = {
            "status": "completed",
            "episode_id": "ep_1",
            "job_id": "job_1",
            "result": {"episode_id": "ep_1", "domain": "Psicología"},
        }
        self.search_result = {"query": "memoria", "results": []}
        self.upsert_result = {"concept": {"uid": "cn_1"}, "created": True}
        self.link_result = {"status": "linked"}
        self.neighborhood_result = {"concept": {"uid": "cn_1"}, "nodes": [], "relations": [], "claims": [], "episodes": []}
        self.link_error: Exception | None = None

    async def ingest_fragment_and_wait(self, **_: Any) -> dict[str, Any]:
        return self.fragment_result

    async def search_candidates(self, **_: Any) -> dict[str, Any]:
        return self.search_result

    async def upsert_concept(self, **_: Any) -> dict[str, Any]:
        return self.upsert_result

    async def link_concepts(self, **_: Any) -> dict[str, Any]:
        if self.link_error:
            raise self.link_error
        return self.link_result

    async def get_neighborhood(self, **_: Any) -> dict[str, Any]:
        return self.neighborhood_result

    async def check_ready(self) -> tuple[bool, dict[str, Any]]:
        return self.ready

    async def add_knowledge_fragment(self, **_: Any) -> dict[str, Any]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def _settings() -> Settings:
    return Settings(
        app_env="test",
        API_KEY="test-api-key",
        ARCADEDB_ROOT_PASSWORD="test-password",
        MCP_BEARER_TOKEN="test-bearer-token",
        KG_API_KEY="internal-api-key",
        KG_API_BASE_URL="http://backend.test",
    )


def test_mcp_http_requires_bearer_token_for_mcp_only():
    backend = FakeBackendClient()
    app = create_app(settings=_settings(), backend_client=backend)

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200

        missing = client.get("/mcp")
        assert missing.status_code == 401
        assert missing.json() == {"detail": "invalid bearer token"}

        invalid = client.post("/mcp", headers={"Authorization": "Bearer nope"}, json={})
        assert invalid.status_code == 401
        assert invalid.json() == {"detail": "invalid bearer token"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client_session() -> AsyncGenerator[Any]:
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    server = create_mcp_server(settings=_settings(), backend_client=FakeBackendClient())
    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        yield session


@pytest.mark.anyio
async def test_mcp_server_exposes_exactly_five_tools(client_session):
    tools = await client_session.list_tools()

    assert sorted(tool.name for tool in tools.tools) == [
        "add_knowledge_fragment",
        "get_neighborhood",
        "link_concepts",
        "search_candidates",
        "upsert_concept",
    ]

    tool_map = {tool.name: tool for tool in tools.tools}
    assert tool_map["add_knowledge_fragment"].inputSchema["properties"]["text"]["type"] == "string"
    assert tool_map["search_candidates"].inputSchema["properties"]["limit"]["default"] == 10
    assert "from" in tool_map["link_concepts"].inputSchema["properties"]
    assert "depth" in tool_map["get_neighborhood"].inputSchema["properties"]


@pytest.mark.anyio
async def test_mcp_server_translates_tool_results_and_errors():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    backend = FakeBackendClient()
    server = create_mcp_server(settings=_settings(), backend_client=backend)
    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        fragment = await session.call_tool("add_knowledge_fragment", {"text": "Texto"})
        assert fragment.isError in {False, None}
        assert fragment.structuredContent["status"] == "completed"
        assert fragment.structuredContent["job_id"] == "job_1"

        backend.link_error = MCPBackendError("source or target concept not found", status_code=404)
        link = await session.call_tool(
            "link_concepts",
            {"from": "cn_1", "relation": "RELATED_TO", "to": "cn_2"},
        )
        assert link.isError is True
        assert any(
            "source or target concept not found" in getattr(content, "text", "") for content in link.content
        )