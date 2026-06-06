from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
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
        self.learning_context_result = {
            "query": "memoria",
            "domain_hint": "Psicología",
            "status": "sparse",
            "primary_concepts": [
                {
                    "uid": "cn_1",
                    "canonical_name": "Memoria episódica",
                    "domain": "Psicología",
                    "description": "Sistema de memoria autobiográfica.",
                    "retrieval_score": 0.98,
                    "retrieval_reason": "alias",
                    "quality_flags": [],
                }
            ],
            "relations": [],
            "claims": [],
            "episodes": [],
            "warnings": ["no_supporting_claims"],
            "debug": {
                "candidate_count": 1,
                "selected_concept_uids": ["cn_1"],
                "selection_reasons": ["cn_1:alias:0.98:exact_or_alias_match"],
            },
        }
        self.tutor_context_result = {
            "resolved_reference": {
                "input_type": "query",
                "input_value": "memoria",
                "resolved_concept_uid": "cn_1",
                "resolved_concept_name": "Memoria episódica",
                "resolved_episode_id": None,
                "resolved_job_id": None,
                "resolution_reason": "alias",
            },
            "status": "ok",
            "concepts": [
                {
                    "uid": "cn_1",
                    "canonical_name": "Memoria episódica",
                    "domain": "Psicología",
                    "description": "Sistema de memoria autobiográfica.",
                    "aliases": ["recuerdo autobiográfico"],
                }
            ],
            "claims": [
                {
                    "uid": "cl_1",
                    "text": "La memoria episódica recupera experiencias personales con contexto temporal.",
                    "confidence": 0.91,
                    "evidence_episode_ids": ["ep_1"],
                }
            ],
            "relations": [],
            "source_fragments": [
                {
                    "episode_id": "ep_1",
                    "text": "Fragmento base.",
                    "status": "processed",
                    "source_type": "manual_input",
                    "tags": ["Psicología"],
                    "language": "es",
                }
            ],
            "evidence": [{"subject_type": "claim", "subject_uid": "cl_1", "episode_id": "ep_1"}],
            "warnings": [],
            "failure_reason": None,
        }
        self.upsert_result = {"concept": {"uid": "cn_1"}, "created": True}
        self.link_result = {"status": "linked"}
        self.neighborhood_result = {"concept": {"uid": "cn_1"}, "nodes": [], "relations": [], "claims": [], "episodes": []}
        self.link_error: Exception | None = None

    async def ingest_fragment_and_wait(self, **_: Any) -> dict[str, Any]:
        return self.fragment_result

    async def search_candidates(self, **_: Any) -> dict[str, Any]:
        return self.search_result

    async def get_learning_context(self, **_: Any) -> dict[str, Any]:
        return self.learning_context_result

    async def get_tutor_context(self, **_: Any) -> dict[str, Any]:
        return self.tutor_context_result

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
async def test_mcp_server_exposes_exactly_seven_tools(client_session):
    tools = await client_session.list_tools()

    assert sorted(tool.name for tool in tools.tools) == [
        "add_knowledge_fragment",
        "get_learning_context",
        "get_tutor_context",
        "get_neighborhood",
        "link_concepts",
        "search_candidates",
        "upsert_concept",
    ]

    tool_map = {tool.name: tool for tool in tools.tools}
    assert tool_map["add_knowledge_fragment"].inputSchema["properties"]["text"]["type"] == "string"
    assert tool_map["search_candidates"].inputSchema["properties"]["limit"]["default"] == 10
    assert tool_map["get_learning_context"].inputSchema["properties"]["candidate_limit"]["default"] == 8
    assert tool_map["get_tutor_context"].inputSchema["properties"]["include_evidence"]["default"] is True
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

        learning_context = await session.call_tool(
            "get_learning_context",
            {"query": "memoria", "domain_hint": "Psicología"},
        )
        assert learning_context.isError in {False, None}
        assert learning_context.structuredContent["status"] == "sparse"
        assert learning_context.structuredContent["primary_concepts"][0]["uid"] == "cn_1"

        tutor_context = await session.call_tool(
            "get_tutor_context",
            {"query": "memoria"},
        )
        assert tutor_context.isError in {False, None}
        assert tutor_context.structuredContent["status"] == "ok"
        assert tutor_context.structuredContent["resolved_reference"]["resolved_concept_uid"] == "cn_1"

        backend.link_error = MCPBackendError("source or target concept not found", status_code=404)
        link = await session.call_tool(
            "link_concepts",
            {"from": "cn_1", "relation": "RELATED_TO", "to": "cn_2"},
        )
        assert link.isError is True
        assert any(
            "source or target concept not found" in getattr(content, "text", "") for content in link.content
        )


@pytest.mark.anyio
async def test_mcp_streamable_http_client_works_with_documented_url():
    pytest.importorskip("mcp")
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    app = create_app(settings=_settings(), backend_client=FakeBackendClient())
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"Authorization": "Bearer test-bearer-token"},
        ) as http_client:
            async with streamable_http_client(
                "http://testserver/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()

    assert sorted(tool.name for tool in tools.tools) == [
        "add_knowledge_fragment",
        "get_learning_context",
        "get_tutor_context",
        "get_neighborhood",
        "link_concepts",
        "search_candidates",
        "upsert_concept",
    ]
