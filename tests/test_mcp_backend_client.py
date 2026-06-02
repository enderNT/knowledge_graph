from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.mcp_backend_client import (
    MCPBackendAuthError,
    MCPBackendClient,
    MCPBackendError,
    MCPBackendNotFoundError,
)


def _settings(**overrides) -> Settings:
    base_kwargs = {
        "app_env": "test",
        "API_KEY": "test-api-key",
        "ARCADEDB_ROOT_PASSWORD": "test-password",
        "KG_API_KEY": "internal-api-key",
        "KG_API_BASE_URL": "http://backend.test",
        "MCP_POLL_INTERVAL_SECONDS": 0.0,
        "MCP_INGESTION_TIMEOUT_SECONDS": 0.01,
    }
    base_kwargs.update(overrides)
    return Settings(**base_kwargs)


def _response(request: httpx.Request, status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=payload, request=request)


@pytest.mark.asyncio
async def test_ingestion_returns_completed_job_result():
    calls = {"jobs": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/knowledge/fragments":
            return _response(request, 202, {"episode_id": "ep_1", "job_id": "job_1", "status": "queued"})
        if request.url.path == "/v1/jobs/job_1":
            calls["jobs"] += 1
            if calls["jobs"] == 1:
                return _response(request, 200, {"uid": "job_1", "episode_id": "ep_1", "status": "processing"})
            return _response(
                request,
                200,
                {
                    "uid": "job_1",
                    "episode_id": "ep_1",
                    "status": "completed",
                    "result": {"episode_id": "ep_1", "domain": "Psicología", "relations": []},
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        result = await client.ingest_fragment_and_wait(text="Texto")
    finally:
        await client.close()

    assert result == {
        "status": "completed",
        "episode_id": "ep_1",
        "job_id": "job_1",
        "result": {"episode_id": "ep_1", "domain": "Psicología", "relations": []},
    }


@pytest.mark.asyncio
async def test_ingestion_returns_failed_status_without_raising():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/knowledge/fragments":
            return _response(request, 202, {"episode_id": "ep_1", "job_id": "job_1", "status": "queued"})
        if request.url.path == "/v1/jobs/job_1":
            return _response(
                request,
                200,
                {"uid": "job_1", "episode_id": "ep_1", "status": "failed", "error": "llm extraction failed"},
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        result = await client.ingest_fragment_and_wait(text="Texto")
    finally:
        await client.close()

    assert result == {
        "status": "failed",
        "episode_id": "ep_1",
        "job_id": "job_1",
        "error": "llm extraction failed",
    }


@pytest.mark.asyncio
async def test_ingestion_timeout_returns_processing_status():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/knowledge/fragments":
            return _response(request, 202, {"episode_id": "ep_1", "job_id": "job_1", "status": "queued"})
        if request.url.path == "/v1/jobs/job_1":
            return _response(request, 200, {"uid": "job_1", "episode_id": "ep_1", "status": "processing"})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = MCPBackendClient(_settings(MCP_INGESTION_TIMEOUT_SECONDS=0.0), transport=httpx.MockTransport(handler))
    try:
        result = await client.ingest_fragment_and_wait(text="Texto")
    finally:
        await client.close()

    assert result == {"status": "processing", "episode_id": "ep_1", "job_id": "job_1"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (401, MCPBackendAuthError),
        (404, MCPBackendNotFoundError),
        (500, MCPBackendError),
    ],
)
async def test_backend_error_translation(status_code: int, error_type: type[Exception]):
    async def handler(request: httpx.Request) -> httpx.Response:
        return _response(request, status_code, {"detail": f"error-{status_code}"})

    client = MCPBackendClient(_settings(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(error_type) as exc_info:
            await client.search_candidates(query="memoria")
    finally:
        await client.close()

    assert str(exc_info.value) == f"error-{status_code}"