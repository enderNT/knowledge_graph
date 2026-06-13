from __future__ import annotations

import logging

import httpx
import pytest

from app.arcadedb_client import ArcadeDBClient
from app.config import Settings


def _settings(**overrides) -> Settings:
    base = {
        "app_env": "test",
        "API_KEY": "test-api-key",
        "ARCADEDB_ROOT_PASSWORD": "test-password",
        "ARCADEDB_URL": "http://arcadedb.test:2480",
        "ARCADEDB_DATABASE": "knowledge",
    }
    base.update(overrides)
    return Settings(**base)


def _client_with_handler(handler) -> ArcadeDBClient:
    client = ArcadeDBClient(_settings())
    # Swap in a mock transport so we can simulate ArcadeDB error responses.
    client.client = httpx.AsyncClient(
        base_url=client.client.base_url,
        transport=httpx.MockTransport(handler),
    )
    return client


@pytest.mark.anyio
async def test_query_failure_logs_error_body(caplog):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=500,
            json={"error": "java.lang.IllegalArgumentException: bad query"},
            request=request,
        )

    client = _client_with_handler(handler)
    try:
        with caplog.at_level(logging.ERROR, logger="app.arcadedb_client"):
            with pytest.raises(httpx.HTTPStatusError):
                await client.query("SELECT FROM Nope")
    finally:
        await client.close()

    records = [r for r in caplog.records if r.message == "arcadedb error"]
    assert records, "expected an 'arcadedb error' log record"
    record = records[0]
    assert record.status_code == 500
    assert record.operation == "query"
    assert "bad query" in record.body


@pytest.mark.anyio
async def test_successful_query_does_not_log_error(caplog):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json={"result": [{"uid": "x"}]}, request=request)

    client = _client_with_handler(handler)
    try:
        with caplog.at_level(logging.ERROR, logger="app.arcadedb_client"):
            rows = await client.query("SELECT FROM Concept")
    finally:
        await client.close()

    assert rows == [{"uid": "x"}]
    assert not [r for r in caplog.records if r.message == "arcadedb error"]
